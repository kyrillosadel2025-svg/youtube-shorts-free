#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path.cwd()
WORK = ROOT / "work"
OUT = ROOT / "output"
WORK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

URL = os.environ.get("YOUTUBE_URL", "").strip()
API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
CLIPS_COUNT = int(os.environ.get("CLIPS_COUNT", "3") or "3")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")

if not URL:
    raise SystemExit("Missing YOUTUBE_URL")
if not API_KEY:
    raise SystemExit("Missing GEMINI_API_KEY secret")

def run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)

def capture(cmd):
    print("+", " ".join(cmd))
    return subprocess.check_output(cmd, text=True).strip()

def ts_to_seconds(ts):
    parts = ts.replace(",", ".").split(":")
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h = "0"
        m, s = parts
    else:
        raise ValueError(ts)
    return int(h) * 3600 + int(m) * 60 + float(s)

def seconds_to_ts(sec):
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    sec -= h * 3600
    m = int(sec // 60)
    s = sec - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def clean_vtt(vtt_text):
    lines = vtt_text.splitlines()
    chunks = []
    current = None

    for line in lines:
        line = line.strip()
        if "-->" in line:
            left, right = line.split("-->", 1)
            start = left.strip().split()[0]
            end = right.strip().split()[0]
            current = {"start": start, "end": end, "text": []}
            chunks.append(current)
        elif current and line and not line.startswith(("WEBVTT", "NOTE", "Kind:", "Language:")):
            # Remove VTT tags and duplicated auto-caption markup.
            txt = re.sub(r"<[^>]+>", "", line)
            txt = re.sub(r"\s+", " ", txt).strip()
            if txt and (not current["text"] or current["text"][-1] != txt):
                current["text"].append(txt)

    cleaned = []
    last_text = None
    for ch in chunks:
        text = " ".join(ch["text"]).strip()
        if not text:
            continue
        # Auto captions often repeat cumulative text; keep only meaningful unique cues.
        if text == last_text:
            continue
        last_text = text
        cleaned.append({
            "start": ts_to_seconds(ch["start"]),
            "end": ts_to_seconds(ch["end"]),
            "text": text
        })
    return cleaned

def transcript_for_prompt(cues, max_chars=450000):
    lines = []
    total = 0
    for c in cues:
        line = f"[{seconds_to_ts(c['start'])} --> {seconds_to_ts(c['end'])}] {c['text']}"
        if total + len(line) + 1 > max_chars:
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)

def call_gemini(prompt):
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{MODEL}:generateContent?key={API_KEY}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json"
        }
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)

    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected Gemini response: {json.dumps(data)[:2000]}") from e

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Defensive cleanup if a model ever wraps JSON.
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))

def normalize_clips(data, video_duration):
    clips = data.get("clips", [])
    result = []
    used = []

    for c in clips:
        try:
            start = float(c["start_seconds"])
            duration = float(c["duration"])
        except Exception:
            continue

        # Hard guardrail requested by the workflow.
        duration = min(45.0, max(35.0, duration))
        start = max(0.0, start)
        if video_duration > 0 and start + duration > video_duration:
            start = max(0.0, video_duration - duration)

        end = start + duration

        # Reject strong overlaps.
        overlap = False
        for a, b in used:
            intersection = max(0.0, min(end, b) - max(start, a))
            if intersection > 5.0:
                overlap = True
                break
        if overlap:
            continue

        used.append((start, end))
        result.append({
            "start_seconds": round(start, 3),
            "duration": round(duration, 3),
            "title": str(c.get("title", "")).strip(),
            "hook": str(c.get("hook", "")).strip(),
            "reason": str(c.get("reason", "")).strip(),
        })
        if len(result) >= CLIPS_COUNT:
            break

    if not result:
        raise RuntimeError("Gemini did not return usable clips.")
    return result

def safe_name(s, fallback):
    s = re.sub(r'[\\/:*?"<>|]+', "", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return (s[:80] or fallback)

# 1) Download the source video.
video_tpl = str(WORK / "source.%(ext)s")
run([
    "yt-dlp",
    "--no-playlist",
    "--js-runtimes", "deno",
    "--remote-components", "ejs:npm",
    "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
    "--merge-output-format", "mp4",
    "-o", video_tpl,
    URL,
])

video_files = sorted(WORK.glob("source.*"))
video_files = [p for p in video_files if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}]
if not video_files:
    raise RuntimeError("Video download failed.")
video = video_files[0]

# 2) Download existing or auto-generated subtitles. Arabic first, English fallback.
run([
    "yt-dlp",
    "--no-playlist",
    "--skip-download",
    "--write-subs",
    "--write-auto-subs",
    "--sub-format", "vtt",
    "--sub-langs", "ar.*,en.*",
    "-o", str(WORK / "subs.%(ext)s"),
    URL,
])

vtts = sorted(WORK.glob("subs*.vtt"))
if not vtts:
    raise RuntimeError(
        "No Arabic/English YouTube captions were found. "
        "This free MVP needs captions on the source video."
    )

# Prefer Arabic, otherwise use first available.
preferred = [p for p in vtts if ".ar" in p.name.lower()]
vtt = preferred[0] if preferred else vtts[0]
cues = clean_vtt(vtt.read_text(encoding="utf-8", errors="ignore"))
if not cues:
    raise RuntimeError("Caption file was found but no transcript cues could be parsed.")

transcript = transcript_for_prompt(cues)

# 3) Read source duration.
duration_text = capture([
    "ffprobe", "-v", "error",
    "-show_entries", "format=duration",
    "-of", "default=noprint_wrappers=1:nokey=1",
    str(video)
])
video_duration = float(duration_text)

# 4) Ask Gemini to select the moments.
prompt = f"""
You are editing a long-form video into YouTube Shorts.

Select exactly {CLIPS_COUNT} strong, non-overlapping clips from the timestamped transcript.

Hard requirements:
- Each selected clip must be between 35 and 45 seconds.
- Never exceed 45 seconds.
- Prefer a complete thought that can stand alone without earlier context.
- Start close to the beginning of a sentence.
- End after a complete sentence or complete idea.
- Avoid greetings, channel intros, sponsor reads, outros, and filler.
- Prefer strong hooks, useful insights, surprising facts, concise stories, lessons, mistakes,
  memorable opinions, or emotionally engaging moments.
- Do not invent timestamps that are outside the transcript.
- The output must be JSON only.

Return exactly this structure:
{{
  "clips": [
    {{
      "start_seconds": 0.0,
      "duration": 40.0,
      "title": "short Arabic title if the transcript is Arabic",
      "hook": "short on-screen hook",
      "reason": "brief reason"
    }}
  ]
}}

TRANSCRIPT:
{transcript}
""".strip()

selection = call_gemini(prompt)
clips = normalize_clips(selection, video_duration)

# 5) Render vertical shorts.
manifest = {"source_url": URL, "model": MODEL, "clips": []}

for i, clip in enumerate(clips, start=1):
    title = safe_name(clip["title"], f"short-{i}")
    outfile = OUT / f"{i:02d}-{title}.mp4"

    # Free MVP: vertical 9:16 center-crop, no face tracking.
    vf = (
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920"
    )
    run([
        "ffmpeg", "-y",
        "-ss", str(clip["start_seconds"]),
        "-i", str(video),
        "-t", str(clip["duration"]),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "21",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(outfile),
    ])

    manifest["clips"].append({
        **clip,
        "file": outfile.name,
    })

(OUT / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print(json.dumps(manifest, ensure_ascii=False, indent=2))
