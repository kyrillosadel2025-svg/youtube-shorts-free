#!/usr/bin/env python3
import json
import math
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

print("OpenCV:", getattr(cv2, "__version__", "unknown"))
print("OpenCV module:", getattr(cv2, "__file__", "unknown"))

if not hasattr(cv2, "CascadeClassifier"):
    raise RuntimeError(
        "OpenCV installation is incomplete: cv2.CascadeClassifier is missing. "
        "Use opencv-python-headless==4.10.0.84."
    )

ROOT = Path.cwd()
WORK = ROOT / "work"
OUT = ROOT / "output"
WORK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

YOUTUBE_URL = os.environ.get("YOUTUBE_URL", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
CLIPS_COUNT = int(os.environ.get("CLIPS_COUNT", "3") or "3")
YOUTUBE_COOKIES_FILE = os.environ.get("YOUTUBE_COOKIES_FILE", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small").strip()

OUT_W = 1080
OUT_H = 1920

if not YOUTUBE_URL:
    raise SystemExit("Missing YOUTUBE_URL")
if not GEMINI_API_KEY:
    raise SystemExit("Missing GEMINI_API_KEY")

def run(cmd):
    print("\n+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)

def capture(cmd):
    print("\n+", " ".join(map(str, cmd)), flush=True)
    return subprocess.check_output([str(x) for x in cmd], text=True).strip()

def sec_to_ass_time(sec):
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    sec -= h * 3600
    m = int(sec // 60)
    sec -= m * 60
    s = int(sec)
    cs = int(round((sec - s) * 100))
    if cs >= 100:
        s += 1
        cs = 0
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def clean_filename(text, fallback):
    text = re.sub(r'[\\/:*?"<>|]+', "", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:80] or fallback

def escape_ass(text):
    text = (text or "").replace("\\", r"\\")
    text = text.replace("{", r"\{").replace("}", r"\}")
    return text.replace("\n", r"\N")

def wrap_caption(text, max_chars=26):
    words = text.split()
    if not words:
        return ""
    lines, current = [], []
    for w in words:
        candidate = " ".join(current + [w])
        if len(candidate) > max_chars and current:
            lines.append(" ".join(current))
            current = [w]
        else:
            current.append(w)
    if current:
        lines.append(" ".join(current))
    return r"\N".join(lines[:3])

def call_gemini(prompt):
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        },
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"Gemini API failed HTTP {e.code}\nModel: {GEMINI_MODEL}\n{body[:5000]}"
        ) from e

    text = data["candidates"][0]["content"]["parts"][0]["text"]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise RuntimeError("Gemini returned invalid JSON:\n" + text[:3000])
        return json.loads(m.group(0))

def transcribe_with_whisper(video_path):
    print("\n===== WHISPER TRANSCRIPTION =====\n", flush=True)
    audio = WORK / "audio.wav"
    run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", audio
    ])

    from faster_whisper import WhisperModel

    model = None
    last_error = None

    for attempt in range(1, 5):
        try:
            print(
                f"Loading Whisper model '{WHISPER_MODEL}' (attempt {attempt}/4)...",
                flush=True
            )
            model = WhisperModel(
                WHISPER_MODEL,
                device="cpu",
                compute_type="int8",
                download_root=str(WORK / "whisper-models"),
                local_files_only=False,
            )
            break
        except Exception as exc:
            last_error = exc
            print(f"Whisper model load failed: {str(exc)[:1200]}", flush=True)
            if attempt < 4:
                wait_seconds = 70 * attempt
                print(f"Waiting {wait_seconds}s before retry...", flush=True)
                time.sleep(wait_seconds)

    if model is None:
        raise RuntimeError(
            "Could not load/download the Whisper model after retries. "
            "If the log contains HTTP 429, add a free Hugging Face access token "
            "as GitHub secret HF_TOKEN. Last error: " + str(last_error)
        )
    segments, info = model.transcribe(
        str(audio),
        beam_size=5,
        vad_filter=True,
        word_timestamps=False,
    )
    cues = []
    for seg in segments:
        text = (seg.text or "").strip()
        if text:
            cues.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "text": text,
            })
    if not cues:
        raise RuntimeError("Whisper returned no transcript.")
    print(f"Detected language: {info.language}; segments: {len(cues)}", flush=True)
    return cues

def transcript_for_prompt(cues, max_chars=360000):
    lines, total = [], 0
    for c in cues:
        line = f"[{c['start']:.2f}-{c['end']:.2f}] {c['text']}"
        if total + len(line) + 1 > max_chars:
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)

def normalize_clips(data, video_duration):
    result, used = [], []
    for c in data.get("clips", []):
        try:
            start = float(c["start_seconds"])
            duration = float(c["duration"])
        except Exception:
            continue

        duration = max(35.0, min(45.0, duration))
        start = max(0.0, start)
        if video_duration > 0 and start + duration > video_duration:
            start = max(0.0, video_duration - duration)
        end = start + duration

        if any(max(0.0, min(end, b) - max(start, a)) > 5.0 for a, b in used):
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

def build_ass_for_clip(cues, clip_start, clip_duration, hook, out_path):
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {OUT_W}
PlayResY: {OUT_H}
ScaledBorderAndShadow: yes
WrapStyle: 2

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Caption,DejaVu Sans,58,&H00FFFFFF,&H000000FF,&H00000000,&H78000000,-1,0,0,0,100,100,0,0,1,5,0,2,80,80,260,1
Style: Hook,DejaVu Sans,66,&H00FFFFFF,&H000000FF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,6,0,8,90,90,180,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    events = []

    if hook:
        events.append(
            f"Dialogue: 1,{sec_to_ass_time(0)},{sec_to_ass_time(min(3.0, clip_duration))},Hook,,0,0,0,,{escape_ass(wrap_caption(hook, 22))}"
        )

    clip_end = clip_start + clip_duration
    for cue in cues:
        if cue["end"] <= clip_start or cue["start"] >= clip_end:
            continue
        start = max(0.0, cue["start"] - clip_start)
        end = min(clip_duration, cue["end"] - clip_start)
        if end - start < 0.15:
            continue
        text = wrap_caption(cue["text"], 28)
        events.append(
            f"Dialogue: 0,{sec_to_ass_time(start)},{sec_to_ass_time(end)},Caption,,0,0,0,,{escape_ass(text)}"
        )

    out_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")

def smart_reframe(input_clip, output_silent):
    """
    Hybrid smart camera for both real people and animation.

    Priority:
    1) Human face detection (OpenCV Haar cascade).
    2) Motion/subject tracking fallback when faces are not found.
    3) Safe centered framing for static scenes.
    """
    cap = cv2.VideoCapture(str(input_clip))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {input_clip}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if W <= 0 or H <= 0:
        raise RuntimeError("Invalid source video dimensions")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_silent), fourcc, fps, (OUT_W, OUT_H))
    if not writer.isOpened():
        raise RuntimeError("Could not create VideoWriter")

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError("Could not load OpenCV Haar face cascade")

    face_detect_every = max(1, int(round(fps / 6)))
    motion_scale = min(1.0, 640.0 / max(W, H))

    smooth_center = 0.075
    smooth_zoom = 0.045
    dead_zone_x = W * 0.025
    dead_zone_y = H * 0.025

    x_center = W / 2
    y_center = H / 2
    crop_h = float(H)
    target_x = x_center
    target_y = y_center
    target_h = crop_h

    last_face_center = None
    no_face_scans = 0
    prev_gray_small = None
    motion_center = None
    motion_strength = 0.0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if motion_scale < 1.0:
            small = cv2.resize(frame, None, fx=motion_scale, fy=motion_scale, interpolation=cv2.INTER_AREA)
        else:
            small = frame

        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.GaussianBlur(gray_small, (7, 7), 0)
        motion_center = None
        motion_strength = 0.0

        if prev_gray_small is not None:
            diff = cv2.absdiff(gray_small, prev_gray_small)
            _, mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.dilate(mask, kernel, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                candidates = []
                frame_area_small = mask.shape[0] * mask.shape[1]
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < frame_area_small * 0.003:
                        continue
                    x, y, w, h = cv2.boundingRect(cnt)
                    candidates.append((area, x + w/2, y + h/2))
                if candidates:
                    candidates.sort(reverse=True, key=lambda c: c[0])
                    top = candidates[:3]
                    total_area = sum(c[0] for c in top)
                    if total_area > 0:
                        mx = sum(c[0] * c[1] for c in top) / total_area
                        my = sum(c[0] * c[2] for c in top) / total_area
                        inv = 1.0 / motion_scale
                        motion_center = (mx * inv, my * inv)
                        motion_strength = min(1.0, total_area / (frame_area_small * 0.18))

        prev_gray_small = gray_small
        faces = []

        if frame_idx % face_detect_every == 0:
            detect_scale = min(1.0, 960.0 / max(W, H))
            if detect_scale < 1.0:
                detect_frame = cv2.resize(frame, None, fx=detect_scale, fy=detect_scale, interpolation=cv2.INTER_AREA)
            else:
                detect_frame = frame
            gray = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            found = face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(35, 35), flags=cv2.CASCADE_SCALE_IMAGE)
            inv = 1.0 / detect_scale
            for (x, y, w, h) in found:
                x1, y1 = int(x*inv), int(y*inv)
                x2, y2 = int((x+w)*inv), int((y+h)*inv)
                x1 = max(0, min(x1, W-1)); y1 = max(0, min(y1, H-1))
                x2 = max(x1+1, min(x2, W)); y2 = max(y1+1, min(y2, H))
                area = (x2-x1)*(y2-y1)
                faces.append((area, (x1+x2)/2, (y1+y2)/2, x1, y1, x2, y2))

            if faces:
                no_face_scans = 0
                faces.sort(reverse=True, key=lambda item: item[0])
                chosen = faces[0]
                if last_face_center is not None:
                    nearest = min(faces, key=lambda f: (f[1]-last_face_center[0])**2 + (f[2]-last_face_center[1])**2)
                    if nearest[0] >= faces[0][0] * 0.50:
                        chosen = nearest
                group = [chosen]
                others = [f for f in faces if f is not chosen]
                if others:
                    second = min(others, key=lambda f: abs(f[1]-chosen[1]))
                    if abs(second[1]-chosen[1]) < W*0.44 and second[0] > chosen[0]*0.42:
                        group.append(second)

                min_x = min(f[3] for f in group); min_y = min(f[4] for f in group)
                max_x = max(f[5] for f in group); max_y = max(f[6] for f in group)
                target_x = (min_x + max_x)/2
                target_y = ((min_y + max_y)/2) + H*0.055
                union_w = max_x-min_x; union_h = max_y-min_y
                if len(group) == 1:
                    desired_h = max(union_h/0.27, union_w/0.31)
                    min_crop_ratio = 0.68
                else:
                    desired_h = max(union_h/0.31, union_w/0.46)
                    min_crop_ratio = 0.78
                target_h = float(np.clip(desired_h, H*min_crop_ratio, H))
                last_face_center = (chosen[1], chosen[2])
            else:
                no_face_scans += 1

        if no_face_scans >= 2 and motion_center is not None:
            mx, my = motion_center
            motion_weight = 0.45 + 0.30 * motion_strength
            target_x = target_x*(1.0-motion_weight) + mx*motion_weight
            target_y = target_y*(1.0-motion_weight) + my*motion_weight
            target_h = max(target_h, H*0.78)
        elif no_face_scans >= 5 and motion_center is None:
            target_x = W/2
            target_y = H/2
            target_h = H*0.94

        dx = target_x - x_center
        dy = target_y - y_center
        if abs(dx) < dead_zone_x: dx = 0.0
        if abs(dy) < dead_zone_y: dy = 0.0

        x_center += dx * smooth_center
        y_center += dy * smooth_center
        crop_h += (target_h - crop_h) * smooth_zoom
        crop_h = float(np.clip(crop_h, H*0.66, H))
        crop_w = crop_h * OUT_W / OUT_H
        if crop_w > W:
            crop_w = float(W)
            crop_h = crop_w * OUT_H / OUT_W

        crop_w_i = max(2, int(round(crop_w)))
        crop_h_i = max(2, int(round(crop_h)))
        x1 = int(round(x_center - crop_w_i/2))
        y1 = int(round(y_center - crop_h_i/2))
        x1 = max(0, min(x1, W-crop_w_i))
        y1 = max(0, min(y1, H-crop_h_i))
        roi = frame[y1:y1+crop_h_i, x1:x1+crop_w_i]
        if roi.size == 0:
            roi = frame
        out_frame = cv2.resize(roi, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
        writer.write(out_frame)

        frame_idx += 1
        if frame_idx % max(1, int(fps*10)) == 0:
            mode = 'face' if no_face_scans < 2 else ('motion' if motion_center is not None else 'center')
            print(f"Smart reframe: {frame_idx}/{total} frames | mode={mode}", flush=True)

    writer.release()
    cap.release()

def render_short(source_video, cues, clip, index):
    title = clean_filename(clip["title"], f"short-{index}")
    base_clip = WORK / f"clip_{index:02d}_base.mp4"
    tracked_silent = WORK / f"clip_{index:02d}_tracked_silent.mp4"
    ass_file = WORK / f"clip_{index:02d}.ass"
    final_out = OUT / f"{index:02d}-{title}.mp4"

    run([
        "ffmpeg", "-y",
        "-ss", str(clip["start_seconds"]),
        "-i", source_video,
        "-t", str(clip["duration"]),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        base_clip,
    ])

    print(f"\n===== SMART CAMERA {index} =====\n", flush=True)
    smart_reframe(base_clip, tracked_silent)

    build_ass_for_clip(
        cues,
        clip["start_seconds"],
        clip["duration"],
        clip.get("hook", ""),
        ass_file,
    )

    # Use video from smart crop + audio from base clip + captions.
    run([
        "ffmpeg", "-y",
        "-i", tracked_silent,
        "-i", base_clip,
        "-vf", f"ass={ass_file}",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        "-movflags", "+faststart",
        final_out,
    ])

    return final_out

# ----------------- Download source -----------------

YT_COMMON = [
    "yt-dlp",
    "--js-runtimes", "deno",
    "--remote-components", "ejs:npm",
    "--no-playlist",
]
if YOUTUBE_COOKIES_FILE:
    YT_COMMON += ["--cookies", YOUTUBE_COOKIES_FILE]

print("\n===== DOWNLOAD VIDEO =====\n", flush=True)
run(YT_COMMON + [
    "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
    "--merge-output-format", "mp4",
    "-o", str(WORK / "source.%(ext)s"),
    YOUTUBE_URL,
])

video_files = [
    p for p in WORK.glob("source.*")
    if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}
]
if not video_files:
    raise RuntimeError("Video download failed.")
source_video = sorted(video_files)[0]

video_duration = float(capture([
    "ffprobe", "-v", "error",
    "-show_entries", "format=duration",
    "-of", "default=noprint_wrappers=1:nokey=1",
    source_video,
]))

# Transcribe the source once. This is used for both AI selection and captions.
cues = transcribe_with_whisper(source_video)
transcript = transcript_for_prompt(cues)

print("\n===== GEMINI CLIP SELECTION =====\n", flush=True)
prompt = f"""
You are a professional short-form video editor.

Select exactly {CLIPS_COUNT} strong non-overlapping YouTube Shorts from the timestamped transcript.

Requirements:
- 35 to 45 seconds each.
- Never exceed 45 seconds.
- Standalone and understandable without previous context.
- Start close to the beginning of a complete sentence.
- End after a complete idea.
- Avoid greetings, intros, sponsor sections, filler and outros.
- Prefer strong hooks, useful insights, stories, mistakes, lessons, surprising facts and memorable opinions.
- Do not invent timestamps.

Return JSON only:
{{
  "clips": [
    {{
      "start_seconds": 0.0,
      "duration": 40.0,
      "title": "short title",
      "hook": "very short on-screen hook",
      "reason": "brief reason"
    }}
  ]
}}

If the transcript is Arabic, write title and hook in Arabic.

TRANSCRIPT:
{transcript}
""".strip()

selection = call_gemini(prompt)
clips = normalize_clips(selection, video_duration)

manifest = {
    "source_url": YOUTUBE_URL,
    "gemini_model": GEMINI_MODEL,
    "whisper_model": WHISPER_MODEL,
    "editing": {
        "smart_camera": True,
        "face_tracking": True,
        "smooth_reframe": True,
        "auto_zoom": True,
        "burned_captions": True,
        "hook_overlay": True,
    },
    "clips": [],
}

print("\n===== RENDER SMART SHORTS =====\n", flush=True)
for i, clip in enumerate(clips, 1):
    out_file = render_short(source_video, cues, clip, i)
    manifest["clips"].append({**clip, "file": out_file.name})

(OUT / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\n===== DONE =====\n", flush=True)
print(json.dumps(manifest, ensure_ascii=False, indent=2))
