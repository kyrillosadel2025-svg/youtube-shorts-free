#!/usr/bin/env python3

import json
import os
import re
import subprocess
import urllib.request
from pathlib import Path


ROOT = Path.cwd()
WORK = ROOT / "work"
OUT = ROOT / "output"

WORK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

YOUTUBE_URL = os.environ.get("YOUTUBE_URL", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
CLIPS_COUNT = int(os.environ.get("CLIPS_COUNT", "3") or "3")
YOUTUBE_COOKIES_FILE = os.environ.get(
    "YOUTUBE_COOKIES_FILE",
    ""
).strip()

GEMINI_MODEL = os.environ.get(
    "GEMINI_MODEL",
    "gemini-2.5-flash-lite"
)

if not YOUTUBE_URL:
    raise SystemExit("Missing YOUTUBE_URL")

if not GEMINI_API_KEY:
    raise SystemExit("Missing GEMINI_API_KEY")


def run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def capture(cmd):
    print("+", " ".join(cmd))
    return subprocess.check_output(cmd, text=True).strip()


def timestamp_to_seconds(value):
    parts = value.replace(",", ".").split(":")

    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = "0"
        minutes, seconds = parts
    else:
        raise ValueError(value)

    return (
        int(hours) * 3600
        + int(minutes) * 60
        + float(seconds)
    )


def seconds_to_timestamp(seconds):
    seconds = max(0.0, float(seconds))

    hours = int(seconds // 3600)
    seconds -= hours * 3600

    minutes = int(seconds // 60)
    seconds -= minutes * 60

    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


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

            current = {
                "start": start,
                "end": end,
                "text": []
            }

            chunks.append(current)

        elif current and line:
            if line.startswith(
                (
                    "WEBVTT",
                    "NOTE",
                    "Kind:",
                    "Language:"
                )
            ):
                continue

            text = re.sub(r"<[^>]+>", "", line)
            text = re.sub(r"\s+", " ", text).strip()

            if text:
                if (
                    not current["text"]
                    or current["text"][-1] != text
                ):
                    current["text"].append(text)

    cleaned = []
    last_text = None

    for chunk in chunks:
        text = " ".join(chunk["text"]).strip()

        if not text:
            continue

        if text == last_text:
            continue

        last_text = text

        cleaned.append(
            {
                "start": timestamp_to_seconds(chunk["start"]),
                "end": timestamp_to_seconds(chunk["end"]),
                "text": text
            }
        )

    return cleaned


def transcript_for_prompt(cues, max_chars=450000):
    lines = []
    total = 0

    for cue in cues:
        line = (
            f"[{seconds_to_timestamp(cue['start'])}"
            f" --> "
            f"{seconds_to_timestamp(cue['end'])}] "
            f"{cue['text']}"
        )

        if total + len(line) + 1 > max_chars:
            break

        lines.append(line)
        total += len(line) + 1

    return "\n".join(lines)


def call_gemini(prompt):
    endpoint = (
        "https://generativelanguage.googleapis.com/"
        "v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
        f"?key={GEMINI_API_KEY}"
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json"
        }
    }

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    with urllib.request.urlopen(
        request,
        timeout=120
    ) as response:
        data = json.load(response)

    try:
        text = (
            data["candidates"][0]
            ["content"]["parts"][0]["text"]
        )

    except (KeyError, IndexError) as error:
        raise RuntimeError(
            "Unexpected Gemini response:\n"
            + json.dumps(
                data,
                ensure_ascii=False
            )[:3000]
        ) from error

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        match = re.search(
            r"\{.*\}",
            text,
            flags=re.S
        )

        if not match:
            raise

        return json.loads(
            match.group(0)
        )


def normalize_clips(data, video_duration):
    clips = data.get("clips", [])

    result = []
    used = []

    for clip in clips:
        try:
            start = float(clip["start_seconds"])
            duration = float(clip["duration"])

        except Exception:
            continue

        duration = min(
            45.0,
            max(
                35.0,
                duration
            )
        )

        start = max(
            0.0,
            start
        )

        if (
            video_duration > 0
            and start + duration > video_duration
        ):
            start = max(
                0.0,
                video_duration - duration
            )

        end = start + duration

        overlap = False

        for used_start, used_end in used:
            intersection = max(
                0.0,
                min(
                    end,
                    used_end
                )
                - max(
                    start,
                    used_start
                )
            )

            if intersection > 5.0:
                overlap = True
                break

        if overlap:
            continue

        used.append(
            (
                start,
                end
            )
        )

        result.append(
            {
                "start_seconds": round(
                    start,
                    3
                ),
                "duration": round(
                    duration,
                    3
                ),
                "title": str(
                    clip.get(
                        "title",
                        ""
                    )
                ).strip(),
                "hook": str(
                    clip.get(
                        "hook",
                        ""
                    )
                ).strip(),
                "reason": str(
                    clip.get(
                        "reason",
                        ""
                    )
                ).strip()
            }
        )

        if len(result) >= CLIPS_COUNT:
            break

    if not result:
        raise RuntimeError(
            "Gemini did not return usable clips"
        )

    return result


def safe_filename(text, fallback):
    text = re.sub(
        r'[\\/:*?"<>|]+',
        "",
        text or ""
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text[:80] or fallback


YT_COMMON = [
    "yt-dlp",

    "--js-runtimes",
    "deno",

    "--remote-components",
    "ejs:npm",

    "--no-playlist"
]

if YOUTUBE_COOKIES_FILE:
    YT_COMMON += [
        "--cookies",
        YOUTUBE_COOKIES_FILE
    ]


print(
    "\n===== DOWNLOAD VIDEO =====\n"
)

video_template = str(
    WORK / "source.%(ext)s"
)

run(
    YT_COMMON
    + [
        "-f",
        (
            "bv*[ext=mp4]+ba[ext=m4a]"
            "/b[ext=mp4]"
            "/b"
        ),

        "--merge-output-format",
        "mp4",

        "-o",
        video_template,

        YOUTUBE_URL
    ]
)


video_files = sorted(
    [
        path
        for path in WORK.glob("source.*")
        if path.suffix.lower()
        in {
            ".mp4",
            ".mkv",
            ".webm",
            ".mov"
        }
    ]
)

if not video_files:
    raise RuntimeError(
        "Video download failed"
    )

video_path = video_files[0]


print(
    "\n===== DOWNLOAD SUBTITLES =====\n"
)

run(
    YT_COMMON
    + [
        "--skip-download",

        "--write-subs",

        "--write-auto-subs",

        "--sub-format",
        "vtt",

        "--sub-langs",
        "ar.*,en.*",

        "-o",
        str(
            WORK
            / "subs.%(ext)s"
        ),

        YOUTUBE_URL
    ]
)


vtt_files = sorted(
    WORK.glob(
        "subs*.vtt"
    )
)

if not vtt_files:
    raise RuntimeError(
        "No Arabic or English captions were found."
    )


arabic_files = [
    path
    for path in vtt_files
    if ".ar" in path.name.lower()
]

if arabic_files:
    subtitle_path = arabic_files[0]
else:
    subtitle_path = vtt_files[0]


subtitle_text = subtitle_path.read_text(
    encoding="utf-8",
    errors="ignore"
)

cues = clean_vtt(
    subtitle_text
)

if not cues:
    raise RuntimeError(
        "Subtitle file exists but could not be parsed"
    )


transcript = transcript_for_prompt(
    cues
)


print(
    "\n===== VIDEO DURATION =====\n"
)

duration_text = capture(
    [
        "ffprobe",
        "-v",
        "error",

        "-show_entries",
        "format=duration",

        "-of",
        "default=noprint_wrappers=1:nokey=1",

        str(
            video_path
        )
    ]
)

video_duration = float(
    duration_text
)


print(
    "\n===== GEMINI CLIP SELECTION =====\n"
)

prompt = f"""
You are editing a long-form video into YouTube Shorts.

Select exactly {CLIPS_COUNT} strong,
non-overlapping clips from the transcript.

Hard requirements:

- Each selected clip must be between 35 and 45 seconds.
- Never exceed 45 seconds.
- Each clip must work independently.
- Start close to the beginning of a complete sentence.
- End after a complete thought.
- Avoid greetings.
- Avoid channel intros.
- Avoid sponsor sections.
- Avoid outros.
- Avoid filler.

Prefer:

- strong hooks
- surprising facts
- useful information
- concise stories
- mistakes
- lessons
- memorable opinions
- emotional moments

Do not invent timestamps.

Return JSON only.

Exact structure:

{{
  "clips": [
    {{
      "start_seconds": 0.0,
      "duration": 40.0,
      "title": "title",
      "hook": "short hook",
      "reason": "brief reason"
    }}
  ]
}}

If the transcript is Arabic,
write title and hook in Arabic.

TRANSCRIPT:

{transcript}
""".strip()


selection = call_gemini(
    prompt
)


clips = normalize_clips(
    selection,
    video_duration
)


print(
    "\n===== RENDER SHORTS =====\n"
)


manifest = {
    "source_url": YOUTUBE_URL,
    "model": GEMINI_MODEL,
    "clips": []
}


for index, clip in enumerate(
    clips,
    start=1
):

    title = safe_filename(
        clip["title"],
        f"short-{index}"
    )

    output_file = (
        OUT
        / f"{index:02d}-{title}.mp4"
    )

    video_filter = (
        "scale="
        "1080:1920:"
        "force_original_aspect_ratio=increase,"
        "crop=1080:1920"
    )

    run(
        [
            "ffmpeg",
            "-y",

            "-ss",
            str(
                clip["start_seconds"]
            ),

            "-i",
            str(
                video_path
            ),

            "-t",
            str(
                clip["duration"]
            ),

            "-vf",
            video_filter,

            "-c:v",
            "libx264",

            "-preset",
            "veryfast",

            "-crf",
            "21",

            "-c:a",
            "aac",

            "-b:a",
            "128k",

            "-movflags",
            "+faststart",

            str(
                output_file
            )
        ]
    )

    manifest["clips"].append(
        {
            **clip,
            "file": output_file.name
        }
    )


manifest_path = (
    OUT
    / "manifest.json"
)


manifest_path.write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)


print(
    "\n===== DONE =====\n"
)

print(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2
    )
)
