#!/usr/bin/env python3

import json
import os
import re
import subprocess
import urllib.request
import urllib.error
from pathlib import Path


# =========================================================
# CONFIG
# =========================================================

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

# Optional.
# If empty, the script automatically discovers a usable Gemini model.
GEMINI_MODEL = os.environ.get(
    "GEMINI_MODEL",
    ""
).strip()


if not YOUTUBE_URL:
    raise SystemExit("Missing YOUTUBE_URL")

if not GEMINI_API_KEY:
    raise SystemExit("Missing GEMINI_API_KEY")


# =========================================================
# HELPERS
# =========================================================

def run(cmd):
    print("\n+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def capture(cmd):
    print("\n+", " ".join(cmd))
    return subprocess.check_output(
        cmd,
        text=True
    ).strip()


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
    seconds = max(
        0.0,
        float(seconds)
    )

    hours = int(
        seconds // 3600
    )

    seconds -= hours * 3600

    minutes = int(
        seconds // 60
    )

    seconds -= minutes * 60

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:06.3f}"
    )


# =========================================================
# SUBTITLE PARSER
# =========================================================

def clean_vtt(vtt_text):

    lines = vtt_text.splitlines()

    chunks = []
    current = None

    for line in lines:

        line = line.strip()

        if "-->" in line:

            left, right = line.split(
                "-->",
                1
            )

            start = (
                left.strip()
                .split()[0]
            )

            end = (
                right.strip()
                .split()[0]
            )

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

            text = re.sub(
                r"<[^>]+>",
                "",
                line
            )

            text = re.sub(
                r"\s+",
                " ",
                text
            ).strip()

            if text:

                if (
                    not current["text"]
                    or current["text"][-1] != text
                ):
                    current["text"].append(
                        text
                    )

    cleaned = []
    last_text = None

    for chunk in chunks:

        text = " ".join(
            chunk["text"]
        ).strip()

        if not text:
            continue

        if text == last_text:
            continue

        last_text = text

        cleaned.append(
            {
                "start": timestamp_to_seconds(
                    chunk["start"]
                ),
                "end": timestamp_to_seconds(
                    chunk["end"]
                ),
                "text": text
            }
        )

    return cleaned


def transcript_for_prompt(
    cues,
    max_chars=400000
):

    lines = []
    total = 0

    for cue in cues:

        line = (
            f"["
            f"{seconds_to_timestamp(cue['start'])}"
            f" --> "
            f"{seconds_to_timestamp(cue['end'])}"
            f"] "
            f"{cue['text']}"
        )

        if (
            total
            + len(line)
            + 1
            > max_chars
        ):
            break

        lines.append(line)

        total += (
            len(line)
            + 1
        )

    return "\n".join(lines)


# =========================================================
# GEMINI
# =========================================================

def get_available_gemini_model():

    if GEMINI_MODEL:

        print(
            f"\nUsing forced Gemini model: "
            f"{GEMINI_MODEL}"
        )

        return GEMINI_MODEL

    print(
        "\n===== FIND GEMINI MODEL =====\n"
    )

    endpoint = (
        "https://generativelanguage.googleapis.com/"
        "v1beta/models"
        f"?key={GEMINI_API_KEY}"
    )

    try:

        with urllib.request.urlopen(
            endpoint,
            timeout=60
        ) as response:

            data = json.load(response)

    except urllib.error.HTTPError as error:

        body = error.read().decode(
            "utf-8",
            errors="ignore"
        )

        raise RuntimeError(
            f"Could not list Gemini models. "
            f"HTTP {error.code}\n"
            f"{body[:3000]}"
        ) from error

    available = []

    for model in data.get(
        "models",
        []
    ):

        methods = model.get(
            "supportedGenerationMethods",
            []
        )

        if (
            "generateContent"
            not in methods
        ):
            continue

        name = model.get(
            "name",
            ""
        )

        name = name.replace(
            "models/",
            ""
        )

        if not name:
            continue

        available.append(name)

    if not available:

        raise RuntimeError(
            "No Gemini model supporting "
            "generateContent is available "
            "for this API key."
        )

    print(
        "Available Gemini models:"
    )

    for name in available:
        print(" -", name)

    # Try to prefer Flash / Flash-Lite models,
    # without depending on a specific model version.
    preferred = []

    for name in available:

        lower = name.lower()

        if (
            "flash-lite" in lower
            and "preview" not in lower
        ):
            preferred.append(
                (
                    1,
                    name
                )
            )

        elif (
            "flash" in lower
            and "preview" not in lower
        ):
            preferred.append(
                (
                    2,
                    name
                )
            )

        elif "flash-lite" in lower:
            preferred.append(
                (
                    3,
                    name
                )
            )

        elif "flash" in lower:
            preferred.append(
                (
                    4,
                    name
                )
            )

        else:
            preferred.append(
                (
                    10,
                    name
                )
            )

    preferred.sort(
        key=lambda item: (
            item[0],
            item[1]
        )
    )

    model_name = preferred[0][1]

    print(
        f"\nSelected Gemini model: "
        f"{model_name}\n"
    )

    return model_name


SELECTED_GEMINI_MODEL = None


def call_gemini(prompt):

    global SELECTED_GEMINI_MODEL

    if (
        SELECTED_GEMINI_MODEL
        is None
    ):

        SELECTED_GEMINI_MODEL = (
            get_available_gemini_model()
        )

    model_name = (
        SELECTED_GEMINI_MODEL
    )

    endpoint = (
        "https://generativelanguage.googleapis.com/"
        "v1beta/models/"
        f"{model_name}:generateContent"
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
        data=json.dumps(
            payload
        ).encode("utf-8"),
        headers={
            "Content-Type":
            "application/json"
        },
        method="POST"
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=180
        ) as response:

            data = json.load(
                response
            )

    except urllib.error.HTTPError as error:

        body = error.read().decode(
            "utf-8",
            errors="ignore"
        )

        raise RuntimeError(
            f"Gemini API failed. "
            f"HTTP {error.code}\n"
            f"Model: {model_name}\n"
            f"{body[:5000]}"
        ) from error

    try:

        text = (
            data["candidates"][0]
            ["content"]["parts"][0]["text"]
        )

    except (
        KeyError,
        IndexError
    ) as error:

        raise RuntimeError(
            "Unexpected Gemini response:\n"
            + json.dumps(
                data,
                ensure_ascii=False
            )[:5000]
        ) from error

    try:

        return json.loads(
            text
        )

    except json.JSONDecodeError:

        match = re.search(
            r"\{.*\}",
            text,
            flags=re.S
        )

        if not match:

            raise RuntimeError(
                "Gemini returned invalid JSON:\n"
                + text[:5000]
            )

        return json.loads(
            match.group(0)
        )


# =========================================================
# CLIP VALIDATION
# =========================================================

def normalize_clips(
    data,
    video_duration
):

    clips = data.get(
        "clips",
        []
    )

    result = []
    used = []

    for clip in clips:

        try:

            start = float(
                clip[
                    "start_seconds"
                ]
            )

            duration = float(
                clip[
                    "duration"
                ]
            )

        except Exception:
            continue

        # Hard limits
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
            and start + duration
            > video_duration
        ):

            start = max(
                0.0,
                video_duration
                - duration
            )

        end = (
            start
            + duration
        )

        overlap = False

        for (
            used_start,
            used_end
        ) in used:

            intersection = max(
                0.0,
                min(
                    end,
                    used_end
                )
                -
                max(
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
                "start_seconds":
                    round(
                        start,
                        3
                    ),

                "duration":
                    round(
                        duration,
                        3
                    ),

                "title":
                    str(
                        clip.get(
                            "title",
                            ""
                        )
                    ).strip(),

                "hook":
                    str(
                        clip.get(
                            "hook",
                            ""
                        )
                    ).strip(),

                "reason":
                    str(
                        clip.get(
                            "reason",
                            ""
                        )
                    ).strip()
            }
        )

        if (
            len(result)
            >= CLIPS_COUNT
        ):
            break

    if not result:

        raise RuntimeError(
            "Gemini did not return "
            "any usable clips."
        )

    return result


def safe_filename(
    text,
    fallback
):

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

    return (
        text[:80]
        or fallback
    )


# =========================================================
# YT-DLP COMMON OPTIONS
# =========================================================

YT_COMMON = [
    "yt-dlp",

    "--js-runtimes",
    "deno",

    "--remote-components",
    "ejs:npm",

    "--no-playlist"
]


if YOUTUBE_COOKIES_FILE:

    print(
        "\nYouTube cookies enabled."
    )

    YT_COMMON += [
        "--cookies",
        YOUTUBE_COOKIES_FILE
    ]


# =========================================================
# 1. DOWNLOAD VIDEO
# =========================================================

print(
    "\n"
    "===== DOWNLOAD VIDEO ====="
    "\n"
)

video_template = str(
    WORK
    / "source.%(ext)s"
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

        for path
        in WORK.glob(
            "source.*"
        )

        if (
            path.suffix.lower()
            in {
                ".mp4",
                ".mkv",
                ".webm",
                ".mov"
            }
        )
    ]
)


if not video_files:

    raise RuntimeError(
        "Video download failed."
    )


video_path = (
    video_files[0]
)


print(
    f"\nVideo file: "
    f"{video_path}\n"
)


# =========================================================
# 2. DOWNLOAD CAPTIONS
# =========================================================

print(
    "\n"
    "===== DOWNLOAD SUBTITLES ====="
    "\n"
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
        "No Arabic or English "
        "YouTube captions were found."
    )


arabic_files = [
    path

    for path
    in vtt_files

    if (
        ".ar"
        in path.name.lower()
    )
]


if arabic_files:

    subtitle_path = (
        arabic_files[0]
    )

else:

    subtitle_path = (
        vtt_files[0]
    )


print(
    f"\nUsing subtitles: "
    f"{subtitle_path}\n"
)


subtitle_text = (
    subtitle_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )
)


cues = clean_vtt(
    subtitle_text
)


if not cues:

    raise RuntimeError(
        "Subtitle file exists, "
        "but could not be parsed."
    )


print(
    f"Parsed subtitle cues: "
    f"{len(cues)}"
)


transcript = (
    transcript_for_prompt(
        cues
    )
)


# =========================================================
# 3. VIDEO DURATION
# =========================================================

print(
    "\n"
    "===== VIDEO DURATION ====="
    "\n"
)


duration_text = capture(
    [
        "ffprobe",

        "-v",
        "error",

        "-show_entries",
        "format=duration",

        "-of",
        (
            "default="
            "noprint_wrappers=1:"
            "nokey=1"
        ),

        str(
            video_path
        )
    ]
)


video_duration = float(
    duration_text
)


print(
    f"\nVideo duration: "
    f"{video_duration:.2f} seconds"
)


# =========================================================
# 4. GEMINI SELECTS CLIPS
# =========================================================

print(
    "\n"
    "===== GEMINI CLIP SELECTION ====="
    "\n"
)


prompt = f"""
You are a professional short-form video editor.

Your task is to turn a long-form YouTube video into
high-quality YouTube Shorts.

Select exactly {CLIPS_COUNT} strong,
non-overlapping moments from the timestamped transcript.

HARD REQUIREMENTS:

- Each clip must be between 35 and 45 seconds.
- Never exceed 45 seconds.
- Each clip must work as a standalone short.
- The viewer should understand the clip without seeing the full video.
- Start close to the beginning of a complete sentence.
- End after a complete thought or complete sentence.
- Do not select overlapping clips.

AVOID:

- Greetings
- Channel introductions
- Sponsor sections
- Long setup with no payoff
- Outros
- Calls to subscribe
- Repetitive filler

PREFER MOMENTS WITH:

- A strong hook
- A surprising statement
- Useful information
- A clear lesson
- A mistake and its solution
- An interesting opinion
- A concise story
- An emotional moment
- A memorable conclusion
- A strong question and answer

IMPORTANT:

Use ONLY timestamps that actually exist in the transcript.

Return JSON ONLY.

Return this exact structure:

{{
  "clips": [
    {{
      "start_seconds": 0.0,
      "duration": 40.0,
      "title": "Short title",
      "hook": "Short on-screen hook",
      "reason": "Why this is a strong short"
    }}
  ]
}}

If the transcript is Arabic:

- Write the title in Arabic.
- Write the hook in Arabic.
- Keep the hook short and natural.
- Do not translate the spoken content.

TIMESTAMPED TRANSCRIPT:

{transcript}
""".strip()


selection = call_gemini(
    prompt
)


print(
    "\nGemini result:\n"
)

print(
    json.dumps(
        selection,
        ensure_ascii=False,
        indent=2
    )
)


clips = normalize_clips(
    selection,
    video_duration
)


print(
    "\nSelected clips:\n"
)

print(
    json.dumps(
        clips,
        ensure_ascii=False,
        indent=2
    )
)


# =========================================================
# 5. RENDER SHORTS
# =========================================================

print(
    "\n"
    "===== RENDER SHORTS ====="
    "\n"
)


manifest = {
    "source_url":
        YOUTUBE_URL,

    "model":
        SELECTED_GEMINI_MODEL,

    "clips":
        []
}


for (
    index,
    clip
) in enumerate(
    clips,
    start=1
):

    title = safe_filename(
        clip["title"],
        f"short-{index}"
    )


    output_file = (
        OUT
        /
        f"{index:02d}-{title}.mp4"
    )


    print(
        f"\nRendering Short "
        f"{index}/{len(clips)}"
    )

    print(
        "Start:",
        clip["start_seconds"]
    )

    print(
        "Duration:",
        clip["duration"]
    )


    # MVP:
    # Center crop to 9:16.
    # Face tracking will be added later.

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
                clip[
                    "start_seconds"
                ]
            ),

            "-i",
            str(
                video_path
            ),

            "-t",
            str(
                clip[
                    "duration"
                ]
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


    manifest[
        "clips"
    ].append(
        {
            **clip,
            "file":
                output_file.name
        }
    )


# =========================================================
# MANIFEST
# =========================================================

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
    "\n"
    "===== DONE ====="
    "\n"
)


print(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2
    )
)


print(
    "\nRendered files:"
)


for file in OUT.iterdir():

    print(
        " -",
        file.name
    )
