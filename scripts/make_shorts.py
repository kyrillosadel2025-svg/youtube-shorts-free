#!/usr/bin/env python3
import json
import math
import os
import re
import shutil
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
    """
    v8.1 Real Glow captions.

    Each caption is drawn in 3 layers:
    1) Wide blurred cyan glow.
    2) Tighter blue/cyan glow.
    3) Crisp white foreground text.

    The glow uses ASS \\blur override tags, so FFmpeg/libass renders
    an actual soft halo instead of a simple thick outline.
    """

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {OUT_W}
PlayResY: {OUT_H}
ScaledBorderAndShadow: yes
WrapStyle: 2

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: CaptionGlowWide,DejaVu Sans,60,&H66FFFFFF,&H000000FF,&HAA00E5FF,&H00000000,-1,0,0,0,100,100,0,0,1,8,0,2,80,80,250,1
Style: CaptionGlowTight,DejaVu Sans,59,&H33FFFFFF,&H000000FF,&HCC00BFFF,&H00000000,-1,0,0,0,100,100,0,0,1,5,0,2,80,80,250,1
Style: Caption,DejaVu Sans,58,&H00FFFFFF,&H000000FF,&H00101010,&H50000000,-1,0,0,0,100,100,0,0,1,3.2,0,2,80,80,250,1

Style: HookGlowWide,DejaVu Sans,68,&H66FFFFFF,&H000000FF,&HAA00E5FF,&H00000000,-1,0,0,0,100,100,0,0,1,9,0,8,90,90,175,1
Style: HookGlowTight,DejaVu Sans,67,&H33FFFFFF,&H000000FF,&HCC00BFFF,&H00000000,-1,0,0,0,100,100,0,0,1,5.5,0,8,90,90,175,1
Style: Hook,DejaVu Sans,66,&H00FFFFFF,&H000000FF,&H00101010,&H50000000,-1,0,0,0,100,100,0,0,1,3.6,0,8,90,90,175,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""

    events = []

    def add_glow_triplet(start_t, end_t, base_style, text):
        if base_style == "Caption":
            events.append(
                f"Dialogue: 0,{sec_to_ass_time(start_t)},{sec_to_ass_time(end_t)},CaptionGlowWide,,0,0,0,,{{\\blur10\\bord9\\1a&H99&\\3a&H55&}}{text}"
            )
            events.append(
                f"Dialogue: 1,{sec_to_ass_time(start_t)},{sec_to_ass_time(end_t)},CaptionGlowTight,,0,0,0,,{{\\blur4.5\\bord6\\1a&H77&\\3a&H33&}}{text}"
            )
            events.append(
                f"Dialogue: 2,{sec_to_ass_time(start_t)},{sec_to_ass_time(end_t)},Caption,,0,0,0,,{{\\blur0.6}}{text}"
            )
        else:
            events.append(
                f"Dialogue: 0,{sec_to_ass_time(start_t)},{sec_to_ass_time(end_t)},HookGlowWide,,0,0,0,,{{\\blur11\\bord10\\1a&H99&\\3a&H55&}}{text}"
            )
            events.append(
                f"Dialogue: 1,{sec_to_ass_time(start_t)},{sec_to_ass_time(end_t)},HookGlowTight,,0,0,0,,{{\\blur5\\bord6.5\\1a&H77&\\3a&H33&}}{text}"
            )
            events.append(
                f"Dialogue: 2,{sec_to_ass_time(start_t)},{sec_to_ass_time(end_t)},Hook,,0,0,0,,{{\\blur0.6}}{text}"
            )

    if hook:
        hook_text = escape_ass(wrap_caption(hook, 22))
        hook_end = min(3.0, clip_duration)
        add_glow_triplet(0, hook_end, "Hook", hook_text)

    clip_end = clip_start + clip_duration

    for cue in cues:
        if cue["end"] <= clip_start or cue["start"] >= clip_end:
            continue

        start = max(0.0, cue["start"] - clip_start)
        end = min(clip_duration, cue["end"] - clip_start)

        if end - start < 0.15:
            continue

        text = escape_ass(
            wrap_caption(
                cue["text"],
                28
            )
        )

        add_glow_triplet(start, end, "Caption", text)

    out_path.write_text(
        header + "\n".join(events) + "\n",
        encoding="utf-8"
    )


def smart_reframe(input_clip, output_silent):
    """
    v8 Scene-Based Reframe

    The camera does NOT continuously chase subjects.

    Workflow:
    - Detect hard/meaningful scene changes.
    - Analyze each new scene for faces + motion.
    - Pick ONE stable vertical crop for that scene.
    - Hold the crop until the next scene.
    - Use only a short eased transition when the scene framing changes.

    This works better for animation / educational content where the important
    elements can be letters, animals, characters, props and text.
    """

    cap = cv2.VideoCapture(str(input_clip))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {input_clip}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if W <= 0 or H <= 0:
        raise RuntimeError("Invalid source dimensions")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_silent),
        fourcc,
        fps,
        (OUT_W, OUT_H)
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create VideoWriter")

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError("Could not load OpenCV face cascade")

    # Scene detection tuning
    scene_threshold = 0.50
    min_scene_frames = max(1, int(round(fps * 0.90)))
    analysis_frames = max(6, int(round(fps * 0.75)))

    # Stable/wide framing
    default_crop_ratio = 0.92
    subject_crop_ratio = 0.84
    face_crop_ratio = 0.80

    # Short transition only when switching scenes
    transition_frames = max(1, int(round(fps * 0.22)))

    prev_hist = None
    prev_gray_small = None

    current_scene_start = 0
    scene_id = 1
    frames_since_scene = 0

    # Current crop
    cur_x = W / 2
    cur_y = H / 2
    cur_h = H * default_crop_ratio

    # Target crop for this scene
    target_x = cur_x
    target_y = cur_y
    target_h = cur_h

    # Transition state
    trans_left = 0
    trans_from = (cur_x, cur_y, cur_h)

    # Accumulators used only during the first part of each scene
    face_samples = []
    motion_samples = []

    def clamp(v, lo, hi):
        return max(lo, min(v, hi))

    def calc_hist(frame):
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    def analyze_faces(frame):
        detect_scale = min(1.0, 900.0 / max(W, H))
        if detect_scale < 1.0:
            dframe = cv2.resize(
                frame, None,
                fx=detect_scale, fy=detect_scale,
                interpolation=cv2.INTER_AREA
            )
        else:
            dframe = frame

        gray = cv2.cvtColor(dframe, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        found = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.09,
            minNeighbors=6,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        inv = 1.0 / detect_scale
        boxes = []

        for (x, y, w, h) in found:
            x1 = int(x * inv)
            y1 = int(y * inv)
            x2 = int((x + w) * inv)
            y2 = int((y + h) * inv)

            x1 = int(clamp(x1, 0, W - 1))
            y1 = int(clamp(y1, 0, H - 1))
            x2 = int(clamp(x2, x1 + 1, W))
            y2 = int(clamp(y2, y1 + 1, H))

            area = (x2 - x1) * (y2 - y1)
            boxes.append((area, x1, y1, x2, y2))

        return boxes

    def analyze_motion(frame, prev_gray):
        scale = min(1.0, 640.0 / max(W, H))
        if scale < 1.0:
            small = cv2.resize(
                frame, None,
                fx=scale, fy=scale,
                interpolation=cv2.INTER_AREA
            )
        else:
            small = frame

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)

        result = None

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            _, mask = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)

            kernel = np.ones((7, 7), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.dilate(mask, kernel, iterations=1)

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            frame_area = mask.shape[0] * mask.shape[1]
            candidates = []

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < frame_area * 0.012:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)

                if w < mask.shape[1] * 0.07 or h < mask.shape[0] * 0.07:
                    continue

                candidates.append((area, x, y, w, h))

            if candidates:
                candidates.sort(reverse=True, key=lambda c: c[0])
                top = candidates[:3]

                min_x = min(c[1] for c in top)
                min_y = min(c[2] for c in top)
                max_x = max(c[1] + c[3] for c in top)
                max_y = max(c[2] + c[4] for c in top)

                inv = 1.0 / scale

                result = (
                    min_x * inv,
                    min_y * inv,
                    max_x * inv,
                    max_y * inv,
                )

        return gray, result

    def choose_scene_crop():
        """
        One framing decision for the whole scene.
        Priority:
        - meaningful face group if present
        - meaningful motion/subject region
        - safe center crop
        """
        if face_samples:
            boxes = []
            for sample in face_samples:
                boxes.extend(sample)

            if boxes:
                boxes.sort(reverse=True, key=lambda b: b[0])

                # Use up to two strongest faces to preserve context.
                chosen = boxes[:2]

                min_x = min(b[1] for b in chosen)
                min_y = min(b[2] for b in chosen)
                max_x = max(b[3] for b in chosen)
                max_y = max(b[4] for b in chosen)

                cx = (min_x + max_x) / 2
                cy = ((min_y + max_y) / 2) + H * 0.04

                union_w = max_x - min_x
                union_h = max_y - min_y

                desired_h = max(
                    union_h / 0.28,
                    union_w / 0.42,
                    H * face_crop_ratio
                )

                desired_h = clamp(desired_h, H * face_crop_ratio, H)

                return float(cx), float(cy), float(desired_h), "face"

        if motion_samples:
            min_x = min(b[0] for b in motion_samples)
            min_y = min(b[1] for b in motion_samples)
            max_x = max(b[2] for b in motion_samples)
            max_y = max(b[3] for b in motion_samples)

            cx = (min_x + max_x) / 2
            cy = (min_y + max_y) / 2

            union_w = max_x - min_x
            union_h = max_y - min_y

            desired_h = max(
                union_h / 0.46,
                union_w / 0.62,
                H * subject_crop_ratio
            )

            desired_h = clamp(desired_h, H * subject_crop_ratio, H)

            return float(cx), float(cy), float(desired_h), "subject"

        return W / 2, H / 2, H * default_crop_ratio, "center"

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frames_since_scene += 1

        # Scene cut detection
        hist = calc_hist(frame)

        scene_cut = False

        if prev_hist is not None and frames_since_scene >= min_scene_frames:
            similarity = cv2.compareHist(
                prev_hist,
                hist,
                cv2.HISTCMP_CORREL
            )

            if similarity < scene_threshold:
                scene_cut = True

        prev_hist = hist

        if scene_cut:
            # Lock the crop selected from the previous scene analysis,
            # then reset analysis for the new scene.
            new_x, new_y, new_h, mode = choose_scene_crop()

            trans_from = (cur_x, cur_y, cur_h)
            target_x = new_x
            target_y = new_y
            target_h = new_h
            trans_left = transition_frames

            print(
                f"Scene {scene_id}: crop mode={mode} "
                f"x={target_x:.0f} y={target_y:.0f} h={target_h:.0f}",
                flush=True
            )

            scene_id += 1
            current_scene_start = frame_idx
            frames_since_scene = 0
            face_samples = []
            motion_samples = []
            prev_gray_small = None

        # Analyze only the beginning of each scene.
        if frames_since_scene <= analysis_frames:
            if frame_idx % max(1, int(round(fps / 4))) == 0:
                faces = analyze_faces(frame)
                if faces:
                    face_samples.append(faces)

            prev_gray_small, motion_box = analyze_motion(
                frame,
                prev_gray_small
            )

            if motion_box is not None:
                motion_samples.append(motion_box)

            # Once enough frames are analyzed, choose and lock this scene's crop.
            if frames_since_scene == analysis_frames:
                new_x, new_y, new_h, mode = choose_scene_crop()

                trans_from = (cur_x, cur_y, cur_h)
                target_x = new_x
                target_y = new_y
                target_h = new_h
                trans_left = transition_frames

                print(
                    f"Scene {scene_id} locked: mode={mode} "
                    f"x={target_x:.0f} y={target_y:.0f} h={target_h:.0f}",
                    flush=True
                )

        # Only animate during a short transition.
        if trans_left > 0:
            done = transition_frames - trans_left + 1
            t = done / transition_frames

            # Smoothstep easing
            t = t * t * (3.0 - 2.0 * t)

            fx, fy, fh = trans_from

            cur_x = fx + (target_x - fx) * t
            cur_y = fy + (target_y - fy) * t
            cur_h = fh + (target_h - fh) * t

            trans_left -= 1
        else:
            # Locked shot
            cur_x = target_x
            cur_y = target_y
            cur_h = target_h

        cur_h = clamp(cur_h, H * face_crop_ratio, H)
        crop_w = cur_h * OUT_W / OUT_H

        if crop_w > W:
            crop_w = float(W)
            cur_h = crop_w * OUT_H / OUT_W

        crop_w_i = max(2, int(round(crop_w)))
        crop_h_i = max(2, int(round(cur_h)))

        x1 = int(round(cur_x - crop_w_i / 2))
        y1 = int(round(cur_y - crop_h_i / 2))

        x1 = max(0, min(x1, W - crop_w_i))
        y1 = max(0, min(y1, H - crop_h_i))

        roi = frame[
            y1:y1 + crop_h_i,
            x1:x1 + crop_w_i
        ]

        if roi.size == 0:
            roi = frame

        out_frame = cv2.resize(
            roi,
            (OUT_W, OUT_H),
            interpolation=cv2.INTER_AREA
        )

        writer.write(out_frame)

        frame_idx += 1

        if frame_idx % max(1, int(fps * 10)) == 0:
            print(
                f"Scene-based reframe: {frame_idx}/{total} frames | scene={scene_id}",
                flush=True
            )

    # Final scene diagnostic
    final_x, final_y, final_h, final_mode = choose_scene_crop()
    print(
        f"Final scene {scene_id}: mode={final_mode}",
        flush=True
    )

    writer.release()
    cap.release()



def probe_media(path):
    """Return ffprobe JSON for a rendered short."""
    raw = capture([
        "ffprobe", "-v", "error",
        "-show_streams",
        "-show_format",
        "-of", "json",
        path,
    ])
    return json.loads(raw)


def estimate_edge_brightness(video_path, seconds=0.7):
    """
    Sample the first and last ~0.7s.
    Returns average luma for both ends.
    Helps catch black/blank openings or endings.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_n = max(2, int(round(fps * seconds)))

    def avg_at(indices):
        vals = []
        for frame_no in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_no)))
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vals.append(float(np.mean(gray)))
        return float(np.mean(vals)) if vals else None

    first_idx = np.linspace(0, max(0, sample_n - 1), min(sample_n, 8))
    last_start = max(0, total - sample_n)
    last_idx = np.linspace(last_start, max(last_start, total - 1), min(sample_n, 8))

    first = avg_at(first_idx)
    last = avg_at(last_idx)
    cap.release()
    return first, last


def measure_mean_volume(video_path):
    """
    Uses FFmpeg volumedetect.
    Returns mean_volume in dB, or None if unavailable.
    """
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostats",
            "-i", str(video_path),
            "-af", "volumedetect",
            "-f", "null", "-"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    log = (proc.stderr or "") + "\n" + (proc.stdout or "")
    m = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", log)
    return float(m.group(1)) if m else None


def caption_coverage(cues, clip_start, clip_duration):
    """
    Rough fraction of the clip covered by transcript cues.
    This does not OCR the burned subtitles; it verifies that
    the caption source actually covers enough of the selected clip.
    """
    clip_end = clip_start + clip_duration
    spans = []

    for cue in cues:
        a = max(clip_start, float(cue["start"]))
        b = min(clip_end, float(cue["end"]))
        if b > a:
            spans.append((a, b))

    if not spans:
        return 0.0

    spans.sort()
    merged = [list(spans[0])]

    for a, b in spans[1:]:
        if a <= merged[-1][1] + 0.08:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])

    covered = sum(b - a for a, b in merged)
    return max(0.0, min(1.0, covered / max(0.001, clip_duration)))



def measure_audio_polish(video_path):
    """
    Collect post-render audio metrics.
    loudnorm output is not required here; this is a light validation pass.
    """
    metrics = {
        "mean_volume_db": None,
        "max_volume_db": None,
    }

    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostats",
            "-i", str(video_path),
            "-af", "volumedetect",
            "-f", "null", "-"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    log = (proc.stderr or "") + "\n" + (proc.stdout or "")

    m = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", log)
    p = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", log)

    if m:
        metrics["mean_volume_db"] = float(m.group(1))
    if p:
        metrics["max_volume_db"] = float(p.group(1))

    return metrics


def quality_gate(video_path, clip, cues):
    """
    Automatic post-render Quality Gate.

    Hard failures:
    - wrong/missing video stream
    - wrong output dimensions
    - duration outside Shorts target window
    - missing audio
    - effectively silent audio
    - near-black beginning/end

    Soft warnings:
    - low caption coverage
    - unusually quiet audio

    Score is informational; hard failures determine approval.
    """
    report = {
        "file": Path(video_path).name,
        "approved": True,
        "score": 100,
        "checks": {},
        "warnings": [],
        "failures": [],
    }

    try:
        meta = probe_media(video_path)
    except Exception as exc:
        report["approved"] = False
        report["score"] = 0
        report["failures"].append(f"ffprobe failed: {exc}")
        return report

    streams = meta.get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    # Video stream / resolution
    if not video_streams:
        report["approved"] = False
        report["score"] -= 50
        report["failures"].append("missing video stream")
        width = height = 0
    else:
        width = int(video_streams[0].get("width") or 0)
        height = int(video_streams[0].get("height") or 0)

    resolution_ok = (width == OUT_W and height == OUT_H)
    report["checks"]["resolution"] = {
        "width": width,
        "height": height,
        "expected": f"{OUT_W}x{OUT_H}",
        "ok": resolution_ok,
    }

    if not resolution_ok:
        report["approved"] = False
        report["score"] -= 25
        report["failures"].append("output resolution is not 1080x1920")

    # Duration
    duration = float(meta.get("format", {}).get("duration") or 0.0)
    duration_ok = 34.0 <= duration <= 46.0
    report["checks"]["duration"] = {
        "seconds": round(duration, 3),
        "ok": duration_ok,
    }

    if not duration_ok:
        report["approved"] = False
        report["score"] -= 20
        report["failures"].append(
            f"duration {duration:.2f}s is outside 34–46s gate"
        )

    # Audio presence
    has_audio = bool(audio_streams)
    report["checks"]["audio_stream"] = {"present": has_audio, "ok": has_audio}

    if not has_audio:
        report["approved"] = False
        report["score"] -= 30
        report["failures"].append("missing audio stream")

    # Audio level
    mean_volume = measure_mean_volume(video_path) if has_audio else None
    polish_metrics = measure_audio_polish(video_path) if has_audio else {
        "mean_volume_db": None,
        "max_volume_db": None,
    }
    report["checks"]["audio_polish"] = {
        **polish_metrics,
        "target_integrated_lufs": -14,
        "target_true_peak_db": -1.5,
        "processing": [
            "highpass 70Hz",
            "light FFT denoise",
            "loudness normalization",
            "peak limiter",
        ],
    }
    audio_level_ok = mean_volume is None or mean_volume > -45.0
    report["checks"]["mean_volume"] = {
        "db": mean_volume,
        "ok": audio_level_ok,
    }

    if mean_volume is not None:
        if mean_volume <= -45.0:
            report["approved"] = False
            report["score"] -= 30
            report["failures"].append(
                f"audio is effectively silent ({mean_volume:.1f} dB)"
            )
        elif mean_volume < -28.0:
            report["score"] -= 8
            report["warnings"].append(
                f"audio is quiet ({mean_volume:.1f} dB)"
            )

    # Beginning/end brightness
    first_luma, last_luma = estimate_edge_brightness(video_path)
    first_ok = first_luma is None or first_luma >= 8.0
    last_ok = last_luma is None or last_luma >= 8.0

    report["checks"]["edge_brightness"] = {
        "first_luma": None if first_luma is None else round(first_luma, 2),
        "last_luma": None if last_luma is None else round(last_luma, 2),
        "first_ok": first_ok,
        "last_ok": last_ok,
    }

    if not first_ok:
        report["approved"] = False
        report["score"] -= 15
        report["failures"].append("opening appears nearly black")

    if not last_ok:
        report["approved"] = False
        report["score"] -= 15
        report["failures"].append("ending appears nearly black")

    # Caption source coverage
    coverage = caption_coverage(
        cues,
        float(clip["start_seconds"]),
        float(clip["duration"]),
    )

    report["checks"]["caption_coverage"] = {
        "ratio": round(coverage, 3),
        "percent": round(coverage * 100, 1),
        "ok": coverage >= 0.45,
    }

    if coverage < 0.30:
        report["score"] -= 15
        report["warnings"].append(
            f"very low caption coverage ({coverage*100:.1f}%)"
        )
    elif coverage < 0.45:
        report["score"] -= 7
        report["warnings"].append(
            f"low caption coverage ({coverage*100:.1f}%)"
        )

    report["score"] = max(0, min(100, int(report["score"])))
    return report


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
        "-af", "highpass=f=70,afftdn=nf=-28:tn=1,loudnorm=I=-14:LRA=9:TP=-1.5,alimiter=limit=0.95",
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
        "scene_based_reframe": True,
        "continuous_camera_tracking": False,
        "real_glow_captions": True,
        "burned_captions": True,
        "hook_overlay": True,
        "quality_gate": True,
        "audio_polish": True,
        "target_loudness_lufs": -14,
        "true_peak_db": -1.5,
    },
    "clips": [],
    "quality_summary": {
        "approved": 0,
        "rejected": 0,
    },
}

quality_reports = []

print("\n===== RENDER + QUALITY GATE =====\n", flush=True)

for i, clip in enumerate(clips, 1):
    out_file = render_short(source_video, cues, clip, i)

    print(f"\n===== QUALITY GATE {i} =====\n", flush=True)
    report = quality_gate(out_file, clip, cues)
    quality_reports.append(report)

    entry = {
        **clip,
        "file": out_file.name,
        "quality": report,
    }

    if report["approved"]:
        manifest["quality_summary"]["approved"] += 1
        entry["status"] = "approved"
        print(
            f"APPROVED: {out_file.name} | score={report['score']}",
            flush=True
        )
    else:
        manifest["quality_summary"]["rejected"] += 1
        entry["status"] = "rejected"

        rejected_dir = WORK / "rejected"
        rejected_dir.mkdir(exist_ok=True)

        rejected_path = rejected_dir / out_file.name
        shutil.move(str(out_file), str(rejected_path))

        entry["file"] = None
        entry["rejected_file"] = rejected_path.name

        print(
            f"REJECTED: {rejected_path.name} | "
            f"score={report['score']} | "
            f"failures={report['failures']}",
            flush=True
        )

    manifest["clips"].append(entry)

(OUT / "quality_report.json").write_text(
    json.dumps(
        {
            "summary": manifest["quality_summary"],
            "reports": quality_reports,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

(OUT / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

if manifest["quality_summary"]["approved"] == 0:
    raise RuntimeError(
        "Quality Gate rejected every rendered short. "
        "See output/quality_report.json in the job logs/artifact context."
    )

print("\n===== DONE =====\n", flush=True)
print(json.dumps(manifest, ensure_ascii=False, indent=2))
