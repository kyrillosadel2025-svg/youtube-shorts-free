# YouTube Long Video → 3 Shorts (No monthly add-on subscription)

This repository is designed for **n8n Cloud + GitHub Actions + Gemini free tier**.

## What it does

1. Receives a YouTube URL.
2. Downloads the video with `yt-dlp`.
3. Pulls Arabic or English YouTube captions with timestamps.
4. Uses Gemini 2.5 Flash-Lite to choose the strongest 35–45 second moments.
5. Renders vertical 1080×1920 MP4 clips with FFmpeg.
6. Uploads the finished clips as a GitHub Actions artifact for review.

## Important limitation

This MVP needs the source video to have YouTube captions (manual or auto-generated).
It does **not** transcribe audio locally yet.

The first version also uses a simple center crop for 9:16. It does not yet do face tracking.

## Setup

### 1. Create a GitHub repository

Upload these files, preserving:

- `.github/workflows/make-shorts.yml`
- `scripts/make_shorts.py`

A public repository can use standard GitHub-hosted runners without Actions minute charges under GitHub's current public-repo policy. Check GitHub's current limits before relying on this at scale.

### 2. Create a Gemini API key

Create an API key in Google AI Studio.

In your GitHub repository:

Settings → Secrets and variables → Actions → New repository secret

Name:

`GEMINI_API_KEY`

Value:

your Gemini API key

### 3. Test manually

GitHub → Actions → Make Shorts → Run workflow

Paste a YouTube URL and use `3` clips.

When the run finishes:

Actions run → Artifacts → `youtube-shorts`

Download the ZIP and review the MP4 files.

## Trigger from n8n Cloud

Use an n8n **HTTP Request** node.

### Endpoint

`POST https://api.github.com/repos/OWNER/REPO/actions/workflows/make-shorts.yml/dispatches`

### Headers

- `Accept: application/vnd.github+json`
- `Authorization: Bearer YOUR_GITHUB_TOKEN`
- `X-GitHub-Api-Version: 2026-03-10`
- `Content-Type: application/json`

### JSON body

```json
{
  "ref": "main",
  "inputs": {
    "youtube_url": "{{$json.youtube_url}}",
    "clips_count": "3"
  }
}
```

For a fine-grained GitHub token, give it access to the target repository and **Actions: Read and write**.

## Alternative trigger

The workflow also accepts a `repository_dispatch` event called `make_shorts`.
