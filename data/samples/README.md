# Sample Videos

This directory should contain sample soccer videos for testing and development.

## How to Get Sample Videos

### Option 1: Record Your Own (Recommended)
- Record a 10-20 second soccer clip on your phone
- Keep the camera steady and make sure players are visible
- Transfer to this directory

### Option 2: Download from YouTube
Use `yt-dlp` or similar to download short clips:

```bash
# Install yt-dlp
pip install yt-dlp

# Download a short clip (first 30 seconds)
yt-dlp -f "best[height<=720]" --download-sections "*0-30" -o "sample.mp4" "VIDEO_URL"
```

**Note:** Make sure to use Creative Commons or appropriately licensed content.

### Option 3: Use Test Footage
- Search for "soccer test footage" or "football test video"
- Look for royalty-free or CC-licensed content
- Download and place in this directory

## Recommended Video Properties

For best results, use videos with:
- **Resolution:** 720p or higher (1080p ideal)
- **FPS:** 25-60 fps
- **Format:** MP4, MOV, or AVI
- **Content:** Clear view of players and ball
- **Lighting:** Good visibility (avoid night games with poor lighting)

## What Makes a Good Test Video?

- Multiple players visible
- Ball is in frame frequently
- Camera relatively stable (Veo-style following is fine)
- At least one clear action (pass, shot, etc.)
- Duration: 10-60 seconds for quick tests

## Privacy Notice

Do NOT commit full match videos or videos containing identifiable people without proper consent. This directory is gitignored by default to prevent accidental commits.
