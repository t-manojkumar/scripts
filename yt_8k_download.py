import os
import subprocess
import yt_dlp
from typing import Dict, List

# ===================== Progress Hook =====================
def progress_hook(d: Dict):
    if d['status'] == 'downloading':
        total = d.get('total_bytes') or d.get('total_bytes_estimate')
        downloaded = d.get('downloaded_bytes', 0)
        percent = downloaded / total * 100 if total else 0
        speed = (d.get('speed') or 0) / (1024 * 1024)
        eta = d.get('eta') or 0

        bar_len = 30
        filled = int(bar_len * percent // 100)
        bar = '█' * filled + '-' * (bar_len - filled)

        print(
            f"\r[{bar}] {percent:5.1f}% | "
            f"Speed: {speed:5.2f} MB/s | ETA: {eta}s",
            end=''
        )
    elif d['status'] == 'finished':
        print(f"\n✔ Download finished: {d['filename']}")


# ===================== User Input =====================
url = input("Video URL: ").strip()
out_dir = input("Save directory: ").strip()
os.makedirs(out_dir, exist_ok=True)

# ===================== Probe Formats (Reliable Highest Detection) =====================
# Use clients that do NOT require PO Tokens
probe_opts = {
    'quiet': True,
    'extractor_args': {
        'youtube': {
            'player_client': ['web', 'tv_embedded', 'web_safari']
        }
    }
}

with yt_dlp.YoutubeDL(probe_opts) as ydl:
    info = ydl.extract_info(url, download=False)

formats = info.get('formats', [])

# Filter real video formats only (exclude storyboards)
video_formats = [
    f for f in formats
    if f.get('vcodec') not in (None, 'none') and f.get('height')
]

if not video_formats:
    raise RuntimeError("No downloadable video formats found")

# Find highest resolution available
max_height = max(f['height'] for f in video_formats)

print(f"\n✔ Highest available resolution detected: {max_height}p")

# Prefer codec order: AV1 > VP9 > H.264
codec_priority = ['av01', 'vp9', 'avc1']

selected_codec = None
for codec in codec_priority:
    if any(codec in (f.get('vcodec') or '') for f in video_formats):
        selected_codec = codec
        break

print(f"✔ Selected codec: {selected_codec.upper()}")

# Detect HDR availability
hdr_available = any(f.get('dynamic_range') == 'HDR' or f.get('hdr') for f in video_formats)
print(f"✔ HDR available: {'YES' if hdr_available else 'NO'}")

# ===================== Format Selector =====================
format_selector = (
    f"bestvideo[vcodec*={selected_codec}][height={max_height}]/"
    f"bestvideo[height={max_height}]+bestaudio/best"
)

# ===================== Download Options =====================
ydl_opts = {
    'format': format_selector,
    'outtmpl': os.path.join(out_dir, '%(title)s.%(ext)s'),
    'merge_output_format': 'mkv',
    'progress_hooks': [progress_hook],
    'quiet': True,
    'noplaylist': True,

    # Performance
    'concurrent_fragment_downloads': 8,
    'continuedl': True,
    'retries': 10,
    'fragment_retries': 10,

    # aria2 for max speed
    'external_downloader': 'aria2c',
    'external_downloader_args': ['-x16', '-k1M'],

    # Same extractor args for actual download
    'extractor_args': {
        'youtube': {
            'player_client': ['web', 'tv_embedded', 'web_safari']
        }
    }
}

# ===================== Download =====================
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])
