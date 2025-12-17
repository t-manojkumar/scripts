import os
import yt_dlp
from typing import Dict

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

print("\nSelect video codec:")
print("1) AV1 (best compression, slow decode)")
print("2) VP9 (recommended)")
print("3) H.264 (max compatibility)")

codec_choice = input("Choice [1-3]: ").strip()
codec_map = {
    '1': 'av1',
    '2': 'vp9',
    '3': 'avc1'
}
selected_codec = codec_map.get(codec_choice, 'vp9')

print("Select resolution:")
print("1) 1080p")
print("2) 4K (2160p)")
print("3) 8K (4320p)")
res_choice = input("Choice [1-3]: ").strip()
res_map = {
    '1': 1080,
    '2': 2160,
    '3': 4320
}
selected_height = res_map.get(res_choice, None)

print("HDR option:")
print("1) HDR (if available)")
print("2) SDR only")
hdr_choice = input("Choice [1-2]: ").strip()

hdr_filter = '[hdr=1]' if hdr_choice == '1' else ''
height_filter = f"[height={selected_height}]" if selected_height else ''

# ===================== yt-dlp Options =====================
format_selector = (
    f"bestvideo[vcodec*={selected_codec}]{hdr_filter}{height_filter}/"
    f"bestvideo[vcodec*={selected_codec}]+bestaudio/best"
)

ydl_opts = {
    'format': format_selector,
    'outtmpl': os.path.join(out_dir, '%(title)s.%(ext)s'),
    'merge_output_format': 'mkv',
    'progress_hooks': [progress_hook],
    'quiet': True,
    'noplaylist': True,

    # Performance optimizations
    'concurrent_fragment_downloads': 8,
    'retries': 10,
    'fragment_retries': 10,
    'continuedl': True,

    # External downloader (aria2)
    'external_downloader': 'aria2c',
    'external_downloader_args': ['-x16', '-k1M'],
}

# ===================== Download =====================
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])
