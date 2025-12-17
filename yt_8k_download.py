import os
import sys
import yt_dlp

# ---------- Progress Hook ----------
def progress_hook(d):
    if d['status'] == 'downloading':
        total_bytes = d.get('total_bytes') or d.get('total_bytes_estimate')
        downloaded_bytes = d.get('downloaded_bytes', 0)
        percentage = (downloaded_bytes / total_bytes * 100) if total_bytes else 0
        downloaded_mb = downloaded_bytes / (1024 * 1024)
        total_mb = total_bytes / (1024 * 1024) if total_bytes else 0
        speed = d.get('speed') or 0
        speed_mb = speed / (1024 * 1024) if speed else 0
        eta = d.get('eta') or 0

        bar_length = 30
        filled_length = int(bar_length * percentage // 100)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        eta_str = f"{int(eta)}s" if eta else "Unknown"
        speed_str = f"{speed_mb:.2f} MB/s" if speed else "Unknown"

        print(f"\r[{bar}] {percentage:3.0f}% "
              f"{downloaded_mb:.2f}/{total_mb:.2f} MB "
              f"Speed: {speed_str} ETA: {eta_str}", end='')
    elif d['status'] == 'finished':
        print(f"\nDownload completed: {d['filename']}")

# ---------- User Inputs ----------
video_url = input("Video link: ").strip()
save_folder = input("Where to save: ").strip()
os.makedirs(save_folder, exist_ok=True)

# ---------- Get Video Info ----------
ydl_opts_info = {'quiet': True}
with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
    info = ydl.extract_info(video_url, download=False)
    title = info.get('title', 'Unknown Title')
    formats = info.get('formats', [])

# ---------- Available Resolutions ----------
resolutions = sorted({f['height'] for f in formats if f.get('height')}, reverse=True)

# Print in single row
print("\nAvailable formats to download:")
res_row = " | ".join([f"{i+1}: {res}p" for i, res in enumerate(resolutions)])
print(res_row)

choice = input("Choose resolution (number): ").strip()
if choice.isdigit() and 1 <= int(choice) <= len(resolutions):
    desired_res = str(resolutions[int(choice) - 1])
else:
    desired_res = str(resolutions[0])  # default to highest


print(f"\nVideo Title: {title}\nDownloading in {desired_res}p...\n")

# ---------- Download ----------
outtmpl = os.path.join(save_folder, '%(title)s.%(ext)s')
ydl_opts_download = {
    'format': f'bestvideo[height={desired_res}]+bestaudio/best',
    'outtmpl': outtmpl,
    'progress_hooks': [progress_hook],
    'quiet': True,
    'noplaylist': True,
}

with yt_dlp.YoutubeDL(ydl_opts_download) as ydl:
    ydl.download([video_url])
