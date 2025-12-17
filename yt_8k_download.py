import os
import yt_dlp

# ---------- Progress Bar Function ----------
def progress_hook(d):
    if d['status'] == 'downloading':
        total_bytes = d.get('total_bytes') or d.get('total_bytes_estimate')
        downloaded_bytes = d.get('downloaded_bytes', 0)
        percentage = (downloaded_bytes / total_bytes * 100) if total_bytes else 0
        downloaded_mb = downloaded_bytes / (1024 * 1024)
        total_mb = total_bytes / (1024 * 1024) if total_bytes else 0
        print(f"\rDownloading: {percentage:3.0f}% [{downloaded_mb:.2f}/{total_mb:.2f} MB]", end='')
    elif d['status'] == 'finished':
        print(f"\nDownload completed: {d['filename']}")

# ---------- User Inputs ----------
video_url = input("Enter YouTube video link: ").strip()
save_folder = input("Enter folder to save the video: ").strip()
os.makedirs(save_folder, exist_ok=True)

# ---------- List Available Resolutions ----------
ydl_opts_list = {'quiet': True}
with yt_dlp.YoutubeDL(ydl_opts_list) as ydl:
    info_dict = ydl.extract_info(video_url, download=False)
formats = info_dict.get('formats', [])
resolutions = sorted(set([str(f['height']) for f in formats if f.get('height')]))
print("\nAvailable resolutions:")
for r in resolutions:
    print(f"  {r}p")

desired_res = input("Enter desired resolution (e.g., 1080): ").strip()

# ---------- yt-dlp Download ----------
ydl_opts_download = {
    'format': f'bestvideo[height={desired_res}]+bestaudio/best',
    'outtmpl': os.path.join(save_folder, '%(title)s.%(ext)s'),
    'progress_hooks': [progress_hook],
    'noplaylist': True,
}

with yt_dlp.YoutubeDL(ydl_opts_download) as ydl:
    ydl.download([video_url])
