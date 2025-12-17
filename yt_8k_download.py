import os
import yt_dlp

# Progress bar function
def progress_hook(d):
    if d['status'] == 'downloading':
        total_bytes = d.get('total_bytes') or d.get('total_bytes_estimate')
        downloaded_bytes = d.get('downloaded_bytes', 0)
        speed = d.get('speed') or 0
        eta = d.get('eta') or 0

        percentage = (downloaded_bytes / total_bytes * 100) if total_bytes else 0
        total_mb = total_bytes / (1024 * 1024) if total_bytes else 0
        downloaded_mb = downloaded_bytes / (1024 * 1024)
        speed_mb = speed / (1024 * 1024) if speed else 0

        bar_length = 30
        filled_length = int(bar_length * percentage // 100)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)

        eta_str = f"{int(eta)}s" if eta else "Unknown"
        speed_str = f"{speed_mb:.2f}MB/s" if speed else "Unknown"

        print(f"\r{d['filename']}\n{percentage:3.0f}%|{bar}| "
              f"{downloaded_mb:.2f}MB/{total_mb:.2f}MB: [{eta_str} left, {speed_str}]", end='')
    elif d['status'] == 'finished':
        print(f"\nDownload completed: {d['filename']}")

# User inputs
video_url = input("Enter YouTube video link: ").strip()
save_folder = input("Enter folder to save the video (e.g., C:\\Users\\YourName\\Videos): ").strip()
os.makedirs(save_folder, exist_ok=True)

# Temporary yt-dlp to list formats
ydl_opts_list = {
    'quiet': True,
    'extractor_args': {'youtube': 'player_client=default,web,web_safari,android_vr,tv'},
    'js_runtimes': {'deno': {}},
}
with yt_dlp.YoutubeDL(ydl_opts_list) as ydl:
    info_dict = ydl.extract_info(video_url, download=False)
    formats = info_dict.get('formats', [])

# Collect available resolutions
resolutions = sorted(set([f"{f['height']}p" for f in formats if f.get('height')]))
print("\nAvailable resolutions:")
for r in resolutions:
    print(f"  {r}")

desired_res = input("Enter desired resolution (e.g., 1080): ").strip()

# Choose codec
print("\nAvailable codecs:")
print("  1: H.264 (avc1)")
print("  2: H.265 / HEVC (hevc)")
print("  3: AV1 (av01)")
codec_choice = input("Choose codec (1-3, default 1): ").strip()
codec_map = {'1': 'avc1', '2': 'hevc', '3': 'av01'}
selected_codec = codec_map.get(codec_choice, 'avc1')

# yt-dlp options for download
ydl_opts_download = {
    'format': f'bestvideo[height={desired_res}][vcodec~={selected_codec}]+bestaudio/best',
    'outtmpl': os.path.join(save_folder, '%(title)s.%(ext)s'),
    'progress_hooks': [progress_hook],
    'extractor_args': {'youtube': 'player_client=default,web,web_safari,android_vr,tv'},
    'js_runtimes': {'deno': {}},
    'noplaylist': True,
}

with yt_dlp.YoutubeDL(ydl_opts_download) as ydl:
    ydl.download([video_url])
