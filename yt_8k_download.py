import yt_dlp
import os

# Ask for YouTube link
video_url = input("Enter YouTube video link: ").strip()

# Ask where to save
save_path = input("Enter folder to save the video (e.g., C:\\Users\\YourName\\Videos): ").strip()
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Function to display progress
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

        # Safe formatting
        eta_str = f"{int(eta)}s" if eta else "Unknown"
        speed_str = f"{speed_mb:.2f}MB/s" if speed else "Unknown"

        print(f"\r{d['filename']}\n{percentage:3.0f}%|{bar}| "
              f"{downloaded_mb:.2f}MB/{total_mb:.2f}MB:  [{eta_str} left, {speed_str}]", end='')
    elif d['status'] == 'finished':
        print(f"\nDownload completed: {d['filename']}")

# Extract available formats and ask user which resolution
with yt_dlp.YoutubeDL({
    'quiet': True,
    'extractor_args': {'youtube': 'player_client=default,web,web_safari,android_vr,tv'},
    'js_runtimes': {'deno': {}, 'node': {}}  # Correct format
}) as ydl:
    info = ydl.extract_info(video_url, download=False)
    formats = info['formats']
    res_list = sorted({f"{f['height']}p" for f in formats if f.get('height')}, reverse=True)
    print("\nAvailable resolutions:")
    for res in res_list:
        print(f"  {res}")
    chosen_res = input("Enter desired resolution (e.g., 1080): ").strip()

    # Filter for chosen resolution
    selected_format = None
    for f in formats:
        if str(f.get('height')) == chosen_res:
            selected_format = f['format_id']
            break

    if not selected_format:
        print(f"No format found for {chosen_res}p. Downloading best available.")
        selected_format = 'bestvideo+bestaudio/best'

# yt-dlp options
ydl_opts = {
    'format': selected_format,
    'outtmpl': os.path.join(save_path, '%(title)s.%(ext)s'),
    'progress_hooks': [progress_hook],
    'noprogress': False,
    'extractor_args': {'youtube': 'player_client=default,web,web_safari,android_vr,tv'},
    'js_runtimes': {'deno': {}, 'node': {}}
}

# Start download
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([video_url])
