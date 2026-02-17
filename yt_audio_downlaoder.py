#!/usr/bin/env python3

import argparse
from pathlib import Path
from yt_dlp import YoutubeDL


def progress_hook(d):
    if d["status"] == "downloading":
        total = d.get("total_bytes") or d.get("total_bytes_estimate")
        downloaded = d.get("downloaded_bytes", 0)
        speed = d.get("speed")
        eta = d.get("eta")

        if total:
            percent = downloaded * 100 / total
            speed_str = f"{speed/1024/1024:.2f} MB/s" if speed else "N/A"
            eta_str = f"{eta}s" if eta else "?"
            print(
                f"\r{percent:6.2f}% | {speed_str} | ETA: {eta_str}",
                end="",
                flush=True,
            )

    elif d["status"] == "finished":
        print("\nDone")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="YouTube video or playlist URL")
    parser.add_argument(
        "-o",
        "--output",
        help="Custom output template (overrides Music folder)",
        default=None,
    )
    parser.add_argument(
        "-p",
        "--parallel",
        type=int,
        default=4,
        help="Parallel fragment downloads",
    )
    args = parser.parse_args()

    # Cross-platform Music directory
    if args.output:
        outtmpl = args.output
    else:
        music_dir = Path.home() / "Music"
        music_dir.mkdir(exist_ok=True)
        outtmpl = str(music_dir / "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,

        "quiet": True,
        "no_warnings": True,

        # ✅ Smart filename sanitization
        "restrictfilenames": True,

        "writethumbnail": True,
        "embedthumbnail": True,

        "user_agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),

        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"],
            }
        },

        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "m4a",
                "preferredquality": "0",
            },
            {
                "key": "FFmpegThumbnailsConvertor",
                "format": "jpg",
            },
            {
                "key": "EmbedThumbnail",
            },
        ],

        "concurrent_fragment_downloads": args.parallel,
        "retries": 10,
        "fragment_retries": 10,

        # ✅ Progress with speed + ETA
        "progress_hooks": [progress_hook],
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([args.url])


if __name__ == "__main__":
    main()
