#!/usr/bin/env python3
"""
yt_audio_downloader.py
A robust YouTube audio downloader built on yt-dlp.
"""

import argparse
import shutil
import sys
from pathlib import Path

from yt_dlp import YoutubeDL


# ─────────────────────────── Dependency check ────────────────────────────────

def check_dependencies() -> None:
    """Fail fast with a helpful message if ffmpeg is not installed."""
    if shutil.which("ffmpeg") is None:
        print(
            "[Error] ffmpeg is not installed or not in PATH.\n"
            "  • macOS:   brew install ffmpeg\n"
            "  • Ubuntu:  sudo apt install ffmpeg\n"
            "  • Windows: https://ffmpeg.org/download.html",
            file=sys.stderr,
        )
        sys.exit(1)


# ─────────────────────────── Custom logger ───────────────────────────────────

class QuietLogger:
    """
    Suppress yt-dlp's default stdout chatter while still surfacing
    genuine warnings and errors to stderr.
    """

    def debug(self, msg: str) -> None:
        pass

    def info(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        print(f"[yt-dlp warning] {msg}", file=sys.stderr)

    def error(self, msg: str) -> None:
        print(f"[yt-dlp error] {msg}", file=sys.stderr)


# ─────────────────────────── Progress hook ───────────────────────────────────

def progress_hook(d: dict) -> None:
    status = d["status"]

    if status == "downloading":
        total      = d.get("total_bytes") or d.get("total_bytes_estimate")
        downloaded = d.get("downloaded_bytes", 0)
        speed      = d.get("speed")
        eta        = d.get("eta")

        speed_str = f"{speed / 1024 / 1024:.2f} MB/s" if speed else "N/A"

        if total:
            percent = downloaded * 100 / total
            eta_str = f"{eta}s" if eta is not None else "?"
            print(
                f"\r  {percent:6.2f}% | {speed_str} | ETA: {eta_str}   ",
                end="",
                flush=True,
            )
        else:
            # Size unknown (live stream, some DASH formats)
            print(
                f"\r  {downloaded / 1024 / 1024:.2f} MB downloaded | {speed_str}   ",
                end="",
                flush=True,
            )

    elif status == "finished":
        print("\n  ✓ Download complete, processing…")

    elif status == "error":
        print("\n  ✗ A fragment error occurred (yt-dlp will retry).", file=sys.stderr)


# ─────────────────────────── Output template ─────────────────────────────────

def build_output_template(custom: str | None, is_playlist: bool) -> str:
    """
    Return an appropriate yt-dlp output template.
    Playlist downloads get an index prefix to prevent filename collisions.
    """
    if custom:
        return custom

    music_dir = Path.home() / "Music"
    music_dir.mkdir(parents=True, exist_ok=True)

    if is_playlist:
        return str(music_dir / "%(playlist_title)s" / "%(playlist_index)s - %(title)s.%(ext)s")
    return str(music_dir / "%(title)s.%(ext)s")


def is_playlist_url(url: str) -> bool:
    """Heuristic: treat the URL as a playlist if it contains 'list='."""
    return "list=" in url


# ─────────────────────────── CLI ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download audio from YouTube videos or playlists.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "url",
        help="YouTube video or playlist URL",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Custom yt-dlp output template (overrides default ~/Music path)",
    )
    parser.add_argument(
        "-c", "--codec",
        default="m4a",
        choices=["m4a", "mp3", "opus", "flac"],
        help="Audio codec to encode to",
    )
    parser.add_argument(
        "-q", "--quality",
        default="192",
        help=(
            "Audio bitrate for lossy codecs (kbps). "
            "Ignored for flac. Use '0' for VBR best on mp3."
        ),
    )
    parser.add_argument(
        "-p", "--parallel",
        type=int,
        default=1,
        help="Parallel fragment downloads (only useful for DASH/HLS streams)",
    )
    parser.add_argument(
        "--no-thumbnail",
        action="store_true",
        help="Skip downloading and embedding the cover thumbnail",
    )
    return parser.parse_args()


# ─────────────────────────── Postprocessors ──────────────────────────────────

def build_postprocessors(codec: str, quality: str, embed_thumbnail: bool) -> list:
    """
    Build the postprocessor chain in the correct order:
      1. Write metadata tags (title, artist, album, etc.)
      2. Extract / re-encode audio
      3. Convert thumbnail to JPEG  <- only if embedding
      4. Embed thumbnail            <- only if embedding
    """
    postprocessors = [
        # Must come BEFORE FFmpegExtractAudio so tags are applied to final file
        {
            "key": "FFmpegMetadata",
            "add_metadata": True,
        },
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": codec,
            # quality "0" = VBR best for mp3; for flac quality is irrelevant
            "preferredquality": quality if codec != "flac" else "0",
        },
    ]

    if embed_thumbnail:
        postprocessors += [
            {
                "key": "FFmpegThumbnailsConvertor",
                "format": "jpg",
            },
            {
                "key": "EmbedThumbnail",
            },
        ]

    return postprocessors


# ─────────────────────────── Main ────────────────────────────────────────────

def main() -> None:
    check_dependencies()
    args = parse_args()

    playlist = is_playlist_url(args.url)
    if playlist:
        print("[Info] Playlist URL detected — files will be grouped by playlist title.")

    outtmpl        = build_output_template(args.output, playlist)
    embed_thumb    = not args.no_thumbnail
    postprocessors = build_postprocessors(args.codec, args.quality, embed_thumb)

    ydl_opts = {
        # ── Format ──────────────────────────────────────────────────────────
        "format": "bestaudio/best",
        "outtmpl": outtmpl,

        # ── Logging — custom logger instead of quiet+no_warnings ─────────────
        "logger": QuietLogger(),

        # ── Filenames ────────────────────────────────────────────────────────
        "restrictfilenames": True,

        # ── Thumbnail (only download if we plan to embed it) ─────────────────
        "writethumbnail": embed_thumb,
        # NOTE: Do NOT set "embedthumbnail": True here.
        # The EmbedThumbnail postprocessor handles it.
        # Setting both causes double-embedding and potential file corruption.

        # ── Network / extractor ──────────────────────────────────────────────
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"],
            }
        },

        # ── Retries ──────────────────────────────────────────────────────────
        "retries": 10,
        "fragment_retries": 10,

        # ── Parallelism ──────────────────────────────────────────────────────
        "concurrent_fragment_downloads": args.parallel,

        # ── Postprocessing ───────────────────────────────────────────────────
        "postprocessors": postprocessors,

        # ── Progress ─────────────────────────────────────────────────────────
        "progress_hooks": [progress_hook],
    }

    print(
        f"[Info] Codec: {args.codec.upper()} | Quality: {args.quality} kbps "
        f"| Thumbnail: {'yes' if embed_thumb else 'no'} | Output: {outtmpl}\n"
    )

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([args.url])
    except KeyboardInterrupt:
        print("\n[Aborted] Download cancelled by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"\n[Error] Download failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n[Done] All downloads finished.")


if __name__ == "__main__":
    main()