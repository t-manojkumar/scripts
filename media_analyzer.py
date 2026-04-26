#!/usr/bin/env python3
"""
Media Analyzer — Fast file scanner with timeline HTML report.
Usage: python media_analyzer.py <directory> [--workers N] [--output report.html]
"""

import os
import sys
import json
import mimetypes
import argparse
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import Optional

# ─── Optional deps ────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn,
        TextColumn, TimeRemainingColumn, MofNCompleteColumn, TaskProgressColumn,
    )
    from rich.panel import Panel
    from rich.table import Table
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    console = None

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ─── Constants ────────────────────────────────────────────────────────────────
IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif",
    ".webp", ".heic", ".heif", ".raw", ".cr2", ".nef", ".arw",
    ".dng", ".avif",
}
VIDEO_EXTS = {
    ".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm",
    ".m4v", ".3gp", ".mpeg", ".mpg", ".ts", ".mts", ".m2ts",
}


# ─── File helpers ─────────────────────────────────────────────────────────────
def get_exif_date(filepath: Path) -> Optional[datetime]:
    if not HAS_PIL:
        return None
    try:
        with Image.open(filepath) as img:
            exif = img._getexif()
            if exif:
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
                        return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return None


def get_file_date(filepath: Path, ext: str) -> datetime:
    if ext in IMAGE_EXTS:
        d = get_exif_date(filepath)
        if d:
            return d
    stat = filepath.stat()
    return datetime.fromtimestamp(min(stat.st_mtime, stat.st_ctime))


def get_file_type(ext: str) -> Optional[str]:
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    mime, _ = mimetypes.guess_type(f"x{ext}")
    if mime:
        if mime.startswith("image/"):
            return "image"
        if mime.startswith("video/"):
            return "video"
    return None


def format_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def analyze_file(filepath: Path) -> Optional[dict]:
    try:
        stat = filepath.stat()
        if stat.st_size == 0:
            return None
        ext = filepath.suffix.lower()
        ftype = get_file_type(ext)
        if ftype is None:
            return None
        date = get_file_date(filepath, ext)
        return {
            "name": filepath.name,
            "path": str(filepath),
            "type": ftype,
            "size": stat.st_size,
            "size_fmt": format_size(stat.st_size),
            "date": date.isoformat(),
            "year": date.year,
            "month": date.month,
            "day": date.day,
            "ext": ext.lstrip("."),
        }
    except Exception:
        return None


# ─── Scanner ──────────────────────────────────────────────────────────────────
def scan_directory(root: Path, max_workers: int = 12):
    if HAS_RICH:
        console.print(f"[dim]Collecting file list from[/dim] [cyan]{root}[/cyan]")

    all_files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fname in filenames:
            if not fname.startswith("."):
                all_files.append(Path(dirpath) / fname)

    total = len(all_files)
    results: list[dict] = []
    t0 = time.perf_counter()

    if HAS_RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=36),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=15,
        ) as progress:
            task = progress.add_task("Analyzing…", total=total)
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(analyze_file, f): f for f in all_files}
                for fut in as_completed(futures):
                    r = fut.result()
                    if r:
                        results.append(r)
                    progress.advance(task)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(analyze_file, f): f for f in all_files}
            done = 0
            for fut in as_completed(futures):
                r = fut.result()
                if r:
                    results.append(r)
                done += 1
                if done % 200 == 0 or done == total:
                    pct = done / max(total, 1) * 100
                    print(f"\r  [{pct:5.1f}%] {done}/{total} files", end="", flush=True)
        print()

    elapsed = time.perf_counter() - t0
    return results, total, elapsed


# ─── HTML report ──────────────────────────────────────────────────────────────
def build_html(results: list[dict], root: str, total_scanned: int, elapsed: float) -> str:
    results.sort(key=lambda x: x["date"])

    grouped: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for item in results:
        grouped[item["year"]][item["month"]][item["day"]].append(item)

    images = [r for r in results if r["type"] == "image"]
    videos = [r for r in results if r["type"] == "video"]
    total_size = sum(r["size"] for r in results)

    # ── timeline HTML ─────────────────────────────────────────────────────────
    tl = []
    for year in sorted(grouped, reverse=True):
        year_count = sum(len(v) for m in grouped[year].values() for v in m.values())
        tl.append(f'<div class="y-group">'
                   f'<div class="y-hdr" onclick="tog(this)">'
                   f'<span class="tri">▼</span>{year}'
                   f'<span class="badge">{year_count}</span></div>'
                   f'<div class="y-body">')
        for month in sorted(grouped[year], reverse=True):
            month_name = datetime(year, month, 1).strftime("%B")
            month_count = sum(len(v) for v in grouped[year][month].values())
            tl.append(f'<div class="m-group">'
                       f'<div class="m-hdr" onclick="tog(this)">'
                       f'<span class="tri">▼</span>{month_name}'
                       f'<span class="badge">{month_count}</span></div>'
                       f'<div class="m-body">')
            for day in sorted(grouped[year][month], reverse=True):
                items = grouped[year][month][day]
                date_label = datetime(year, month, day).strftime("%A, %d %B %Y")
                tl.append(f'<div class="d-group">'
                           f'<div class="d-hdr">{date_label}'
                           f'<span class="badge">{len(items)}</span></div>'
                           f'<div class="grid">')
                for item in sorted(items, key=lambda x: x["date"]):
                    icon = "🖼️" if item["type"] == "image" else "🎬"
                    tl.append(
                        f'<div class="card {item["type"]}" data-name="{item["name"].lower()}"'
                        f' data-type="{item["type"]}" title="{item["path"]}">'
                        f'<div class="card-icon">{icon}</div>'
                        f'<div class="card-name">{item["name"]}</div>'
                        f'<div class="card-meta">'
                        f'<span class="sz">{item["size_fmt"]}</span>'
                        f'<span class="ext">{item["ext"].upper()}</span>'
                        f'</div></div>'
                    )
                tl.append("</div></div>")
            tl.append("</div></div>")
        tl.append("</div></div>")

    timeline_html = "\n".join(tl)
    scan_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Media Analyzer</title>
<style>
:root[data-theme="dark"]{{
  --bg:#0d0d0d;--bg2:#161616;--bg3:#202020;--bg4:#2a2a2a;
  --bd:#2e2e2e;--tx:#e0e0e0;--tx2:#888;--tx3:#555;
  --acc:#4f9cf9;--img:#2563eb;--vid:#7c3aed;
  --shad:rgba(0,0,0,.5);
}}
:root[data-theme="light"]{{
  --bg:#f3f4f6;--bg2:#fff;--bg3:#f0f0f0;--bg4:#e5e7eb;
  --bd:#d1d5db;--tx:#111;--tx2:#6b7280;--tx3:#9ca3af;
  --acc:#2563eb;--img:#1d4ed8;--vid:#7c3aed;
  --shad:rgba(0,0,0,.08);
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  background:var(--bg);color:var(--tx);font-size:13px;line-height:1.5}}

/* ── header ── */
header{{background:var(--bg2);border-bottom:1px solid var(--bd);
  padding:14px 22px;display:flex;align-items:center;
  justify-content:space-between;position:sticky;top:0;z-index:50;
  backdrop-filter:blur(12px)}}
.h-title{{font-size:16px;font-weight:700;letter-spacing:-.02em}}
.h-path{{font-size:11px;color:var(--tx2);font-family:monospace;margin-top:2px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:55vw}}
.theme-btn{{background:var(--bg3);border:1px solid var(--bd);color:var(--tx);
  padding:5px 12px;border-radius:20px;cursor:pointer;font-size:12px;
  display:flex;align-items:center;gap:6px;transition:.15s}}
.theme-btn:hover{{background:var(--bg4)}}

/* ── stats bar ── */
.stats{{background:var(--bg2);border-bottom:1px solid var(--bd);
  padding:10px 22px;display:flex;gap:20px;flex-wrap:wrap;align-items:center}}
.stat{{text-align:center}}
.stat-v{{font-size:20px;font-weight:800;line-height:1}}
.stat-l{{font-size:10px;color:var(--tx2);text-transform:uppercase;
  letter-spacing:.06em;margin-top:2px}}
.sep{{width:1px;height:36px;background:var(--bd)}}

/* ── toolbar ── */
.toolbar{{background:var(--bg2);border-bottom:1px solid var(--bd);
  padding:9px 22px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}}
.search{{flex:1;max-width:360px;background:var(--bg3);border:1px solid var(--bd);
  color:var(--tx);padding:7px 12px;border-radius:8px;font-size:12px;outline:none;
  transition:.15s}}
.search:focus{{border-color:var(--acc)}}
.fbtn{{background:var(--bg3);border:1px solid var(--bd);color:var(--tx2);
  padding:5px 12px;border-radius:8px;cursor:pointer;font-size:11px;
  font-weight:600;transition:.15s}}
.fbtn:hover,.fbtn.on{{background:var(--acc);color:#fff;border-color:var(--acc)}}
.fbtn.img.on{{background:var(--img);border-color:var(--img)}}
.fbtn.vid.on{{background:var(--vid);border-color:var(--vid)}}

/* ── main ── */
main{{padding:18px 22px;max-width:1440px;margin:0 auto}}

/* ── year / month groups ── */
.y-group{{margin-bottom:8px}}
.y-hdr{{display:flex;align-items:center;gap:10px;padding:11px 14px;
  background:var(--bg2);border:1px solid var(--bd);border-radius:10px;
  cursor:pointer;font-size:18px;font-weight:800;user-select:none;
  transition:.12s}}
.y-hdr:hover{{background:var(--bg3)}}
.y-body{{padding-left:14px;margin-top:5px}}
.m-group{{margin-bottom:5px}}
.m-hdr{{display:flex;align-items:center;gap:9px;padding:8px 12px;
  background:var(--bg3);border:1px solid var(--bd);border-radius:8px;
  cursor:pointer;font-size:14px;font-weight:600;user-select:none;transition:.12s}}
.m-hdr:hover{{background:var(--bg4)}}
.m-body{{padding-left:12px;margin-top:4px}}

/* ── day ── */
.d-group{{margin-bottom:12px}}
.d-hdr{{font-size:11px;font-weight:600;color:var(--tx2);padding:5px 0;
  display:flex;align-items:center;gap:8px;text-transform:uppercase;
  letter-spacing:.06em}}

/* ── grid ── */
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(152px,1fr));gap:7px}}
.card{{background:var(--bg2);border:1px solid var(--bd);border-radius:8px;
  padding:11px;transition:.12s;overflow:hidden;cursor:default}}
.card:hover{{background:var(--bg3);border-color:var(--acc);
  transform:translateY(-2px);box-shadow:0 4px 14px var(--shad)}}
.card.image{{border-left:3px solid var(--img)}}
.card.video{{border-left:3px solid var(--vid)}}
.card-icon{{font-size:20px;margin-bottom:5px}}
.card-name{{font-size:11px;font-weight:500;word-break:break-all;
  max-height:2.8em;overflow:hidden;line-height:1.4}}
.card-meta{{display:flex;justify-content:space-between;margin-top:7px;
  align-items:center}}
.sz{{font-size:10px;color:var(--tx2)}}
.ext{{font-size:9px;background:var(--bg4);color:var(--tx2);
  padding:1px 5px;border-radius:4px;font-weight:700}}

/* ── misc ── */
.badge{{margin-left:auto;background:var(--bg4);color:var(--tx2);
  border-radius:10px;padding:1px 8px;font-size:10px;font-weight:600}}
.tri{{font-size:10px;color:var(--tx3);transition:.18s;width:14px;display:inline-block}}
.collapsed>.tri{{transform:rotate(-90deg)}}
.hidden{{display:none}}
footer{{text-align:center;padding:18px;color:var(--tx3);font-size:10px;
  border-top:1px solid var(--bd);margin-top:30px}}

@media(max-width:600px){{
  .grid{{grid-template-columns:repeat(auto-fill,minmax(130px,1fr))}}
  .stats{{gap:12px}}
}}
</style>
</head>
<body>

<header>
  <div>
    <div class="h-title">Media Analyzer</div>
    <div class="h-path">{root}</div>
  </div>
  <button class="theme-btn" onclick="toggleTheme()">
    <span id="ticon">☀️</span><span id="tlbl">Light</span>
  </button>
</header>

<div class="stats">
  <div class="stat">
    <div class="stat-v">{len(results)}</div>
    <div class="stat-l">Media Files</div>
  </div>
  <div class="sep"></div>
  <div class="stat">
    <div class="stat-v" style="color:var(--img)">{len(images)}</div>
    <div class="stat-l">🖼 Images</div>
  </div>
  <div class="sep"></div>
  <div class="stat">
    <div class="stat-v" style="color:var(--vid)">{len(videos)}</div>
    <div class="stat-l">🎬 Videos</div>
  </div>
  <div class="sep"></div>
  <div class="stat">
    <div class="stat-v">{format_size(total_size)}</div>
    <div class="stat-l">Total Size</div>
  </div>
  <div class="sep"></div>
  <div class="stat">
    <div class="stat-v">{total_scanned}</div>
    <div class="stat-l">Files Scanned</div>
  </div>
  <div class="sep"></div>
  <div class="stat">
    <div class="stat-v">{elapsed:.1f}s</div>
    <div class="stat-l">Scan Time</div>
  </div>
</div>

<div class="toolbar">
  <input class="search" type="text" placeholder="Search files…" oninput="doSearch(this.value)">
  <button class="fbtn on"  id="f-all" onclick="setFilter('all')">All</button>
  <button class="fbtn img" id="f-image" onclick="setFilter('image')">🖼 Images</button>
  <button class="fbtn vid" id="f-video" onclick="setFilter('video')">🎬 Videos</button>
</div>

<main id="main">
{timeline_html}
</main>

<footer>
  Scanned {total_scanned} files in {elapsed:.2f}s &nbsp;·&nbsp;
  {len(results)} media files found &nbsp;·&nbsp;
  Generated {scan_ts}
</footer>

<script>
// ── theme ──────────────────────────────────────────────────────────────
const saved = localStorage.getItem('ma-theme') || 'dark';
applyTheme(saved);
function applyTheme(t){{
  document.documentElement.setAttribute('data-theme', t);
  document.getElementById('ticon').textContent = t==='dark'?'☀️':'🌙';
  document.getElementById('tlbl').textContent  = t==='dark'?'Light':'Dark';
}}
function toggleTheme(){{
  const t = document.documentElement.getAttribute('data-theme')==='dark'?'light':'dark';
  applyTheme(t); localStorage.setItem('ma-theme', t);
}}

// ── collapse ───────────────────────────────────────────────────────────
function tog(hdr){{
  hdr.classList.toggle('collapsed');
  const body = hdr.nextElementSibling;
  if(body) body.classList.toggle('hidden');
}}

// ── filter & search ────────────────────────────────────────────────────
let curFilter='all', curSearch='';

function setFilter(f){{
  curFilter=f;
  ['all','image','video'].forEach(x=>{{
    const b=document.getElementById('f-'+x);
    b.classList.toggle('on', x===f);
  }});
  apply();
}}
function doSearch(v){{ curSearch=v.toLowerCase(); apply(); }}

function apply(){{
  document.querySelectorAll('.card').forEach(c=>{{
    const nm=c.dataset.name||'';
    const tp=c.dataset.type||'';
    const show=(curFilter==='all'||tp===curFilter)&&(!curSearch||nm.includes(curSearch));
    c.classList.toggle('hidden',!show);
  }});
  // hide empty day/month/year groups
  document.querySelectorAll('.d-group').forEach(d=>{{
    d.classList.toggle('hidden', !d.querySelector('.card:not(.hidden)'));
  }});
  document.querySelectorAll('.m-group').forEach(m=>{{
    m.classList.toggle('hidden', !m.querySelector('.d-group:not(.hidden)'));
  }});
  document.querySelectorAll('.y-group').forEach(y=>{{
    y.classList.toggle('hidden', !y.querySelector('.m-group:not(.hidden)'));
  }});
}}
</script>
</body>
</html>"""


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Analyze and timeline-sort media files in a directory."
    )
    parser.add_argument("directory", help="Root directory to scan")
    parser.add_argument(
        "--workers", type=int, default=12, help="Parallel worker threads (default: 12)"
    )
    parser.add_argument(
        "--output", default="media_report.html", help="Output HTML path (default: media_report.html)"
    )
    args = parser.parse_args()

    root = Path(args.directory).resolve()
    if not root.is_dir():
        print(f"Error: '{root}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    if HAS_RICH:
        console.print(Panel(
            f"[bold]Media Analyzer[/bold]\n"
            f"[dim]Target:[/dim] [cyan]{root}[/cyan]\n"
            f"[dim]Workers:[/dim] {args.workers}  "
            f"[dim]EXIF:[/dim] {'✓' if HAS_PIL else '✗ (pip install Pillow)'}",
            border_style="bright_blue", expand=False,
        ))
    else:
        print(f"\nMedia Analyzer\nTarget: {root}\nWorkers: {args.workers}\n")

    results, total_scanned, elapsed = scan_directory(root, args.workers)

    images = sum(1 for r in results if r["type"] == "image")
    videos = sum(1 for r in results if r["type"] == "video")
    total_size = sum(r["size"] for r in results)

    if HAS_RICH:
        tbl = Table(show_header=False, box=None, padding=(0, 2))
        tbl.add_column(style="dim")
        tbl.add_column(style="bold")
        tbl.add_row("Files scanned", str(total_scanned))
        tbl.add_row("Media found",   str(len(results)))
        tbl.add_row("Images",        f"[blue]{images}[/blue]")
        tbl.add_row("Videos",        f"[purple]{videos}[/purple]")
        tbl.add_row("Total size",    format_size(total_size))
        tbl.add_row("Elapsed",       f"{elapsed:.2f}s")
        console.print(tbl)
    else:
        print(f"\nResults: {len(results)} media ({images} images, {videos} videos)")
        print(f"Total size: {format_size(total_size)}  |  Elapsed: {elapsed:.2f}s\n")

    html = build_html(results, str(root), total_scanned, elapsed)
    out = Path(args.output)
    out.write_text(html, encoding="utf-8")

    if HAS_RICH:
        console.print(f"\n[green]✓[/green] Report saved → [bold]{out.resolve()}[/bold]")
    else:
        print(f"Report saved: {out.resolve()}")


if __name__ == "__main__":
    main()
