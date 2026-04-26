#!/usr/bin/env python3
"""
Media Analyzer — Windows GUI App
Requires: pip install customtkinter Pillow
"""

import os
import sys
import time
import threading
import mimetypes
import webbrowser
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import Optional
import queue

import customtkinter as ctk
from tkinter import filedialog, messagebox

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ─── File type sets ───────────────────────────────────────────────────────────
IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif",
    ".webp", ".heic", ".heif", ".raw", ".cr2", ".nef", ".arw",
    ".dng", ".avif",
}
VIDEO_EXTS = {
    ".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm",
    ".m4v", ".3gp", ".mpeg", ".mpg", ".ts", ".mts", ".m2ts",
}

# ─── Colours ──────────────────────────────────────────────────────────────────
DARK = {
    "bg":      "#0f0f0f",
    "panel":   "#181818",
    "card":    "#212121",
    "border":  "#2e2e2e",
    "accent":  "#4f9cf9",
    "img":     "#2563eb",
    "vid":     "#7c3aed",
    "success": "#22c55e",
    "text":    "#e0e0e0",
    "muted":   "#6b7280",
}
LIGHT = {
    "bg":      "#f3f4f6",
    "panel":   "#ffffff",
    "card":    "#f9fafb",
    "border":  "#d1d5db",
    "accent":  "#2563eb",
    "img":     "#1d4ed8",
    "vid":     "#7c3aed",
    "success": "#16a34a",
    "text":    "#111827",
    "muted":   "#6b7280",
}


# ─── Core analysis logic (same as CLI) ───────────────────────────────────────
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


def build_html(results, root, total_scanned, elapsed):
    results = sorted(results, key=lambda x: x["date"])
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for item in results:
        grouped[item["year"]][item["month"]][item["day"]].append(item)

    images = [r for r in results if r["type"] == "image"]
    videos = [r for r in results if r["type"] == "video"]
    total_size = sum(r["size"] for r in results)

    tl = []
    for year in sorted(grouped, reverse=True):
        year_count = sum(len(v) for m in grouped[year].values() for v in m.values())
        tl.append(f'<div class="y-group"><div class="y-hdr" onclick="tog(this)"><span class="tri">▼</span>{year}<span class="badge">{year_count}</span></div><div class="y-body">')
        for month in sorted(grouped[year], reverse=True):
            month_name = datetime(year, month, 1).strftime("%B")
            month_count = sum(len(v) for v in grouped[year][month].values())
            tl.append(f'<div class="m-group"><div class="m-hdr" onclick="tog(this)"><span class="tri">▼</span>{month_name}<span class="badge">{month_count}</span></div><div class="m-body">')
            for day in sorted(grouped[year][month], reverse=True):
                items = grouped[year][month][day]
                date_label = datetime(year, month, day).strftime("%A, %d %B %Y")
                tl.append(f'<div class="d-group"><div class="d-hdr">{date_label}<span class="badge">{len(items)}</span></div><div class="grid">')
                for item in sorted(items, key=lambda x: x["date"]):
                    icon = "🖼️" if item["type"] == "image" else "🎬"
                    tl.append(f'<div class="card {item["type"]}" data-name="{item["name"].lower()}" data-type="{item["type"]}" title="{item["path"]}"><div class="card-icon">{icon}</div><div class="card-name">{item["name"]}</div><div class="card-meta"><span class="sz">{item["size_fmt"]}</span><span class="ext">{item["ext"].upper()}</span></div></div>')
                tl.append("</div></div>")
            tl.append("</div></div>")
        tl.append("</div></div>")

    scan_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timeline_html = "\n".join(tl)

    return f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Media Analyzer</title>
<style>
:root[data-theme="dark"]{{--bg:#0d0d0d;--bg2:#161616;--bg3:#202020;--bg4:#2a2a2a;--bd:#2e2e2e;--tx:#e0e0e0;--tx2:#888;--tx3:#555;--acc:#4f9cf9;--img:#2563eb;--vid:#7c3aed;--shad:rgba(0,0,0,.5)}}
:root[data-theme="light"]{{--bg:#f3f4f6;--bg2:#fff;--bg3:#f0f0f0;--bg4:#e5e7eb;--bd:#d1d5db;--tx:#111;--tx2:#6b7280;--tx3:#9ca3af;--acc:#2563eb;--img:#1d4ed8;--vid:#7c3aed;--shad:rgba(0,0,0,.08)}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);color:var(--tx);font-size:13px;line-height:1.5}}
header{{background:var(--bg2);border-bottom:1px solid var(--bd);padding:14px 22px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:50}}
.h-title{{font-size:16px;font-weight:700}}.h-path{{font-size:11px;color:var(--tx2);font-family:monospace;margin-top:2px}}
.theme-btn{{background:var(--bg3);border:1px solid var(--bd);color:var(--tx);padding:5px 12px;border-radius:20px;cursor:pointer;font-size:12px;display:flex;align-items:center;gap:6px}}
.stats{{background:var(--bg2);border-bottom:1px solid var(--bd);padding:10px 22px;display:flex;gap:20px;flex-wrap:wrap;align-items:center}}
.stat{{text-align:center}}.stat-v{{font-size:20px;font-weight:800;line-height:1}}.stat-l{{font-size:10px;color:var(--tx2);text-transform:uppercase;letter-spacing:.06em;margin-top:2px}}
.sep{{width:1px;height:36px;background:var(--bd)}}
.toolbar{{background:var(--bg2);border-bottom:1px solid var(--bd);padding:9px 22px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}}
.search{{flex:1;max-width:360px;background:var(--bg3);border:1px solid var(--bd);color:var(--tx);padding:7px 12px;border-radius:8px;font-size:12px;outline:none}}
.search:focus{{border-color:var(--acc)}}
.fbtn{{background:var(--bg3);border:1px solid var(--bd);color:var(--tx2);padding:5px 12px;border-radius:8px;cursor:pointer;font-size:11px;font-weight:600}}
.fbtn.on{{background:var(--acc);color:#fff;border-color:var(--acc)}}.fbtn.img.on{{background:var(--img);border-color:var(--img)}}.fbtn.vid.on{{background:var(--vid);border-color:var(--vid)}}
main{{padding:18px 22px;max-width:1440px;margin:0 auto}}
.y-group{{margin-bottom:8px}}.y-hdr{{display:flex;align-items:center;gap:10px;padding:11px 14px;background:var(--bg2);border:1px solid var(--bd);border-radius:10px;cursor:pointer;font-size:18px;font-weight:800;user-select:none}}
.y-hdr:hover{{background:var(--bg3)}}.y-body{{padding-left:14px;margin-top:5px}}
.m-group{{margin-bottom:5px}}.m-hdr{{display:flex;align-items:center;gap:9px;padding:8px 12px;background:var(--bg3);border:1px solid var(--bd);border-radius:8px;cursor:pointer;font-size:14px;font-weight:600;user-select:none}}
.m-hdr:hover{{background:var(--bg4)}}.m-body{{padding-left:12px;margin-top:4px}}
.d-group{{margin-bottom:12px}}.d-hdr{{font-size:11px;font-weight:600;color:var(--tx2);padding:5px 0;display:flex;align-items:center;gap:8px;text-transform:uppercase;letter-spacing:.06em}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(152px,1fr));gap:7px}}
.card{{background:var(--bg2);border:1px solid var(--bd);border-radius:8px;padding:11px;transition:.12s;overflow:hidden;cursor:default}}
.card:hover{{background:var(--bg3);border-color:var(--acc);transform:translateY(-2px);box-shadow:0 4px 14px var(--shad)}}
.card.image{{border-left:3px solid var(--img)}}.card.video{{border-left:3px solid var(--vid)}}
.card-icon{{font-size:20px;margin-bottom:5px}}.card-name{{font-size:11px;font-weight:500;word-break:break-all;max-height:2.8em;overflow:hidden;line-height:1.4}}
.card-meta{{display:flex;justify-content:space-between;margin-top:7px;align-items:center}}
.sz{{font-size:10px;color:var(--tx2)}}.ext{{font-size:9px;background:var(--bg4);color:var(--tx2);padding:1px 5px;border-radius:4px;font-weight:700}}
.badge{{margin-left:auto;background:var(--bg4);color:var(--tx2);border-radius:10px;padding:1px 8px;font-size:10px;font-weight:600}}
.tri{{font-size:10px;color:var(--tx3);transition:.18s;width:14px;display:inline-block}}
.collapsed>.tri{{transform:rotate(-90deg)}}.hidden{{display:none}}
footer{{text-align:center;padding:18px;color:var(--tx3);font-size:10px;border-top:1px solid var(--bd);margin-top:30px}}
</style></head><body>
<header>
  <div><div class="h-title">Media Analyzer</div><div class="h-path">{root}</div></div>
  <button class="theme-btn" onclick="toggleTheme()"><span id="ticon">☀️</span><span id="tlbl">Light</span></button>
</header>
<div class="stats">
  <div class="stat"><div class="stat-v">{len(results)}</div><div class="stat-l">Media Files</div></div>
  <div class="sep"></div>
  <div class="stat"><div class="stat-v" style="color:var(--img)">{len(images)}</div><div class="stat-l">🖼 Images</div></div>
  <div class="sep"></div>
  <div class="stat"><div class="stat-v" style="color:var(--vid)">{len(videos)}</div><div class="stat-l">🎬 Videos</div></div>
  <div class="sep"></div>
  <div class="stat"><div class="stat-v">{format_size(total_size)}</div><div class="stat-l">Total Size</div></div>
  <div class="sep"></div>
  <div class="stat"><div class="stat-v">{total_scanned}</div><div class="stat-l">Files Scanned</div></div>
  <div class="sep"></div>
  <div class="stat"><div class="stat-v">{elapsed:.1f}s</div><div class="stat-l">Scan Time</div></div>
</div>
<div class="toolbar">
  <input class="search" type="text" placeholder="Search files…" oninput="doSearch(this.value)">
  <button class="fbtn on" id="f-all" onclick="setFilter('all')">All</button>
  <button class="fbtn img" id="f-image" onclick="setFilter('image')">🖼 Images</button>
  <button class="fbtn vid" id="f-video" onclick="setFilter('video')">🎬 Videos</button>
</div>
<main>{timeline_html}</main>
<footer>Scanned {total_scanned} files in {elapsed:.2f}s · {len(results)} media files · Generated {scan_ts}</footer>
<script>
const saved=localStorage.getItem('ma-theme')||'dark';applyTheme(saved);
function applyTheme(t){{document.documentElement.setAttribute('data-theme',t);document.getElementById('ticon').textContent=t==='dark'?'☀️':'🌙';document.getElementById('tlbl').textContent=t==='dark'?'Light':'Dark';}}
function toggleTheme(){{const t=document.documentElement.getAttribute('data-theme')==='dark'?'light':'dark';applyTheme(t);localStorage.setItem('ma-theme',t);}}
function tog(h){{h.classList.toggle('collapsed');const b=h.nextElementSibling;if(b)b.classList.toggle('hidden');}}
let curFilter='all',curSearch='';
function setFilter(f){{curFilter=f;['all','image','video'].forEach(x=>{{document.getElementById('f-'+x).classList.toggle('on',x===f);}});apply();}}
function doSearch(v){{curSearch=v.toLowerCase();apply();}}
function apply(){{
  document.querySelectorAll('.card').forEach(c=>{{const show=(curFilter==='all'||c.dataset.type===curFilter)&&(!curSearch||c.dataset.name.includes(curSearch));c.classList.toggle('hidden',!show);}});
  document.querySelectorAll('.d-group').forEach(d=>{{d.classList.toggle('hidden',!d.querySelector('.card:not(.hidden)'));}});
  document.querySelectorAll('.m-group').forEach(m=>{{m.classList.toggle('hidden',!m.querySelector('.d-group:not(.hidden)'));}});
  document.querySelectorAll('.y-group').forEach(y=>{{y.classList.toggle('hidden',!y.querySelector('.m-group:not(.hidden)'));}});
}}
</script></body></html>"""


# ─── GUI App ──────────────────────────────────────────────────────────────────
class MediaAnalyzerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Media Analyzer")
        self.geometry("700x600")
        self.minsize(600, 520)
        self.resizable(True, True)

        # State
        self._scanning = False
        self._cancel   = threading.Event()
        self._q        = queue.Queue()
        self._results  = []
        self._output_path = ""

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._build_ui()
        self.after(100, self._poll_queue)

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = ctk.CTkFrame(self, corner_radius=0, height=54)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(hdr, text="  Media Analyzer",
                     font=ctk.CTkFont(size=18, weight="bold")).grid(
            row=0, column=0, padx=4, pady=12, sticky="w")

        self._theme_btn = ctk.CTkButton(
            hdr, text="☀  Light", width=90, height=30, corner_radius=16,
            fg_color="transparent", border_width=1,
            command=self._toggle_theme)
        self._theme_btn.grid(row=0, column=2, padx=12, pady=12)

        # ── Config panel ──────────────────────────────────────────────────────
        cfg = ctk.CTkFrame(self, corner_radius=0)
        cfg.grid(row=1, column=0, sticky="ew", padx=0, pady=0)
        cfg.grid_columnconfigure(1, weight=1)

        # Directory row
        ctk.CTkLabel(cfg, text="Directory", width=80,
                     font=ctk.CTkFont(size=12)).grid(
            row=0, column=0, padx=(16, 8), pady=(14, 4), sticky="w")

        self._dir_var = ctk.StringVar()
        self._dir_entry = ctk.CTkEntry(
            cfg, textvariable=self._dir_var,
            placeholder_text="Choose a folder to scan…",
            height=34)
        self._dir_entry.grid(row=0, column=1, padx=(0, 8), pady=(14, 4), sticky="ew")

        ctk.CTkButton(cfg, text="Browse", width=80, height=34,
                      command=self._browse_dir).grid(
            row=0, column=2, padx=(0, 16), pady=(14, 4))

        # Output row
        ctk.CTkLabel(cfg, text="Output", width=80,
                     font=ctk.CTkFont(size=12)).grid(
            row=1, column=0, padx=(16, 8), pady=4, sticky="w")

        self._out_var = ctk.StringVar(value="media_report.html")
        self._out_entry = ctk.CTkEntry(cfg, textvariable=self._out_var, height=34)
        self._out_entry.grid(row=1, column=1, padx=(0, 8), pady=4, sticky="ew")

        ctk.CTkButton(cfg, text="Browse", width=80, height=34,
                      command=self._browse_output).grid(
            row=1, column=2, padx=(0, 16), pady=4)

        # Workers row
        workers_frame = ctk.CTkFrame(cfg, fg_color="transparent")
        workers_frame.grid(row=2, column=0, columnspan=3,
                           padx=16, pady=(4, 12), sticky="ew")
        workers_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(workers_frame, text="Workers",
                     font=ctk.CTkFont(size=12), width=80).grid(
            row=0, column=0, sticky="w")

        self._workers_var = ctk.IntVar(value=12)
        self._workers_lbl = ctk.CTkLabel(
            workers_frame, text="12",
            font=ctk.CTkFont(size=12, weight="bold"), width=28)
        self._workers_lbl.grid(row=0, column=2, padx=(8, 0))

        slider = ctk.CTkSlider(
            workers_frame, from_=1, to=32, number_of_steps=31,
            variable=self._workers_var,
            command=lambda v: self._workers_lbl.configure(text=str(int(v))))
        slider.grid(row=0, column=1, padx=8, sticky="ew")

        # ── Main area ─────────────────────────────────────────────────────────
        mid = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        mid.grid(row=2, column=0, sticky="nsew", padx=16, pady=8)
        mid.grid_columnconfigure(0, weight=1)
        mid.grid_rowconfigure(1, weight=1)

        # Stats row
        stats = ctk.CTkFrame(mid, height=70)
        stats.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        for i in range(6):
            stats.grid_columnconfigure(i, weight=1)

        self._stat_labels = {}
        stat_defs = [
            ("total",   "—",  "Files"),
            ("images",  "—",  "Images 🖼"),
            ("videos",  "—",  "Videos 🎬"),
            ("size",    "—",  "Size"),
            ("scanned", "—",  "Scanned"),
            ("elapsed", "—",  "Time"),
        ]
        for col, (key, val, label) in enumerate(stat_defs):
            f = ctk.CTkFrame(stats, fg_color="transparent")
            f.grid(row=0, column=col, padx=6, pady=10, sticky="nsew")
            v_lbl = ctk.CTkLabel(f, text=val,
                                  font=ctk.CTkFont(size=20, weight="bold"))
            v_lbl.pack()
            ctk.CTkLabel(f, text=label,
                          font=ctk.CTkFont(size=10),
                          text_color="gray").pack()
            self._stat_labels[key] = v_lbl

        # Log box
        self._log = ctk.CTkTextbox(mid, font=ctk.CTkFont(family="Consolas", size=11),
                                    wrap="word", state="disabled")
        self._log.grid(row=1, column=0, sticky="nsew")

        # ── Bottom bar ────────────────────────────────────────────────────────
        bot = ctk.CTkFrame(self, corner_radius=0, height=70)
        bot.grid(row=3, column=0, sticky="ew")
        bot.grid_columnconfigure(1, weight=1)

        # Progress bar
        prog_frame = ctk.CTkFrame(bot, fg_color="transparent")
        prog_frame.grid(row=0, column=0, columnspan=3,
                        padx=16, pady=(8, 0), sticky="ew")
        prog_frame.grid_columnconfigure(0, weight=1)

        self._progress = ctk.CTkProgressBar(prog_frame, height=6)
        self._progress.set(0)
        self._progress.grid(row=0, column=0, sticky="ew")

        self._prog_lbl = ctk.CTkLabel(
            prog_frame, text="Ready",
            font=ctk.CTkFont(size=11), text_color="gray")
        self._prog_lbl.grid(row=0, column=1, padx=(10, 0))

        # Buttons
        btn_frame = ctk.CTkFrame(bot, fg_color="transparent")
        btn_frame.grid(row=1, column=0, columnspan=3,
                       padx=16, pady=(6, 10), sticky="e")

        self._open_btn = ctk.CTkButton(
            btn_frame, text="Open Report", width=110, height=34,
            state="disabled", command=self._open_report)
        self._open_btn.grid(row=0, column=0, padx=(0, 8))

        self._scan_btn = ctk.CTkButton(
            btn_frame, text="▶  Start Scan", width=130, height=34,
            command=self._toggle_scan)
        self._scan_btn.grid(row=0, column=1)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _toggle_theme(self):
        mode = ctk.get_appearance_mode()
        if mode == "Dark":
            ctk.set_appearance_mode("light")
            self._theme_btn.configure(text="🌙  Dark")
        else:
            ctk.set_appearance_mode("dark")
            self._theme_btn.configure(text="☀  Light")

    def _browse_dir(self):
        path = filedialog.askdirectory(title="Select folder to scan")
        if path:
            self._dir_var.set(path)
            # Auto-set output next to selected folder
            out = str(Path(path).parent / "media_report.html")
            self._out_var.set(out)

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            title="Save report as",
            defaultextension=".html",
            filetypes=[("HTML file", "*.html")])
        if path:
            self._out_var.set(path)

    def _log_write(self, msg: str):
        self._log.configure(state="normal")
        self._log.insert("end", msg + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")

    def _set_stat(self, key: str, val: str):
        self._stat_labels[key].configure(text=val)

    def _open_report(self):
        webbrowser.open(f"file:///{Path(self._output_path).resolve()}")

    # ── Scan control ──────────────────────────────────────────────────────────
    def _toggle_scan(self):
        if self._scanning:
            self._cancel.set()
            self._scan_btn.configure(text="Stopping…", state="disabled")
        else:
            self._start_scan()

    def _start_scan(self):
        directory = self._dir_var.get().strip()
        if not directory:
            messagebox.showwarning("No folder", "Please select a folder to scan.")
            return
        if not Path(directory).is_dir():
            messagebox.showerror("Not found", f"Directory not found:\n{directory}")
            return

        self._output_path = self._out_var.get().strip() or "media_report.html"
        workers = int(self._workers_var.get())

        self._scanning = True
        self._cancel.clear()
        self._results = []
        self._open_btn.configure(state="disabled")
        self._scan_btn.configure(text="■  Stop", fg_color="#dc2626", hover_color="#b91c1c")
        self._progress.set(0)
        self._prog_lbl.configure(text="Collecting files…")

        for key in self._stat_labels:
            self._set_stat(key, "—")

        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")
        self._log_write(f"Scanning: {directory}")
        self._log_write(f"Workers:  {workers}  |  EXIF: {'✓' if HAS_PIL else '✗'}\n")

        t = threading.Thread(
            target=self._scan_worker,
            args=(directory, workers),
            daemon=True)
        t.start()

    def _scan_worker(self, directory: str, workers: int):
        root = Path(directory)
        q = self._q

        # Collect files
        all_files = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for fname in filenames:
                if not fname.startswith("."):
                    all_files.append(Path(dirpath) / fname)

        total = len(all_files)
        q.put(("total_files", total))

        results = []
        done = 0
        t0 = time.perf_counter()

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(analyze_file, f): f for f in all_files}
            for fut in as_completed(futures):
                if self._cancel.is_set():
                    ex.shutdown(wait=False, cancel_futures=True)
                    q.put(("cancelled", None))
                    return
                r = fut.result()
                if r:
                    results.append(r)
                done += 1
                if done % 50 == 0 or done == total:
                    q.put(("progress", (done, total, len(results))))

        elapsed = time.perf_counter() - t0
        q.put(("done", (results, total, elapsed)))

    # ── Queue polling (UI thread) ─────────────────────────────────────────────
    def _poll_queue(self):
        try:
            while True:
                msg, data = self._q.get_nowait()

                if msg == "total_files":
                    self._log_write(f"Found {data:,} files — analyzing…")

                elif msg == "progress":
                    done, total, found = data
                    pct = done / max(total, 1)
                    self._progress.set(pct)
                    self._prog_lbl.configure(
                        text=f"{done:,} / {total:,}  ({found:,} media)")
                    self._set_stat("scanned", f"{done:,}")
                    self._set_stat("total",   f"{found:,}")

                elif msg == "done":
                    results, total_scanned, elapsed = data
                    self._finish_scan(results, total_scanned, elapsed)

                elif msg == "cancelled":
                    self._scan_aborted()

        except queue.Empty:
            pass

        self.after(60, self._poll_queue)

    def _finish_scan(self, results, total_scanned, elapsed):
        self._results = results
        self._scanning = False
        self._scan_btn.configure(
            text="▶  Start Scan", state="normal",
            fg_color=("#3B82F6", "#1D4ED8"), hover_color=("#2563EB", "#1E40AF"))
        self._progress.set(1)
        self._prog_lbl.configure(text="Done")

        images = sum(1 for r in results if r["type"] == "image")
        videos = sum(1 for r in results if r["type"] == "video")
        total_size = sum(r["size"] for r in results)

        self._set_stat("total",   str(len(results)))
        self._set_stat("images",  str(images))
        self._set_stat("videos",  str(videos))
        self._set_stat("size",    format_size(total_size))
        self._set_stat("scanned", str(total_scanned))
        self._set_stat("elapsed", f"{elapsed:.1f}s")

        self._log_write(f"\n✓ Scan complete in {elapsed:.2f}s")
        self._log_write(f"  {len(results):,} media files  ({images:,} images · {videos:,} videos)")
        self._log_write(f"  Total size: {format_size(total_size)}")

        # Write report
        try:
            html = build_html(results, self._dir_var.get(), total_scanned, elapsed)
            out = Path(self._output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(html, encoding="utf-8")
            self._log_write(f"\n  Report → {out.resolve()}")
            self._open_btn.configure(state="normal")
        except Exception as e:
            self._log_write(f"\n  ✗ Could not write report: {e}")

    def _scan_aborted(self):
        self._scanning = False
        self._scan_btn.configure(
            text="▶  Start Scan", state="normal",
            fg_color=("#3B82F6", "#1D4ED8"), hover_color=("#2563EB", "#1E40AF"))
        self._progress.set(0)
        self._prog_lbl.configure(text="Cancelled")
        self._log_write("\n  Scan cancelled.")


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = MediaAnalyzerApp()
    app.mainloop()
