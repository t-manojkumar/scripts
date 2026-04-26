#!/usr/bin/env python3
"""
Media Organizer — Windows GUI App
Scans a folder, previews the Year/Month/Day tree, then moves or copies files.

Requirements: pip install customtkinter Pillow
"""

import os
import shutil
import threading
import mimetypes
import webbrowser
import queue
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import Optional

import customtkinter as ctk
from tkinter import filedialog, messagebox

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ── Media types ───────────────────────────────────────────────────────────────
IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif",
    ".webp", ".heic", ".heif", ".raw", ".cr2", ".nef", ".arw", ".dng", ".avif",
}
VIDEO_EXTS = {
    ".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm",
    ".m4v", ".3gp", ".mpeg", ".mpg", ".ts", ".mts", ".m2ts",
}

LAYOUTS = {
    "Year / Month / Day":  lambda d: Path(str(d.year)) / d.strftime("%B")       / f"{d.day:02d}",
    "Year / Month":        lambda d: Path(str(d.year)) / d.strftime("%B"),
    "Year / MM-Month":     lambda d: Path(str(d.year)) / d.strftime("%m-%B"),
    "Year / YYYY-MM-DD":   lambda d: Path(str(d.year)) / d.strftime("%Y-%m-%d"),
    "YYYY / MM / DD":      lambda d: Path(str(d.year)) / f"{d.month:02d}"       / f"{d.day:02d}",
    "Flat YYYY-MM-DD":     lambda d: Path(d.strftime("%Y-%m-%d")),
}


# ── Analysis helpers ──────────────────────────────────────────────────────────
def _exif_date(path: Path) -> Optional[datetime]:
    if not HAS_PIL:
        return None
    try:
        with Image.open(path) as img:
            exif = img._getexif()
            if exif:
                for tid, val in exif.items():
                    if TAGS.get(tid) in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
                        return datetime.strptime(val, "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return None


def _file_date(path: Path, ext: str) -> datetime:
    if ext in IMAGE_EXTS:
        d = _exif_date(path)
        if d:
            return d
    st = path.stat()
    return datetime.fromtimestamp(min(st.st_mtime, st.st_ctime))


def _file_type(ext: str) -> Optional[str]:
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    mime, _ = mimetypes.guess_type(f"x{ext}")
    if mime:
        if mime.startswith("image/"): return "image"
        if mime.startswith("video/"): return "video"
    return None


def fmt_size(n: int) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024: return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def analyze_file(path: Path) -> Optional[dict]:
    try:
        st = path.stat()
        if st.st_size == 0:
            return None
        ext = path.suffix.lower()
        ftype = _file_type(ext)
        if not ftype:
            return None
        date = _file_date(path, ext)
        return {
            "name": path.name, "path": str(path),
            "type": ftype, "size": st.st_size, "ext": ext.lstrip("."),
            "date": date,          # datetime object kept in memory
            "year": date.year, "month": date.month, "day": date.day,
        }
    except Exception:
        return None


# ── Move / copy one file ──────────────────────────────────────────────────────
def do_transfer(item: dict, dest_root: Path,
                layout_fn, action: str, conflict: str) -> str:
    """Returns 'ok' | 'skip' | 'error:<msg>'"""
    src = Path(item["path"])
    folder = dest_root / layout_fn(item["date"])
    folder.mkdir(parents=True, exist_ok=True)
    dst = folder / src.name

    if dst.exists():
        if conflict == "Skip":
            return "skip"
        if conflict == "Rename":
            i = 1
            while True:
                cand = folder / f"{src.stem} ({i}){src.suffix}"
                if not cand.exists():
                    dst = cand
                    break
                i += 1
        # Overwrite: dst stays as-is

    try:
        if action == "Move":
            shutil.move(str(src), dst)
        else:
            shutil.copy2(str(src), dst)
        return "ok"
    except Exception as e:
        return f"error:{e}"


# ─── Main window ──────────────────────────────────────────────────────────────
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Media Organizer")
        self.geometry("820x720")
        self.minsize(700, 580)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._results:   list[dict] = []
        self._scanning   = False
        self._organising = False
        self._cancel     = threading.Event()
        self._q          = queue.Queue()

        self._build()
        self.after(50, self._poll)

    # ─── Build UI ─────────────────────────────────────────────────────────────
    def _build(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)  # tree row expands

        # ── Header ────────────────────────────────────────────────────────────
        hdr = ctk.CTkFrame(self, corner_radius=0, height=50)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(hdr, text="  Media Organizer",
                     font=ctk.CTkFont(size=17, weight="bold")).grid(
            row=0, column=0, padx=6, pady=10, sticky="w")
        self._theme_btn = ctk.CTkButton(
            hdr, text="☀  Light", width=92, height=28, corner_radius=14,
            fg_color="transparent", border_width=1, command=self._toggle_theme)
        self._theme_btn.grid(row=0, column=2, padx=12)

        # ── Config panel ──────────────────────────────────────────────────────
        cfg = ctk.CTkFrame(self, corner_radius=0)
        cfg.grid(row=1, column=0, sticky="ew")
        cfg.grid_columnconfigure(1, weight=1)

        def path_row(row, label, var, cmd, padtop=12):
            ctk.CTkLabel(cfg, text=label, width=88,
                         font=ctk.CTkFont(size=12)).grid(
                row=row, column=0, padx=(14, 6), pady=(padtop, 4), sticky="w")
            ctk.CTkEntry(cfg, textvariable=var, height=32).grid(
                row=row, column=1, padx=(0, 6), pady=(padtop, 4), sticky="ew")
            ctk.CTkButton(cfg, text="Browse", width=72, height=32,
                          command=cmd).grid(
                row=row, column=2, padx=(0, 14), pady=(padtop, 4))

        self._src_var  = ctk.StringVar()
        self._dest_var = ctk.StringVar()
        path_row(0, "Source folder", self._src_var,  self._browse_src,  12)
        path_row(1, "Destination",   self._dest_var, self._browse_dest, 4)

        # Options row
        opt = ctk.CTkFrame(cfg, fg_color="transparent")
        opt.grid(row=2, column=0, columnspan=3,
                 padx=14, pady=(4, 10), sticky="ew")
        for i in (1, 3, 5, 7):
            opt.grid_columnconfigure(i, weight=1)

        def lbl(col, text):
            ctk.CTkLabel(opt, text=text, font=ctk.CTkFont(size=11),
                         text_color="gray").grid(row=0, column=col, sticky="w")

        lbl(0, "Layout"); lbl(2, "Action"); lbl(4, "Files"); lbl(6, "Conflicts")

        self._layout_var   = ctk.StringVar(value=list(LAYOUTS.keys())[0])
        self._action_var   = ctk.StringVar(value="Copy")
        self._media_var    = ctk.StringVar(value="Images + Videos")
        self._conflict_var = ctk.StringVar(value="Rename")
        self._workers_var  = ctk.IntVar(value=12)

        ctk.CTkOptionMenu(opt, variable=self._layout_var,
                          values=list(LAYOUTS.keys()), height=30).grid(
            row=1, column=0, columnspan=2, padx=(0, 8), sticky="ew")
        ctk.CTkOptionMenu(opt, variable=self._action_var,
                          values=["Copy", "Move"], height=30, width=90).grid(
            row=1, column=2, columnspan=2, padx=(0, 8), sticky="ew")
        ctk.CTkOptionMenu(opt, variable=self._media_var,
                          values=["Images + Videos", "Images only", "Videos only"],
                          height=30).grid(
            row=1, column=4, columnspan=2, padx=(0, 8), sticky="ew")
        ctk.CTkOptionMenu(opt, variable=self._conflict_var,
                          values=["Rename", "Skip", "Overwrite"],
                          height=30, width=100).grid(
            row=1, column=6, columnspan=2, sticky="ew")

        # ── Stats strip ───────────────────────────────────────────────────────
        sf = ctk.CTkFrame(self, corner_radius=0, height=60)
        sf.grid(row=2, column=0, sticky="ew")
        for i in range(7):
            sf.grid_columnconfigure(i, weight=1)

        self._sv = {}
        for col, (k, lbl_) in enumerate([
            ("files",   "Total"),
            ("images",  "Images 🖼"),
            ("videos",  "Videos 🎬"),
            ("size",    "Size"),
            ("done",    "Organised"),
            ("skipped", "Skipped"),
            ("elapsed", "Time"),
        ]):
            f = ctk.CTkFrame(sf, fg_color="transparent")
            f.grid(row=0, column=col, padx=2, pady=8, sticky="nsew")
            v = ctk.CTkLabel(f, text="—", font=ctk.CTkFont(size=17, weight="bold"))
            v.pack()
            ctk.CTkLabel(f, text=lbl_, font=ctk.CTkFont(size=10),
                         text_color="gray").pack()
            self._sv[k] = v

        # ── Folder tree (scrollable) ───────────────────────────────────────
        tree_wrap = ctk.CTkFrame(self, corner_radius=0)
        tree_wrap.grid(row=3, column=0, sticky="nsew", padx=0, pady=0)
        tree_wrap.grid_columnconfigure(0, weight=1)
        tree_wrap.grid_rowconfigure(1, weight=1)

        tree_hdr = ctk.CTkFrame(tree_wrap, fg_color="transparent", height=32)
        tree_hdr.grid(row=0, column=0, sticky="ew", padx=14, pady=(8, 0))
        tree_hdr.grid_columnconfigure(0, weight=1)
        self._tree_title = ctk.CTkLabel(
            tree_hdr, text="Folder preview  —  scan first",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="gray",
            anchor="w")
        self._tree_title.grid(row=0, column=0, sticky="w")
        self._preview_btn = ctk.CTkButton(
            tree_hdr, text="Refresh preview", width=120, height=26,
            fg_color="transparent", border_width=1,
            command=self._refresh_preview, state="disabled")
        self._preview_btn.grid(row=0, column=1)

        self._tree_scroll = ctk.CTkScrollableFrame(
            tree_wrap, corner_radius=6, fg_color=("gray95", "gray10"))
        self._tree_scroll.grid(row=1, column=0, sticky="nsew",
                               padx=10, pady=(4, 6))
        self._tree_scroll.grid_columnconfigure(0, weight=1)
        self._tree_rows: list[ctk.CTkLabel] = []

        # ── Bottom bar ────────────────────────────────────────────────────────
        bot = ctk.CTkFrame(self, corner_radius=0, height=72)
        bot.grid(row=4, column=0, sticky="ew")
        bot.grid_columnconfigure(0, weight=1)

        pf = ctk.CTkFrame(bot, fg_color="transparent")
        pf.grid(row=0, column=0, sticky="ew", padx=14, pady=(10, 0))
        pf.grid_columnconfigure(0, weight=1)
        self._pbar = ctk.CTkProgressBar(pf, height=8, corner_radius=4)
        self._pbar.set(0)
        self._pbar.grid(row=0, column=0, sticky="ew")
        self._plbl = ctk.CTkLabel(pf, text="Ready",
                                   font=ctk.CTkFont(size=11), text_color="gray",
                                   width=180, anchor="e")
        self._plbl.grid(row=0, column=1, padx=(8, 0))

        bf = ctk.CTkFrame(bot, fg_color="transparent")
        bf.grid(row=1, column=0, sticky="e", padx=14, pady=(6, 10))

        self._rpt_btn = ctk.CTkButton(
            bf, text="Open Report", width=110, height=34,
            fg_color="transparent", border_width=1,
            state="disabled", command=self._open_report)
        self._rpt_btn.grid(row=0, column=0, padx=(0, 8))

        self._scan_btn = ctk.CTkButton(
            bf, text="🔍  Scan", width=110, height=34,
            command=self._toggle_scan)
        self._scan_btn.grid(row=0, column=1, padx=(0, 8))

        self._org_btn = ctk.CTkButton(
            bf, text="📁  Organise Files", width=150, height=34,
            state="disabled", command=self._toggle_org)
        self._org_btn.grid(row=0, column=2)

        self._report_path = ""

    # ─── Theme ────────────────────────────────────────────────────────────────
    def _toggle_theme(self):
        if ctk.get_appearance_mode() == "Dark":
            ctk.set_appearance_mode("light")
            self._theme_btn.configure(text="🌙  Dark")
        else:
            ctk.set_appearance_mode("dark")
            self._theme_btn.configure(text="☀  Light")

    # ─── Browse ───────────────────────────────────────────────────────────────
    def _browse_src(self):
        p = filedialog.askdirectory(title="Select source folder")
        if p:
            self._src_var.set(p)
            if not self._dest_var.get():
                self._dest_var.set(str(Path(p).parent / "Organised"))

    def _browse_dest(self):
        p = filedialog.askdirectory(title="Select destination folder")
        if p:
            self._dest_var.set(p)

    def _open_report(self):
        if self._report_path:
            webbrowser.open(f"file:///{Path(self._report_path).resolve()}")

    # ─── Tree preview ─────────────────────────────────────────────────────────
    def _get_items(self) -> list[dict]:
        mv = self._media_var.get()
        if mv == "Images only":
            return [r for r in self._results if r["type"] == "image"]
        if mv == "Videos only":
            return [r for r in self._results if r["type"] == "video"]
        return list(self._results)

    def _refresh_preview(self):
        items = self._get_items()
        if not items:
            return
        layout_fn = LAYOUTS[self._layout_var.get()]
        dest = self._dest_var.get().strip() or "Organised"

        # Build tree: year → month-day-path → count/size
        tree: dict[str, dict] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for item in items:
            sub = layout_fn(item["date"])
            parts = sub.parts
            year_key  = parts[0]
            child_key = str(Path(*parts[1:])) if len(parts) > 1 else ""
            tree[year_key][child_key][0] += 1
            tree[year_key][child_key][1] += item["size"]

        # Clear old rows
        for w in self._tree_rows:
            w.destroy()
        self._tree_rows.clear()

        def add_row(text, indent=0, color=None):
            lbl = ctk.CTkLabel(
                self._tree_scroll,
                text="  " * indent + text,
                font=ctk.CTkFont(family="Consolas", size=12),
                text_color=color or ("gray20", "gray85"),
                anchor="w")
            lbl.grid(row=len(self._tree_rows), column=0, sticky="ew", pady=1)
            self._tree_rows.append(lbl)

        add_row(f"📂  {dest}", color=("#2563eb", "#4f9cf9"))
        for year in sorted(tree, reverse=True):
            year_files = sum(v[0] for v in tree[year].values())
            year_size  = sum(v[1] for v in tree[year].values())
            add_row(f"📁  {year}    {year_files} files   {fmt_size(year_size)}", 1,
                    color=("#1d4ed8", "#60a5fa"))
            for child in sorted(tree[year], reverse=True):
                count, size = tree[year][child]
                label = child if child else "(root)"
                add_row(f"📄  {label}    {count} files   {fmt_size(size)}", 2)

        total_files = len(items)
        total_size  = sum(i["size"] for i in items)
        images_n    = sum(1 for i in items if i["type"] == "image")
        videos_n    = sum(1 for i in items if i["type"] == "video")
        add_row("")
        add_row(
            f"→  {total_files} files to {self._action_var.get().lower()}  "
            f"({images_n} images · {videos_n} videos · {fmt_size(total_size)})  "
            f"Conflicts: {self._conflict_var.get()}",
            color=("#16a34a", "#4ade80"))

        self._tree_title.configure(
            text=f"Folder preview  —  {total_files} files  →  {dest}",
            text_color=("gray30", "gray70"))

    # ─── Scan ─────────────────────────────────────────────────────────────────
    def _toggle_scan(self):
        if self._scanning:
            self._cancel.set()
            self._scan_btn.configure(text="Stopping…", state="disabled")
        else:
            self._start_scan()

    def _start_scan(self):
        src = self._src_var.get().strip()
        if not src or not Path(src).is_dir():
            messagebox.showwarning("No folder", "Select a valid source folder.")
            return

        self._scanning = True
        self._cancel.clear()
        self._results = []
        self._org_btn.configure(state="disabled")
        self._rpt_btn.configure(state="disabled")
        self._preview_btn.configure(state="disabled")
        self._scan_btn.configure(text="■  Stop",
                                  fg_color="#dc2626", hover_color="#b91c1c")
        self._pbar.set(0)
        self._plbl.configure(text="Collecting files…")
        for k in self._sv:
            self._sv[k].configure(text="—")

        # Clear tree
        for w in self._tree_rows:
            w.destroy()
        self._tree_rows.clear()
        self._tree_title.configure(text="Scanning…", text_color="gray")

        threading.Thread(
            target=self._scan_worker, args=(src,), daemon=True).start()

    def _scan_worker(self, src: str):
        root = Path(src)
        all_files = []
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if not d.startswith(".")]
            for fn in fns:
                if not fn.startswith("."):
                    all_files.append(Path(dp) / fn)

        total = len(all_files)
        self._q.put(("scan_total", total))

        results, done = [], 0
        t0 = time.perf_counter()
        workers = 12

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(analyze_file, f): f for f in all_files}
            for fut in as_completed(futs):
                if self._cancel.is_set():
                    ex.shutdown(wait=False, cancel_futures=True)
                    self._q.put(("scan_cancel", None))
                    return
                r = fut.result()
                if r:
                    results.append(r)
                done += 1
                if done % 50 == 0 or done == total:
                    self._q.put(("scan_prog", (done, total, len(results))))

        self._q.put(("scan_done", (results, total, time.perf_counter() - t0, src)))

    # ─── Organize ─────────────────────────────────────────────────────────────
    def _toggle_org(self):
        if self._organising:
            self._cancel.set()
            self._org_btn.configure(text="Stopping…", state="disabled")
        else:
            self._start_org()

    def _start_org(self):
        items = self._get_items()
        if not items:
            messagebox.showinfo("Nothing to do", "No files match selected type.")
            return
        dest = self._dest_var.get().strip()
        if not dest:
            messagebox.showwarning("No destination", "Set a destination folder.")
            return

        action = self._action_var.get()
        src_resolved = str(Path(self._src_var.get()).resolve())
        dest_resolved = str(Path(dest).resolve())
        if dest_resolved.startswith(src_resolved):
            if not messagebox.askyesno(
                "Destination inside source",
                "Destination is inside the source folder.\n"
                "This is safe for Copy but may cause loops with Move.\n\nContinue?",
                icon="warning"):
                return

        if action == "Move":
            if not messagebox.askyesno(
                "Confirm Move",
                f"MOVE {len(items)} files into:\n{dest}\n\n"
                "Original files will be deleted from the source.\n\nProceed?",
                icon="warning"):
                return

        self._organising = True
        self._cancel.clear()
        self._org_btn.configure(text="■  Stop",
                                 fg_color="#dc2626", hover_color="#b91c1c")
        self._scan_btn.configure(state="disabled")
        self._pbar.set(0)
        self._plbl.configure(text="Organising…")
        for k in ("done", "skipped", "elapsed"):
            self._sv[k].configure(text="—")

        layout_fn = LAYOUTS[self._layout_var.get()]
        conflict  = self._conflict_var.get()

        threading.Thread(
            target=self._org_worker,
            args=(items, Path(dest), layout_fn, action, conflict),
            daemon=True).start()

    def _org_worker(self, items, dest_root, layout_fn, action, conflict):
        total = len(items)
        ok = skipped = errors = 0
        t0 = time.perf_counter()

        for i, item in enumerate(items, 1):
            if self._cancel.is_set():
                self._q.put(("org_cancel", (ok, skipped, errors,
                                             time.perf_counter() - t0)))
                return
            result = do_transfer(item, dest_root, layout_fn, action, conflict)
            if result == "ok":
                ok += 1
            elif result == "skip":
                skipped += 1
            else:
                errors += 1
                self._q.put(("org_err", result.replace("error:", "")))

            if i % 10 == 0 or i == total:
                self._q.put(("org_prog", (i, total, ok, skipped, errors)))

        self._q.put(("org_done", (ok, skipped, errors, time.perf_counter() - t0, action)))

    # ─── Queue poll ───────────────────────────────────────────────────────────
    def _poll(self):
        try:
            while True:
                msg, data = self._q.get_nowait()

                if msg == "scan_total":
                    self._tree_title.configure(
                        text=f"Scanning — {data:,} files found…", text_color="gray")

                elif msg == "scan_prog":
                    done, total, found = data
                    self._pbar.set(done / max(total, 1))
                    self._plbl.configure(
                        text=f"{done:,} / {total:,}  ({found:,} media)")
                    self._sv["files"].configure(text=f"{found:,}")

                elif msg == "scan_done":
                    results, scanned, elapsed, src = data
                    self._scanning = False
                    self._results  = results
                    self._scan_btn.configure(
                        text="🔍  Scan", state="normal",
                        fg_color=("#3B82F6", "#1D4ED8"),
                        hover_color=("#2563EB", "#1E40AF"))
                    self._pbar.set(1)

                    images   = sum(1 for r in results if r["type"] == "image")
                    videos   = sum(1 for r in results if r["type"] == "video")
                    tot_size = sum(r["size"] for r in results)

                    self._sv["files"].configure(text=str(len(results)))
                    self._sv["images"].configure(text=str(images))
                    self._sv["videos"].configure(text=str(videos))
                    self._sv["size"].configure(text=fmt_size(tot_size))
                    self._sv["elapsed"].configure(text=f"{elapsed:.1f}s")
                    self._plbl.configure(
                        text=f"Scan done — {len(results):,} media in {elapsed:.1f}s")

                    # Write HTML report
                    try:
                        html   = _build_html(results, src, scanned, elapsed)
                        rpt    = Path(src).parent / "media_report.html"
                        rpt.write_text(html, encoding="utf-8")
                        self._report_path = str(rpt)
                        self._rpt_btn.configure(state="normal")
                    except Exception:
                        pass

                    # Show folder preview & enable organize
                    self._preview_btn.configure(state="normal")
                    self._org_btn.configure(state="normal")
                    self._refresh_preview()

                elif msg == "scan_cancel":
                    self._scanning = False
                    self._scan_btn.configure(
                        text="🔍  Scan", state="normal",
                        fg_color=("#3B82F6", "#1D4ED8"),
                        hover_color=("#2563EB", "#1E40AF"))
                    self._pbar.set(0)
                    self._plbl.configure(text="Scan cancelled")
                    self._tree_title.configure(text="Cancelled", text_color="gray")

                elif msg == "org_prog":
                    done, total, ok, skipped, errors = data
                    self._pbar.set(done / max(total, 1))
                    self._plbl.configure(
                        text=f"{done:,}/{total:,}  ✓{ok:,}  skip {skipped:,}  ✗{errors:,}")
                    self._sv["done"].configure(text=str(ok))
                    self._sv["skipped"].configure(text=str(skipped))

                elif msg == "org_err":
                    pass  # silently counted; shown in plbl

                elif msg == "org_done":
                    ok, skipped, errors, elapsed, action = data
                    self._organising = False
                    self._org_btn.configure(
                        text="📁  Organise Files", state="normal",
                        fg_color=("#3B82F6", "#1D4ED8"),
                        hover_color=("#2563EB", "#1E40AF"))
                    self._scan_btn.configure(state="normal")
                    self._pbar.set(1)
                    self._sv["done"].configure(text=str(ok))
                    self._sv["skipped"].configure(text=str(skipped))
                    self._sv["elapsed"].configure(text=f"{elapsed:.1f}s")
                    self._plbl.configure(
                        text=f"Done — {ok:,} {action.lower()}d · {skipped:,} skipped · {errors:,} errors")

                    if action == "Move":
                        self._results = []
                        self._org_btn.configure(state="disabled")
                        self._preview_btn.configure(state="disabled")
                        self._tree_title.configure(
                            text="Files moved — re-scan to refresh", text_color="gray")
                        for w in self._tree_rows:
                            w.destroy()
                        self._tree_rows.clear()
                    else:
                        self._refresh_preview()  # re-draw to keep it in sync

                elif msg == "org_cancel":
                    ok, skipped, errors, elapsed = data
                    self._organising = False
                    self._org_btn.configure(
                        text="📁  Organise Files", state="normal",
                        fg_color=("#3B82F6", "#1D4ED8"),
                        hover_color=("#2563EB", "#1E40AF"))
                    self._scan_btn.configure(state="normal")
                    self._pbar.set(0)
                    self._plbl.configure(
                        text=f"Cancelled — {ok:,} done · {skipped:,} skipped")

        except queue.Empty:
            pass
        self.after(50, self._poll)


# ─── Minimal HTML report ──────────────────────────────────────────────────────
def _build_html(results, root, scanned, elapsed):
    results = sorted(results, key=lambda x: x["date"])
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in results:
        grouped[r["year"]][r["month"]][r["day"]].append(r)

    images   = sum(1 for r in results if r["type"] == "image")
    videos   = sum(1 for r in results if r["type"] == "video")
    tot_size = sum(r["size"] for r in results)

    tl = []
    for yr in sorted(grouped, reverse=True):
        yc = sum(len(v) for m in grouped[yr].values() for v in m.values())
        tl.append(f'<div class="yg"><div class="yh" onclick="tog(this)"><span class="t">▼</span>{yr}<span class="b">{yc}</span></div><div class="yb">')
        for mo in sorted(grouped[yr], reverse=True):
            mn = datetime(yr, mo, 1).strftime("%B")
            mc = sum(len(v) for v in grouped[yr][mo].values())
            tl.append(f'<div class="mg"><div class="mh" onclick="tog(this)"><span class="t">▼</span>{mn}<span class="b">{mc}</span></div><div class="mb">')
            for dy in sorted(grouped[yr][mo], reverse=True):
                items = grouped[yr][mo][dy]
                dl = datetime(yr, mo, dy).strftime("%A, %d %B %Y")
                tl.append(f'<div class="dg"><div class="dh">{dl}<span class="b">{len(items)}</span></div><div class="grid">')
                for item in sorted(items, key=lambda x: x["date"]):
                    icon = "🖼️" if item["type"] == "image" else "🎬"
                    sz   = fmt_size(item["size"])
                    tl.append(f'<div class="card {item["type"]}" title="{item["path"]}"><div class="ci">{icon}</div><div class="cn">{item["name"]}</div><div class="cm"><span class="sz">{sz}</span><span class="ex">{item["ext"].upper()}</span></div></div>')
                tl.append("</div></div>")
            tl.append("</div></div>")
        tl.append("</div></div>")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!DOCTYPE html><html lang="en" data-theme="dark"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Media Report</title>
<style>
:root[data-theme="dark"]{{--bg:#0d0d0d;--bg2:#161616;--bg3:#202020;--bg4:#2a2a2a;--bd:#2e2e2e;--tx:#e0e0e0;--tx2:#888;--tx3:#555;--acc:#4f9cf9;--img:#2563eb;--vid:#7c3aed;--sh:rgba(0,0,0,.5)}}
:root[data-theme="light"]{{--bg:#f3f4f6;--bg2:#fff;--bg3:#f0f0f0;--bg4:#e5e7eb;--bd:#d1d5db;--tx:#111;--tx2:#6b7280;--tx3:#9ca3af;--acc:#2563eb;--img:#1d4ed8;--vid:#7c3aed;--sh:rgba(0,0,0,.08)}}
*{{box-sizing:border-box;margin:0;padding:0}}body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);color:var(--tx);font-size:13px}}
header{{background:var(--bg2);border-bottom:1px solid var(--bd);padding:14px 22px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:50}}
.ht{{font-size:16px;font-weight:700}}.hp{{font-size:11px;color:var(--tx2);font-family:monospace;margin-top:2px}}
.tb{{background:var(--bg3);border:1px solid var(--bd);color:var(--tx);padding:5px 12px;border-radius:20px;cursor:pointer;font-size:12px}}
.stats{{background:var(--bg2);border-bottom:1px solid var(--bd);padding:10px 22px;display:flex;gap:20px;flex-wrap:wrap}}
.stat{{text-align:center}}.sv{{font-size:20px;font-weight:800}}.sl{{font-size:10px;color:var(--tx2);text-transform:uppercase}}
.sep{{width:1px;height:36px;background:var(--bd)}}
.tb2{{background:var(--bg2);border-bottom:1px solid var(--bd);padding:9px 22px;display:flex;gap:8px;align-items:center}}
.si{{flex:1;max-width:360px;background:var(--bg3);border:1px solid var(--bd);color:var(--tx);padding:7px 12px;border-radius:8px;font-size:12px;outline:none}}
.si:focus{{border-color:var(--acc)}}
.fb{{background:var(--bg3);border:1px solid var(--bd);color:var(--tx2);padding:5px 12px;border-radius:8px;cursor:pointer;font-size:11px;font-weight:600}}
.fb.on{{background:var(--acc);color:#fff;border-color:var(--acc)}}.fb.im.on{{background:var(--img);border-color:var(--img)}}.fb.vi.on{{background:var(--vid);border-color:var(--vid)}}
main{{padding:18px 22px;max-width:1440px;margin:0 auto}}
.yg{{margin-bottom:8px}}.yh{{display:flex;align-items:center;gap:10px;padding:11px 14px;background:var(--bg2);border:1px solid var(--bd);border-radius:10px;cursor:pointer;font-size:18px;font-weight:800;user-select:none}}
.yh:hover{{background:var(--bg3)}}.yb{{padding-left:14px;margin-top:5px}}
.mg{{margin-bottom:5px}}.mh{{display:flex;align-items:center;gap:9px;padding:8px 12px;background:var(--bg3);border:1px solid var(--bd);border-radius:8px;cursor:pointer;font-size:14px;font-weight:600;user-select:none}}
.mh:hover{{background:var(--bg4)}}.mb{{padding-left:12px;margin-top:4px}}
.dg{{margin-bottom:12px}}.dh{{font-size:11px;font-weight:600;color:var(--tx2);padding:5px 0;display:flex;align-items:center;gap:8px;text-transform:uppercase;letter-spacing:.06em}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(152px,1fr));gap:7px}}
.card{{background:var(--bg2);border:1px solid var(--bd);border-radius:8px;padding:11px;transition:.12s;overflow:hidden}}
.card:hover{{background:var(--bg3);border-color:var(--acc);transform:translateY(-2px);box-shadow:0 4px 14px var(--sh)}}
.card.image{{border-left:3px solid var(--img)}}.card.video{{border-left:3px solid var(--vid)}}
.ci{{font-size:20px;margin-bottom:5px}}.cn{{font-size:11px;font-weight:500;word-break:break-all;max-height:2.8em;overflow:hidden;line-height:1.4}}
.cm{{display:flex;justify-content:space-between;margin-top:7px}}.sz{{font-size:10px;color:var(--tx2)}}
.ex{{font-size:9px;background:var(--bg4);color:var(--tx2);padding:1px 5px;border-radius:4px;font-weight:700}}
.b{{margin-left:auto;background:var(--bg4);color:var(--tx2);border-radius:10px;padding:1px 8px;font-size:10px;font-weight:600}}
.t{{font-size:10px;color:var(--tx3);transition:.18s;width:14px;display:inline-block}}
.collapsed>.t{{transform:rotate(-90deg)}}.hidden{{display:none}}
footer{{text-align:center;padding:18px;color:var(--tx3);font-size:10px;border-top:1px solid var(--bd);margin-top:30px}}
</style></head><body>
<header><div><div class="ht">Media Report</div><div class="hp">{root}</div></div>
<button class="tb" onclick="tg()"><span id="ti">☀️</span><span id="tl">Light</span></button></header>
<div class="stats">
<div class="stat"><div class="sv">{len(results)}</div><div class="sl">Files</div></div><div class="sep"></div>
<div class="stat"><div class="sv" style="color:var(--img)">{images}</div><div class="sl">Images 🖼</div></div><div class="sep"></div>
<div class="stat"><div class="sv" style="color:var(--vid)">{videos}</div><div class="sl">Videos 🎬</div></div><div class="sep"></div>
<div class="stat"><div class="sv">{fmt_size(tot_size)}</div><div class="sl">Size</div></div><div class="sep"></div>
<div class="stat"><div class="sv">{scanned}</div><div class="sl">Scanned</div></div><div class="sep"></div>
<div class="stat"><div class="sv">{elapsed:.1f}s</div><div class="sl">Time</div></div>
</div>
<div class="tb2">
<input class="si" type="text" placeholder="Search…" oninput="ds(this.value)">
<button class="fb on" id="fa" onclick="sf('all')">All</button>
<button class="fb im" id="fi" onclick="sf('image')">🖼 Images</button>
<button class="fb vi" id="fv" onclick="sf('video')">🎬 Videos</button>
</div>
<main>{"".join(tl)}</main>
<footer>Scanned {scanned} files in {elapsed:.2f}s · {len(results)} media · {ts}</footer>
<script>
const sv=localStorage.getItem('t')||'dark';at(sv);
function at(t){{document.documentElement.setAttribute('data-theme',t);document.getElementById('ti').textContent=t==='dark'?'☀️':'🌙';document.getElementById('tl').textContent=t==='dark'?'Light':'Dark';}}
function tg(){{const t=document.documentElement.getAttribute('data-theme')==='dark'?'light':'dark';at(t);localStorage.setItem('t',t);}}
function tog(h){{h.classList.toggle('collapsed');const b=h.nextElementSibling;if(b)b.classList.toggle('hidden');}}
let cf='all',cs='';
function sf(f){{cf=f;['all','image','video'].forEach(x=>document.getElementById('f'+x[0]).classList.toggle('on',x===f));ap();}}
function ds(v){{cs=v.toLowerCase();ap();}}
function ap(){{
document.querySelectorAll('.card').forEach(c=>{{const ok=(cf==='all'||c.classList.contains(cf))&&(!cs||c.querySelector('.cn').textContent.toLowerCase().includes(cs));c.classList.toggle('hidden',!ok);}});
document.querySelectorAll('.dg').forEach(d=>d.classList.toggle('hidden',!d.querySelector('.card:not(.hidden)')));
document.querySelectorAll('.mg').forEach(m=>m.classList.toggle('hidden',!m.querySelector('.dg:not(.hidden)')));
document.querySelectorAll('.yg').forEach(y=>y.classList.toggle('hidden',!y.querySelector('.mg:not(.hidden)')));
}}
</script></body></html>"""


# ─── Entry ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    App().mainloop()
