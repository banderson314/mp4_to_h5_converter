"""
mp4_to_h5_gui.py
----------------
GUI tool: convert MP4 videos → HDF5 (.h5) for Ilastik Animal Tracking.

Requirements:
    pip install opencv-python h5py numpy

Usage:
    python mp4_to_h5_gui.py
"""

import json
import threading
from pathlib import Path

import cv2
import h5py
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# ─────────────────────────────────────────────────────────────────────────────
# Axistags  — THIS is what fixes the "no time axis" error in Ilastik
# ─────────────────────────────────────────────────────────────────────────────

def make_axistags(grayscale: bool) -> str:
    """
    Build a vigra-style axistags JSON string.

    Ilastik reads this attribute from the HDF5 dataset to identify
    which array dimension corresponds to time (t), space (x/y), or
    channel (c).  Without it, Ilastik cannot detect a time axis and
    will refuse to load the file as a video.

    typeFlags:
        1 = channel (c)
        2 = space   (x, y, z)
        8 = time    (t)

    Grayscale shape  (T, Y, X)     → axes: t, y, x
    Colour    shape  (T, Y, X, C)  → axes: t, y, x, c
    """
    axes = [
        {"key": "t", "typeFlags": 8, "resolution": 0, "description": ""},
        {"key": "y", "typeFlags": 2, "resolution": 0, "description": ""},
        {"key": "x", "typeFlags": 2, "resolution": 0, "description": ""},
    ]
    if not grayscale:
        axes.append({"key": "c", "typeFlags": 1, "resolution": 0, "description": ""})
    return json.dumps({"axes": axes})


# ─────────────────────────────────────────────────────────────────────────────
# Core conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert_mp4_to_h5(
    input_path:   Path,
    output_path:  Path,
    grayscale:    bool,
    frame_skip:   int,
    scale:        float,
    compress:     bool,
    dataset_name: str,
    start_frame:  int,
    end_frame,          # int or None
    log_fn,             # callable(str)
    progress_fn,        # callable(float 0-100)
) -> bool:

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        log_fn(f"  [ERROR] Could not open: {input_path.name}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    orig_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    end_frame = min(end_frame, total_frames) if end_frame is not None else total_frames
    new_w = max(1, int(orig_w * scale))
    new_h = max(1, int(orig_h * scale))
    span  = end_frame - start_frame

    log_fn(f"  Source  : {total_frames} frames | {fps:.2f} fps | {orig_w}×{orig_h}")
    log_fn(f"  Extract : frames {start_frame}–{end_frame - 1} | "
           f"every {frame_skip} frame(s) → ~{len(range(start_frame, end_frame, frame_skip))} frames")
    log_fn(f"  Output  : {'grayscale' if grayscale else 'colour (RGB)'} | "
           f"{new_w}×{new_h} | {'gzip compressed' if compress else 'uncompressed'}")

    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for i in range(span):
        ret, frame = cap.read()
        if not ret:
            log_fn(f"  [WARNING] Stream ended early at frame {start_frame + i}")
            break

        if i % frame_skip != 0:
            continue

        if scale != 1.0:
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY if grayscale else cv2.COLOR_BGR2RGB)
        frames.append(frame)

        if i % 30 == 0:
            progress_fn(i / span * 88)   # 0 → 88 % while reading

    cap.release()

    if not frames:
        log_fn("  [ERROR] No frames extracted.")
        return False

    data = np.array(frames, dtype=np.uint8)
    # Grayscale → (T, Y, X)   |   Colour → (T, Y, X, C)
    log_fn(f"  Array   : shape {data.shape} | dtype {data.dtype}")
    progress_fn(92)

    # ── Write HDF5 ────────────────────────────────────────────────────────
    h5_kwargs = {"chunks": True}
    if compress:
        h5_kwargs.update(compression="gzip", compression_opts=4)

    with h5py.File(output_path, "w") as f:
        ds = f.create_dataset(dataset_name, data=data, **h5_kwargs)

        # !! Critical fix: tell Ilastik which axis is time !!
        ds.attrs["axistags"] = make_axistags(grayscale)

        # Convenience metadata (visible in HDFView / h5py)
        f.attrs.update(
            source_file     = input_path.name,
            original_fps    = fps,
            original_width  = orig_w,
            original_height = orig_h,
            frame_skip      = frame_skip,
            scale           = scale,
            grayscale       = int(grayscale),
            start_frame     = start_frame,
            end_frame       = end_frame,
        )

    size_mb = output_path.stat().st_size / (1024 ** 2)
    log_fn(f"  Saved   : {output_path.name}  ({size_mb:.1f} MB)")
    progress_fn(100)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────

class App(tk.Tk):

    PAD = {"padx": 12, "pady": 6}

    def __init__(self):
        super().__init__()
        self.title("MP4 → HDF5 Converter for Ilastik")
        self.resizable(False, False)
        self._build_ui()

    # ── Build ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        p = self.PAD

        # ── Folder selection ──────────────────────────────────────────────
        folder_frame = ttk.LabelFrame(self, text=" Folders ", padding=10)
        folder_frame.grid(row=0, column=0, sticky="ew", **p)
        folder_frame.columnconfigure(1, weight=1)

        ttk.Label(folder_frame, text="Input folder:").grid(
            row=0, column=0, sticky="w")
        self.input_var = tk.StringVar()
        ttk.Entry(folder_frame, textvariable=self.input_var, width=50).grid(
            row=0, column=1, padx=6, sticky="ew")
        ttk.Button(folder_frame, text="Browse…",
                   command=self._browse_input).grid(row=0, column=2)

        ttk.Label(folder_frame, text="Output folder:").grid(
            row=1, column=0, sticky="w", pady=(8, 0))
        self.output_var = tk.StringVar()
        ttk.Entry(folder_frame, textvariable=self.output_var, width=50).grid(
            row=1, column=1, padx=6, sticky="ew", pady=(8, 0))
        ttk.Button(folder_frame, text="Browse…",
                   command=self._browse_output).grid(row=1, column=2, pady=(8, 0))
        ttk.Label(folder_frame,
                  text="Leave blank to save H5 files alongside source MP4s",
                  foreground="gray").grid(row=2, column=1, sticky="w", padx=6)

        # ── Image options ─────────────────────────────────────────────────
        img_frame = ttk.LabelFrame(self, text=" Image Options ", padding=10)
        img_frame.grid(row=1, column=0, sticky="ew", **p)
        img_frame.columnconfigure(2, weight=1)

        self.grayscale_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            img_frame,
            text="Convert to grayscale  (recommended — halves file size;"
                 " sufficient for mouse foreground/background segmentation)",
            variable=self.grayscale_var,
        ).grid(row=0, column=0, columnspan=3, sticky="w")

        ttk.Label(img_frame, text="Spatial scale:").grid(
            row=1, column=0, sticky="w", pady=(8, 0))
        self.scale_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(img_frame, from_=0.1, to=2.0, increment=0.05,
                    textvariable=self.scale_var, width=7,
                    format="%.2f").grid(row=1, column=1, sticky="w",
                                        padx=6, pady=(8, 0))
        ttk.Label(img_frame,
                  text="1.0 = original resolution  |  0.5 = half  |  0.25 = quarter",
                  foreground="gray").grid(row=1, column=2, sticky="w", pady=(8, 0))

        # ── Temporal options ──────────────────────────────────────────────
        time_frame = ttk.LabelFrame(self, text=" Temporal Options ", padding=10)
        time_frame.grid(row=2, column=0, sticky="ew", **p)
        time_frame.columnconfigure(2, weight=1)

        ttk.Label(time_frame, text="Frame skip:").grid(row=0, column=0, sticky="w")
        self.frame_skip_var = tk.IntVar(value=1)
        ttk.Spinbox(time_frame, from_=1, to=100,
                    textvariable=self.frame_skip_var,
                    width=7).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(time_frame,
                  text="1 = keep every frame  |  2 = keep every other frame  |  etc.",
                  foreground="gray").grid(row=0, column=2, sticky="w")

        ttk.Label(time_frame, text="Start frame:").grid(
            row=1, column=0, sticky="w", pady=(8, 0))
        self.start_frame_var = tk.StringVar(value="0")
        ttk.Entry(time_frame, textvariable=self.start_frame_var,
                  width=9).grid(row=1, column=1, sticky="w", padx=6, pady=(8, 0))
        ttk.Label(time_frame, text="First frame index to extract  (0 = beginning of video)",
                  foreground="gray").grid(row=1, column=2, sticky="w", pady=(8, 0))

        ttk.Label(time_frame, text="End frame:").grid(
            row=2, column=0, sticky="w", pady=(6, 0))
        self.end_frame_var = tk.StringVar(value="")
        ttk.Entry(time_frame, textvariable=self.end_frame_var,
                  width=9).grid(row=2, column=1, sticky="w", padx=6, pady=(6, 0))
        ttk.Label(time_frame, text="Last frame index (leave blank = end of video)",
                  foreground="gray").grid(row=2, column=2, sticky="w", pady=(6, 0))

        # ── HDF5 options ──────────────────────────────────────────────────
        h5_frame = ttk.LabelFrame(self, text=" HDF5 Options ", padding=10)
        h5_frame.grid(row=3, column=0, sticky="ew", **p)
        h5_frame.columnconfigure(2, weight=1)

        self.compress_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            h5_frame,
            text="Enable gzip compression  (smaller files, slower to write — "
                 "useful for archiving or tight disk space)",
            variable=self.compress_var,
        ).grid(row=0, column=0, columnspan=3, sticky="w")

        ttk.Label(h5_frame, text="Dataset name:").grid(
            row=1, column=0, sticky="w", pady=(8, 0))
        self.dataset_name_var = tk.StringVar(value="data")
        ttk.Entry(h5_frame, textvariable=self.dataset_name_var,
                  width=14).grid(row=1, column=1, sticky="w", padx=6, pady=(8, 0))
        ttk.Label(h5_frame,
                  text="Internal HDF5 path Ilastik will show — 'data' is the standard default",
                  foreground="gray").grid(row=1, column=2, sticky="w", pady=(8, 0))

        # ── Progress & log ────────────────────────────────────────────────
        log_frame = ttk.LabelFrame(self, text=" Progress ", padding=10)
        log_frame.grid(row=4, column=0, sticky="ew", **p)
        log_frame.columnconfigure(0, weight=1)

        self.progress_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(log_frame, variable=self.progress_var,
                        maximum=100, length=560).grid(
            row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))

        self.log_text = tk.Text(
            log_frame, height=11, width=76, state="disabled",
            bg="#1e1e1e", fg="#d4d4d4", font=("Courier", 9),
            relief="flat",
        )
        self.log_text.grid(row=1, column=0, sticky="ew")
        sb = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        sb.grid(row=1, column=1, sticky="ns")
        self.log_text["yscrollcommand"] = sb.set

        # ── Action buttons ────────────────────────────────────────────────
        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=5, column=0, pady=12)

        self.convert_btn = ttk.Button(
            btn_frame, text="▶   Convert All MP4s",
            command=self._start_conversion)
        self.convert_btn.grid(row=0, column=0, padx=8, ipadx=6, ipady=3)

        ttk.Button(btn_frame, text="Clear Log",
                   command=self._clear_log).grid(row=0, column=1, padx=8)

    # ── Folder browsers ───────────────────────────────────────────────────

    def _browse_input(self):
        d = filedialog.askdirectory(title="Select folder containing MP4 files")
        if d:
            self.input_var.set(d)

    def _browse_output(self):
        d = filedialog.askdirectory(title="Select output folder for H5 files")
        if d:
            self.output_var.set(d)

    # ── Logging ───────────────────────────────────────────────────────────

    def _log(self, message: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self.progress_var.set(0.0)

    def _set_progress(self, value: float):
        self.progress_var.set(value)
        self.update_idletasks()

    # ── Validation ────────────────────────────────────────────────────────

    def _validate(self) -> bool:
        if not self.input_var.get():
            messagebox.showerror("Missing Input", "Please select an input folder.")
            return False
        if not Path(self.input_var.get()).is_dir():
            messagebox.showerror("Invalid Folder",
                                 f"Input folder not found:\n{self.input_var.get()}")
            return False
        try:
            start = int(self.start_frame_var.get())
            assert start >= 0
        except (ValueError, AssertionError):
            messagebox.showerror("Invalid Value",
                                 "Start frame must be a non-negative integer.")
            return False
        end_str = self.end_frame_var.get().strip()
        if end_str:
            try:
                end = int(end_str)
                assert end > int(self.start_frame_var.get())
            except (ValueError, AssertionError):
                messagebox.showerror("Invalid Value",
                                     "End frame must be an integer greater than start frame.")
                return False
        try:
            scale = float(self.scale_var.get())
            assert 0.05 <= scale <= 4.0
        except (ValueError, AssertionError):
            messagebox.showerror("Invalid Value", "Scale must be between 0.05 and 4.0.")
            return False
        if not self.dataset_name_var.get().strip():
            messagebox.showerror("Invalid Value", "Dataset name cannot be empty.")
            return False
        return True

    # ── Conversion thread ─────────────────────────────────────────────────

    def _start_conversion(self):
        if not self._validate():
            return
        self.convert_btn.configure(state="disabled")
        threading.Thread(target=self._run_conversion, daemon=True).start()

    def _run_conversion(self):
        input_dir    = Path(self.input_var.get())
        out_str      = self.output_var.get().strip()
        output_dir   = Path(out_str) if out_str else input_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        grayscale    = self.grayscale_var.get()
        scale        = float(self.scale_var.get())
        frame_skip   = int(self.frame_skip_var.get())
        compress     = self.compress_var.get()
        dataset_name = self.dataset_name_var.get().strip()
        start_frame  = int(self.start_frame_var.get())
        end_str      = self.end_frame_var.get().strip()
        end_frame    = int(end_str) if end_str else None

        mp4_files = sorted(
            p for p in input_dir.iterdir() if p.suffix.lower() == ".mp4"
        )

        if not mp4_files:
            self.after(0, lambda: messagebox.showwarning(
                "No Files Found", f"No MP4 files found in:\n{input_dir}"))
            self.after(0, lambda: self.convert_btn.configure(state="normal"))
            return

        def log(msg):
            self.after(0, lambda m=msg: self._log(m))

        log("═" * 56)
        log(f"  Found {len(mp4_files)} MP4 file(s)  →  output: {output_dir.name}/")
        log("═" * 56)

        results = {"ok": [], "fail": []}

        for idx, mp4_path in enumerate(mp4_files):
            output_path = output_dir / (mp4_path.stem + ".h5")
            log(f"\n[{idx + 1}/{len(mp4_files)}]  {mp4_path.name}")
            self.after(0, lambda: self._set_progress(0))

            success = convert_mp4_to_h5(
                input_path   = mp4_path,
                output_path  = output_path,
                grayscale    = grayscale,
                frame_skip   = frame_skip,
                scale        = scale,
                compress     = compress,
                dataset_name = dataset_name,
                start_frame  = start_frame,
                end_frame    = end_frame,
                log_fn       = log,
                progress_fn  = lambda v: self.after(0, lambda val=v: self._set_progress(val)),
            )

            (results["ok"] if success else results["fail"]).append(mp4_path.name)

        log("\n" + "═" * 56)
        log(f"  Finished — ✓ {len(results['ok'])} succeeded  "
            f"✗ {len(results['fail'])} failed")
        for name in results["fail"]:
            log(f"    ✗  {name}")
        log("═" * 56)

        self.after(0, lambda: self.convert_btn.configure(state="normal"))
        self.after(0, lambda: messagebox.showinfo(
            "Conversion Complete",
            f"✓  {len(results['ok'])} file(s) converted successfully\n"
            f"✗  {len(results['fail'])} file(s) failed\n\n"
            f"Output folder:\n{output_dir}"
        ))


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    App().mainloop()