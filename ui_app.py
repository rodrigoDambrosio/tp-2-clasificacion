import io
import os
import sys
import threading
import tkinter as tk
from contextlib import redirect_stderr, redirect_stdout
from tkinter import filedialog, ttk

import cv2
import numpy as np
from joblib import load
from PIL import Image, ImageTk

from generator import append_row
import trainer
from pipeline import PipelineConfig, analyze_frame
from vision_utils import load_label_map


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Tetris Classifier UI")
        self.root.option_add("*Font", ("Segoe UI", 10))
        self.process = None
        self.cap = None
        self.preview_running = False
        self.last_result = None
        self.model = None
        self.label_map = {}
        self.dragging = False
        self.drag_start = None
        self.preview_half = (1, 1)
        self.preview_frame_scale = (1.0, 1.0)
        self.preview_pad = (0, 0)
        self.preview_scale = 1.0
        self.frame_tile_scale = 1.0
        self.frame_tile_pad = (0, 0)

        self.mode = tk.StringVar(value="Generator")
        self.method = tk.StringVar(value="otsu")
        self.invert = tk.BooleanVar(value=False)
        self.raw_hu = tk.BooleanVar(value=False)
        self.manual_thresh = tk.IntVar(value=127)

        self.min_area = tk.IntVar(value=800)
        self.morph_size = tk.IntVar(value=5)
        self.dilate_iter = tk.IntVar(value=1)

        self.use_roi = tk.BooleanVar(value=False)
        self.roi_x = tk.IntVar(value=0)
        self.roi_y = tk.IntVar(value=0)
        self.roi_w = tk.IntVar(value=200)
        self.roi_h = tk.IntVar(value=200)

        self.gen_label = tk.IntVar(value=1)
        self.gen_output = tk.StringVar(value="dataset.csv")

        self.cls_model = tk.StringVar(value="model.joblib")
        self.cls_labels = tk.StringVar(value="labels.json")
        self.pred_var = tk.StringVar(value="Pred: -")

        self.train_data = tk.StringVar(value="dataset.csv")
        self.train_model = tk.StringVar(value="model.joblib")
        self.test_split = tk.DoubleVar(value=0.2)
        self.max_depth = tk.IntVar(value=0)
        self.min_samples_leaf = tk.IntVar(value=1)

        self._build_layout()
        self._bind_events()

    def _build_layout(self):
        self.root.geometry("1200x780")
        self.root.configure(bg="#1f1f1f")

        main = tk.Frame(self.root, bg="#1f1f1f")
        main.pack(fill=tk.BOTH, expand=True)

        left_container = tk.Frame(main, bg="#242424", width=320)
        left_container.pack(side=tk.LEFT, fill=tk.Y)
        left_container.pack_propagate(False)

        left_canvas = tk.Canvas(left_container, bg="#242424", highlightthickness=0)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_scroll = ttk.Scrollbar(left_container, orient=tk.VERTICAL, command=left_canvas.yview)
        left_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        left_canvas.configure(yscrollcommand=left_scroll.set)

        left = tk.Frame(left_canvas, bg="#242424")
        left_window = left_canvas.create_window((0, 0), window=left, anchor="nw")

        def _on_left_configure(event):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))

        def _on_canvas_configure(event):
            left_canvas.itemconfigure(left_window, width=event.width)

        left.bind("<Configure>", _on_left_configure)
        left_canvas.bind("<Configure>", _on_canvas_configure)

        right = tk.PanedWindow(main, orient=tk.VERTICAL, bg="#1f1f1f", sashwidth=6, sashrelief=tk.RAISED)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Style().theme_use("clam")
        style = ttk.Style()
        style.configure("TLabel", background="#242424", foreground="#e6e6e6")
        style.configure("TButton", background="#3a3a3a", foreground="#ffffff", padding=(10, 6))
        style.map(
            "TButton",
            background=[("active", "#4a4a4a"), ("pressed", "#2f2f2f")],
            foreground=[("disabled", "#8a8a8a")],
        )
        style.configure("TFrame", background="#242424")
        style.configure("TNotebook", background="#242424")
        style.configure("TNotebook.Tab", background="#2a2a2a", foreground="#e6e6e6")

        header = tk.Label(left, text="Tetris ML", bg="#242424", fg="#ffffff", font=("Segoe UI", 16, "bold"))
        header.pack(padx=12, pady=(16, 8), anchor="w")

        mode_row = tk.Frame(left, bg="#242424")
        mode_row.pack(fill=tk.X, padx=12, pady=4)
        tk.Label(mode_row, text="Mode", bg="#242424", fg="#cfcfcf").pack(anchor="w")
        ttk.Combobox(mode_row, textvariable=self.mode, values=["Generator", "Trainer", "Classifier"], state="readonly").pack(fill=tk.X, pady=4)

        self.pred_row = tk.Frame(left, bg="#242424")
        self.pred_row.pack(fill=tk.X, padx=12, pady=(2, 6))
        tk.Label(self.pred_row, text="Prediction", bg="#242424", fg="#cfcfcf").pack(anchor="w")
        tk.Label(
            self.pred_row,
            textvariable=self.pred_var,
            bg="#242424",
            fg="#ffffff",
            font=("Segoe UI", 11, "bold"),
            anchor="w",
        ).pack(fill=tk.X)

        ttk.Separator(left).pack(fill=tk.X, padx=12, pady=8)

        self.gen_section = tk.Frame(left, bg="#242424")
        self.gen_section.pack(fill=tk.X, padx=12, pady=(4, 8))
        self._build_generator_tab()

        self.cls_section = tk.Frame(left, bg="#242424")
        self.cls_section.pack(fill=tk.X, padx=12, pady=(4, 8))
        self._build_classifier_tab()

        self.train_section = tk.Frame(left, bg="#242424")
        self.train_section.pack(fill=tk.X, padx=12, pady=(4, 8))
        self._build_trainer_tab()

        ttk.Separator(left).pack(fill=tk.X, padx=12, pady=8)

        self._build_config_panel(left)

        preview_frame = tk.Frame(right, bg="#1f1f1f")
        self.preview_label = tk.Label(preview_frame, bg="#1f1f1f")
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=16, pady=(16, 8))
        self.preview_label.bind("<ButtonPress-1>", self.on_preview_mouse_down)
        self.preview_label.bind("<B1-Motion>", self.on_preview_mouse_move)
        self.preview_label.bind("<ButtonRelease-1>", self.on_preview_mouse_up)

        pred = tk.Label(preview_frame, textvariable=self.pred_var, bg="#1f1f1f", fg="#e6e6e6", font=("Segoe UI", 12, "bold"))
        pred.pack(anchor="w", padx=16)

        log_frame = tk.Frame(right, bg="#1f1f1f")
        self.log = tk.Text(log_frame, height=8, bg="#1b1b1b", fg="#e6e6e6", insertbackground="#ffffff")
        self.log.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        right.add(preview_frame, stretch="always")
        right.add(log_frame, stretch="never", height=180)
        self.log_msg("UI ready.")

    def _build_generator_tab(self):
        frame = self.gen_section
        ttk.Label(frame, text="Label (int)").pack(anchor="w", pady=(6, 2))
        ttk.Entry(frame, textvariable=self.gen_label).pack(fill=tk.X)

        ttk.Label(frame, text="Output CSV").pack(anchor="w", pady=(8, 2))
        row = tk.Frame(frame, bg="#242424")
        row.pack(fill=tk.X)
        ttk.Entry(row, textvariable=self.gen_output).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="Browse", command=self.browse_output).pack(side=tk.RIGHT, padx=4)

        ttk.Button(frame, text="Capture Hu (preview)", command=self.capture_hu).pack(fill=tk.X, pady=8)

    def _build_trainer_tab(self):
        frame = self.train_section
        ttk.Label(frame, text="Dataset CSV").pack(anchor="w", pady=(6, 2))
        row = tk.Frame(frame, bg="#242424")
        row.pack(fill=tk.X)
        ttk.Entry(row, textvariable=self.train_data).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="Browse", command=self.browse_train_data).pack(side=tk.RIGHT, padx=4)

        ttk.Label(frame, text="Model Output").pack(anchor="w", pady=(8, 2))
        row = tk.Frame(frame, bg="#242424")
        row.pack(fill=tk.X)
        ttk.Entry(row, textvariable=self.train_model).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="Browse", command=self.browse_train_model).pack(side=tk.RIGHT, padx=4)

        ttk.Label(frame, text="Test Split (0-1)").pack(anchor="w", pady=(8, 2))
        ttk.Entry(frame, textvariable=self.test_split).pack(fill=tk.X)

        ttk.Label(frame, text="Max Depth (0 = None)").pack(anchor="w", pady=(8, 2))
        ttk.Entry(frame, textvariable=self.max_depth).pack(fill=tk.X)

        ttk.Label(frame, text="Min Samples Leaf").pack(anchor="w", pady=(8, 2))
        ttk.Entry(frame, textvariable=self.min_samples_leaf).pack(fill=tk.X)

        ttk.Button(frame, text="Train Model", command=self.run_training).pack(fill=tk.X, pady=8)

    def _build_classifier_tab(self):
        frame = self.cls_section
        ttk.Label(frame, text="Model File").pack(anchor="w", pady=(6, 2))
        row = tk.Frame(frame, bg="#242424")
        row.pack(fill=tk.X)
        ttk.Entry(row, textvariable=self.cls_model).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="Browse", command=self.browse_model).pack(side=tk.RIGHT, padx=4)

        ttk.Label(frame, text="Labels JSON").pack(anchor="w", pady=(8, 2))
        row = tk.Frame(frame, bg="#242424")
        row.pack(fill=tk.X)
        ttk.Entry(row, textvariable=self.cls_labels).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="Browse", command=self.browse_labels).pack(side=tk.RIGHT, padx=4)

        ttk.Button(frame, text="Load Model (preview)", command=self.load_model).pack(fill=tk.X, pady=8)

    def _build_config_panel(self, parent):
        self.config_panel = tk.Frame(parent, bg="#242424")
        self.config_panel.pack(fill=tk.X, padx=12, pady=4)

        ttk.Label(self.config_panel, text="Threshold Method").pack(anchor="w")
        ttk.Combobox(
            self.config_panel,
            textvariable=self.method,
            values=["otsu", "manual", "canny"],
            state="readonly",
        ).pack(fill=tk.X, pady=4)

        ttk.Checkbutton(self.config_panel, text="Invert", variable=self.invert).pack(anchor="w")
        ttk.Checkbutton(self.config_panel, text="Raw Hu (no log)", variable=self.raw_hu).pack(anchor="w")

        ttk.Label(self.config_panel, text="Manual Threshold").pack(anchor="w", pady=(6, 0))
        ttk.Scale(self.config_panel, from_=0, to=255, variable=self.manual_thresh, orient=tk.HORIZONTAL).pack(fill=tk.X)

        ttk.Label(self.config_panel, text="Min Area").pack(anchor="w", pady=(6, 0))
        ttk.Scale(self.config_panel, from_=1, to=5000, variable=self.min_area, orient=tk.HORIZONTAL).pack(fill=tk.X)

        ttk.Label(self.config_panel, text="Morph Size").pack(anchor="w", pady=(6, 0))
        ttk.Scale(self.config_panel, from_=1, to=21, variable=self.morph_size, orient=tk.HORIZONTAL).pack(fill=tk.X)

        ttk.Label(self.config_panel, text="Dilate Iterations").pack(anchor="w", pady=(6, 0))
        ttk.Scale(self.config_panel, from_=0, to=5, variable=self.dilate_iter, orient=tk.HORIZONTAL).pack(fill=tk.X)

        ttk.Separator(self.config_panel).pack(fill=tk.X, pady=6)
        ttk.Checkbutton(self.config_panel, text="Use ROI (preview)", variable=self.use_roi).pack(anchor="w")

        roi_row = tk.Frame(self.config_panel, bg="#242424")
        roi_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(roi_row, text="x").pack(side=tk.LEFT)
        ttk.Entry(roi_row, textvariable=self.roi_x, width=5).pack(side=tk.LEFT, padx=4)
        ttk.Label(roi_row, text="y").pack(side=tk.LEFT)
        ttk.Entry(roi_row, textvariable=self.roi_y, width=5).pack(side=tk.LEFT, padx=4)
        ttk.Label(roi_row, text="w").pack(side=tk.LEFT)
        ttk.Entry(roi_row, textvariable=self.roi_w, width=5).pack(side=tk.LEFT, padx=4)
        ttk.Label(roi_row, text="h").pack(side=tk.LEFT)
        ttk.Entry(roi_row, textvariable=self.roi_h, width=5).pack(side=tk.LEFT, padx=4)

    def _bind_events(self):
        self.mode.trace_add("write", lambda *_: self._sync_tabs())
        self._sync_tabs()

    def _sync_tabs(self):
        mode = self.mode.get()
        if mode == "Generator":
            self.gen_section.pack(fill=tk.X, padx=12, pady=(4, 8))
            self.cls_section.pack_forget()
            self.train_section.pack_forget()
            self.config_panel.pack(fill=tk.X, padx=12, pady=4)
            self.pred_row.pack_forget()
            self.pred_var.set("Pred: -")
            self.start_preview()
        elif mode == "Classifier":
            self.cls_section.pack(fill=tk.X, padx=12, pady=(4, 8))
            self.gen_section.pack_forget()
            self.train_section.pack_forget()
            self.config_panel.pack(fill=tk.X, padx=12, pady=4)
            self.pred_row.pack(fill=tk.X, padx=12, pady=(2, 6))
            self.start_preview()
        else:
            self.train_section.pack(fill=tk.X, padx=12, pady=(4, 8))
            self.gen_section.pack_forget()
            self.cls_section.pack_forget()
            self.config_panel.pack_forget()
            self.pred_row.pack_forget()
            self.pred_var.set("Pred: -")
            self.stop_preview()

    def log_msg(self, msg):
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)

    def browse_output(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            self.gen_output.set(path)

    def browse_train_data(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if path:
            self.train_data.set(path)

    def browse_train_model(self):
        path = filedialog.asksaveasfilename(defaultextension=".joblib", filetypes=[("Joblib", "*.joblib")])
        if path:
            self.train_model.set(path)

    def browse_model(self):
        path = filedialog.askopenfilename(filetypes=[("Joblib", "*.joblib")])
        if path:
            self.cls_model.set(path)

    def browse_labels(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if path:
            self.cls_labels.set(path)

    def stop_process(self):
        self.process = None

    def run_training(self):
        def _worker():
            args = [
                "trainer.py",
                "--data",
                self.train_data.get(),
                "--model",
                self.train_model.get(),
                "--test-split",
                str(float(self.test_split.get())),
                "--min-samples-leaf",
                str(int(self.min_samples_leaf.get())),
            ]
            max_depth = int(self.max_depth.get())
            if max_depth > 0:
                args.extend(["--max-depth", str(max_depth)])

            buf = io.StringIO()
            with redirect_stdout(buf), redirect_stderr(buf):
                old_argv = sys.argv
                try:
                    sys.argv = args
                    trainer.main()
                finally:
                    sys.argv = old_argv

            output = buf.getvalue().strip()
            if output:
                self.log_msg(output)

        threading.Thread(target=_worker, daemon=True).start()

    def start_preview(self):
        if self.preview_running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.log_msg("Could not open camera")
            return
        self.preview_running = True
        self.update_preview()
        self.log_msg("Preview started")

    def stop_preview(self):
        self.preview_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.pred_var.set("Pred: -")
        self.log_msg("Preview stopped")

    def _build_preview_config(self):
        method = self.method.get()
        edges = method == "canny"
        return PipelineConfig(
            invert=self.invert.get(),
            raw_hu=self.raw_hu.get(),
            edges=edges,
            min_area=int(self.min_area.get()),
            morph_size=max(1, int(self.morph_size.get()) | 1),
            dilate_iter=int(self.dilate_iter.get()),
            method=method,
            manual_thresh=int(self.manual_thresh.get()),
        )

    def _get_preview_roi(self):
        if not self.use_roi.get():
            return None
        return (
            int(self.roi_x.get()),
            int(self.roi_y.get()),
            int(self.roi_w.get()),
            int(self.roi_h.get()),
        )

    def on_preview_mouse_down(self, event):
        if not self.preview_running:
            return
        px, py = self._map_preview_point(event.x, event.y)
        if px is None:
            return
        if px >= self.preview_half[0] or py >= self.preview_half[1]:
            return
        self.dragging = True
        self.drag_start = (px, py)

    def on_preview_mouse_move(self, event):
        if not self.dragging or self.drag_start is None:
            return
        x0, y0 = self.drag_start
        px, py = self._map_preview_point(event.x, event.y)
        if px is None:
            return
        x1 = min(max(px, 0), self.preview_half[0] - 1)
        y1 = min(max(py, 0), self.preview_half[1] - 1)
        self._update_roi_from_preview(x0, y0, x1, y1)

    def on_preview_mouse_up(self, event):
        if not self.dragging or self.drag_start is None:
            return
        x0, y0 = self.drag_start
        px, py = self._map_preview_point(event.x, event.y)
        if px is None:
            self.dragging = False
            self.drag_start = None
            return
        x1 = min(max(px, 0), self.preview_half[0] - 1)
        y1 = min(max(py, 0), self.preview_half[1] - 1)
        self._update_roi_from_preview(x0, y0, x1, y1)
        self.dragging = False
        self.drag_start = None

    def _map_preview_point(self, x, y):
        x_adj = x - self.preview_pad[0]
        y_adj = y - self.preview_pad[1]
        if x_adj < 0 or y_adj < 0:
            return None, None
        px = int(x_adj / self.preview_scale)
        py = int(y_adj / self.preview_scale)
        if px < 0 or py < 0:
            return None, None
        if px >= self.preview_half[0] * 2 or py >= self.preview_half[1] * 2:
            return None, None
        return px, py

    def _update_roi_from_preview(self, x0, y0, x1, y1):
        scale = self.frame_tile_scale
        pad_x, pad_y = self.frame_tile_pad
        tile_w, tile_h = self.preview_half
        max_x = pad_x + max(1, tile_w - pad_x * 2)
        max_y = pad_y + max(1, tile_h - pad_y * 2)
        x0c = min(max(min(x0, x1), pad_x), max_x)
        x1c = min(max(max(x0, x1), pad_x), max_x)
        y0c = min(max(min(y0, y1), pad_y), max_y)
        y1c = min(max(max(y0, y1), pad_y), max_y)
        px0 = int((x0c - pad_x) / scale)
        py0 = int((y0c - pad_y) / scale)
        px1 = int((x1c - pad_x) / scale)
        py1 = int((y1c - pad_y) / scale)
        w = max(1, px1 - px0)
        h = max(1, py1 - py0)
        self.roi_x.set(px0)
        self.roi_y.set(py0)
        self.roi_w.set(w)
        self.roi_h.set(h)
        self.use_roi.set(True)

    def _prepare_previews(self, frame, thresh, crop):
        if thresh is None:
            thresh = np.zeros(frame.shape[:2], dtype=np.uint8)
        if crop is None:
            crop = np.zeros_like(frame)

        thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        h, w = frame.shape[:2]
        half_w = max(1, w // 2)
        half_h = max(1, h // 2)

        frame_small, frame_scale, frame_pad = self._letterbox_bgr(frame, half_w, half_h)
        thresh_small, _, _ = self._letterbox_bgr(thresh_bgr, half_w, half_h)
        crop_small, _, _ = self._letterbox_bgr(crop, half_w, half_h)
        self.frame_tile_scale = frame_scale
        self.frame_tile_pad = frame_pad
        blank = np.zeros_like(frame_small)

        top = np.concatenate([frame_small, thresh_small], axis=1)
        bottom = np.concatenate([crop_small, blank], axis=1)
        return np.concatenate([top, bottom], axis=0)

    def update_preview(self):
        if not self.preview_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.log_msg("Failed to read frame")
            self.stop_preview()
            return

        config = self._build_preview_config()
        roi = self._get_preview_roi()
        result = analyze_frame(frame, config, roi=roi)
        self.last_result = result

        display = frame.copy()
        contour = result["contour"]
        if result["roi"] is not None:
            x, y, w, h = result["roi"]
            cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if contour is not None:
                roi_view = display[y : y + h, x : x + w]
                cv2.drawContours(roi_view, [contour], -1, (0, 255, 0), 2)
        elif contour is not None:
            cv2.drawContours(display, [contour], -1, (0, 255, 0), 2)

        if self.mode.get() == "Classifier":
            if self.model is None:
                self.pred_var.set("Pred: load model")
            elif result["hu"] is not None:
                pred = self.model.predict([result["hu"]])[0]
                label_text = self.label_map.get(int(pred), str(int(pred)))
                self.pred_var.set(f"Pred: {label_text} ({int(pred)})")
                cv2.putText(
                    display,
                    f"Pred: {label_text} ({int(pred)})",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 200, 0),
                    2,
                    cv2.LINE_AA,
                )
            else:
                self.pred_var.set("Pred: -")
                cv2.putText(
                    display,
                    "No contour",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 40, 40),
                    2,
                    cv2.LINE_AA,
                )

        thresh = result["thresh"]
        crop = result["crop"]
        if crop is not None and contour is not None:
            cv2.drawContours(crop, [contour], -1, (0, 255, 0), 2)
        preview = self._prepare_previews(display, thresh, crop)
        preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(preview_rgb)
        target_w = max(1, self.preview_label.winfo_width())
        target_h = max(1, self.preview_label.winfo_height())
        if target_w < 10 or target_h < 10:
            target_w, target_h = 800, 450
        resized, scale, pad_x, pad_y = self._resize_with_letterbox(pil, target_w, target_h)
        self.preview_half = (preview.shape[1] // 2, preview.shape[0] // 2)
        self.preview_scale = scale
        self.preview_pad = (pad_x, pad_y)
        photo = ImageTk.PhotoImage(resized)
        self.preview_label.configure(image=photo)
        self.preview_label.image = photo

        self.root.after(30, self.update_preview)

    def capture_hu(self):
        if self.last_result is None or self.last_result["hu"] is None:
            self.log_msg("No contour to capture (preview)")
            return
        label = int(self.gen_label.get())
        output = self.gen_output.get().strip()
        if not output:
            self.log_msg("Set output CSV path")
            return
        row = list(self.last_result["hu"]) + [label]
        write_header = not os.path.exists(output) or os.path.getsize(output) == 0
        append_row(output, row, write_header=write_header)
        hu_str = ", ".join(f"{v:.6g}" for v in self.last_result["hu"])
        self.log_msg(f"HU: [{hu_str}]")
        self.log_msg(f"Saved sample label={label} -> {output}")

    def load_model(self):
        try:
            self.model = load(self.cls_model.get())
            self.label_map = load_label_map(self.cls_labels.get())
            self.log_msg("Model loaded for preview")
        except Exception as exc:
            self.log_msg(f"Load failed: {exc}")

    def run_selected(self):
        self.log_msg("CLI launch removed. Run generator.py/trainer.py/classifier.py manually.")

    def _resize_with_letterbox(self, image, target_w, target_h):
        src_w, src_h = image.size
        scale = min(target_w / src_w, target_h / src_h)
        new_w = max(1, int(src_w * scale))
        new_h = max(1, int(src_h * scale))
        resized = image.resize((new_w, new_h))
        canvas = Image.new("RGB", (target_w, target_h), (20, 20, 20))
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        canvas.paste(resized, (pad_x, pad_y))
        return canvas, scale, pad_x, pad_y

    def _letterbox_bgr(self, image_bgr, target_w, target_h, bg_color=(20, 20, 20)):
        src_h, src_w = image_bgr.shape[:2]
        scale = min(target_w / src_w, target_h / src_h)
        new_w = max(1, int(src_w * scale))
        new_h = max(1, int(src_h * scale))
        resized = cv2.resize(image_bgr, (new_w, new_h))
        canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
        return canvas, scale, (pad_x, pad_y)


def main():
    root = tk.Tk()
    app = App(root)
    def on_close():
        app.stop_preview()
        app.stop_process()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
