"""Simple desktop app: open an image or use the camera, see the predicted class.

Usage:
    python src/gui.py

Needs Tkinter (bundled with Python on Windows/macOS; on Ubuntu/Debian run
`sudo apt install python3-tk`). The result is colour-coded:
green = confident, orange = unsure, red = unknown.
"""
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from PIL import Image, ImageTk

import config
from predict import classify, format_top3, load_model

PREVIEW = 360                      # preview size in pixels


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Office item classifier")
        self.resizable(False, False)
        self.model = load_model()
        self.cap = None            # cv2.VideoCapture while the camera is on

        bar = tk.Frame(self)
        bar.pack(padx=10, pady=8)
        tk.Button(bar, text="Open image...", command=self.open_image).pack(side="left", padx=4)
        self.cam_btn = tk.Button(bar, text="Start camera", command=self.toggle_camera)
        self.cam_btn.pack(side="left", padx=4)

        self.canvas = tk.Label(self, width=PREVIEW, height=PREVIEW // 2,
                               bg="#eeeeee", text="No image yet")
        self.canvas.pack(padx=10)
        self.result = tk.Label(self, text="", font=("Arial", 18, "bold"))
        self.result.pack(pady=(8, 0))
        self.detail = tk.Label(self, text="", font=("Arial", 10), fg="#555555")
        self.detail.pack(pady=(0, 10))
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # -- showing results -----------------------------------------------------
    def show(self, bgr_or_pil, label, conf, top3):
        img = bgr_or_pil
        if not isinstance(img, Image.Image):                 # camera frame
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = img.copy()
        img.thumbnail((PREVIEW, PREVIEW))
        self.photo = ImageTk.PhotoImage(img)                 # keep a reference
        self.canvas.config(image=self.photo, text="", width=img.width, height=img.height)

        if label == "unknown":
            color = "#cc0000"
        elif conf >= config.HIGH_CONFIDENCE:
            color = "#1a8f3c"
        else:
            color = "#e08a00"
        self.result.config(text=f"{label} ({conf:.0%})", fg=color)
        self.detail.config(text=format_top3(top3))

    # -- image file ----------------------------------------------------------
    def open_image(self):
        self.stop_camera()
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        try:
            label, conf, top3 = classify(self.model, path)
            self.show(Image.open(path), label, conf, top3)
        except Exception as exc:                              # corrupt file etc.
            messagebox.showerror("Could not read image", str(exc))

    # -- camera --------------------------------------------------------------
    def toggle_camera(self):
        if self.cap is not None:
            self.stop_camera()
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap = None
            messagebox.showerror("Camera", "Could not open camera 0.")
            return
        self.cam_btn.config(text="Stop camera")
        self.update_frame()

    def update_frame(self):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if ok:
            label, conf, top3 = classify(self.model, frame)
            self.show(frame, label, conf, top3)
        self.after(30, self.update_frame)

    def stop_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.cam_btn.config(text="Start camera")

    def on_close(self):
        self.stop_camera()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
