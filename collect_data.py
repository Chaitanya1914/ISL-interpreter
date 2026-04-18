import customtkinter as ctk
import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time
from PIL import Image
import threading

# ─────────────────────────────────────────────────────────
# VOCABULARY: These are the 10 words your AI will learn
# ─────────────────────────────────────────────────────────
WORDS = ["Hello", "Thank You", "Yes", "No", "Help",
         "Water", "Food", "Good", "Sorry", "Please"]

SAMPLES_PER_WORD = 300   # Frames captured per word (higher = better model)
CSV_FILE = "my_isl_dataset.csv"

# ─────────────────────────────────────────────────────────
# CSV HEADER BUILDER
# ─────────────────────────────────────────────────────────
def build_header():
    header = ["class_name"]
    for hand in ["lh", "rh"]:
        for i in range(21):
            for axis in ["x", "y", "z"]:
                header.append(f"{hand}_{i}_{axis}")
    return header

# ─────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────
class DataCollector(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ISL Data Collection Studio")
        self.geometry("1100x700")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.current_word_idx = 0
        self.collecting = False
        self.sample_count = 0
        self.countdown_active = False
        self.countdown_val = 3
        self.total_recorded = 0

        # ── LAYOUT ──────────────────────────────────────────
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)

        # LEFT — Webcam
        self.video_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.video_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.video_label = ctk.CTkLabel(self.video_frame, text="Initializing Camera...")
        self.video_label.pack(expand=True, fill="both")

        # RIGHT — Control Sidebar
        self.sidebar = ctk.CTkFrame(self, width=340, fg_color="#18181A")
        self.sidebar.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")
        self.sidebar.grid_propagate(False)

        ctk.CTkLabel(self.sidebar, text="ISL Studio",
                     font=ctk.CTkFont("Roboto", 28, "bold")).pack(pady=(30, 5))
        ctk.CTkLabel(self.sidebar, text="Custom Dataset Builder",
                     font=ctk.CTkFont("Roboto", 13), text_color="#888").pack(pady=(0, 25))

        # Current sign display
        ctk.CTkLabel(self.sidebar, text="Sign This Word:",
                     font=ctk.CTkFont(size=14)).pack()
        self.word_label = ctk.CTkLabel(self.sidebar, text=WORDS[0],
                                        font=ctk.CTkFont("Roboto", 46, "bold"),
                                        text_color="#1FA56A")
        self.word_label.pack(pady=(5, 15))

        # Status
        self.status_label = ctk.CTkLabel(self.sidebar, text="Press START to begin collection",
                                          font=ctk.CTkFont(size=13), text_color="#AAAAAA",
                                          wraplength=280)
        self.status_label.pack(pady=(0, 20))

        # Progress bar for samples
        ctk.CTkLabel(self.sidebar, text="Samples Captured:",
                     font=ctk.CTkFont(size=13)).pack()
        self.progress_bar = ctk.CTkProgressBar(self.sidebar, width=270, height=18,
                                                progress_color="#1FA56A")
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=(5, 5))
        self.progress_count_label = ctk.CTkLabel(self.sidebar,
                                                  text=f"0 / {SAMPLES_PER_WORD}",
                                                  font=ctk.CTkFont(size=12),
                                                  text_color="#888")
        self.progress_count_label.pack(pady=(0, 20))

        # Word list
        ctk.CTkLabel(self.sidebar, text="Vocabulary List:",
                     font=ctk.CTkFont(size=13)).pack(pady=(10, 5))
        self.word_list_box = ctk.CTkTextbox(self.sidebar, height=180, width=280,
                                             fg_color="#212124", text_color="#E0E0E0",
                                             font=ctk.CTkFont(size=13))
        self.word_list_box.pack(padx=20, pady=(0, 20))
        for i, w in enumerate(WORDS):
            self.word_list_box.insert("end", f"  {'✅' if False else '⬜'}  {i+1:02d}. {w}\n")
        self.word_list_box.configure(state="disabled")

        # Start button
        self.start_btn = ctk.CTkButton(self.sidebar, text="▶  START COLLECTION",
                                        font=ctk.CTkFont(size=15, weight="bold"),
                                        fg_color="#1FA56A", hover_color="#178a56",
                                        height=48, command=self.begin_countdown)
        self.start_btn.pack(padx=20, pady=(5, 10), fill="x")

        # Skip button
        self.skip_btn = ctk.CTkButton(self.sidebar, text="⏭  Skip This Word",
                                       font=ctk.CTkFont(size=13),
                                       fg_color="#333", hover_color="#444",
                                       height=36, command=self.skip_word)
        self.skip_btn.pack(padx=20, pady=(0, 10), fill="x")

        # Total count
        self.total_label = ctk.CTkLabel(self.sidebar, text="Total recorded: 0 samples",
                                         font=ctk.CTkFont(size=12), text_color="#666")
        self.total_label.pack(pady=5)

        # ── MediaPipe ──────────────────────────────────────
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(max_num_hands=2,
                                          min_detection_confidence=0.6,
                                          min_tracking_confidence=0.5)

        # ── CSV Init ───────────────────────────────────────
        if not os.path.exists(CSV_FILE):
            with open(CSV_FILE, "w", newline="") as f:
                csv.writer(f).writerow(build_header())
            print(f"Created fresh dataset: {CSV_FILE}")
        else:
            print(f"Appending to existing: {CSV_FILE}")

        # ── Webcam ─────────────────────────────────────────
        self.cap = cv2.VideoCapture(0)
        self.overlay_text = ""
        self.overlay_color = (255, 255, 255)

        self.update_video()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ── Video Loop ─────────────────────────────────────────
    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(frame_rgb)

            lh_row = [0.0] * 63
            rh_row = [0.0] * 63
            hand_detected = False

            if result.multi_hand_landmarks and result.multi_handedness:
                hand_detected = True
                for idx, hand_lm in enumerate(result.multi_hand_landmarks):
                    label = result.multi_handedness[idx].classification[0].label
                    coords = []
                    for lm in hand_lm.landmark:
                        coords.extend([lm.x, lm.y, lm.z])
                    if label == "Left":
                        lh_row = coords
                        self.mp_drawing.draw_landmarks(
                            frame_rgb, hand_lm, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=(250, 44, 121), thickness=2, circle_radius=4))
                    else:
                        rh_row = coords
                        self.mp_drawing.draw_landmarks(
                            frame_rgb, hand_lm, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=(16, 245, 117), thickness=2, circle_radius=4))

            # If actively collecting, save a row
            if self.collecting and hand_detected and self.sample_count < SAMPLES_PER_WORD:
                row = [WORDS[self.current_word_idx]] + lh_row + rh_row
                with open(CSV_FILE, "a", newline="") as f:
                    csv.writer(f).writerow(row)
                self.sample_count += 1
                self.total_recorded += 1
                prog = self.sample_count / SAMPLES_PER_WORD
                self.progress_bar.set(prog)
                self.progress_count_label.configure(text=f"{self.sample_count} / {SAMPLES_PER_WORD}")
                self.total_label.configure(text=f"Total recorded: {self.total_recorded} samples")

                if self.sample_count >= SAMPLES_PER_WORD:
                    self.collecting = False
                    self.on_word_complete()

            # Draw live collection indicator
            frame_rgb = cv2.flip(frame_rgb, 1)
            if self.collecting:
                cv2.rectangle(frame_rgb, (0, 0), (720, 60), (31, 165, 106), -1)
                cv2.putText(frame_rgb, f"RECORDING: {WORDS[self.current_word_idx]}  [{self.sample_count}/{SAMPLES_PER_WORD}]",
                            (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            elif self.overlay_text:
                cv2.rectangle(frame_rgb, (0, 0), (720, 60), (40, 40, 40), -1)
                cv2.putText(frame_rgb, self.overlay_text,
                            (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.overlay_color, 2)

            if not hand_detected:
                cv2.putText(frame_rgb, "No hand detected", (10, 530),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 200), 2)

            img = Image.fromarray(frame_rgb).resize((720, 540))
            imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(720, 540))
            self.video_label.configure(image=imgtk, text="")

        self.after(15, self.update_video)

    # ── Countdown before recording ──────────────────────────
    def begin_countdown(self):
        if self.collecting or self.countdown_active:
            return
        if self.current_word_idx >= len(WORDS):
            self.status_label.configure(text="✅ All words collected! Run train_custom.py to build your model.", text_color="#1FA56A")
            return

        self.start_btn.configure(state="disabled")
        self.countdown_active = True
        self.countdown_val = 3
        self._do_countdown()

    def _do_countdown(self):
        if self.countdown_val > 0:
            self.overlay_text = f"Get Ready: {self.countdown_val}..."
            self.overlay_color = (255, 200, 0)
            self.status_label.configure(
                text=f"Starting in {self.countdown_val}... Hold the sign for '{WORDS[self.current_word_idx]}'",
                text_color="#FFCC00")
            self.countdown_val -= 1
            self.after(1000, self._do_countdown)
        else:
            self.countdown_active = False
            self.overlay_text = ""
            self.collecting = True
            self.sample_count = 0
            self.progress_bar.set(0)
            self.progress_count_label.configure(text=f"0 / {SAMPLES_PER_WORD}")
            self.status_label.configure(text=f"🔴 RECORDING... Keep signing '{WORDS[self.current_word_idx]}'!",
                                        text_color="#FF4444")

    # ── Word complete ───────────────────────────────────────
    def on_word_complete(self):
        # Mark word done in list
        self._refresh_word_list(self.current_word_idx)
        self.current_word_idx += 1
        self.start_btn.configure(state="normal")

        if self.current_word_idx >= len(WORDS):
            self.status_label.configure(text="🎉 All 10 words collected! Run train_custom.py now!", text_color="#1FA56A")
            self.word_label.configure(text="Done!", text_color="#1FA56A")
            self.start_btn.configure(state="disabled", text="All Complete ✅")
        else:
            self.word_label.configure(text=WORDS[self.current_word_idx])
            self.status_label.configure(
                text=f"✅ '{WORDS[self.current_word_idx - 1]}' done! Next: '{WORDS[self.current_word_idx]}'. Press START when ready.",
                text_color="#1FA56A")
            self.progress_bar.set(0)
            self.progress_count_label.configure(text=f"0 / {SAMPLES_PER_WORD}")

    def _refresh_word_list(self, done_idx):
        self.word_list_box.configure(state="normal")
        self.word_list_box.delete("1.0", "end")
        for i, w in enumerate(WORDS):
            tick = "✅" if i <= done_idx else "⬜"
            self.word_list_box.insert("end", f"  {tick}  {i+1:02d}. {w}\n")
        self.word_list_box.configure(state="disabled")

    # ── Skip word ───────────────────────────────────────────
    def skip_word(self):
        if self.collecting:
            self.collecting = False
        self.on_word_complete()

    # ── Cleanup ─────────────────────────────────────────────
    def on_close(self):
        self.collecting = False
        self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = DataCollector()
    app.mainloop()
