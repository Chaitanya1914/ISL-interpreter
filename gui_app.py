import customtkinter as ctk
from PIL import Image
import cv2
import mediapipe as mp
import numpy as np
import collections
from collections import Counter
import threading
import pickle
import time
from gtts import gTTS
import pygame
import os
import io
from datetime import datetime

pygame.mixer.init()

# --- EXACT NORMALISATION MATH ---
def normalise(landmarks):
    lms = np.array(landmarks).reshape(21, 3)
    wrist = lms[0]
    lms_shifted = lms - wrist
    dist = np.linalg.norm(lms_shifted[9])
    if dist < 1e-6:
        dist = 1e-6
    return (lms_shifted / dist).flatten()

# --- ASYNC AUDIO BACKGROUND WORKER (RAM STREAMING) ---
def play_audio_async(text):
    def _play():
        try:
            tts = gTTS(text=text, lang='en')
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            pygame.mixer.music.load(fp, 'mp3')
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            pygame.mixer.music.unload()
        except: pass
    threading.Thread(target=_play, daemon=True).start()


# ═══════════════════════════════════════════════════════════
# ULTIMATE GRAPHICAL DASHBOARD WITH ADVANCED FILTERING
# ═══════════════════════════════════════════════════════════
class ISLApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Indian Sign Language — Advanced Recognition Hub")
        self.geometry("1280x780")
        self.minsize(1100, 700)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # ── INTERNAL DATA STORES ────────────────────────────
        self.word_log = []           # [{word, confidence, timestamp}, ...]
        self.active_filters = set()  # words currently allowed (empty = all)
        self.sort_mode = "Time ↓"    # default sort
        self.tts_enabled = True
        self.paused = False

        # ── 3-COLUMN LAYOUT ─────────────────────────────────
        self.grid_columnconfigure(0, weight=0)   # Left: Filter panel
        self.grid_columnconfigure(1, weight=1)   # Center: Webcam
        self.grid_columnconfigure(2, weight=0)   # Right: Live HUD
        self.grid_rowconfigure(0, weight=1)

        self._build_filter_panel()
        self._build_video_panel()
        self._build_hud_panel()

        # ── BACKEND ─────────────────────────────────────────
        self.load_ml_backend()
        self.cap = cv2.VideoCapture(0)

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5)

        self.buffer = collections.deque(maxlen=20)
        self.last_emit_time = time.time()
        self.confirmed_word = ""

        self.update_video()

    # ════════════════════════════════════════════════════════
    #  LEFT PANEL — ADVANCED FILTERING & SORTING
    # ════════════════════════════════════════════════════════
    def _build_filter_panel(self):
        self.filter_panel = ctk.CTkScrollableFrame(
            self, width=230, fg_color="#111113",
            label_text="  ⚙ Filters & Sorting",
            label_font=ctk.CTkFont(size=15, weight="bold"),
            label_fg_color="#1A1A1C")
        self.filter_panel.grid(row=0, column=0, padx=(15, 0), pady=15, sticky="nsew")

        # ── CONFIDENCE THRESHOLD SLIDER ─────────────────────
        ctk.CTkLabel(self.filter_panel, text="Min Confidence",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color="#CCCCCC").pack(pady=(15, 2), anchor="w", padx=10)

        self.conf_slider = ctk.CTkSlider(
            self.filter_panel, from_=0.3, to=0.99,
            number_of_steps=69, width=200,
            progress_color="#1FA56A", button_color="#1FA56A",
            command=self._on_conf_change)
        self.conf_slider.set(0.70)
        self.conf_slider.pack(padx=10, pady=(0, 2))

        self.conf_val_label = ctk.CTkLabel(
            self.filter_panel, text="Threshold: 70%",
            font=ctk.CTkFont(size=12), text_color="#888")
        self.conf_val_label.pack(anchor="w", padx=10)

        # ── SORT MODE ───────────────────────────────────────
        ctk.CTkLabel(self.filter_panel, text="Sort History By",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color="#CCCCCC").pack(pady=(20, 4), anchor="w", padx=10)

        self.sort_var = ctk.StringVar(value="Time ↓")
        sort_options = ["Time ↓", "Time ↑", "A → Z", "Z → A", "Confidence ↓", "Frequency ↓"]
        self.sort_menu = ctk.CTkOptionMenu(
            self.filter_panel, values=sort_options,
            variable=self.sort_var, width=200,
            fg_color="#2A2A2E", button_color="#1FA56A",
            command=self._on_sort_change)
        self.sort_menu.pack(padx=10)

        # ── WORD FILTER CHECKBOXES ──────────────────────────
        ctk.CTkLabel(self.filter_panel, text="Word Filter",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color="#CCCCCC").pack(pady=(20, 4), anchor="w", padx=10)

        ctk.CTkLabel(self.filter_panel, text="(Uncheck to ignore a sign)",
                     font=ctk.CTkFont(size=11), text_color="#666").pack(anchor="w", padx=10)

        self.word_checkboxes = {}
        self._placeholder_label = ctk.CTkLabel(
            self.filter_panel, text="Load model first...",
            font=ctk.CTkFont(size=11), text_color="#555")
        self._placeholder_label.pack(pady=5, padx=10, anchor="w")

        # ── CONTROLS ────────────────────────────────────────
        sep = ctk.CTkFrame(self.filter_panel, height=2, fg_color="#333")
        sep.pack(fill="x", padx=10, pady=(25, 15))

        self.tts_switch = ctk.CTkSwitch(
            self.filter_panel, text="Audio Feedback",
            font=ctk.CTkFont(size=13),
            progress_color="#1FA56A",
            command=self._toggle_tts)
        self.tts_switch.select()
        self.tts_switch.pack(padx=10, pady=5, anchor="w")

        self.pause_btn = ctk.CTkButton(
            self.filter_panel, text="⏸  Pause Detection",
            font=ctk.CTkFont(size=13), height=36,
            fg_color="#333", hover_color="#444",
            command=self._toggle_pause)
        self.pause_btn.pack(padx=10, pady=(10, 5), fill="x")

        self.clear_btn = ctk.CTkButton(
            self.filter_panel, text="🗑  Clear History",
            font=ctk.CTkFont(size=13), height=36,
            fg_color="#6B2020", hover_color="#8B3030",
            command=self._clear_history)
        self.clear_btn.pack(padx=10, pady=5, fill="x")

        # ── LIVE STATS ──────────────────────────────────────
        sep2 = ctk.CTkFrame(self.filter_panel, height=2, fg_color="#333")
        sep2.pack(fill="x", padx=10, pady=(20, 10))

        ctk.CTkLabel(self.filter_panel, text="Session Stats",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color="#CCCCCC").pack(anchor="w", padx=10)

        self.stats_label = ctk.CTkLabel(
            self.filter_panel,
            text="Words: 0\nUnique: 0\nAvg Conf: —\nTop Word: —",
            font=ctk.CTkFont(size=12),
            text_color="#888", justify="left")
        self.stats_label.pack(anchor="w", padx=10, pady=(5, 15))

    # ════════════════════════════════════════════════════════
    #  CENTER — VIDEO FEED
    # ════════════════════════════════════════════════════════
    def _build_video_panel(self):
        self.video_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.video_frame.grid(row=0, column=1, padx=10, pady=15, sticky="nsew")
        self.video_label = ctk.CTkLabel(self.video_frame, text="Initializing Camera Array...")
        self.video_label.pack(expand=True, fill="both", padx=5, pady=5)

    # ════════════════════════════════════════════════════════
    #  RIGHT PANEL — LIVE HUD
    # ════════════════════════════════════════════════════════
    def _build_hud_panel(self):
        self.sidebar = ctk.CTkFrame(self, width=300, fg_color="#18181A")
        self.sidebar.grid(row=0, column=2, padx=(0, 15), pady=15, sticky="nsew")

        ctk.CTkLabel(self.sidebar, text="Live Interpreter",
                     font=ctk.CTkFont(size=26, weight="bold")).pack(pady=(25, 15))

        ctk.CTkLabel(self.sidebar, text="Sign Designation:",
                     font=ctk.CTkFont(size=13)).pack(pady=(15, 0))
        self.pred_val = ctk.CTkLabel(
            self.sidebar, text="-- Ready --",
            font=ctk.CTkFont(size=38, weight="bold"),
            text_color="#1FA56A")
        self.pred_val.pack(pady=(0, 15))

        ctk.CTkLabel(self.sidebar, text="Neural Confidence:",
                     font=ctk.CTkFont(size=13)).pack(pady=(8, 0))
        self.conf_progressbar = ctk.CTkProgressBar(
            self.sidebar, width=220, height=15,
            progress_color="#1FA56A")
        self.conf_progressbar.set(0)
        self.conf_progressbar.pack(pady=(4, 20))

        # Auto-brightness indicator
        self.enhancement_indicator = ctk.CTkLabel(
            self.sidebar, text="Auto-Brightness: Standby",
            font=ctk.CTkFont(size=13), text_color="#A0A0A0")
        self.enhancement_indicator.pack(pady=8)

        # Sentence box
        ctk.CTkLabel(self.sidebar, text="Constructed Sentence:",
                     font=ctk.CTkFont(size=13)).pack(pady=(20, 4))
        self.history_textbox = ctk.CTkTextbox(
            self.sidebar, height=120, width=260,
            fg_color="#212124", text_color="#E0E0E0",
            wrap="word", font=ctk.CTkFont(size=16))
        self.history_textbox.pack(padx=15, pady=(0, 10))

        # Filtered history log
        ctk.CTkLabel(self.sidebar, text="Filtered Log:",
                     font=ctk.CTkFont(size=13)).pack(pady=(10, 4))
        self.log_textbox = ctk.CTkTextbox(
            self.sidebar, height=160, width=260,
            fg_color="#1A1A1E", text_color="#B0B0B0",
            wrap="word", font=ctk.CTkFont(size=12))
        self.log_textbox.pack(padx=15, pady=(0, 15))

    # ════════════════════════════════════════════════════════
    #  ML BACKEND
    # ════════════════════════════════════════════════════════
    def load_ml_backend(self):
        self.model = None
        self.encoder = None
        self.all_classes = []
        try:
            with open('isl_model.pkl', 'rb') as f:
                model_dict = pickle.load(f)
            self.model = model_dict['model']
            self.encoder = model_dict['encoder']
            self.all_classes = list(self.encoder.classes_)
            self._build_word_checkboxes()
        except:
            self.history_textbox.insert("0.0", "[Error]: Train Model First! ")

    def _build_word_checkboxes(self):
        """Dynamically create a toggle for every class the model knows."""
        self._placeholder_label.destroy()
        self.check_vars = {}
        for cls in sorted(self.all_classes):
            clean = cls.split('.')[-1].strip() if '.' in cls else cls
            var = ctk.IntVar(value=1)
            cb = ctk.CTkCheckBox(
                self.filter_panel, text=clean,
                variable=var, font=ctk.CTkFont(size=12),
                fg_color="#1FA56A", hover_color="#178A56",
                command=self._on_filter_change)
            cb.pack(anchor="w", padx=15, pady=2)
            self.word_checkboxes[cls] = cb
            self.check_vars[cls] = var

    # ════════════════════════════════════════════════════════
    #  FILTER / SORT CALLBACKS
    # ════════════════════════════════════════════════════════
    def _on_conf_change(self, val):
        pct = int(val * 100)
        self.conf_val_label.configure(text=f"Threshold: {pct}%")

    def _on_sort_change(self, _=None):
        self.sort_mode = self.sort_var.get()
        self._refresh_log()

    def _on_filter_change(self):
        self.active_filters = {
            cls for cls, var in self.check_vars.items() if var.get() == 1
        }
        self._refresh_log()

    def _toggle_tts(self):
        self.tts_enabled = self.tts_switch.get()

    def _toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.pause_btn.configure(text="▶  Resume Detection", fg_color="#1FA56A", hover_color="#178A56")
        else:
            self.pause_btn.configure(text="⏸  Pause Detection", fg_color="#333", hover_color="#444")

    def _clear_history(self):
        self.word_log.clear()
        self.history_textbox.delete("1.0", "end")
        self.log_textbox.delete("1.0", "end")
        self.confirmed_word = ""
        self.pred_val.configure(text="-- Ready --")
        self._update_stats()

    # ════════════════════════════════════════════════════════
    #  SORTED + FILTERED LOG RENDERER
    # ════════════════════════════════════════════════════════
    def _refresh_log(self):
        """Rebuild the filtered log textbox based on current sort & filter settings."""
        filtered = self.word_log
        if self.active_filters:
            filtered = [e for e in filtered if e["raw_class"] in self.active_filters]

        mode = self.sort_mode
        if mode == "Time ↓":
            sorted_data = sorted(filtered, key=lambda e: e["ts"], reverse=True)
        elif mode == "Time ↑":
            sorted_data = sorted(filtered, key=lambda e: e["ts"])
        elif mode == "A → Z":
            sorted_data = sorted(filtered, key=lambda e: e["word"].lower())
        elif mode == "Z → A":
            sorted_data = sorted(filtered, key=lambda e: e["word"].lower(), reverse=True)
        elif mode == "Confidence ↓":
            sorted_data = sorted(filtered, key=lambda e: e["conf"], reverse=True)
        elif mode == "Frequency ↓":
            freq = Counter(e["word"] for e in filtered)
            sorted_data = sorted(filtered, key=lambda e: (-freq[e["word"]], e["ts"]))
        else:
            sorted_data = filtered

        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", "end")
        for entry in sorted_data:
            line = f"[{entry['time_str']}]  {entry['word']}  ({entry['conf']*100:.0f}%)\n"
            self.log_textbox.insert("end", line)
        self.log_textbox.configure(state="disabled")

    def _update_stats(self):
        total = len(self.word_log)
        if total == 0:
            self.stats_label.configure(text="Words: 0\nUnique: 0\nAvg Conf: —\nTop Word: —")
            return
        unique = len(set(e["word"] for e in self.word_log))
        avg_conf = np.mean([e["conf"] for e in self.word_log]) * 100
        top_word = Counter(e["word"] for e in self.word_log).most_common(1)[0][0]
        self.stats_label.configure(
            text=f"Words: {total}\nUnique: {unique}\nAvg Conf: {avg_conf:.1f}%\nTop Word: {top_word}")

    # ════════════════════════════════════════════════════════
    #  CLAHE
    # ════════════════════════════════════════════════════════
    def apply_clahe(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8,8))
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl,a,b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # ════════════════════════════════════════════════════════
    #  MAIN VIDEO LOOP
    # ════════════════════════════════════════════════════════
    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            # Auto-CLAHE
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            if brightness < 90:
                frame = self.apply_clahe(frame)
                self.enhancement_indicator.configure(
                    text="Auto-Brightness: ACTIVE", text_color="#1FA56A")
            else:
                self.enhancement_indicator.configure(
                    text="Auto-Brightness: Standby", text_color="#A0A0A0")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            predicted_sign = None
            max_prob = 0.0

            if not self.paused:
                result = self.hands.process(frame_rgb)

                lh_flat = np.zeros(63)
                rh_flat = np.zeros(63)

                if result.multi_hand_landmarks and result.multi_handedness:
                    for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                        label = result.multi_handedness[idx].classification[0].label
                        if label == "Left":
                            self.mp_drawing.draw_landmarks(
                                frame_rgb, hand_landmarks,
                                self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing.DrawingSpec(
                                    color=(250,44,121), thickness=2, circle_radius=4))
                            lh_flat = normalise(
                                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                        else:
                            self.mp_drawing.draw_landmarks(
                                frame_rgb, hand_landmarks,
                                self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing.DrawingSpec(
                                    color=(16,245,117), thickness=2, circle_radius=4))
                            rh_flat = normalise(
                                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

                    combined = np.concatenate([lh_flat, rh_flat])

                    if self.model and np.any(combined):
                        try:
                            probas = self.model.predict_proba([combined])[0]
                            max_idx = np.argmax(probas)
                            max_prob = probas[max_idx]

                            threshold = self.conf_slider.get()
                            if max_prob > threshold:
                                raw_class = self.encoder.inverse_transform([max_idx])[0]

                                # WORD FILTER: only accept if checkbox is ON
                                if not self.check_vars or self.check_vars.get(raw_class, ctk.IntVar(value=1)).get() == 1:
                                    predicted_sign = raw_class
                        except: pass

            self.conf_progressbar.set(max_prob)

            if predicted_sign:
                self.buffer.append(predicted_sign)

            current_time = time.time()
            if len(self.buffer) > 0:
                most_common_sign, _ = Counter(self.buffer).most_common(1)[0]
                clean = most_common_sign.split('.')[-1].strip() if '.' in most_common_sign else most_common_sign
                self.pred_val.configure(text=clean.upper())

                if most_common_sign != self.confirmed_word and (current_time - self.last_emit_time) > 0.6:
                    self.confirmed_word = most_common_sign
                    self.last_emit_time = current_time

                    clean_word = most_common_sign.split('.')[-1].strip().lower() if '.' in most_common_sign else most_common_sign.lower()

                    # Log the entry
                    entry = {
                        "word": clean_word,
                        "raw_class": most_common_sign,
                        "conf": max_prob,
                        "ts": time.time(),
                        "time_str": time.strftime('%H:%M:%S')
                    }
                    self.word_log.append(entry)

                    # Sentence builder
                    self.history_textbox.insert("end", clean_word + " ")
                    self.history_textbox.see("end")

                    # Filtered log
                    self._refresh_log()
                    self._update_stats()

                    # TTS
                    if self.tts_enabled:
                        play_audio_async(clean_word)

                    self.buffer.clear()

            # Flip for mirror display only
            frame_rgb = cv2.flip(frame_rgb, 1)

            img = Image.fromarray(frame_rgb).resize((680, 510))
            imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(680, 510))
            self.video_label.configure(image=imgtk, text="")

        self.after(15, self.update_video)

    def on_closing(self):
        self.cap.release()
        self.destroy()


if __name__ == "__main__":
    app = ISLApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
