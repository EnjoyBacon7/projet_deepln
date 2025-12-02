import os
import json
import threading
import multiprocessing as mp
from collections import deque
from typing import Optional, Tuple

import numpy as np
import pygame
import cv2
try:
    import sounddevice as sd
except Exception:
    sd = None
import torch
import whisper
try:
    from faster_whisper import WhisperModel as FasterWhisperModel
except Exception:
    FasterWhisperModel = None
try:
    import vosk
except Exception:
    vosk = None
try:
    import webrtcvad
except Exception:
    webrtcvad = None

# Local imports from the inference module directory
from predict_on_video import load_deva


def asr_worker(audio_q: "mp.Queue", text_q: "mp.Queue", device: str, model_name: str, use_faster: bool, engine: str, vosk_model_path: Optional[str]):
    """Process target: consume audio chunks, produce text pieces via queues.

    engine: 'whisper' | 'faster-whisper' | 'vosk'
    """
    try:
        # Allow MPS to fall back gracefully for unsupported kernels
        if device == "mps":
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        # Initialize engine
        rec = None
        fw = None
        ow = None
        if engine == "vosk" and vosk is not None and vosk_model_path and os.path.isdir(vosk_model_path):
            try:
                model = vosk.Model(vosk_model_path)
                rec = vosk.KaldiRecognizer(model, 16000)
            except Exception:
                rec = None
                engine = "whisper"
        if engine == "faster-whisper" and FasterWhisperModel is not None:
            fw_device = "cuda" if device == "cuda" else "cpu"
            if device == "cuda":
                compute_type = "float16"
            elif device == "mps":
                compute_type = "float32"
            else:
                compute_type = "int8"
            fw = FasterWhisperModel(model_name, device=fw_device, compute_type=compute_type)
        if engine == "whisper" or (fw is None and rec is None):
            ow = whisper.load_model(model_name, device=device)
        # Optional VAD gate
        vad = webrtcvad.Vad(2) if webrtcvad is not None else None
        while True:
            chunk = audio_q.get()
            if chunk is None:
                break
            try:
                if chunk.size == 0:
                    continue
                # VAD gate
                if vad is not None:
                    pcm16 = np.clip(chunk * 32768.0, -32768, 32767).astype(np.int16).tobytes()
                    frame_ms = 30
                    frame_len = int(16000 * frame_ms / 1000) * 2
                    speech, total = 0, 0
                    for i in range(0, len(pcm16) - frame_len + 1, frame_len):
                        frame = pcm16[i:i+frame_len]
                        total += 1
                        try:
                            if vad.is_speech(frame, 16000):
                                speech += 1
                        except Exception:
                            pass
                    if total > 0 and (speech / total) < 0.2:
                        continue
                else:
                    if float(np.mean(chunk**2)) < 1e-4:
                        continue

                if rec is not None:
                    pcm16 = np.clip(chunk * 32768.0, -32768, 32767).astype(np.int16).tobytes()
                    text_piece = ""
                    if rec.AcceptWaveform(pcm16):
                        try:
                            j = json.loads(rec.Result())
                            text_piece = (j.get("text") or "").strip()
                        except Exception:
                            text_piece = ""
                    else:
                        try:
                            j = json.loads(rec.PartialResult())
                            text_piece = (j.get("partial") or "").strip()
                        except Exception:
                            text_piece = ""
                elif fw is not None:
                    segs, _ = fw.transcribe(chunk, language="en", beam_size=1, vad_filter=True)
                    text_piece = " ".join((s.text or "").strip() for s in segs).strip()
                else:
                    res = ow.transcribe(
                        chunk,
                        language="en",
                        fp16=(device == "cuda"),
                        beam_size=1,
                        best_of=1,
                        temperature=0.0,
                        condition_on_previous_text=False,
                        without_timestamps=True,
                    )
                    text_piece = (res.get("text") or "").strip()
                if text_piece:
                    text_q.put(text_piece)
            except Exception:
                continue
    except Exception:
        pass


def infer_worker(
    in_q: "mp.Queue",
    out_q: "mp.Queue",
    models_dir: str,
    device: str,
    text_max_len_fallback: int = 32,
):
    """Process target: consume (text, audio) and produce (score, label)."""
    try:
        # Load model and tokenizer once in this process
        model, tokenizer, cfg, device = load_deva(models_dir=models_dir, device=device)
        try:
            tokenizer.truncation_side = "left"
        except Exception:
            pass
        visual_dim = int(cfg.get("visual_cnn_dim", 8))
        visual_vec = np.zeros((visual_dim,), dtype=np.float32)
        visual_cnn_tensor = torch.tensor(visual_vec, dtype=torch.float32, device=device).unsqueeze(0)
        text_max_len = int(cfg.get("text_max_len", text_max_len_fallback))

        while True:
            item = in_q.get()
            if item is None:
                break
            try:
                text: str = item.get("text", "")
                audio_vec: np.ndarray = item.get("audio", np.zeros((0,), dtype=np.float32))

                tokens = tokenizer(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=text_max_len,
                )
                input_ids = tokens["input_ids"].to(device)
                attention_mask = tokens["attention_mask"].to(device)

                n_mfcc = int(cfg.get("audio_mfcc_dim", 20))
                mfcc_vec = _compute_mfcc_from_audio_array(audio_vec, sr=16000, n_mfcc=n_mfcc)
                audio_mfcc = torch.tensor(mfcc_vec, dtype=torch.float32, device=device).unsqueeze(0)

                with torch.no_grad():
                    if device == "cuda":
                        with torch.autocast("cuda", dtype=torch.float16):
                            pred = model(input_ids, attention_mask, audio_mfcc, visual_cnn_tensor, D_a=None, D_v=None).squeeze(-1)
                    else:
                        pred = model(input_ids, attention_mask, audio_mfcc, visual_cnn_tensor, D_a=None, D_v=None).squeeze(-1)
                    score = float(pred.item())
                label = 0
                if score > 0.1:
                    label = 1
                elif score < -0.1:
                    label = -1
                try:
                    out_q.put_nowait((score, label))
                except Exception:
                    # If full, drop older results and put latest
                    try:
                        _ = out_q.get_nowait()
                    except Exception:
                        pass
                    try:
                        out_q.put_nowait((score, label))
                    except Exception:
                        pass
            except Exception:
                continue
    except Exception:
        pass


def _compute_mfcc_from_audio_array(audio: np.ndarray, sr: int = 16000, n_mfcc: int = 20) -> np.ndarray:
    import librosa
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    mfcc = librosa.feature.mfcc(y=audio.astype(np.float64), sr=sr, n_mfcc=n_mfcc)
    return mfcc.mean(axis=1).astype(np.float32)


def _preferred_device(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class RingAudioBuffer:
    def __init__(self, samplerate: int, max_seconds: float = 20.0):
        self.sr = samplerate
        self.max_samples = int(max_seconds * samplerate)
        self.buf = deque(maxlen=self.max_samples)
        self.lock = threading.Lock()

    def append(self, samples: np.ndarray):
        with self.lock:
            self.buf.extend(samples.tolist())

    def pop_chunk(self, seconds: float) -> Optional[np.ndarray]:
        need = int(seconds * self.sr)
        with self.lock:
            if len(self.buf) < need:
                return None
            out = [self.buf.popleft() for _ in range(need)]
        return np.asarray(out, dtype=np.float32)

    def snapshot(self, seconds: float) -> np.ndarray:
        take = int(seconds * self.sr)
        with self.lock:
            if take <= 0 or len(self.buf) == 0:
                return np.zeros((0,), dtype=np.float32)
            arr = np.asarray(self.buf, dtype=np.float32)
            if len(arr) <= take:
                return arr.copy()
            return arr[-take:].copy()


class LivePredictor:
    def __init__(
        self,
        models_dir: str = "models",
        whisper_model: str = "tiny.en",
        audio_sr: int = 16000,
        audio_chunk_seconds: float = 1.5,
        device: Optional[str] = None,
        camera_index: int = 0,
        window_size: Tuple[int, int] = (1280, 720),
    ):
        self.device = _preferred_device(device)

        # Decide ASR engine (env overrides)
        env_engine = os.environ.get("ASR_ENGINE", "").strip().lower()
        self.vosk_model_path = os.environ.get("VOSK_MODEL_PATH")
        if env_engine == "vosk" and vosk is not None and self.vosk_model_path:
            self.asr_engine = "vosk"
        elif self.device == "mps":
            self.asr_engine = "whisper"  # leverage MPS GPU
        elif FasterWhisperModel is not None and env_engine in ("", "faster-whisper", "fast", "whisper"):
            self.asr_engine = "faster-whisper"
        else:
            self.asr_engine = "whisper"
        self.use_faster = (self.asr_engine == "faster-whisper")
        # ASR in separate process (spawn context for macOS)
        self.ctx = mp.get_context("spawn")
        self.audio_q: mp.Queue = self.ctx.Queue(maxsize=4)
        self.text_q: mp.Queue = self.ctx.Queue(maxsize=8)
        self.asr_proc: Optional[mp.Process] = None
        self.whisper_model_name = whisper_model

        # Inference in separate process
        self.inf_in_q: mp.Queue = self.ctx.Queue(maxsize=1)
        self.inf_out_q: mp.Queue = self.ctx.Queue(maxsize=1)
        self.inf_proc: Optional[mp.Process] = None

        self.audio_sr = audio_sr
        self.audio_chunk_seconds = audio_chunk_seconds
        self.audio_buf = RingAudioBuffer(audio_sr, max_seconds=30.0)
        self.transcript = ""
        self.last_sentiment = (0.0, 0)

        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        pygame.init()
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Live Sentiment (Video + Whisper + DEVA)")
        self.font = pygame.font.SysFont(None, 24)
        self.bigfont = pygame.font.SysFont(None, 40)
        self.clock = pygame.time.Clock()

        self.running = True

        # Start ASR process
        self._start_asr_process()
        # Start inference process
        self._start_infer_process(models_dir)

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            # Avoid stdout spam; ignore for now
            pass
        samples = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        self.audio_buf.append(samples)

    def _start_audio(self):
        if sd is None:
            raise RuntimeError(
                "sounddevice is not installed. Install it with 'pip install sounddevice'. "
                "On macOS, if installation fails, run 'brew install portaudio' then reinstall."
            )
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.audio_sr,
            dtype="float32",
            callback=self._audio_callback,
        )
        self.stream.start()

    def _stop_audio(self):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass

    def _start_asr_process(self):
        self.asr_proc = self.ctx.Process(
            target=asr_worker,
            args=(self.audio_q, self.text_q, self.device, self.whisper_model_name, self.use_faster, self.asr_engine, self.vosk_model_path),
            daemon=True,
        )
        self.asr_proc.start()

    def _stop_asr_process(self):
        try:
            if self.asr_proc is not None:
                self.audio_q.put(None)
                self.asr_proc.join(timeout=2.0)
        except Exception:
            pass

    def _start_infer_process(self, models_dir: str):
        self.inf_proc = self.ctx.Process(
            target=infer_worker,
            args=(self.inf_in_q, self.inf_out_q, models_dir, self.device),
            daemon=True,
        )
        self.inf_proc.start()

    def _stop_infer_process(self):
        try:
            if self.inf_proc is not None:
                self.inf_in_q.put(None)
                self.inf_proc.join(timeout=2.0)
        except Exception:
            pass

    def _kick_transcription(self):
        # Move audio chunk to ASR process if queue has space
        chunk = self.audio_buf.pop_chunk(self.audio_chunk_seconds)
        if chunk is None:
            return
        try:
            self.audio_q.put_nowait(chunk)
        except Exception:
            pass

    def _poll_asr_and_infer(self):
        # Drain available text pieces and trigger inference on the latest
        new_texts = []
        while True:
            try:
                t = self.text_q.get_nowait()
                new_texts.append(t)
            except Exception:
                break
        if not new_texts:
            return
        for t in new_texts:
            if t:
                self.transcript = (self.transcript + " " + t).strip()
        audio_for_mfcc = self.audio_buf.snapshot(3.0)
        # Send latest task to inference process; drop if queue busy
        try:
            self.inf_in_q.put_nowait({"text": self.transcript, "audio": audio_for_mfcc})
        except Exception:
            # Replace pending with latest
            try:
                _ = self.inf_in_q.get_nowait()
            except Exception:
                pass
            try:
                self.inf_in_q.put_nowait({"text": self.transcript, "audio": audio_for_mfcc})
            except Exception:
                pass

    def _poll_infer_results(self):
        try:
            while True:
                score, label = self.inf_out_q.get_nowait()
                self.last_sentiment = (score, label)
        except Exception:
            pass

    def _render(self, frame_bgr: np.ndarray):
        self.screen.fill((20, 20, 20))
        # Left panel area
        panel_w = int(self.screen.get_width() * 0.6)
        panel_h = self.screen.get_height()
        if frame_bgr is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            # Preserve aspect ratio to fit into (panel_w, panel_h)
            scale = min(panel_w / w, panel_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame_resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # Pygame surface expects (width, height, 3) array; transpose axes accordingly
            frame_for_pg = np.transpose(frame_resized, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame_for_pg)
            # Center within the left panel
            x_off = (panel_w - new_w) // 2
            y_off = (panel_h - new_h) // 2
            self.screen.blit(surf, (x_off, y_off))

        panel_x = panel_w + 16
        # right panel starts at panel_x

        score, label = self.last_sentiment
        lbl_text = {1: "Positive", 0: "Neutral", -1: "Negative"}.get(label, "Neutral")
        color = (80, 200, 120) if label == 1 else (220, 80, 80) if label == -1 else (200, 200, 200)
        header = self.bigfont.render(f"Sentiment: {lbl_text}", True, color)
        self.screen.blit(header, (panel_x, 20))
        score_surf = self.font.render(f"Score: {score:.3f}", True, (220, 220, 220))
        self.screen.blit(score_surf, (panel_x, 70))

        # Wrap transcript
        transcript = self.transcript[-1000:]
        words = transcript.split()
        lines = []
        line = ""
        max_chars = 60
        for w in words:
            if len(line) + 1 + len(w) > max_chars:
                lines.append(line)
                line = w
            else:
                line = (line + " " + w).strip()
        if line:
            lines.append(line)
        y = 110
        title = self.font.render("Transcript (latest):", True, (180, 180, 180))
        self.screen.blit(title, (panel_x, y))
        y += 24
        for ln in lines[-18:]:
            txt = self.font.render(ln, True, (230, 230, 230))
            self.screen.blit(txt, (panel_x, y))
            y += 22

        hint = self.font.render("Press ESC or close window to quit", True, (140, 140, 140))
        self.screen.blit(hint, (panel_x, self.screen.get_height() - 30))

        pygame.display.flip()

    def run(self):
        self._start_audio()
        try:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        self.running = False

                ok, frame = self.cap.read()
                if not ok:
                    frame = None

                self._kick_transcription()
                self._poll_asr_and_infer()
                self._poll_infer_results()
                self._render(frame)
                self.clock.tick(30)
        finally:
            self._stop_audio()
            self._stop_asr_process()
            self._stop_infer_process()
            try:
                self.cap.release()
            except Exception:
                pass
            pygame.quit()


if __name__ == "__main__":
    LivePredictor().run()
