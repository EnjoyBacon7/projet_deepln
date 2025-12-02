import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
from torch import nn
from transformers import BertTokenizer, BertModel
import whisper


# -----------------------------------------------------------------------------
# Model definitions (mirrors the notebook)
# -----------------------------------------------------------------------------
class TextEncoder(nn.Module):
    def __init__(self, bert_model: str = "bert-base-uncased", bert_dim: int = 768, num_output_tokens: int = 8, nhead: int = 8, device: str = "cpu"):
        super().__init__()
        self.bert_dim = bert_dim
        self.T = num_output_tokens
        self.special_token = nn.Parameter(torch.randn(1, 1, bert_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=bert_dim,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # External BERT instance; tokenizer created alongside
        self._bert_name = bert_model
        self._device = device
        self._bert = BertModel.from_pretrained(self._bert_name).to(self._device)
        for p in self._bert.parameters():
            p.requires_grad = False
        self._bert.eval()

    @property
    def bert(self):
        return self._bert

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self._bert(input_ids=input_ids, attention_mask=attention_mask)
            seq = out.last_hidden_state  # [B, L, 768]
        B = seq.size(0)
        tok = self.special_token.expand(B, -1, -1)
        seq = torch.cat([tok, seq], dim=1)
        enc = self.transformer_encoder(seq)
        if enc.size(1) >= self.T:
            return enc[:, :self.T, :]
        pad = torch.zeros(B, self.T - enc.size(1), self.bert_dim, device=enc.device)
        return torch.cat([enc, pad], dim=1)


class CrossModalAttention(nn.Module):
    def __init__(self, dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        B = query.size(0)
        Q = self.query_proj(query).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(key_value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(key_value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.dim)
        return self.out_proj(out)


class AudioVisualFeatureProjector(nn.Module):
    def __init__(self, input_dim, output_dim=768, num_tokens=8):
        super().__init__()
        self.num_tokens = num_tokens
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim * num_tokens),
        )

    def forward(self, x):
        B = x.size(0)
        proj = self.projection(x)
        return proj.view(B, self.num_tokens, -1)

class MFU(nn.Module):
    def __init__(self, dim=768, num_heads=8):
        super().__init__()
        self.cross_attn_audio = CrossModalAttention(dim=dim, num_heads=num_heads)
        self.cross_attn_visual = CrossModalAttention(dim=dim, num_heads=num_heads)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.norm = nn.LayerNorm(dim)

    def forward(self, X_t, X_a, X_v, prev_fusion=None):
        a = self.cross_attn_audio(query=X_a, key_value=X_t)
        v = self.cross_attn_visual(query=X_v, key_value=X_t)
        prev = X_t if prev_fusion is None else prev_fusion
        return self.norm(prev + self.alpha * a + self.beta * v)


class DEVANet(nn.Module):
    def __init__(self, text_encoder: TextEncoder, audio_mfcc_dim=20, visual_cnn_dim=8,
                 hidden_dim=768, num_tokens=8, num_mfu_layers=1, num_heads=8):
        super().__init__()
        self.text_encoder = text_encoder
        self.hidden_dim = hidden_dim
        self.audio_projector = AudioVisualFeatureProjector(audio_mfcc_dim, hidden_dim, num_tokens)
        self.visual_projector = AudioVisualFeatureProjector(visual_cnn_dim, hidden_dim, num_tokens)
        self.mfu_layers = nn.ModuleList([MFU(dim=hidden_dim, num_heads=num_heads) for _ in range(num_mfu_layers)])
        self.audio_desc_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.audio_desc_transform = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.LayerNorm(hidden_dim))
        self.visual_desc_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.visual_desc_transform = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.LayerNorm(hidden_dim))
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1)
        )

    def forward(self, input_ids, attention_mask, audio_features, visual_features, D_a=None, D_v=None):
        X_t = self.text_encoder(input_ids, attention_mask)
        X_a = self.audio_projector(audio_features)
        X_v = self.visual_projector(visual_features)
        fusion = X_t
        for m in self.mfu_layers:
            fusion = m(X_t, X_a, X_v, prev_fusion=fusion)
        pooled = fusion.mean(dim=1)
        return self.prediction_head(pooled)


# -----------------------------------------------------------------------------
# DEVA loading utilities
# -----------------------------------------------------------------------------
_CACHED: Dict[str, Any] = {"model": None, "tokenizer": None, "cfg": None, "device": None}

def _preferred_device(explicit: Optional[str] = None) -> str:
    """Pick the best available device or honor explicit override."""
    if explicit:
        return explicit
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _pick_latest_model_files(models_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    cfgs = sorted(models_dir.glob("deva_config_*.json"))
    states = sorted(models_dir.glob("deva_state_dict_*.pt"))
    if not cfgs or not states:
        return None, None
    return cfgs[-1], states[-1]


def load_deva(models_dir: str = "models", device: Optional[str] = None):
    models_dir_path = Path(models_dir)
    device = _preferred_device(device)

    c_auto, s_auto = _pick_latest_model_files(models_dir_path)
    cfg_path = str(c_auto) if c_auto else None
    state_path = str(s_auto) if s_auto else None

    if not cfg_path or not state_path:
        raise FileNotFoundError(
            f"Could not find DEVA model/config. Looked in {models_dir_path}. "
            f"Ensure you ran the notebook cell that saves models (Cell 13)."
        )

    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    # Tokenizer must match training
    tokenizer = BertTokenizer.from_pretrained(cfg.get("tokenizer", "bert-base-uncased"))

    # Build text encoder and model
    text_encoder = TextEncoder(
        bert_model=cfg.get("bert_model", "bert-base-uncased"),
        bert_dim=cfg.get("hidden_dim", 768),
        num_output_tokens=cfg.get("num_tokens", 8),
        nhead=cfg.get("num_heads", 8),
        device=device,
    )

    model = DEVANet(
        text_encoder=text_encoder,
        audio_mfcc_dim=int(cfg.get("audio_mfcc_dim", 20)),
        visual_cnn_dim=int(cfg.get("visual_cnn_dim", 8)),
        hidden_dim=int(cfg.get("hidden_dim", 768)),
        num_tokens=int(cfg.get("num_tokens", 8)),
        num_mfu_layers=int(cfg.get("num_mfu_layers", 1)),
        num_heads=int(cfg.get("num_heads", 8)),
    ).to(device)

    state = torch.load(state_path, map_location=device)

    # Accept checkpoints saved in different wrappers and allow missing BERT weights.
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    incompatible = model.load_state_dict(state, strict=False)
    miss, unexp = incompatible.missing_keys, incompatible.unexpected_keys
    if miss:
        keep = [k for k in miss if not k.startswith("text_encoder._bert.")]
        if keep:
            print(f"Warning: Missing keys while loading model: {len(keep)} keys (showing first 5): {keep[:5]}")
    if unexp:
        print(f"Warning: Unexpected keys in checkpoint: {len(unexp)} keys (showing first 5): {unexp[:5]}")
    model.eval()

    _CACHED.update({"model": model, "tokenizer": tokenizer, "cfg": cfg, "device": device})
    return model, tokenizer, cfg, device


def _compute_mfcc_from_audio_array(audio: np.ndarray, sr: int = 16000, n_mfcc: int = 20) -> np.ndarray:
    import librosa  # ensured above
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    mfcc = librosa.feature.mfcc(y=audio.astype(np.float64), sr=sr, n_mfcc=n_mfcc)
    return mfcc.mean(axis=1).astype(np.float32)


def _transcribe_with_whisper(video_path: str, model_name: str = "base", device: str = "cpu") -> str:
    """Return a transcript string for `video_path` using Whisper, honoring device.

    Disables FP16 on CPU/MPS to avoid warnings and ensure correctness.
    """
    if not hasattr(whisper, "load_model") or not hasattr(whisper, "load_audio"):
        raise RuntimeError(
            "The 'whisper' module does not expose load_model/load_audio. Install 'openai-whisper'."
        )
    asr = whisper.load_model(model_name, device=device)
    use_fp16 = device == "cuda"
    result = asr.transcribe(video_path, language="en", fp16=use_fp16)
    return (result.get("text") or "").strip()


def predict_on_video(video_path: str,
                     whisper_model: str = "base",
                     models_dir: str = "models",
                     device: Optional[str] = None,
                     max_len: Optional[int] = None) -> Dict[str, Any]:
    """
    Transcribe a video, build features, and run the DEVA model.

    Returns a dict with: {
        'text': transcript,
        'score': float sentiment score,
        'label': +1/-1/0,
    }
    """
    if _CACHED.get("model") is None:
        load_deva(models_dir=models_dir, device=device)

    model = _CACHED["model"]
    tokenizer = _CACHED["tokenizer"]
    cfg = _CACHED["cfg"]
    device = _CACHED["device"]

    text_max_len = int(cfg.get("text_max_len", 32)) if max_len is None else int(max_len)

    print("\n[Step 1/3] Transcribing with Whisper...")
    text = _transcribe_with_whisper(video_path, whisper_model, device=device)
    if not text:
        print("Warning: Empty transcript. Proceeding with empty text.")
        text = ""

    print("[Step 2/3] Extracting audio MFCCs...")
    audio = whisper.load_audio(video_path)  # float32 mono, sr=16000
    n_mfcc = int(cfg.get("audio_mfcc_dim", 20))
    audio_mfcc_vec = _compute_mfcc_from_audio_array(audio, sr=16000, n_mfcc=n_mfcc)

    print("[Step 3/3] Building tensors and running model...")
    tokens = tokenizer(text,
                       return_tensors='pt',
                       padding='max_length',
                       truncation=True,
                       max_length=text_max_len)
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    visual_dim = int(cfg.get('visual_cnn_dim', 8))
    visual_vec = np.zeros((visual_dim,), dtype=np.float32)

    audio_mfcc = torch.tensor(audio_mfcc_vec, dtype=torch.float32, device=device).unsqueeze(0)
    visual_cnn = torch.tensor(visual_vec, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        pred = model(input_ids, attention_mask, audio_mfcc, visual_cnn, D_a=None, D_v=None).squeeze(-1)
        score = float(pred.item())

    label = 0
    if score > 0.1:
        label = 1
    elif score < -0.1:
        label = -1

    print(f"\nTranscript: {text[:200]}{'...' if len(text)>200 else ''}")
    print(f"Predicted sentiment score: {score:.3f}  => label {label}")

    return {"text": text, "score": score, "label": label}


# -----------------------------------------------------------------------------
# CLI entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run DEVA sentiment inference on a video file.")
    parser.add_argument("video", help="Path to video file")

    args = parser.parse_args()

    out = predict_on_video(video_path=args.video)

    print("\n=== Inference Result ===")
    print(f"Score: {out['score']:.3f} | Label: {out['label']}")
    txt = out.get('text', '')
    print("Text:", (txt[:500] + ('...' if len(txt) > 500 else '')))