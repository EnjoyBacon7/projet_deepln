# DEVA: Description-Enhanced Video–Audio Sentiment (CMU-MOSI)

End-to-end pipeline for multimodal sentiment analysis on CMU-MOSI:

- Authentic preprocessing from the source data (audio, video, transcript) with OpenFace AUs.
- Two models:
	- Baseline: Concatenate text/audio/visual features → MLP.
	- DEVA: Text-guided token fusion with cross-modal attention and semantic descriptions.

Repository highlights:
- Notebooks: `Data_Preprocessing.ipynb` (feature generation) and `Projet_musi.ipynb` (models + training).
- Inference scripts in `inference/` for quick tests on new videos.

---

## Environment

Python ≥ 3.12. Dependencies are listed in `pyproject.toml` and resolved via `uv` or `pip`.

Using uv (recommended):
```zsh
cd /Users/camillebizeul/Documents/GitHub/projet_deepln
uv venv .venv
source .venv/bin/activate
uv pip install -r <(uv pip compile pyproject.toml -q)
```

Using pip:
```zsh
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e CMU-MultimodalSDK  # provides mmsdk
pip install -r <(python - <<'PY'
from tomllib import load
import sys
with open('pyproject.toml','rb') as f:
		data = load(f)
for dep in data['project']['dependencies']:
		print(dep)
PY
)
```

Key packages: `torch`, `transformers`, `librosa`, `opencv-python-headless`, `pandas`, `numpy`, `mmsdk`, `tqdm`.

---

## Data

Expected CMU-MOSI layout:
```
MOSI/
	Audio/WAV_16000/Segmented/*.wav
	Video/Segmented/*.mp4
	Transcript/Segmented/*.annotprocessed
```
Labels are loaded from CMU-MultimodalSDK (included in `CMU-MultimodalSDK/`).

---

## OpenFace (AUs)

OpenFace is invoked automatically during preprocessing to extract real Action Unit intensities. Build once:

Option A: our script
```zsh
chmod +x ./openface.sh
./openface.sh
```

Option B: manual build
```zsh
brew update && brew install cmake pkg-config opencv boost tbb openblas wget
git clone https://github.com/TadasBaltrusaitis/OpenFace.git external_libs/openFace/OpenFace
cd external_libs/openFace/OpenFace
bash ./download_models.sh
mkdir -p build && cd build
cmake -D CMAKE_BUILD_TYPE=Release ..
make -j"$(sysctl -n hw.ncpu)"
```

Notebook configuration (already in code):
```python
OPENFACE_BIN = "./external_libs/openFace/OpenFace/build/bin/FeatureExtraction"
OPENFACE_OUTPUT_DIR = "./openface_output"
OPENFACE_AUTO = True  # automatically run OpenFace if CSV missing
```

---

## Preprocessing (Notebook)

Open `Data_Preprocessing.ipynb` and run the cells. It will:
- Audio: load each segment, compute MFCC (20), prosody (pitch, loudness, jitter, shimmer), textify audio description, BERT-encode it.
- Video: auto-run OpenFace on each segment if needed, average AU intensities from CSV, build a visual description, BERT-encode it, and compute a tiny-CNN visual vector for a mid frame.
- Text: load segment transcript, encode via BERT-based `TextEncoder`.
- Output: saves `preprocessed_data/` with:
	- `mosi_preprocessed.pkl` (DataFrame with arrays)
	- `mosi_metadata.csv` (human-readable metadata)
	- `dataset_info.pkl` (feature dimensions)

---

## Model Pipelines

### Baseline

```mermaid
flowchart LR
	subgraph Baseline Model
		A1[Text Transcript] --> A2[TextEncoder (BERT→768)]
		A2 --> T[Text Embedding (768)]
		B1[Audio WAV] --> B2[MFCC (20)] --> AM[Audio Features (20)]
		C1[Video Segment] --> C2[Tiny CNN (8)] --> VM[Visual Features (8)]
		T --> CONCAT[Concat (768+20+8)]
		AM --> CONCAT
		VM --> CONCAT
		CONCAT --> MLP[MLP Regressor] --> Y[(Sentiment Score)]
	end
```

ASCII
```
Text -> TextEncoder (768) ----\
Audio -> MFCC (20) -----------+--> Concat (796) -> MLP -> Score
Video -> Tiny CNN (8) --------/
```

### DEVA (Description-Enhanced)

```mermaid
flowchart LR
	subgraph Inputs
		TTXT[Text Transcript] --> TOK[Tokenize] --> TE[TextEncoder (BERT+Transformer)] --> XT[Text Tokens X_t (T×D)]
		AW[Audio WAV] --> MF[MFCC (20)] --> AP[Audio Projector] --> XA[X_a (T×D)]
		VW[Video Segment] --> VC[Tiny CNN (8)] --> VP[Visual Projector] --> XV[X_v (T×D)]
		DAs[Audio Desc D_a (BERT 768)] --> G1[Audio Gate+Transform] --> XA
		DVs[Visual Desc D_v (BERT 768)] --> G2[Visual Gate+Transform] --> XV
	end
	subgraph MFU_Stack[L layers]
		XA -->|query| ATTa[Cross-Attn Audio<-Text]
		XT -->|key/value| ATTa
		XV -->|query| ATTv[Cross-Attn Visual<-Text]
		XT -->|key/value| ATTv
		Fprev[Residual Fusion (init = X_t)] --> RES[LayerNorm(fusion + α·ATTa + β·ATTv)] --> Fnext
	end
	Fnext --> POOL[Mean Pool over T] --> HEAD[Prediction Head (MLP)] --> Yd[(Sentiment Score)]
```

ASCII
```
Text -> TextEncoder -> X_t (T×D)
Audio MFCC (20) -> Projector -> X_a (T×D)  +  gate(D_a)
Video CNN (8)   -> Projector -> X_v (T×D)  +  gate(D_v)

Repeat L times:
	fusion = LayerNorm(fusion + α·Attn(X_a ← X_t) + β·Attn(X_v ← X_t))

MeanPool(fusion over tokens) -> MLP -> Score
```

---

## Training & Evaluation (Notebook)

Open `Projet_musi.ipynb` and run:
- Baseline training (batched): concatenated features → MSE regression; reports Acc-2, F1, MAE, Pearson Corr.
- DEVA training (batched): token-aligned fusion with MFU; same metrics.

Metrics utilities provided in the notebook (`evaluate_metrics`).

---

## Inference on a New Video

DEVA CLI (transcribes with Whisper, uses MFCC + zero visual vector by default):
```zsh
python inference/predict_on_video.py /path/to/video.mp4
```
Output prints the transcript (trimmed), continuous sentiment score, and a coarse label (+1/0/−1).

Live webcam demo:
```zsh
python inference/live_prediction.py
```

Note: The inference scripts load the latest saved DEVA checkpoint from `models/` (as produced by the notebook).

---

## Troubleshooting

- OpenFace binary not found:
	- Ensure `OPENFACE_BIN` points to your build: `./external_libs/openFace/OpenFace/build/bin/FeatureExtraction`.
	- Re-run `./openface.sh` to build and install model files.
- Empty AUs:
	- OpenFace may fail on some segments; the notebook leaves `aus_intensities` empty and sets a neutral description.
- Import errors:
	- Make sure your virtualenv is active and `CMU-MultimodalSDK` is installed (or editable-installed) so `mmsdk` resolves.

---

## Repository Structure

```
Data_Preprocessing.ipynb   # Feature extraction + OpenFace AUs + embeddings
Projet_musi.ipynb          # Baseline & DEVA models + training/eval
inference/
	predict_on_video.py      # CLI inference (Whisper + DEVA)
	live_prediction.py       # Live webcam demo
CMU-MultimodalSDK/         # SDK (editable install provides mmsdk)
openface.sh                # Build helper for OpenFace (macOS)
preprocessed_data/         # Outputs written by preprocessing notebook
models/                    # Saved model checkpoints/configs
pyproject.toml             # Python dependencies
```

---

## Acknowledgements

- CMU-MultimodalSDK (CMU-MOSI)
- OpenFace (facial Action Units)
- Hugging Face Transformers (BERT)
- Librosa (audio features)