# Google Colab

Open the [vsep Colab notebook](https://colab.research.google.com/github/BF667-IDLE/vsep/blob/main/notebooks/vsep_demo.ipynb) and run each cell in order.

## Notebook Structure

| Cell | Purpose |
|------|---------|
| **1. Install** | Clone repo, install dependencies, check GPU |
| **2. Upload Audio** | Upload file or paste YouTube/URL to download |
| **3. Browse Models** | View all models grouped by task category with filtering |
| **4. Select Model** | Paste display name to resolve to filename |
| **5. Separate** | Run separation with the selected model |
| **6. Listen & Download** | Play results and download to your device |

## Quick Start

1. Click **"Open in Colab"** on the [repository page](https://github.com/BF667-IDLE/vsep)
2. Change runtime to **GPU** (Runtime > Change runtime type > T4 GPU)
3. Run all cells in order
4. In the **Browse Models** cell, select a category and note your desired model's display name
5. Paste the display name in the **Select Model** cell
6. Run the separation and listen to results

## Using YouTube / URL Input

Instead of uploading, you can download audio from YouTube or any direct URL:

In the **Upload Audio** cell, change the source to "YouTube / URL" and paste the link. The notebook uses `yt-dlp` to download and convert the audio automatically.

## Tips for Colab

- **GPU Required**: Most models are too slow on CPU. Always use a GPU runtime.
- **T4 GPU (Free)**: Good for most models. MelBand Big Beta6X and 4-stem Demucs may be tight on 16GB.
- **A100 (Paid)**: Can run any model including the largest Roformer models.
- **Timeouts**: Colab sessions time out after ~90 minutes of inactivity. Keep your session active.
- **Storage**: Models are cached in the runtime's `/content/vsep/models/` directory. Re-downloading on a new session takes a few minutes.
- **Audio Length**: Very long files (>30 min) may cause memory issues on T4. Try chunking (set `chunk_duration=300` in the Separate cell).
- **Model Switching**: You can change the model and re-run the Separate cell without re-running Install or Upload.

## Using the Python API in Colab

The notebook also exposes the Python API for advanced usage:

```python
import sys
sys.path.insert(0, "/content/vsep")

from separator import Separator
import config.variables as cfg

# Speed up downloads
cfg.MAX_DOWNLOAD_WORKERS = 4

separator = Separator(
    model_file_dir="/content/vsep/models",
    output_dir="/content/vsep/output",
    use_soundfile=True,  # Avoids OOM on large files
)

# Load a specific model
separator.load_model("model_bs_roformer_ep_317_sdr_12.9755.ckpt")
output_files = separator.separate("audio_file.wav")
```
