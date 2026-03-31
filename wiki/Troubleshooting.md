# Troubleshooting

## Common Issues

### "No GPU detected — go to Runtime > Change runtime type"

**Cause:** The notebook is running on CPU-only runtime.

**Fix:**
1. Click **Runtime** in the top menu
2. Select **Change runtime type**
3. Under **Hardware accelerator**, select **T4 GPU**
4. Click **Save**
5. Wait for the runtime to restart, then re-run the installation cell

### "ModuleNotFoundError: No module named 'librosa' / 'onnxruntime' / etc."

**Cause:** Dependencies not installed or runtime restarted.

**Fix:** Re-run the installation cell:
```python
!pip install -q -r /content/vsep/requirements.txt
```

### "CUDA out of memory" / "RuntimeError: CUDA out of memory"

**Cause:** The model is too large for available GPU memory (T4 has 16GB).

**Fixes (try in order):**
1. **Use a smaller model** — MDX models (`.onnx`) are smaller than Roformer models (`.ckpt`)
2. **Enable chunking** — Add `chunk_duration=300` to the Separator call
3. **Use Colab Pro** with an A100 (40GB) for the largest models

### "Download failed" / Connection timeout

**Cause:** Slow internet or model repository rate limiting.

**Fixes:**
1. **Re-run** — downloads resume automatically (partial files are kept)
2. **Check internet** — make sure you have internet access in Colab
3. **Try a different model** — some are hosted on different servers
4. **Reduce parallel workers** — set `cfg.MAX_DOWNLOAD_WORKERS = 2`

### "Model not found: <name>"

**Cause:** The display name doesn't match any model in the catalog.

**Fix:**
1. Run the **Browse Models** cell to see the exact display names
2. Copy the display name exactly (case-sensitive)
3. Or use the model filename directly (e.g., `model_bs_roformer_ep_317_sdr_12.9755.ckpt`)

### "AttributeError: 'NoneType' object has no attribute 'version'"

**Cause:** This was a bug in an older version of vsep. Should be fixed in the latest version.

**Fix:** Pull the latest version:
```bash
!cd /content/vsep && git pull
```

### "OSError: [Errno 2] No such file or directory: 'audio_separator'"

**Cause:** Code is trying to import `audio_separator` as a pip package instead of using local paths.

**Fix:** This was fixed in the latest version. Pull the latest:
```bash
!cd /content/vsep && git pull
```

### Slow Separation on CPU

**Cause:** Running models on CPU is 5-10x slower than GPU.

**Fix:**
1. Enable GPU runtime (see above)
2. Use a smaller model (MDX-Net instead of Roformer)
3. Reduce audio length or enable chunking

### "ValueError: model not in MDXC model data"

**Cause:** The model file exists but its parameters aren't in `model-data.json`.

**Fix:**
1. Try running `python utils/cli.py --download_model_only <model_filename>` to force download metadata
2. If still failing, the model may not be compatible

### Sound quality issues (artifacts, clicking, distortion)

**Possible fixes:**
1. Try a different model — some handle specific genres better
2. Increase overlap for MDXC models: `mdxc_params={"overlap": 16}`
3. Try ensemble mode with multiple models
4. Post-process with de-reverb/de-echo models

### Google Colab disconnects / session timeout

**Cause:** Colab sessions time out after ~90 minutes of inactivity.

**Fix:**
1. Keep the browser tab active
2. Reconnecting restores the session but may lose variables — re-run from the Browse Models cell
3. Models are re-downloaded quickly (cached on first download)

## Getting More Help

If you encounter an issue not listed here:

1. **Check existing issues**: [github.com/BF667-IDLE/vsep/issues](https://github.com/BF667-IDLE/vsep/issues)
2. **Open a new issue**: Include your OS, Colab specs, model name, audio format, and the full error traceback
3. **GitHub Discussions**: Ask the community for advice
