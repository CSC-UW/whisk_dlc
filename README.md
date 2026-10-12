# whisker-dlc

DeepLabCut 3 (PyTorch) model that tracks a single whisker and the snout in
720×540 high-speed side-view video. It tracks 7 points:

`whisker_base`, `blob_start`, `blob_end`, `whisker_tip`, `tip_blob_midpoint`,
`blob_base_midpoint`, `snout tip`

The model is ResNet-50, shuffle 1, iteration 0, snapshot `snapshot-best-060.pt`.

## What's in the repo

| Path | Contents |
|---|---|
| `whisker_stim-LL-2026-08-24/` | The DLC project: config, labels, training set, trained model |
| `analyze.py` | Runs the model on videos (fixes project paths automatically) |
| `fix_dlc_gui.py` | Patch for a Windows crash when opening the DLC GUI |
| `pyproject.toml`, `uv.lock` | Exact Python environment (Python 3.12, DeepLabCut 3.0.1) |

**Not in the repo:** videos (`videos/`, up to 7.5 GB each, too large for
GitHub), older snapshots, and the `*_labeled` check images. Copy videos over
separately (external drive, Box, OneDrive, network share).

## Setup (Windows PC with an NVIDIA GPU)

1. Install the NVIDIA driver, [Git](https://git-scm.com/download/win) and
   [uv](https://docs.astral.sh/uv/getting-started/installation/):

   ```powershell
   winget install --id astral-sh.uv
   ```

2. Clone and install the environment. uv downloads Python 3.12 and the exact
   package versions from `uv.lock`:

   ```powershell
   git clone <repo URL> whisker_dlc
   cd whisker_dlc
   uv sync
   ```

3. Check that PyTorch sees the GPU:

   ```powershell
   uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

   If this prints `False`, the installed PyTorch is CPU-only or the driver is
   missing. Analysis will still run, but very slowly.

## Analyzing videos

```powershell
uv run python analyze.py D:\videos\spica-20261007_160406-cam2.mp4
```

Results (`<video>DLC_Resnet50_...h5`) are written next to each video. Options:

| Option | Effect |
|---|---|
| `--crop X1 X2 Y1 Y2` | Analyze only this region; output coordinates stay full-frame. `80 640 260 540` covers the whisker and snout in the spica recordings. **Check that the tracked whisker never leaves the box.** |
| `--batch-size N` | Frames per GPU batch. Try 8–32; too large runs out of GPU memory. |
| `--autocast` | Half-precision inference: faster, predictions can shift by a fraction of a pixel. |
| `--dest FOLDER` | Write outputs to `FOLDER` instead of next to the videos. |
| `--csv` | Also save a `.csv` copy of the results. |

Multiple videos or wildcards work: `analyze.py D:\videos\*.mp4`.

On first run, `analyze.py` sets `project_path` in
`whisker_stim-LL-2026-08-24/config.yaml` to the clone's location. Commit that
change only if you want that machine's path to be the default.

To check tracking quality on a short excerpt, make a labeled video:

```powershell
uv run python -c "import deeplabcut; deeplabcut.create_labeled_video(r'whisker_stim-LL-2026-08-24\config.yaml', [r'D:\videos\clip.mp4'])"
```

Skip this for full-length recordings; rendering 1M+ frames takes about as long
as the analysis.

## Using the GUI

`uv run python -m deeplabcut` starts the GUI. If it crashes on startup, run
`uv run python fix_dlc_gui.py` once (and again after any `uv sync` that
reinstalls deeplabcut).

DLC 3.0.1 bug: in the **Extract Frames** tab, selecting specific videos always
fails with "Video files must be corrupted". Extract from Python instead:

```python
deeplabcut.extract_frames(cfg, mode="automatic", algo="uniform",
                          userfeedback=False, crop=False, videos_list=[r"...\video.mp4"])
```
