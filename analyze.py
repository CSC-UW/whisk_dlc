"""Run the trained whisker_stim model on one or more videos.

The project's config.yaml stores an absolute `project_path`, and DeepLabCut uses
it to find the model. This script first points it at wherever the repo lives on
this machine, so a fresh clone works without hand-editing the config.

    uv run python analyze.py D:\\videos\\spica-20261007_160406-cam2.mp4
    uv run python analyze.py D:\\videos\\*.mp4 --crop 80 640 256 540
    uv run python analyze.py video.mp4 --batch-size 16 --csv

Outputs (.h5, plus .csv with --csv) are written next to each video unless
--dest is given. Videos that already have results are skipped.
"""

import argparse
import glob
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent / "whisker_stim-LL-2026-08-24"
CONFIG = PROJECT / "config.yaml"
# ResNet-50 at output_stride 16: a crop origin off this grid shifts predictions slightly.
GRID = 16


def relocate_project() -> None:
    """Point config.yaml's project_path at this checkout, if it isn't already."""
    from deeplabcut.utils import auxiliaryfunctions

    cfg = auxiliaryfunctions.read_config(str(CONFIG))
    if Path(cfg["project_path"]) != PROJECT:
        auxiliaryfunctions.edit_config(str(CONFIG), {"project_path": str(PROJECT)})
        print(f"project_path updated to {PROJECT}")


def shift_to_full_frame(result: Path, x1: int, y1: int) -> None:
    """DLC 3.0.1 saves cropped-run coordinates relative to the crop; add the offset back."""
    import pandas as pd

    df = pd.read_hdf(result)
    coords = df.columns.get_level_values("coords")
    df.loc[:, coords == "x"] += x1
    df.loc[:, coords == "y"] += y1
    df.to_hdf(result, key="df_with_missing", format="table", mode="w")
    csv = result.with_suffix(".csv")
    if csv.exists():
        df.to_csv(csv)
    print(f"shifted to full-frame coordinates (+{x1}, +{y1}): {result.name}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("videos", nargs="+", help="video files (wildcards allowed)")
    p.add_argument("--crop", nargs=4, type=int, metavar=("X1", "X2", "Y1", "Y2"),
                   help="analyze only this region; saved coordinates are full-frame")
    p.add_argument("--batch-size", type=int, help="frames per GPU batch (default: config.yaml's batch_size)")
    p.add_argument("--dest", help="folder for the output files (default: next to each video)")
    p.add_argument("--csv", action="store_true", help="also save results as .csv")
    args = p.parse_args()

    videos = [Path(v).resolve() for pattern in args.videos for v in (glob.glob(pattern) or [pattern])]
    missing = [str(v) for v in videos if not v.is_file()]
    if missing:
        print("video(s) not found:\n  " + "\n  ".join(missing))
        return 1
    if args.crop and (args.crop[0] % GRID or args.crop[2] % GRID):
        print(f"warning: X1 and Y1 should be multiples of {GRID}; otherwise predictions "
              "shift by up to ~2 px relative to full-frame analysis")

    import deeplabcut

    relocate_project()
    started = time.time()
    scorer = deeplabcut.analyze_videos(
        str(CONFIG),
        [str(v) for v in videos],
        shuffle=1,
        batch_size=args.batch_size,
        cropping=args.crop,
        destfolder=args.dest,
        save_as_csv=args.csv,
    )

    if args.crop:
        for v in videos:
            result = Path(args.dest or v.parent) / f"{v.stem}{scorer}.h5"
            # Results from an earlier run were skipped by DLC and are already shifted.
            if result.exists() and result.stat().st_mtime >= started:
                shift_to_full_frame(result, args.crop[0], args.crop[2])
    return 0


if __name__ == "__main__":
    sys.exit(main())
