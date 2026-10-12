"""Run the trained whisker_stim model on one or more videos.

The project's config.yaml stores an absolute `project_path`, and DeepLabCut uses
it to find the model. This script first points it at wherever the repo lives on
this machine, so a fresh clone works without hand-editing the config.

    uv run python analyze.py D:\videos\spica-20261007_160406-cam2.mp4
    uv run python analyze.py D:\videos\*.mp4 --crop 80 640 260 540
    uv run python analyze.py video.mp4 --batch-size 16 --autocast

Outputs (.h5, plus .csv with --csv) are written next to each video unless
--dest is given.
"""

import argparse
import glob
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent / "whisker_stim-LL-2026-08-24"
CONFIG = PROJECT / "config.yaml"


def relocate_project() -> None:
    """Point config.yaml's project_path at this checkout, if it isn't already."""
    from deeplabcut.utils import auxiliaryfunctions

    cfg = auxiliaryfunctions.read_config(str(CONFIG))
    if Path(cfg["project_path"]) != PROJECT:
        auxiliaryfunctions.edit_config(str(CONFIG), {"project_path": str(PROJECT)})
        print(f"project_path updated to {PROJECT}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("videos", nargs="+", help="video files (wildcards allowed)")
    p.add_argument("--crop", nargs=4, type=int, metavar=("X1", "X2", "Y1", "Y2"),
                   help="analyze only this region; coordinates in the output stay full-frame")
    p.add_argument("--batch-size", type=int, help="frames per GPU batch (default: config.yaml's batch_size)")
    p.add_argument("--autocast", action="store_true",
                   help="half-precision inference: faster, predictions can shift by a fraction of a pixel")
    p.add_argument("--dest", help="folder for the output files (default: next to each video)")
    p.add_argument("--csv", action="store_true", help="also save results as .csv")
    args = p.parse_args()

    videos = [v for pattern in args.videos for v in (glob.glob(pattern) or [pattern])]
    missing = [v for v in videos if not Path(v).is_file()]
    if missing:
        print("video(s) not found:\n  " + "\n  ".join(missing))
        return 1

    import deeplabcut

    relocate_project()
    deeplabcut.analyze_videos(
        str(CONFIG),
        [str(Path(v).resolve()) for v in videos],
        shuffle=1,
        batch_size=args.batch_size,
        cropping=args.crop,
        destfolder=args.dest,
        save_as_csv=args.csv,
        inference_cfg={"autocast": {"enabled": True}} if args.autocast else None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
