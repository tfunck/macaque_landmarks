#!/usr/bin/env python3
r"""Build the landmark GUI's section CSV from autoradiograph file names.

File names look like ``mr#rh#11530#L#musc#5768#37#L_crop.nii.gz``:

    field 3 = subject (11530)      field 4 = hemisphere (L)
    field 5 = ligand (musc)        field 7 = section number (37)

A section's existing labels are ``<image name without .nii.gz>_landmark.nii.gz``
in the labels folder (empty in the CSV when there is none).

Scan folders:

    python make_section_csv.py --root /data/macaque --images preprocessed \
        --labels v1.3 --out /data/macaque/sections.csv

or use a list of paths relative to --root (e.g. from ``find``):

    python make_section_csv.py --root /data/macaque --file-list files.txt \
        --labels v1.3 --out /data/macaque/sections.csv

Paths in the CSV are written relative to the CSV's folder when possible
(the GUI resolves relative paths against the CSV's folder).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd

NAME = re.compile(r"^[^#]+#[^#]+#(?P<sub>[^#]+)#(?P<hemi>[^#]+)#(?P<ligand>[^#]+)#[^#]+#(?P<section>\d+)#")
IMAGE_EXT = (".nii.gz", ".nii", ".png", ".tif", ".tiff")
LABEL_SUFFIX = "_landmark"


def stem(path: str) -> str:
    name = os.path.basename(path)
    for ext in IMAGE_EXT:
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return os.path.splitext(name)[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="folder the image/label paths are relative to")
    ap.add_argument("--images", default="preprocessed", help="image folder (relative to --root)")
    ap.add_argument("--labels", default=None, help="folder with existing <image>_landmark.nii.gz files")
    ap.add_argument("--file-list", help="text file with one path per line (relative to --root) instead of scanning")
    ap.add_argument("--label-suffix", default=LABEL_SUFFIX)
    ap.add_argument("--out", required=True, help="CSV to write")
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser()
    out = Path(args.out).expanduser()
    if args.file_list:
        paths = [ln.strip() for ln in Path(args.file_list).read_text().splitlines() if ln.strip()]
    else:
        base = root / args.images
        paths = [str(p.relative_to(root)) for p in sorted(base.iterdir()) if p.is_file()]
        if args.labels:
            paths += [str(p.relative_to(root)) for p in sorted((root / args.labels).iterdir()) if p.is_file()]

    images_dir = Path(args.images).as_posix().rstrip("/")
    labels_dir = Path(args.labels).as_posix().rstrip("/") if args.labels else None
    images = [p for p in paths if Path(p).parent.as_posix() == images_dir and p.lower().endswith(IMAGE_EXT)]
    label_files = {}
    if labels_dir:
        for p in paths:
            if Path(p).parent.as_posix() == labels_dir and stem(p).endswith(args.label_suffix):
                label_files[stem(p)[: -len(args.label_suffix)]] = p

    def rel(p: str) -> str:
        full = (root / p).resolve() if not Path(p).is_absolute() else Path(p)
        try:
            return os.path.relpath(full, out.resolve().parent)
        except ValueError:          # different drive on Windows
            return str(full)

    rows, skipped = [], []
    for p in images:
        m = NAME.match(os.path.basename(p))
        if not m:
            skipped.append(p)
            continue
        label = label_files.pop(stem(p), None)
        rows.append(dict(base=stem(p), sub=m["sub"], hemisphere=m["hemi"], section=int(m["section"]),
                         ligand=m["ligand"], raw=rel(p), labels=rel(label) if label else ""))
    df = pd.DataFrame(rows).sort_values(["sub", "hemisphere", "section"], kind="stable")
    dupes = df[df.duplicated(["sub", "hemisphere", "section"], keep=False)]
    if not dupes.empty:
        print(f"error: duplicate section numbers:\n{dupes[['base']].to_string(index=False)}", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out}: {len(df)} sections, {int((df['labels'] != '').sum())} with labels")
    for (sub, hemi), g in df.groupby(["sub", "hemisphere"]):
        print(f"  {sub}-{hemi}: {len(g)} sections ({g['section'].min()}-{g['section'].max()}), "
              f"{int((g['labels'] != '').sum())} labelled")
    if skipped:
        print(f"warning: {len(skipped)} file name(s) not in the expected pattern, skipped: {skipped[:5]}")
    if label_files:
        print(f"warning: {len(label_files)} label file(s) without a matching image: {list(label_files.values())[:5]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
