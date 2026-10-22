#!/usr/bin/env python3
"""Build a small synthetic dataset to try the landmark GUI (and to test it).

Creates, under OUT_DIR:
  template.nii.gz            3-D "MRI" (RAS, 0.5 mm)
  template_landmarks.nii.gz  existing 3-D landmark labels (1..N)
  sections/<specimen>/...    coronal autoradiographs: PNG for three hemispheres,
                             2-D NIfTI (BrainBuilder style) for one
  labels_in/...              some pre-existing 2-D labels
  sections.csv               subject, hemisphere, section, ligand, autoradiograph, labels

Two problems are planted on purpose (disable with --clean) so the checks have
something to find: one label is placed in two sections of 11539-R, and 11543-L
has a label 99 that does not exist in the template.

    python make_synthetic_data.py demo_data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

SHAPE = (80, 96, 64)      # x (L->R), y (P->A), z (I->S)
VOX = 0.5
UPSAMPLE = 3              # autoradiograph pixels per template voxel
SPECIMENS = {             # id: (hemisphere, section number = a + b*y, file format, coronal stride offset)
    "11530-L": ("L", 2000, 4, "png", 0),
    "11539-L": ("L", 500, 4, "png", 1),
    "11539-R": ("R", 3000, -4, "png", 0),
    "11543-L": ("L", 10, 2, "nifti", 1),
}
LIGANDS = ("ampa", "flum", "musc")


def make_template(rng):
    x, y, z = np.meshgrid(*[np.linspace(-1, 1, n) for n in SHAPE], indexing="ij")
    r = np.sqrt((x / 0.92) ** 2 + (y / 0.95) ** 2 + (z / 0.85) ** 2)
    brain = r < 1.0
    vol = np.zeros(SHAPE, np.float32)
    vol[brain] = 60
    vol[(r > 0.80) & brain] = 110                                  # "cortex"
    vent = (((np.abs(x) - 0.22) / 0.10) ** 2 + (y / 0.45) ** 2 + ((z - 0.1) / 0.18) ** 2) < 1
    vol[vent] = 15                                                  # "ventricles"
    vol[(np.abs(x) < 0.02) & brain] = 20                            # midline
    deep = (((np.abs(x) - 0.38) / 0.12) ** 2 + ((y + 0.1) / 0.25) ** 2 + ((z + 0.25) / 0.15) ** 2) < 1
    vol[deep] = 140                                                 # "nuclei"
    noise = ndimage.gaussian_filter(rng.normal(size=SHAPE), 1.5).astype(np.float32)
    vol = np.clip(vol + 25 * noise * brain, 0, None)
    return vol, brain


def stamp(vol, c, r, value):
    lo = np.maximum(np.array(c) - r, 0)
    hi = np.minimum(np.array(c) + r + 1, vol.shape)
    sub = tuple(slice(a, b) for a, b in zip(lo, hi))
    grids = np.ogrid[sub]
    d2 = sum((g - ci) ** 2 for g, ci in zip(grids, c))
    vol[sub][d2 <= r * r] = value


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--clean", action="store_true", help="do not plant validation errors")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir).resolve()
    (out / "labels_in").mkdir(parents=True, exist_ok=True)

    affine = np.diag([VOX, VOX, VOX, 1.0])
    affine[:3, 3] = -np.array(SHAPE) * VOX / 2
    vol, brain = make_template(rng)
    nib.save(nib.Nifti1Image(vol, affine), str(out / "template.nii.gz"))

    # Existing 3-D landmarks: 10 in the left hemisphere, 6 in the right.
    labels3d = np.zeros(SHAPE, np.int16)
    inner = ndimage.binary_erosion(brain, iterations=4)
    centers = {}
    lab = 1
    for side, n in (("L", 10), ("R", 6)):
        cand = np.argwhere(inner & ((np.arange(SHAPE[0]) < SHAPE[0] // 2 - 3)[:, None, None] if side == "L"
                                    else (np.arange(SHAPE[0]) > SHAPE[0] // 2 + 3)[:, None, None]))
        for c in cand[rng.choice(len(cand), n, replace=False)]:
            stamp(labels3d, c, 1, lab)
            centers[lab] = (side, c)
            lab += 1
    nib.save(nib.Nifti1Image(labels3d, affine), str(out / "template_landmarks.nii.gz"))

    rows = []
    half = SHAPE[0] // 2
    for sid, (hemi, a, b, fmt, offset) in SPECIMENS.items():
        sub = sid.split("-")[0]
        sdir = out / "sections" / sid
        sdir.mkdir(parents=True, exist_ok=True)
        ys = list(range(6 + offset, SHAPE[1] - 6, 2))
        xs = slice(0, half) if hemi == "L" else slice(half, SHAPE[0])
        for n, y in enumerate(ys):
            lig = LIGANDS[n % 3]
            number = a + b * y
            sl = vol[xs, y, :]                                    # (x, z)
            up = ndimage.zoom(sl, UPSAMPLE, order=1)
            gamma = {"ampa": 1.0, "flum": 0.6, "musc": 1.6}[lig]
            img = (up / max(vol.max(), 1)) ** gamma
            img = np.clip(img + rng.normal(0, 0.03, img.shape), 0, 1)
            root = f"mf#{sid.lower()}#{sub}#{hemi}#{lig}#{number}"
            if fmt == "png":
                raster = np.flipud(img.T)                          # rows: superior->inferior
                path = sdir / f"{root}.png"
                Image.fromarray((raster * 255).astype(np.uint8)).save(path)
            else:
                path = sdir / f"{root}.nii.gz"
                aff = np.diag([VOX / UPSAMPLE, VOX / UPSAMPLE, 1, 1])
                nib.save(nib.Nifti1Image(img.astype(np.float32), aff), str(path))
            rows.append(dict(subject=sub, hemisphere=hemi, section=number, ligand=lig,
                             autoradiograph=str(path), labels="", _y=y, _shape=img.shape, _fmt=fmt))

        # Pre-existing 2-D labels for ~70% of the template landmarks on this side.
        spec_rows = [r for r in rows if f"{r['subject']}-{r['hemisphere']}" == sid]
        by_y = {r["_y"]: r for r in spec_rows}
        per_section: dict[int, np.ndarray] = {}
        for lab_id, (side, c) in centers.items():
            if side != hemi or rng.random() > 0.7:
                continue
            y = min(by_y, key=lambda v: abs(v - c[1]))
            r_ = by_y[y]
            arr = per_section.setdefault(y, np.zeros(r_["_shape"], np.int32))
            xi = (c[0] - (0 if hemi == "L" else half)) * UPSAMPLE + 1
            zi = c[2] * UPSAMPLE + 1
            stamp(arr, (xi, zi), 3, lab_id)
        if not args.clean and sid == "11539-R" and per_section:   # one label placed twice
            placed = sorted({int(v) for a_ in per_section.values() for v in np.unique(a_) if v})
            ys_sorted = sorted(by_y)
            dup = per_section.setdefault(ys_sorted[3], np.zeros(by_y[ys_sorted[3]]["_shape"], np.int32))
            stamp(dup, (20, 40), 3, placed[0])
        if not args.clean and sid == "11543-L":       # label with no 3-D counterpart
            ys_sorted = sorted(by_y)
            arr = per_section.setdefault(ys_sorted[10], np.zeros(by_y[ys_sorted[10]]["_shape"], np.int32))
            stamp(arr, (30, 60), 3, 99)
        for y, arr in per_section.items():
            r_ = by_y[y]
            root = Path(r_["autoradiograph"]).name.replace(".nii.gz", "").replace(".png", "")
            if r_["_fmt"] == "png":
                p = out / "labels_in" / f"{root}_labels.png"
                Image.fromarray(np.flipud(arr.T).astype(np.uint16)).save(p)
            else:
                p = out / "labels_in" / f"{root}_labels.nii.gz"
                nib.save(nib.Nifti1Image(arr, np.diag([VOX / UPSAMPLE, VOX / UPSAMPLE, 1, 1])), str(p))
            r_["labels"] = str(p)

    df = pd.DataFrame(rows).drop(columns=["_y", "_shape", "_fmt"])
    df = df.sample(frac=1, random_state=1)   # the GUI must not rely on CSV order
    df.to_csv(out / "sections.csv", index=False)
    print(f"Wrote {len(df)} sections for {len(SPECIMENS)} hemispheres and {len(centers)} template "
          f"landmarks to {out}")
    print(f"Try:\n  python landmark_gui.py --csv {out/'sections.csv'} --template {out/'template.nii.gz'} "
          f"--template-labels {out/'template_landmarks.nii.gz'} --out-dir {out/'landmarks_out'}")


if __name__ == "__main__":
    main()
