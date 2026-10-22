"""Qt-free core of the landmark labelling tool.

Everything that touches data lives here so it can be tested without a display
and reused by the QC script:

* reading the section CSV (flexible column names, BrainBuilder-style or not),
* reading/writing 2-D autoradiographs and label images (NIfTI, PNG, TIFF),
* the 3-D template and its label volume,
* per-label statistics (voxel count, centre of mass, connected components),
* the uniqueness / 2-D <-> 3-D correspondence checks,
* saving every output.

Images are shown as stored. The GUI builds a *display index map* (an array
whose value at each displayed pixel is the flat index of the native pixel it
shows) and paints through it, so saved labels always have exactly the layout of
the raw image file referenced in the CSV (and of the template voxel grid).
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - nibabel is a hard requirement for the template
    nib = None

try:
    import tifffile
except ImportError:  # pragma: no cover - optional, Pillow reads most TIFFs too
    tifffile = None

log = logging.getLogger("landmarks")

LABEL_DTYPE = np.int32
TEMPLATE_KEY = "__template__"
NIFTI_SUFFIXES = (".nii", ".nii.gz")
TIFF_SUFFIXES = (".tif", ".tiff")

# Output file names inside --out-dir.
OUT_TABLE = "landmark_table.csv"
OUT_COVERAGE = "landmark_coverage.csv"
OUT_REPORT = "validation_report.txt"
SESSION_JSON = "session.json"
STATE_JSON = "gui_state.json"


# --------------------------------------------------------------------------- #
# Display orientation
# --------------------------------------------------------------------------- #
Orientation = tuple  # (flip_lr: bool, flip_ud: bool), applied in display (raster) space


def base_to_raster(arr: np.ndarray, kind: str) -> np.ndarray:
    """Convert a native 2-D array to 'as displayed' raster order (row 0 on top).

    Raster images (PNG/TIFF) already are. NIfTI slices are (x, y): x is shown
    left->right and y bottom->top, as in BrainBuilder's QC figures, i.e.
    ``flipud(arr.T)``.
    """
    if kind == "nifti":
        return np.flipud(arr.T)
    return arr


def display_index(native_shape: tuple[int, int], kind: str,
                  flips: Orientation = (False, False)) -> np.ndarray:
    """Array of native flat indices, laid out as the pixels are displayed.

    Value at display pixel (r, c) is the flat index of the native pixel shown
    there, so painting/hover map back to the file's own layout exactly.
    """
    n = int(np.prod(native_shape))
    dtype = np.int32 if n < 2**31 - 1 else np.int64
    idx = base_to_raster(np.arange(n, dtype=dtype).reshape(native_shape), kind)
    if flips[0]:
        idx = np.fliplr(idx)
    if flips[1]:
        idx = np.flipud(idx)
    return np.ascontiguousarray(idx)


def native_to_display(i, j, native_shape: tuple[int, int], kind: str,
                      flips: Orientation = (False, False)):
    """Display (row, col) of native coordinates (i, j); works for float
    coordinates such as centres of mass. Inverse of ``display_index``."""
    i, j = np.asarray(i, float), np.asarray(j, float)
    if kind == "nifti":
        h, w = native_shape[1], native_shape[0]
        r, c = (h - 1) - j, i
    else:
        h, w = native_shape
        r, c = i, j
    if flips[0]:
        c = (w - 1) - c
    if flips[1]:
        r = (h - 1) - r
    return r, c


def display_spacing(native_spacing: tuple[float, float], kind: str) -> tuple[float, float]:
    """(row, col) pixel spacing of the displayed raster."""
    s0, s1 = float(native_spacing[0]), float(native_spacing[1])
    return (s1, s0) if kind == "nifti" else (s0, s1)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def is_nifti(path: object) -> bool:
    return str(path).lower().endswith(NIFTI_SUFFIXES)


def strip_ext(path: object) -> str:
    """Basename without extension. NIfTI handling matches BrainBuilder's
    ``_strip_ext`` so ``{root}*.nii.gz`` globbing finds our output files."""
    name = os.path.basename(str(path))
    if is_nifti(name):
        return name.replace(".nii.gz", "").replace(".nii", "")
    return os.path.splitext(name)[0]


def is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and value.strip().lower() in {"", "nan", "none"}


def resolve_path(value: object, base_dir: Path) -> Path | None:
    if is_missing(value):
        return None
    p = Path(str(value).strip()).expanduser()
    if p.is_absolute():
        return p.resolve()
    candidate = base_dir / p            # relative paths are relative to the CSV's folder
    if candidate.exists() or not p.exists():
        return candidate.resolve()
    return p.resolve()                  # fallback: relative to the working directory


def atomic_target(path: Path) -> Path:
    """Temporary sibling path that keeps the extension (nibabel picks gzip by it)."""
    name = path.name
    for suffix in (".nii.gz", ".nii", ".png", ".tif", ".tiff", ".csv", ".json", ".txt"):
        if name.lower().endswith(suffix):
            return path.with_name(f".{name[: -len(suffix)]}.tmp{os.getpid()}{suffix}")
    return path.with_name(f".{name}.tmp{os.getpid()}")


def _replace(tmp: Path, final: Path) -> None:
    os.replace(tmp, final)


def write_text_atomic(path: Path, text: str) -> None:
    tmp = atomic_target(path)
    tmp.write_text(text)
    _replace(tmp, path)


def percentile_sample(arr: np.ndarray, n: int = 40000) -> np.ndarray:
    flat = np.asarray(arr, dtype=np.float32).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return np.zeros(1, np.float32)
    step = max(1, flat.size // n)
    return np.sort(flat[::step])


def levels_from_sample(sample: np.ndarray, lo_pct: float, hi_pct: float) -> tuple[float, float]:
    if sample.size == 0:
        return 0.0, 1.0
    lo = float(sample[int(np.clip(round(lo_pct / 100 * (sample.size - 1)), 0, sample.size - 1))])
    hi = float(sample[int(np.clip(round(hi_pct / 100 * (sample.size - 1)), 0, sample.size - 1))])
    if hi <= lo:
        hi = lo + max(1e-6, abs(lo) * 1e-3 + 1e-6)
    return lo, hi


def clean_sub(value: object) -> str:
    return re.sub(r"^sub-", "", str(value).strip())


def specimen_id(sub: object, hemisphere: object) -> str:
    return f"{clean_sub(sub)}-{str(hemisphere).strip()}"


# --------------------------------------------------------------------------- #
# Image IO
# --------------------------------------------------------------------------- #
@dataclass
class ImageMeta:
    path: str
    kind: str                      # "nifti" or "raster"
    shape: tuple[int, int]         # squeezed native 2-D shape
    disk_shape: tuple[int, ...]    # on-disk shape (singleton axes kept) for writing back
    spacing: tuple[float, float]
    affine: np.ndarray | None = None
    header: object | None = None


def _require_nibabel() -> None:
    if nib is None:
        raise ImportError("nibabel is required (pip install nibabel)")


def _nifti_spacing(header, disk_shape) -> tuple[float, float]:
    zooms = list(header.get_zooms()) + [1.0] * 3
    axes = [k for k, n in enumerate(disk_shape) if n > 1][:2]
    sp = [float(zooms[k]) if zooms[k] and np.isfinite(zooms[k]) else 1.0 for k in axes]
    while len(sp) < 2:
        sp.append(1.0)
    return abs(sp[0]) or 1.0, abs(sp[1]) or 1.0


def read_meta(path: str) -> ImageMeta:
    """Shape/geometry of a 2-D image without reading its pixels."""
    if is_nifti(path):
        _require_nibabel()
        img = nib.load(str(path))
        disk_shape = tuple(int(n) for n in img.shape)
        shape = tuple(n for n in disk_shape if n > 1)
        if len(shape) == 1:
            shape = (shape[0], 1)
        if len(shape) == 3 and shape[-1] in (3, 4):  # RGB stored as 3-D
            shape = shape[:2]
        if len(shape) != 2:
            raise ValueError(f"Expected a 2-D image, got shape {disk_shape}: {path}")
        return ImageMeta(str(path), "nifti", shape, disk_shape, _nifti_spacing(img.header, disk_shape),
                         np.asarray(img.affine), img.header.copy())
    if str(path).lower().endswith(TIFF_SUFFIXES) and tifffile is not None:
        with tifffile.TiffFile(str(path)) as tf:
            shp = tf.pages[0].shape
        shape = (int(shp[0]), int(shp[1]))
    else:
        with Image.open(str(path)) as im:
            shape = (int(im.height), int(im.width))
    return ImageMeta(str(path), "raster", shape, shape, (1.0, 1.0))


def _to_gray(arr: np.ndarray, path: str) -> np.ndarray:
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = arr[..., :3].astype(np.float32).mean(axis=-1)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {arr.shape}: {path}")
    return arr


def _read_raster(path: str) -> np.ndarray:
    if str(path).lower().endswith(TIFF_SUFFIXES) and tifffile is not None:
        return np.asarray(tifffile.imread(str(path)))
    with Image.open(str(path)) as im:
        if im.mode == "P":
            im = im.convert("L")
        return np.asarray(im)


def read_image(path: str) -> tuple[np.ndarray, ImageMeta]:
    meta = read_meta(path)
    if meta.kind == "nifti":
        data = np.asanyarray(nib.load(str(path)).dataobj)
    else:
        data = _read_raster(path)
    arr = np.ascontiguousarray(_to_gray(data, path), dtype=np.float32)
    if arr.shape != meta.shape:
        arr = arr.reshape(meta.shape)
    return arr, meta


def read_labels(path: str, expected_shape: tuple[int, int]) -> np.ndarray:
    """Read a 2-D label image as int32, checking it matches the raw image."""
    if is_nifti(path):
        _require_nibabel()
        data = np.asanyarray(nib.load(str(path)).dataobj)
    else:
        data = _read_raster(path)
    data = np.squeeze(data)
    if data.ndim != 2:
        raise ValueError(f"Label image must be single-channel 2-D, got shape {data.shape}: {path}")
    if np.issubdtype(data.dtype, np.floating):
        rounded = np.rint(np.nan_to_num(data))
        if not np.allclose(rounded, np.nan_to_num(data), atol=1e-3):
            log.warning("Non-integer label values rounded in %s", path)
        data = rounded
    data = data.astype(np.int64)
    if (data < 0).any():
        log.warning("Negative label values set to 0 in %s", path)
        data[data < 0] = 0
    if tuple(data.shape) != tuple(expected_shape):
        hint = " (looks transposed)" if tuple(data.shape) == tuple(expected_shape)[::-1] else ""
        raise ValueError(f"Label image {path} has shape {data.shape} but its autoradiograph has "
                         f"shape {tuple(expected_shape)}{hint}")
    return np.ascontiguousarray(data.astype(LABEL_DTYPE))


LABEL_FORMATS = ("match", "nifti", "png", "tif")
_FORMAT_EXT = {"nifti": ".nii.gz", "png": ".png", "tif": ".tif"}


def resolve_label_format(fmt: str, meta: ImageMeta) -> str:
    if fmt != "match":
        return fmt
    if meta.kind == "nifti":
        return "nifti"
    return "tif" if meta.path.lower().endswith(TIFF_SUFFIXES) else "png"


def write_labels(path: Path, labels: np.ndarray, meta: ImageMeta, fmt: str) -> None:
    """Write 2-D labels aligned with (same layout as) the raw image file."""
    path = Path(path)
    tmp = atomic_target(path)
    if fmt == "nifti":
        _require_nibabel()
        if meta.kind == "nifti":
            data = labels.reshape(meta.disk_shape).astype(LABEL_DTYPE)
            hdr = meta.header.copy()
            hdr.set_data_dtype(LABEL_DTYPE)
            img = nib.Nifti1Image(data, meta.affine, hdr)
        else:
            img = nib.Nifti1Image(labels.astype(LABEL_DTYPE), np.eye(4))
            img.header.set_data_dtype(LABEL_DTYPE)
        img.header.set_slope_inter(1.0, 0.0)
        img.header["cal_min"], img.header["cal_max"] = 0, float(labels.max())
        nib.save(img, str(tmp))
    elif fmt == "png":
        if labels.max() > 65535:
            raise ValueError("PNG labels are 16-bit; use --label-format nifti for values > 65535")
        Image.fromarray(labels.astype(np.uint16)).save(str(tmp))
    elif fmt == "tif":
        out = labels.astype(np.uint16 if labels.max() <= 65535 else np.uint32)
        if tifffile is not None:
            tifffile.imwrite(str(tmp), out)
        else:
            Image.fromarray(out).save(str(tmp))
    else:
        raise ValueError(f"Unknown label format {fmt!r}")
    _replace(tmp, path)


# --------------------------------------------------------------------------- #
# Label statistics
# --------------------------------------------------------------------------- #
@dataclass
class LabelStat:
    count: int
    com: tuple[float, ...]        # centre of mass, array index coordinates
    n_components: int
    bbox: tuple[slice, ...]


def _full_structure(ndim: int) -> np.ndarray:
    return ndimage.generate_binary_structure(ndim, ndim)


def _stat_from_mask(mask: np.ndarray, offset: Sequence[int]) -> LabelStat | None:
    coords = np.nonzero(mask)
    if coords[0].size == 0:
        return None
    _, n_comp = ndimage.label(mask, structure=_full_structure(mask.ndim))
    com = tuple(float(c.mean()) + o for c, o in zip(coords, offset))
    bbox = tuple(slice(int(c.min()) + o, int(c.max()) + 1 + o) for c, o in zip(coords, offset))
    return LabelStat(int(coords[0].size), com, int(n_comp), bbox)


def compute_stats(arr: np.ndarray) -> dict[int, LabelStat]:
    """Per-label count, centre of mass, connected-component count (full
    connectivity: 8 in 2-D, 26 in 3-D) and bounding box."""
    if arr.size == 0 or not arr.any():
        return {}
    out: dict[int, LabelStat] = {}
    max_label = int(arr.max())
    if max_label <= 2_000_000:
        objects = ndimage.find_objects(arr)
        items = ((lab, sl) for lab, sl in enumerate(objects, start=1) if sl is not None)
    else:  # pathological label values: fall back to one pass per label
        items = []
        for lab in np.unique(arr[arr > 0]):
            coords = np.nonzero(arr == lab)
            items.append((int(lab), tuple(slice(int(c.min()), int(c.max()) + 1) for c in coords)))
    for lab, sl in items:
        stat = _stat_from_mask(arr[sl] == lab, [s.start for s in sl])
        if stat is not None:
            out[int(lab)] = stat
    return out


def union_bbox(a: Sequence[slice], b: Sequence[slice]) -> tuple[slice, ...]:
    return tuple(slice(min(x.start, y.start), max(x.stop, y.stop)) for x, y in zip(a, b))


def bbox_from_flat(flat: np.ndarray, shape: Sequence[int]) -> tuple[slice, ...]:
    coords = np.unravel_index(np.asarray(flat), tuple(shape))
    return tuple(slice(int(c.min()), int(c.max()) + 1) for c in coords)


def label_near(arr: np.ndarray, r: int, c: int, tol: int) -> int:
    """Label at (r, c), or the nearest non-zero label within ``tol`` pixels."""
    h, w = arr.shape
    if not (0 <= r < h and 0 <= c < w):
        return 0
    v = int(arr[r, c])
    if v or tol <= 0:
        return v
    r0, r1, c0, c1 = max(0, r - tol), min(h, r + tol + 1), max(0, c - tol), min(w, c + tol + 1)
    win = arr[r0:r1, c0:c1]
    nz = np.nonzero(win)
    if nz[0].size == 0:
        return 0
    d2 = (nz[0] + r0 - r) ** 2 + (nz[1] + c0 - c) ** 2
    k = int(np.argmin(d2))
    if d2[k] > tol * tol:
        return 0
    return int(win[nz[0][k], nz[1][k]])


# --------------------------------------------------------------------------- #
# Section CSV
# --------------------------------------------------------------------------- #
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "sub": ("sub", "subject", "subject_id", "sub_id", "specimen", "brain"),
    "hemisphere": ("hemisphere", "hemi", "side"),
    "section": ("section", "sample", "section_number", "section_num", "section_no", "section_id", "slice"),
    "raw": ("raw", "autoradiograph", "autoradiograph_path", "image", "image_path", "img", "raw_path",
            "filepath", "file_path", "file", "path"),
    "labels": ("labels", "label", "landmark", "landmarks", "section_labels", "label_path", "labels_path",
               "landmark_path", "landmarks_path"),
    "chunk": ("chunk", "slab"),
    "acquisition": ("acquisition", "ligand", "acq", "receptor", "tracer"),
}
REQUIRED_ROLES = ("sub", "hemisphere", "section", "raw")


def _norm_col(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", str(name).strip().lower())


def resolve_columns(columns: Iterable[str], overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Map roles (sub, hemisphere, section, raw, labels, ...) to CSV column names."""
    columns = list(columns)
    by_norm = {_norm_col(c): c for c in columns}
    mapping: dict[str, str] = {}
    for role, col in (overrides or {}).items():
        if col:
            if col not in columns:
                raise ValueError(f"Column {col!r} (given for {role}) is not in the CSV. Columns: {columns}")
            mapping[role] = col
    for role, aliases in COLUMN_ALIASES.items():
        if role in mapping:
            continue
        for alias in aliases:
            if alias in by_norm and by_norm[alias] not in mapping.values():
                mapping[role] = by_norm[alias]
                break
    missing = [r for r in REQUIRED_ROLES if r not in mapping and not (r == "hemisphere" and "sub" in mapping)]
    if missing:
        raise ValueError(f"Could not find CSV column(s) for {missing}. Columns present: {columns}. "
                         "Rename them or pass --col-<role>.")
    return mapping


@dataclass
class Section:
    row: int
    specimen: str
    sub: str
    hemisphere: str
    number: int
    raw: str
    labels_in: str | None = None
    chunk: str | None = None
    acquisition: str | None = None

    @property
    def root(self) -> str:
        return strip_ext(self.raw)


def _parse_section_number(value: object, row: int) -> int:
    text = str(value).strip()
    try:
        f = float(text)
    except ValueError:
        m = re.search(r"-?\d+", text)
        if not m:
            raise ValueError(f"Row {row}: section number {value!r} is not a number")
        return int(m.group())
    if not float(f).is_integer():
        raise ValueError(f"Row {row}: section number {value!r} is not an integer")
    return int(f)


def read_sections(csv_path: str | Path, overrides: dict[str, str] | None = None,
                  specimens: Sequence[str] | None = None
                  ) -> tuple[pd.DataFrame, dict[str, str], list[Section], list[str]]:
    """Read the section CSV. Returns (dataframe, column map, sections, warnings)."""
    csv_path = Path(csv_path).expanduser().resolve()
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if df.empty:
        raise ValueError(f"No rows in {csv_path}")
    cols = resolve_columns(df.columns, overrides)
    base_dir = csv_path.parent
    warnings: list[str] = []
    wanted = None
    if specimens:
        wanted = {s.strip().lower(): s for s in specimens}
    sections: list[Section] = []
    missing_raw: list[str] = []
    for row, rec in df.iterrows():
        sub = str(rec[cols["sub"]]).strip()
        if "hemisphere" in cols:
            hemi = str(rec[cols["hemisphere"]]).strip()
        else:  # "11530-L" / "11530_L" in the subject column
            m = re.match(r"^(.*?)[-_]([LRlr])$", sub)
            if not m:
                raise ValueError(f"No hemisphere column and cannot split hemisphere from {sub!r}")
            sub, hemi = m.group(1), m.group(2).upper()
        sid = specimen_id(sub, hemi)
        if wanted is not None and sid.lower() not in wanted:
            continue
        raw = resolve_path(rec[cols["raw"]], base_dir)
        if raw is None or not raw.exists():
            missing_raw.append(f"row {row}: {rec[cols['raw']]}")
            continue
        labels_in = None
        if "labels" in cols:
            lp = resolve_path(rec[cols["labels"]], base_dir)
            if lp is not None:
                if lp.exists():
                    labels_in = str(lp)
                else:
                    warnings.append(f"{sid} row {row}: label file not found, starting empty: {lp}")
        sections.append(Section(
            row=int(row), specimen=sid, sub=clean_sub(sub), hemisphere=hemi,
            number=_parse_section_number(rec[cols["section"]], row),
            raw=str(raw), labels_in=labels_in,
            chunk=(str(rec[cols["chunk"]]).strip() or None) if "chunk" in cols else None,
            acquisition=(str(rec[cols["acquisition"]]).strip() or None) if "acquisition" in cols else None,
        ))
    if missing_raw:
        head = "\n  ".join(missing_raw[:10])
        more = f"\n  ... and {len(missing_raw) - 10} more" if len(missing_raw) > 10 else ""
        raise FileNotFoundError(f"{len(missing_raw)} autoradiograph file(s) not found:\n  {head}{more}")
    if wanted is not None:
        found = {s.specimen.lower() for s in sections}
        absent = [orig for key, orig in wanted.items() if key not in found]
        if absent:
            raise ValueError(f"Specimen(s) not in CSV: {absent}. Available specimen IDs look like "
                             f"'{sections[0].specimen if sections else '11530-L'}'.")
    if not sections:
        raise ValueError("No sections selected from the CSV")
    return df, cols, sections, warnings


def group_specimens(sections: Sequence[Section], order: Sequence[str] | None = None) -> list[list[Section]]:
    groups: OrderedDict[str, list[Section]] = OrderedDict()
    for s in sections:
        groups.setdefault(s.specimen, []).append(s)
    if order:
        lower = {k.lower(): k for k in groups}
        keys = [lower[o.strip().lower()] for o in order if o.strip().lower() in lower]
        keys += [k for k in groups if k not in keys]
        return [groups[k] for k in keys]
    return list(groups.values())


def _chunk_key(chunk: str | None):
    if chunk is None:
        return (0, 0, "")
    try:
        return (1, float(chunk), "")
    except ValueError:
        return (2, 0, chunk)


# --------------------------------------------------------------------------- #
# One hemisphere's section stack
# --------------------------------------------------------------------------- #
class SpecimenStack:
    """Sections of one specimen/hemisphere, ordered by (chunk, section, acquisition)."""

    def __init__(self, sections: Sequence[Section], cache_size: int = 10):
        self.sections: list[Section] = sorted(
            sections, key=lambda s: (_chunk_key(s.chunk), s.number, s.acquisition or "", s.raw))
        first = self.sections[0]
        self.id, self.sub, self.hemisphere = first.specimen, first.sub, first.hemisphere
        self.labels: dict[int, np.ndarray] = {}
        self.stats: dict[int, dict[int, LabelStat]] = {}
        self._meta: dict[int, ImageMeta] = {}
        self._loc_cache: dict[int, list[tuple[int, LabelStat]]] | None = None
        self._cache_size = cache_size
        self._futures: OrderedDict[int, Future] = OrderedDict()
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"load-{self.id}")
        self.multi_chunk = len({s.chunk for s in self.sections}) > 1
        self.flip_lr = False        # display only: show sections mirrored left-right

    @property
    def display_name(self) -> str:
        return f"{self.id}-flipped-lr" if self.flip_lr else self.id

    @property
    def display_flips(self) -> Orientation:
        return (self.flip_lr, False)

    def __len__(self) -> int:
        return len(self.sections)

    # -- geometry / pixels ------------------------------------------------- #
    def meta(self, idx: int) -> ImageMeta:
        m = self._meta.get(idx)
        if m is None:
            m = read_meta(self.sections[idx].raw)
            self._meta[idx] = m
        return m

    def _load(self, idx: int):
        arr, meta = read_image(self.sections[idx].raw)
        self._meta[idx] = meta
        return arr, meta, percentile_sample(arr)

    def _future(self, idx: int) -> Future:
        with self._lock:
            fut = self._futures.get(idx)
            if fut is None or (fut.done() and fut.exception() is not None):
                fut = self._pool.submit(self._load, idx)
                self._futures[idx] = fut
            self._futures.move_to_end(idx)
            while len(self._futures) > self._cache_size:
                _, old = self._futures.popitem(last=False)
                old.cancel()
            return fut

    def image(self, idx: int) -> tuple[np.ndarray, ImageMeta, np.ndarray]:
        """(native float32 image, meta, sorted intensity sample)."""
        return self._future(idx).result()

    def prefetch(self, idx: int) -> None:
        if 0 <= idx < len(self.sections):
            self._future(idx)

    # -- labels ------------------------------------------------------------ #
    def load_labels(self, progress=None) -> None:
        for idx, s in enumerate(self.sections):
            if progress is not None:
                progress(idx, len(self.sections))
            if not s.labels_in:
                continue
            arr = read_labels(s.labels_in, self.meta(idx).shape)
            if arr.any():
                self.labels[idx] = arr
                self.refresh(idx)

    def get_labels(self, idx: int, create: bool = False) -> np.ndarray | None:
        arr = self.labels.get(idx)
        if arr is None and create:
            arr = np.zeros(self.meta(idx).shape, dtype=LABEL_DTYPE)
            self.labels[idx] = arr
        return arr

    def refresh(self, idx: int) -> None:
        """Recompute label statistics for one section (drops empty arrays)."""
        arr = self.labels.get(idx)
        self._loc_cache = None
        if arr is None or not arr.any():
            self.labels.pop(idx, None)
            self.stats.pop(idx, None)
            return
        self.stats[idx] = compute_stats(arr)

    def locations(self) -> dict[int, list[tuple[int, LabelStat]]]:
        """label -> [(section index, stat), ...] (more than one entry = duplicate)."""
        if self._loc_cache is None:
            loc: dict[int, list[tuple[int, LabelStat]]] = {}
            for idx in sorted(self.stats):
                for lab, st in self.stats[idx].items():
                    loc.setdefault(lab, []).append((idx, st))
            self._loc_cache = loc
        return self._loc_cache

    def all_labels(self) -> set[int]:
        return set(self.locations())

    def remove_label(self, label: int) -> list[tuple[int, np.ndarray]]:
        """Erase ``label`` from every section. Returns [(idx, flat indices)]."""
        removed = []
        for idx, st in list(self.locations().get(label, [])):
            arr = self.labels[idx]
            mask = arr[st.bbox] == label
            coords = [c + s.start for c, s in zip(np.nonzero(mask), st.bbox)]
            flat = np.ravel_multi_index(coords, arr.shape)
            arr.ravel()[flat] = 0
            removed.append((idx, flat))
            self.refresh(idx)
        return removed

    # -- navigation helpers ------------------------------------------------ #
    def describe(self, idx: int, short: bool = False) -> str:
        s = self.sections[idx]
        chunk = f"chunk {s.chunk} · " if (self.multi_chunk and s.chunk) else ""
        if short:
            return f"{'c' + s.chunk + ':' if self.multi_chunk and s.chunk else ''}s{s.number}"
        acq = f" · {s.acquisition}" if s.acquisition else ""
        return f"{chunk}section {s.number}{acq}"

    def find(self, text: str) -> tuple[int | None, str]:
        """Index for a typed section number ("123" or "chunk:123")."""
        text = text.strip()
        chunk = None
        if ":" in text:
            chunk, text = (t.strip() for t in text.split(":", 1))
        try:
            num = int(float(text))
        except ValueError:
            return None, f"'{text}' is not a section number"
        cands = [i for i, s in enumerate(self.sections) if chunk is None or str(s.chunk) == chunk]
        if not cands:
            return None, f"no chunk {chunk}"
        exact = [i for i in cands if self.sections[i].number == num]
        if exact:
            return exact[0], ""
        nearest = min(cands, key=lambda i: (abs(self.sections[i].number - num), i))
        return nearest, f"section {num} not in stack; nearest is {self.sections[nearest].number}"


# --------------------------------------------------------------------------- #
# 3-D template + label volume
# --------------------------------------------------------------------------- #
class Template:
    def __init__(self, image_path: str, label_path: str | None = None, coronal_axis: str | int = "auto"):
        _require_nibabel()
        img = nib.load(str(image_path))
        data = np.asanyarray(img.dataobj)
        while data.ndim > 3 and data.shape[-1] == 1:
            data = data[..., 0]
        if data.ndim == 4:
            log.warning("Template is 4-D; using the first volume")
            data = data[..., 0]
        if data.ndim != 3:
            raise ValueError(f"Template must be 3-D, got shape {data.shape}")
        self.image_path = str(image_path)
        self.image = np.ascontiguousarray(np.nan_to_num(data.astype(np.float32)))
        self.shape = self.image.shape
        self.affine = np.asarray(img.affine)
        self.header = img.header.copy()
        self.axcodes = tuple(nib.aff2axcodes(self.affine))
        self.zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
        if coronal_axis in (None, "auto", ""):
            ap = [i for i, c in enumerate(self.axcodes) if c in ("A", "P")]
            self.ax = ap[0] if ap else 1
        else:
            self.ax = int(coronal_axis)
        self.inplane = tuple(a for a in range(3) if a != self.ax)
        self.slice_shape = tuple(self.shape[a] for a in self.inplane)
        self.slice_spacing = tuple(abs(self.zooms[a]) or 1.0 for a in self.inplane)

        self.label_path = str(label_path) if label_path else None
        self.label_affine = self.affine
        self.label_header = None
        self.warnings: list[str] = []
        if label_path and Path(label_path).exists():
            limg = nib.load(str(label_path))
            ldata = np.asanyarray(limg.dataobj)
            while ldata.ndim > 3 and ldata.shape[-1] == 1:
                ldata = ldata[..., 0]
            if ldata.shape != self.shape:
                raise ValueError(f"Template labels {label_path} have shape {ldata.shape}, "
                                 f"template has {self.shape}")
            if np.issubdtype(ldata.dtype, np.floating):
                rounded = np.rint(np.nan_to_num(ldata))
                if not np.allclose(rounded, np.nan_to_num(ldata), atol=1e-3):
                    self.warnings.append("Non-integer values in the 3-D label volume were rounded")
                ldata = rounded
            ldata = ldata.astype(np.int64)
            if (ldata < 0).any():
                self.warnings.append("Negative values in the 3-D label volume were set to 0")
                ldata[ldata < 0] = 0
            if not np.allclose(limg.affine, self.affine, atol=1e-3):
                self.warnings.append("Template and label volume affines differ; labels are shown on the "
                                     "template voxel grid and saved with the label volume's affine")
            self.label_affine = np.asarray(limg.affine)
            self.label_header = limg.header.copy()
            self.labels = np.ascontiguousarray(ldata.astype(LABEL_DTYPE))
        else:
            if label_path:
                self.warnings.append(f"3-D label volume not found, starting empty: {label_path}")
            self.labels = np.zeros(self.shape, dtype=LABEL_DTYPE)
        self.stats: dict[int, LabelStat] = compute_stats(self.labels)
        self.sample = percentile_sample(self.image[self.image != 0] if (self.image != 0).any() else self.image)

    # -- geometry ----------------------------------------------------------- #
    @property
    def n_slices(self) -> int:
        return self.shape[self.ax]

    @property
    def axis_name(self) -> str:
        return "xyz"[self.ax]

    def slicer(self, k: int) -> tuple:
        idx: list = [slice(None)] * 3
        idx[self.ax] = int(k)
        return tuple(idx)

    def image_slice(self, k: int) -> np.ndarray:
        return self.image[self.slicer(k)]

    def label_slice(self, k: int) -> np.ndarray:
        return self.labels[self.slicer(k)]

    def slice_flat_to_volume(self, k: int, flat: np.ndarray) -> np.ndarray:
        u, v = np.unravel_index(np.asarray(flat), self.slice_shape)
        coords: list = [None, None, None]
        coords[self.ax] = np.full(u.shape, int(k))
        coords[self.inplane[0]] = u
        coords[self.inplane[1]] = v
        return np.ravel_multi_index(coords, self.shape)

    def display_flips(self) -> Orientation:
        """Flips that show the coronal slice neurological (subject left on
        screen left) and superior up, given ``base_to_raster`` shows the first
        in-plane axis left->right and the second bottom->top."""
        h_code = self.axcodes[self.inplane[0]]
        v_code = self.axcodes[self.inplane[1]]
        return h_code == "L", v_code in ("I", "P")

    def world(self, ijk: Sequence[float]) -> np.ndarray:
        return (self.affine @ np.r_[np.asarray(ijk, float), 1.0])[:3]

    def com_slice(self, label: int) -> int | None:
        st = self.stats.get(label)
        return None if st is None else int(np.rint(st.com[self.ax]))

    # -- label bookkeeping --------------------------------------------------- #
    def refresh_labels(self, labels: Iterable[int], region: Sequence[slice]) -> None:
        """Update stats for ``labels`` after voxels inside ``region`` changed."""
        for lab in {int(x) for x in labels}:
            if lab == 0:
                continue
            old = self.stats.get(lab)
            reg = tuple(region) if old is None else union_bbox(region, old.bbox)
            stat = _stat_from_mask(self.labels[reg] == lab, [s.start for s in reg])
            if stat is None:
                self.stats.pop(lab, None)
            else:
                self.stats[lab] = stat

    def remove_label(self, label: int) -> np.ndarray:
        st = self.stats.get(label)
        if st is None:
            return np.zeros(0, dtype=np.int64)
        mask = self.labels[st.bbox] == label
        coords = [c + s.start for c, s in zip(np.nonzero(mask), st.bbox)]
        flat = np.ravel_multi_index(coords, self.shape)
        self.labels.ravel()[flat] = 0
        self.stats.pop(label, None)
        return flat


# --------------------------------------------------------------------------- #
# Coverage + validation
# --------------------------------------------------------------------------- #
@dataclass
class Issue:
    severity: str                  # "error" | "warning" | "info"
    message: str
    label: int | None = None
    specimen: str | None = None
    section_idx: int | None = None
    template_slice: int | None = None


@dataclass
class Cell:
    status: str                    # ok | absent | duplicate | multiblob | orphan | missing
    text: str = ""
    idx: int | None = None


def coverage(stacks: Sequence[SpecimenStack], template: Template) -> dict[int, dict[str, Cell]]:
    """label -> {"template": Cell, specimen_id: Cell} for every known label."""
    locs = {st.id: st.locations() for st in stacks}
    labels = set(template.stats)
    for loc in locs.values():
        labels |= set(loc)
    table: dict[int, dict[str, Cell]] = {}
    for lab in sorted(labels):
        row: dict[str, Cell] = {}
        tst = template.stats.get(lab)
        if tst is None:
            row["template"] = Cell("missing", "missing")
        else:
            k = int(np.rint(tst.com[template.ax]))
            if tst.n_components > 1:
                row["template"] = Cell("multiblob", f"{tst.n_components} blobs", k)
            else:
                row["template"] = Cell("ok", f"{template.axis_name}={k}", k)
        for st in stacks:
            places = locs[st.id].get(lab, [])
            if not places:
                row[st.id] = Cell("absent", "")
            elif len(places) > 1:
                row[st.id] = Cell("duplicate", "×%d " % len(places) + ",".join(
                    st.describe(i, short=True) for i, _ in places), places[0][0])
            else:
                idx, s = places[0]
                txt = st.describe(idx, short=True)
                if s.n_components > 1:
                    row[st.id] = Cell("multiblob", f"{txt} ({s.n_components} blobs)", idx)
                elif tst is None:
                    row[st.id] = Cell("orphan", f"{txt} (no 3D)", idx)
                else:
                    row[st.id] = Cell("ok", txt, idx)
        table[lab] = row
    return table


ERROR_STATUSES = {"duplicate", "multiblob", "orphan", "missing"}


def validate(stacks: Sequence[SpecimenStack], template: Template) -> list[Issue]:
    issues: list[Issue] = []
    ax = template.ax
    for lab, st in sorted(template.stats.items()):
        if st.n_components > 1:
            issues.append(Issue("error", f"Template: label {lab} has {st.n_components} separate blobs "
                                         f"(each label must be one blob)", label=lab,
                                template_slice=int(np.rint(st.com[ax]))))
    for stack in stacks:
        locs = stack.locations()
        for lab, places in sorted(locs.items()):
            if len(places) > 1:
                secs = ", ".join(stack.describe(i, short=True) for i, _ in places)
                issues.append(Issue("error", f"{stack.id}: label {lab} appears in {len(places)} sections "
                                             f"({secs}); it must be unique per hemisphere",
                                    label=lab, specimen=stack.id, section_idx=places[0][0]))
            for idx, st in places:
                if st.n_components > 1:
                    issues.append(Issue("error", f"{stack.id} {stack.describe(idx, short=True)}: label {lab} "
                                                 f"has {st.n_components} separate blobs in one section",
                                        label=lab, specimen=stack.id, section_idx=idx))
            if lab not in template.stats:
                issues.append(Issue("error", f"{stack.id}: label {lab} "
                                             f"({stack.describe(places[0][0], short=True)}) has no "
                                             f"counterpart in the 3D template", label=lab,
                                    specimen=stack.id, section_idx=places[0][0]))
        n_missing = len(set(template.stats) - set(locs))
        issues.append(Issue("info", f"{stack.id}: {len(locs)} labels placed; {n_missing} template labels "
                                    f"not identified on this hemisphere", specimen=stack.id))
    roots = [s.root for st in stacks for s in st.sections]
    dupes = sorted({r for r in roots if roots.count(r) > 1}) if len(roots) != len(set(roots)) else []
    for r in dupes[:20]:
        issues.append(Issue("error", f"Two sections share the file name root '{r}'; their label files "
                                     f"would overwrite each other"))
    uniq = sorted(set(roots))
    prefix = [(a, b) for a, b in zip(uniq, uniq[1:]) if b.startswith(a)]
    if prefix:
        ex = "; ".join(f"'{a}' / '{b}'" for a, b in prefix[:3])
        issues.append(Issue("warning", f"{len(prefix)} raw file name(s) are prefixes of others ({ex}). "
                                       f"BrainBuilder finds landmark files with '{{root}}*.nii.gz', which would "
                                       f"match more than one file for these sections."))
    return issues


def blocking_errors(issues: Sequence[Issue], stack: SpecimenStack) -> list[Issue]:
    """Errors that make a hemisphere's 2-D <-> 3-D mapping unusable."""
    labs = stack.all_labels()
    return [i for i in issues if i.severity == "error"
            and (i.specimen == stack.id or (i.specimen is None and i.label in labs))]


def estimate_section(stack: SpecimenStack, template: Template, label: int) -> tuple[int, float, int] | None:
    """Predict where ``label`` should sit in a hemisphere that lacks it, from a
    linear fit of section number vs. template coronal position over the
    landmarks the two share. Returns (stack index, predicted number, n pairs)."""
    tst = template.stats.get(label)
    if tst is None or stack.multi_chunk:
        return None
    xs, ys = [], []
    for lab, places in stack.locations().items():
        if len(places) == 1 and lab in template.stats and lab != label:
            xs.append(template.stats[lab].com[template.ax])
            ys.append(stack.sections[places[0][0]].number)
    if len(xs) < 2 or np.ptp(xs) < 1e-6:
        return None
    a, b = np.polyfit(xs, ys, 1)
    pred = float(a * tst.com[template.ax] + b)
    numbers = np.array([s.number for s in stack.sections], dtype=float)
    return int(np.argmin(np.abs(numbers - pred))), pred, len(xs)


def next_free_label(stacks: Sequence[SpecimenStack], template: Template) -> int:
    used = set(template.stats)
    for st in stacks:
        used |= st.all_labels()
    return (max(used) + 1) if used else 1


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def landmark_table(stacks: Sequence[SpecimenStack], template: Template) -> pd.DataFrame:
    rows = []
    for lab, st in sorted(template.stats.items()):
        w = template.world(st.com)
        rows.append(dict(label=lab, space="template", sub="", hemisphere="", section="", chunk="",
                         acquisition="", i=st.com[0], j=st.com[1], k=st.com[2],
                         coronal_slice=int(np.rint(st.com[template.ax])),
                         x_mm=w[0], y_mm=w[1], z_mm=w[2], n_voxels=st.count, n_blobs=st.n_components,
                         file=""))
    for stack in stacks:
        for lab, places in sorted(stack.locations().items()):
            for idx, st in places:
                s = stack.sections[idx]
                meta = stack.meta(idx)
                x = y = z = np.nan
                if meta.kind == "nifti" and meta.affine is not None:
                    # Map the squeezed (i, j) back to on-disk voxel coordinates.
                    ijk = np.zeros(3)
                    axes = [a for a, n in enumerate(meta.disk_shape) if n > 1][:2]
                    for a, v in zip(axes, st.com):
                        if a < 3:
                            ijk[a] = v
                    x, y, z = (meta.affine @ np.r_[ijk, 1.0])[:3]
                rows.append(dict(label=lab, space=stack.id, sub=s.sub, hemisphere=s.hemisphere,
                                 section=s.number, chunk=s.chunk or "", acquisition=s.acquisition or "",
                                 i=st.com[0], j=st.com[1], k=np.nan, coronal_slice=np.nan,
                                 x_mm=x, y_mm=y, z_mm=z, n_voxels=st.count, n_blobs=st.n_components,
                                 file=s.raw))
    return pd.DataFrame(rows)


def coverage_frame(stacks: Sequence[SpecimenStack], template: Template) -> pd.DataFrame:
    cov = coverage(stacks, template)
    cols = ["template"] + [s.id for s in stacks]
    data = {c: [] for c in cols}
    for lab, row in cov.items():
        for c in cols:
            cell = row[c]
            data[c].append(cell.text if cell.status == "ok" else
                           ("" if cell.status == "absent" else f"{cell.status.upper()}: {cell.text}"))
    return pd.DataFrame(data, index=pd.Index(list(cov), name="label"))


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #
LABEL_SUFFIX = "_landmark"
_VERSION = re.compile(r"v\d+(?:\.\d+)*")


@dataclass
class OutputNames:
    """File names inside the output folder. The layout is flat, like an
    existing landmark folder (e.g. ``v1.3/``):

        <raw root>_landmark.nii.gz              2-D labels, one per labelled section
        <template stem>_<tag>.nii.gz            all 3-D labels (e.g. MEBRAINS_L_landmarks_v1.4.nii.gz)
        <sub>_<hemi>_landmarks_<tag>.nii.gz     3-D labels of one hemisphere (1:1 with its 2-D labels)
        landmark_sections_<tag>.csv             the input CSV, pointing at the new label files

    ``tag`` defaults to the folder name when it looks like a version (v1.4).
    """
    out_dir: Path
    tag: str = ""
    label_suffix: str = LABEL_SUFFIX
    template_stem: str = "template_landmarks"

    @property
    def _t(self) -> str:
        return f"_{self.tag}" if self.tag else ""

    def section(self, section: Section, fmt: str) -> Path:
        return self.out_dir / f"{section.root}{self.label_suffix}{_FORMAT_EXT[fmt]}"

    def template(self) -> Path:
        return self.out_dir / f"{self.template_stem}{self._t}.nii.gz"

    def hemi(self, stack: SpecimenStack) -> Path:
        return self.out_dir / f"{stack.sub}_{stack.hemisphere}_landmarks{self._t}.nii.gz"

    def csv(self) -> Path:
        return self.out_dir / f"landmark_sections{self._t}.csv"

    def aux(self, name: str) -> Path:
        return self.out_dir / name


def output_names(out_dir: str | Path, template_labels_input: str | None = None, tag: str | None = None,
                 label_suffix: str = LABEL_SUFFIX) -> OutputNames:
    out_dir = Path(out_dir).expanduser().resolve()
    if tag is None:
        tag = out_dir.name if _VERSION.fullmatch(out_dir.name) else ""
    stem = "template_landmarks"
    if template_labels_input:
        stem = re.sub(r"_v\d+(?:\.\d+)*$", "", strip_ext(template_labels_input)) or stem
    return OutputNames(out_dir, tag.lstrip("_"), label_suffix, stem)


def folder_contents(out_dir: str | Path) -> list[str]:
    """Visible entries of ``out_dir`` (empty list if it doesn't exist)."""
    out_dir = Path(out_dir).expanduser()
    if not out_dir.is_dir():
        return []
    return sorted(p.name for p in out_dir.iterdir() if not p.name.startswith("."))


def _template_label_image(template: Template, data: np.ndarray):
    hdr = (template.label_header or template.header).copy()
    hdr.set_data_dtype(LABEL_DTYPE)
    img = nib.Nifti1Image(data.astype(LABEL_DTYPE), template.label_affine, hdr)
    img.header.set_slope_inter(1.0, 0.0)
    img.header["cal_min"], img.header["cal_max"] = 0, float(data.max()) if data.size else 0.0
    return img


def _save_nifti_atomic(img, path: Path) -> None:
    tmp = atomic_target(path)
    nib.save(img, str(tmp))
    _replace(tmp, path)


def _rebase(value: object, from_dir: Path | None, to_dir: Path) -> object:
    """Re-express a (possibly relative) path cell relative to ``to_dir``."""
    if from_dir is None or is_missing(value):
        return value
    p = resolve_path(value, from_dir)
    try:
        return os.path.relpath(p, to_dir)
    except ValueError:  # different drive (Windows)
        return str(p)


@dataclass
class SaveReport:
    out_dir: str
    csv: str = ""
    n_sections_written: int = 0
    removed: list[str] = field(default_factory=list)
    kept_stale: list[str] = field(default_factory=list)
    hemi_written: list[str] = field(default_factory=list)
    hemi_skipped: dict[str, str] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def n_errors(self) -> int:
        return sum(i.severity == "error" for i in self.issues)

    def summary(self) -> str:
        parts = [f"{self.n_sections_written} labelled section file(s)", "3D labels"]
        if self.hemi_written:
            parts.append(f"hemisphere 3D labels for {', '.join(self.hemi_written)}")
        msg = "Saved " + ", ".join(parts) + f" and {Path(self.csv).name} to {self.out_dir}."
        if self.hemi_skipped:
            msg += (" Hemisphere 3D labels not written (validation errors): "
                    + ", ".join(k for k, v in self.hemi_skipped.items() if v != "no labels placed") + ".")
        if self.n_errors:
            msg += f" {self.n_errors} validation error(s)."
        if self.kept_stale:
            msg += f" {len(self.kept_stale)} older file(s) left untouched (see validation_report.txt)."
        return msg


def format_report(issues: Sequence[Issue], stacks: Sequence[SpecimenStack], template: Template,
                  kept_stale: Sequence[str] = ()) -> str:
    lines = [f"Landmark validation report — {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"Template: {template.image_path}",
             f"Template labels: {len(template.stats)}", ""]
    for sev in ("error", "warning", "info"):
        group = [i for i in issues if i.severity == sev]
        lines.append(f"{sev.upper()}S ({len(group)})")
        lines += [f"  - {i.message}" for i in group] or ["  none"]
        lines.append("")
    if kept_stale:
        lines.append("FILES IN THIS FOLDER FROM BEFORE THIS TOOL WROTE TO IT (left untouched; may be out of date)")
        lines += [f"  - {n}" for n in kept_stale]
        lines.append("")
    return "\n".join(lines)


def save_session(names: OutputNames, df: pd.DataFrame, columns: dict[str, str],
                 stacks: Sequence[SpecimenStack], template: Template, label_format: str = "match",
                 input_csv_dir: str | Path | None = None, session_info: dict | None = None,
                 gui_state: dict | None = None) -> SaveReport:
    """Write every output into ``names.out_dir``. Input files are never touched
    (unless the output folder *is* their folder and the caller allowed that).

    Files this tool wrote in an earlier save (listed in session.json) are
    removed when they become empty or invalid; any other file is never deleted.
    """
    t0 = time.time()
    out_dir = names.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report = SaveReport(str(out_dir), csv=str(names.csv()))
    issues = validate(stacks, template)
    report.issues = issues
    prev_info = {}
    if names.aux(SESSION_JSON).exists():
        try:
            prev_info = json.loads(names.aux(SESSION_JSON).read_text())
        except json.JSONDecodeError:
            prev_info = {}
    ours_before = set(prev_info.get("written", []))
    written: list[str] = []

    def retire(path: Path) -> None:
        """A file we would no longer write: delete it only if we wrote it."""
        if not path.exists():
            return
        if str(path) in ours_before:
            path.unlink()
            report.removed.append(path.name)
        else:
            report.kept_stale.append(path.name)

    # Plan section outputs first so a name clash aborts before anything is written.
    planned: dict[Path, tuple[SpecimenStack, int, str]] = {}
    for stack in stacks:
        for idx, s in enumerate(stack.sections):
            fmt = resolve_label_format(label_format, stack.meta(idx))
            path = names.section(s, fmt)
            if path in planned:
                other = planned[path]
                raise ValueError(f"Sections {other[0].id} {other[0].describe(other[1], True)} and "
                                 f"{stack.id} {stack.describe(idx, True)} would both write {path.name}")
            planned[path] = (stack, idx, fmt)

    in_dir = Path(input_csv_dir).resolve() if input_csv_dir else None
    out_df = df.copy()
    label_col = columns.get("labels") or "labels"
    if label_col not in out_df.columns:
        out_df[label_col] = ""
    for col in {columns.get("raw"), columns.get("labels")} - {None}:
        if col in out_df.columns:   # keep relative paths valid from the new CSV's folder
            out_df[col] = [_rebase(v, in_dir, out_dir) for v in out_df[col]]
    for path, (stack, idx, fmt) in planned.items():
        s = stack.sections[idx]
        arr = stack.labels.get(idx)
        if arr is not None and arr.any():
            write_labels(path, arr, stack.meta(idx), fmt)
            out_df.at[s.row, label_col] = path.name
            written.append(str(path))
            report.n_sections_written += 1
        else:
            out_df.at[s.row, label_col] = ""
            retire(path)

    tmp = atomic_target(names.csv())
    out_df.to_csv(tmp, index=False)
    _replace(tmp, names.csv())
    written.append(str(names.csv()))

    _save_nifti_atomic(_template_label_image(template, template.labels), names.template())
    written.append(str(names.template()))

    for stack in stacks:
        path = names.hemi(stack)
        errs = blocking_errors(issues, stack)
        labs = sorted(stack.all_labels())
        if errs or not labs:
            report.hemi_skipped[stack.id] = errs[0].message if errs else "no labels placed"
            retire(path)   # a stale 1:1 file must not survive an invalid state
            continue
        data = np.where(np.isin(template.labels, labs), template.labels, 0)
        _save_nifti_atomic(_template_label_image(template, data), path)
        written.append(str(path))
        report.hemi_written.append(stack.id)

    tbl = atomic_target(names.aux(OUT_TABLE))
    landmark_table(stacks, template).to_csv(tbl, index=False)
    _replace(tbl, names.aux(OUT_TABLE))
    cov = atomic_target(names.aux(OUT_COVERAGE))
    coverage_frame(stacks, template).to_csv(cov)
    _replace(cov, names.aux(OUT_COVERAGE))
    write_text_atomic(names.aux(OUT_REPORT), format_report(issues, stacks, template, report.kept_stale))

    info = dict(session_info or {})
    info.update(saved=time.strftime("%Y-%m-%d %H:%M:%S"), columns=columns, label_format=label_format,
                coronal_axis=template.ax, template=template.image_path,
                specimens=[s.id for s in stacks], version_tag=names.tag, label_suffix=names.label_suffix,
                outputs=dict(csv=str(names.csv()), template_labels=str(names.template())),
                written=sorted(set(written)))
    write_text_atomic(names.aux(SESSION_JSON), json.dumps(info, indent=2))
    if gui_state is not None:
        write_text_atomic(names.aux(STATE_JSON), json.dumps(gui_state, indent=2))
    report.seconds = time.time() - t0
    return report


def previous_session(out_dir: str | Path) -> dict | None:
    """Info about a session saved in ``out_dir`` by this tool (None if none)."""
    out_dir = Path(out_dir).expanduser().resolve()
    sj = out_dir / SESSION_JSON
    if not sj.exists():
        return None
    try:
        info = json.loads(sj.read_text())
    except json.JSONDecodeError:
        return None
    outs = info.get("outputs", {})
    csv, tl = outs.get("csv"), outs.get("template_labels")
    if not (csv and tl and Path(csv).exists() and Path(tl).exists()):
        return None
    info["csv_path"], info["template_labels_path"] = csv, tl
    return info


def load_state(out_dir: str | Path) -> dict:
    p = Path(out_dir).expanduser() / STATE_JSON
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            log.warning("Ignoring unreadable %s", p)
    return {}


# --------------------------------------------------------------------------- #
# Convenience loader shared by the GUI and the QC script
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    df: pd.DataFrame
    columns: dict[str, str]
    stacks: list[SpecimenStack]
    template: Template
    warnings: list[str]
    csv_path: str = ""


def load_dataset(csv_path: str, template_path: str, template_labels: str | None,
                 specimens: Sequence[str] | None = None, coronal_axis: str | int = "auto",
                 column_overrides: dict[str, str] | None = None, progress=None,
                 flip_lr: Sequence[str] | None = None) -> Dataset:
    """``flip_lr``: specimen IDs whose sections are *displayed* mirrored
    left-right (GUI and QC). Labels are always saved in the files' own layout."""
    df, cols, sections, warnings = read_sections(csv_path, column_overrides, specimens)
    groups = group_specimens(sections, specimens)
    stacks = [SpecimenStack(g) for g in groups]
    wanted = {f.strip().lower() for f in (flip_lr or [])}
    unknown = wanted - {st.id.lower() for st in stacks}
    if unknown:
        raise ValueError(f"--flip-lr: specimen(s) not loaded: {sorted(unknown)}; "
                         f"loaded: {[st.id for st in stacks]}")
    for st in stacks:
        st.flip_lr = st.id.lower() in wanted
    for si, stack in enumerate(stacks):
        def _p(i, n, si=si, stack=stack):
            if progress is not None:
                progress(f"Loading labels for {stack.id}", si * 1000 + int(1000 * i / max(n, 1)), len(stacks) * 1000)
        stack.load_labels(_p)
    if progress is not None:
        progress("Loading template", len(stacks) * 1000, len(stacks) * 1000)
    template = Template(template_path, template_labels, coronal_axis)
    warnings += template.warnings
    for w in warnings:
        log.warning(w)
    return Dataset(df, cols, stacks, template, warnings, str(Path(csv_path).expanduser().resolve()))
