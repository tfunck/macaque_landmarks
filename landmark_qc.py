#!/usr/bin/env python3
"""QC figures for landmark labels.

For every label: the template's coronal slice through the label's (rounded)
centre of mass, next to the section holding that label in each hemisphere.
The top row shows whole sections with a crosshair at the label, so you can
judge at a glance whether the landmark sits in the same anatomical place; the
bottom row zooms in on it. Labels missing from a hemisphere get an empty panel,
and every problem (duplicate, several blobs, no 3-D counterpart) is written in
the panel title, not just coloured.

Outputs (in <out-dir>/qc by default):
  landmarks_qc.pdf          coverage table, then one page per label
  coverage.png              the coverage table
  label-0012.png ...        one image per label

Run from the GUI ("QC figures…") or on a saved session:

    python landmark_qc.py --out-dir landmarks_out            # everything
    python landmark_qc.py --out-dir landmarks_out --labels 3 12 15
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

import landmark_core as core

log = logging.getLogger("landmarks")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
MARK = "#ffd400"          # same yellow as the GUI marker
TOP_MAX_PX = 900          # whole-section panels are decimated to this size
STATUS_TEXT = {"duplicate": "DUPLICATE", "multiblob": "SEVERAL BLOBS", "orphan": "NO 3D COUNTERPART",
               "missing": "MISSING IN 3D"}
SHORT_TEXT = {"duplicate": "duplicate", "multiblob": "several blobs", "orphan": "no 3D counterpart",
              "missing": "missing in 3D"}


def problem_title(cell: core.Cell) -> str:
    """Panel-title wording for a problem cell (status always spelled out)."""
    if cell.status == "duplicate":
        return f"DUPLICATE: in {cell.text.split(' ', 1)[0].lstrip('×')} sections ({cell.text.split(' ', 1)[-1]})"
    if cell.status == "multiblob":
        return f"SEVERAL BLOBS: {cell.text}"
    return STATUS_TEXT[cell.status]


def cell_text(cell: core.Cell) -> str:
    """Coverage-table wording."""
    if cell.status == "duplicate":
        return f"dup {cell.text}"
    if cell.status == "orphan":
        return cell.text.replace(" (no 3D)", "") + " · no 3D"
    return cell.text


# --------------------------------------------------------------------------- #
# Panel data
# --------------------------------------------------------------------------- #
def _display_pos(native_shape, kind, flips, com) -> tuple[float, float]:
    """Display (row, col) of a native centre of mass, rounded to the pixel."""
    u = float(np.clip(np.rint(com[0]), 0, native_shape[0] - 1))
    v = float(np.clip(np.rint(com[1]), 0, native_shape[1] - 1))
    r, c = core.native_to_display(u, v, native_shape, kind, flips)
    return float(r), float(c)


def template_panel(template: core.Template, label: int, cell: core.Cell) -> dict | None:
    st = template.stats.get(label)
    if st is None:
        return None
    k = int(np.rint(st.com[template.ax]))
    disp = core.display_index(template.slice_shape, "nifti", template.display_flips())
    image = np.take(template.image_slice(k), disp)
    labels = np.take(template.label_slice(k), disp)
    com2 = (st.com[template.inplane[0]], st.com[template.inplane[1]])
    return dict(image=image, labels=labels, pos=_display_pos(template.slice_shape, "nifti", template.display_flips(), com2),
                levels=core.levels_from_sample(template.sample, 0.5, 99.5),
                spacing=core.display_spacing(template.slice_spacing, "nifti"),
                title=f"Template · {template.axis_name}={k}", status=cell.status, cell=cell)


def section_panel(stack: core.SpecimenStack, label: int, cell: core.Cell) -> dict | None:
    places = stack.locations().get(label, [])
    if not places:
        return None
    idx, st = places[0]
    arr, meta, sample = stack.image(idx)
    disp = core.display_index(meta.shape, meta.kind, stack.display_flips)
    return dict(image=np.take(arr, disp), labels=np.take(stack.labels[idx], disp),
                pos=_display_pos(meta.shape, meta.kind, stack.display_flips, st.com),
                levels=core.levels_from_sample(sample, 0.5, 99.5),
                spacing=core.display_spacing(meta.spacing, meta.kind),
                title=f"{stack.display_name} · {stack.describe(idx)}", status=cell.status, cell=cell)


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def _empty(ax, title, text):
    ax.set_facecolor("#ecebe8")
    ax.text(0.5, 0.5, text, ha="center", va="center", color=INK_2, fontsize=9, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=9, color=INK_2, loc="left")


def _title(ax, panel):
    status = panel["status"]
    if status in STATUS_TEXT:
        ax.set_title(f"{panel['title']}\n{problem_title(panel['cell'])}", fontsize=8.5,
                     color=CRITICAL, loc="left", fontweight="bold")
    else:
        ax.set_title(panel["title"], fontsize=9, color=INK, loc="left")


def _draw_full(ax, panel, label):
    img = panel["image"]
    step = max(1, int(np.ceil(max(img.shape) / TOP_MAX_PX)))
    shown = img[::step, ::step]
    sp_r, sp_c = panel["spacing"]
    ax.imshow(shown, cmap="gray", vmin=panel["levels"][0], vmax=panel["levels"][1],
              interpolation="nearest", aspect=sp_r / sp_c)
    r, c = panel["pos"]
    if np.isfinite(r):
        r, c = r / step, c / step
        ax.axhline(r, color=MARK, lw=0.7, alpha=0.75)
        ax.axvline(c, color=MARK, lw=0.7, alpha=0.75)
        ax.plot([c], [r], marker="o", ms=11, mfc="none", mec=MARK, mew=1.6)
    ax.set_xticks([])
    ax.set_yticks([])
    _title(ax, panel)


def _draw_zoom(ax, panel, label):
    img, lab = panel["image"], panel["labels"]
    r, c = panel["pos"]
    if not np.isfinite(r):
        _empty(ax, "", "centre outside slice")
        return
    h, w = img.shape
    half = int(max(12, 0.1 * max(h, w)))
    r0, r1 = int(max(0, r - half)), int(min(h, r + half + 1))
    c0, c1 = int(max(0, c - half)), int(min(w, c + half + 1))
    sp_r, sp_c = panel["spacing"]
    ax.imshow(img[r0:r1, c0:c1], cmap="gray", vmin=panel["levels"][0], vmax=panel["levels"][1],
              interpolation="nearest", aspect=sp_r / sp_c, extent=(c0 - 0.5, c1 - 0.5, r1 - 0.5, r0 - 0.5))
    mask = lab[r0:r1, c0:c1] == label
    if mask.any():
        yy, xx = np.mgrid[r0:r1, c0:c1]
        ax.contour(xx, yy, mask.astype(float), levels=[0.5], colors=[MARK], linewidths=1.2)
    ax.plot([c], [r], marker="+", ms=10, color=MARK, mew=1.4)
    ax.set_xlim(c0 - 0.5, c1 - 0.5)
    ax.set_ylim(r1 - 0.5, r0 - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"zoom ±{half} px", fontsize=8, color=INK_2, loc="left")


def label_figure(label: int, stacks, template, cov_row: dict) -> Figure:
    ncol = 1 + len(stacks)
    fig = Figure(figsize=(3.1 * ncol, 6.6), facecolor=SURFACE)
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(2, ncol, height_ratios=[1.35, 1.0], hspace=0.28, wspace=0.08,
                          left=0.02, right=0.98, top=0.86, bottom=0.03)
    panels = [template_panel(template, label, cov_row["template"])]
    panels += [section_panel(st, label, cov_row[st.id]) for st in stacks]
    names = ["Template"] + [st.display_name for st in stacks]
    present = sum(p is not None for p in panels[1:])
    problems = [f"{n}: {SHORT_TEXT[cov_row[k].status]}" for n, k in
                zip(names, ["template"] + [st.id for st in stacks]) if cov_row[k].status in STATUS_TEXT]
    fig.suptitle(f"Label {label}", x=0.02, y=0.975, ha="left", fontsize=14, fontweight="bold", color=INK)
    sub = f"placed in {present}/{len(stacks)} hemispheres"
    if problems:
        sub += " · PROBLEMS: " + "; ".join(problems)
    fig.text(0.02, 0.925, sub, fontsize=9.5, color=CRITICAL if problems else INK_2)
    for col, (panel, name) in enumerate(zip(panels, names)):
        top, bot = fig.add_subplot(gs[0, col]), fig.add_subplot(gs[1, col])
        top.set_anchor("N")
        bot.set_anchor("N")
        if panel is None:
            msg = "not in 3D template" if col == 0 else "not identified"
            _empty(top, name, msg)
            if col == 0:
                top.set_title(f"{name}\nMISSING IN 3D", fontsize=8.5, color=CRITICAL, loc="left",
                              fontweight="bold")
            _empty(bot, "", "")
            continue
        _draw_full(top, panel, label)
        _draw_zoom(bot, panel, label)
    return fig


def coverage_figures(stacks, template, cov: dict, rows_per_page: int = 40) -> list[Figure]:
    labels = list(cov)
    cols = ["template"] + [st.id for st in stacks]
    headers = ["Template"] + [st.display_name for st in stacks]
    pages = [labels[i:i + rows_per_page] for i in range(0, max(len(labels), 1), rows_per_page)] or [[]]
    figs = []
    n_err = sum(cov[lab][c].status in core.ERROR_STATUSES for lab in labels for c in cols)
    for pi, page in enumerate(pages):
        n = len(page)
        fig = Figure(figsize=(1.6 + 1.55 * len(cols), 1.6 + 0.26 * max(n, 1)), facecolor=SURFACE)
        FigureCanvasAgg(fig)
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.96 - 1.1 / fig.get_figheight()])
        ax.set_xlim(0, len(cols) + 0.6)
        ax.set_ylim(n + 1, 0)
        ax.axis("off")
        title = f"Landmark coverage — {len(labels)} labels, {n_err} problem cell(s)"
        if len(pages) > 1:
            title += f" (page {pi + 1}/{len(pages)})"
        fig.text(0.02, 1 - 0.35 / fig.get_figheight(), title, fontsize=12, fontweight="bold", color=INK,
                 va="top")
        fig.text(0.02, 1 - 0.7 / fig.get_figheight(),
                 "green = placed once (section / slice shown) · grey = not identified · red = problem "
                 "(text says which)", fontsize=8.5, color=INK_2, va="top")
        ax.text(0.3, 0.5, "Label", ha="center", va="center", fontsize=9, fontweight="bold", color=INK)
        for j, h in enumerate(headers):
            ax.text(0.6 + j + 0.5, 0.5, h, ha="center", va="center", fontsize=9, fontweight="bold", color=INK)
        for i, lab in enumerate(page, start=1):
            ax.text(0.3, i + 0.5, str(lab), ha="center", va="center", fontsize=9, color=INK)
            for j, c in enumerate(cols):
                cell = cov[lab][c]
                if cell.status == "ok":
                    face, text, color = "#d6efd6", cell.text, INK
                elif cell.status == "absent":
                    face, text, color = "#ecebe8", "–", INK_2
                else:
                    face, text, color = "#f6d3d3", cell_text(cell), "#6b0f0f"
                ax.add_patch(Rectangle((0.6 + j + 0.03, i + 0.06), 0.94, 0.88, facecolor=face, edgecolor="none"))
                ax.text(0.6 + j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=7.5, color=color,
                        clip_on=True)
        figs.append(fig)
    return figs


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def render_qc(qc_dir, stacks, template, labels=None, progress=None, dpi=110) -> Path | None:
    """Write PNGs and a PDF. ``progress(i, n, msg)`` may return False to cancel."""
    qc_dir = Path(qc_dir)
    qc_dir.mkdir(parents=True, exist_ok=True)
    cov = core.coverage(stacks, template)
    todo = [lab for lab in cov if labels is None or lab in set(labels)]
    pdf_path = qc_dir / "landmarks_qc.pdf"
    tmp_pdf = core.atomic_target(pdf_path)
    with PdfPages(str(tmp_pdf)) as pdf:
        for k, fig in enumerate(coverage_figures(stacks, template, {lab: cov[lab] for lab in todo})):
            pdf.savefig(fig)
            fig.savefig(str(qc_dir / ("coverage.png" if k == 0 else f"coverage-{k + 1}.png")), dpi=dpi)
        for i, lab in enumerate(todo):
            if progress is not None and progress(i, len(todo), f"Label {lab} ({i + 1}/{len(todo)})") is False:
                tmp_pdf.unlink(missing_ok=True)
                return None
            fig = label_figure(lab, stacks, template, cov[lab])
            fig.savefig(str(qc_dir / f"label-{lab:04d}.png"), dpi=dpi, facecolor=SURFACE)
            pdf.savefig(fig, facecolor=SURFACE)
    tmp_pdf.replace(pdf_path)
    if progress is not None:
        progress(len(todo), len(todo), "done")
    return pdf_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True, help="landmark GUI output directory (a saved session)")
    ap.add_argument("--csv", help="section CSV (default: the session's landmark_sections_<tag>.csv)")
    ap.add_argument("--template", help="template MRI (default: from <out-dir>/session.json)")
    ap.add_argument("--template-labels", help="3-D labels (default: the session's 3-D label file)")
    ap.add_argument("--specimens", nargs="+", help="specimens to include (default: those of the session)")
    ap.add_argument("--flip-lr", nargs="*", metavar="SPECIMEN",
                    help="specimens shown mirrored left-right (default: as in the session; "
                         "pass --flip-lr with no names for none)")
    ap.add_argument("--labels", nargs="+", type=int, help="only these labels")
    ap.add_argument("--qc-dir", help="output folder (default: <out-dir>/qc)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    out_dir = Path(args.out_dir).expanduser().resolve()
    prev = core.previous_session(out_dir) or {}
    csv = args.csv or prev.get("csv_path")
    template = args.template or prev.get("template")
    tlabels = args.template_labels or prev.get("template_labels_path")
    if not (csv and template):
        ap.error("no saved session in --out-dir; pass --csv, --template and --template-labels")
    cols = prev.get("columns") or {}
    overrides = {k: cols[k] for k in ("sub", "hemisphere", "section", "raw", "labels") if k in cols}
    ds = core.load_dataset(csv, template, tlabels, args.specimens or prev.get("specimens"),
                           prev.get("coronal_axis", "auto"), overrides,
                           flip_lr=args.flip_lr if args.flip_lr is not None else prev.get("flip_lr"))

    def progress(i, n, msg):
        print(f"\r{msg:<40}", end="", flush=True)
        return True

    pdf = render_qc(args.qc_dir or out_dir / "qc", ds.stacks, ds.template, args.labels, progress)
    print(f"\nWrote {pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
