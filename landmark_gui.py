#!/usr/bin/env python3
r"""Landmark labelling GUI: autoradiograph stacks <-> 3-D template.

One pane per hemisphere stack (coronal autoradiographs) plus one pane with the
coronal slices of the 3-D reference MRI and its existing landmark labels.
Label values are shared between all panes: the same number means the same
anatomical point in every hemisphere and in the template.

    python landmark_gui.py --csv landmark_sections_v1.3.csv --template MEBRAINS_T1.nii.gz \
        --template-labels v1.3/MEBRAINS_L_landmarks_v1.3.nii.gz --out-dir v1.4 \
        --specimens 11530-L 11539-L 11539-R 11543-L --flip-lr 11539-R

Nothing is written unless an annotation changes. Edited labels and a new CSV go
to --out-dir (created on the first save); a folder that already holds files is
only written to after you confirm, or with --overwrite.

Mouse (in a pane)
  wheel / right-drag     zoom          left-drag (Navigate) or middle-drag   pan
  left click / drag      paint the current label (Paint) or erase (Erase)
  Shift+click            place the current label here, removing it elsewhere in
                         this hemisphere (or in the template): "move landmark"
  Ctrl+click             pick the label under the cursor as the current label
  hover                  shows the label number under the cursor

Keys
  P / E / N (Esc)        Paint / Erase / Navigate mode
  Left/Up, Right/Down    previous / next section in the pane under the mouse
  [ and ]                smaller / larger brush (for the pane under the mouse)
  F                      fit the pane under the mouse
  Ctrl+G                 jump to the "Go to label" box
  Ctrl+Z                 undo          Ctrl+S   save
"""
from __future__ import annotations

import argparse
import colorsys
import itertools
import logging
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

import landmark_core as core

pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)
Qt = QtCore.Qt
QAction = getattr(QtGui, "QAction", None) or QtWidgets.QAction
QShortcut = getattr(QtGui, "QShortcut", None) or QtWidgets.QShortcut
log = logging.getLogger("landmarks")

N_COLORS = 4096
NOTE_COLORS = {"error": "#d9534f", "warn": "#e0a030", "muted": "#8a8a8a", "ok": "#5cb85c", "": "#cccccc"}


def _palette(n: int) -> np.ndarray:
    lut = np.zeros((n, 4), np.uint8)
    for k in range(1, n):
        r, g, b = colorsys.hsv_to_rgb((k * 0.618033988749895) % 1.0, 0.85, 1.0)
        lut[k] = (int(r * 255), int(g * 255), int(b * 255), 255)
    return lut


PALETTE = _palette(N_COLORS)


def color_index(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    return np.where(labels > 0, 1 + (labels - 1) % (N_COLORS - 1), 0)


def label_qcolor(lab: int) -> QtGui.QColor:
    r, g, b, _ = PALETTE[int(color_index(np.array([lab]))[0])]
    return QtGui.QColor(int(r), int(g), int(b))


# --------------------------------------------------------------------------- #
# Data sources: one interface for a hemisphere stack and for the template
# --------------------------------------------------------------------------- #
class StackSource:
    is_template = False

    def __init__(self, stack: core.SpecimenStack):
        self.stack = stack
        self.key = stack.id
        self.title = stack.display_name
        self.goto_hint = "section #"

    def __len__(self):
        return len(self.stack)

    def load(self, idx):
        arr, meta, sample = self.stack.image(idx)
        return dict(native=arr, shape=meta.shape, kind=meta.kind, spacing=meta.spacing,
                    sample=sample, flips=self.stack.display_flips)

    def prefetch(self, idx):
        self.stack.prefetch(idx)

    def labels(self, idx):
        """Native 2-D labels for reading (None if the section has none)."""
        return self.stack.get_labels(idx)

    def target(self, idx, create=True):
        return self.stack.get_labels(idx, create=create)

    def to_target_flat(self, idx, flat):
        return np.asarray(flat, dtype=np.int64)

    def target_key(self, idx):
        return (self.stack.id, idx)

    def describe(self, idx):
        return self.stack.describe(idx)

    def slice_stats(self, idx):
        return self.stack.stats.get(idx, {})

    def find(self, text):
        return self.stack.find(text)


class TemplateSource:
    is_template = True

    def __init__(self, template: core.Template):
        self.t = template
        self.key = core.TEMPLATE_KEY
        self.title = f"Template · coronal ({''.join(template.axcodes)}, axis {template.ax})"
        self.goto_hint = "slice #"

    def __len__(self):
        return self.t.n_slices

    def load(self, idx):
        return dict(native=self.t.image_slice(idx), shape=self.t.slice_shape, kind="nifti",
                    spacing=self.t.slice_spacing, sample=self.t.sample, flips=self.t.display_flips())

    def prefetch(self, idx):
        pass

    def labels(self, idx):
        return np.ascontiguousarray(self.t.label_slice(idx))

    def target(self, idx, create=True):
        return self.t.labels

    def to_target_flat(self, idx, flat):
        return self.t.slice_flat_to_volume(idx, flat)

    def target_key(self, idx):
        return (core.TEMPLATE_KEY,)

    def describe(self, idx):
        return f"{self.t.axis_name} = {idx}"

    def slice_stats(self, idx):
        return core.compute_stats(np.ascontiguousarray(self.t.label_slice(idx)))

    def find(self, text):
        try:
            k = int(float(text.strip()))
        except ValueError:
            return None, f"'{text}' is not a slice number"
        if 0 <= k < self.t.n_slices:
            return k, ""
        k2 = int(np.clip(k, 0, self.t.n_slices - 1))
        return k2, f"slice {k} out of range; showing {k2}"


# --------------------------------------------------------------------------- #
# Graphics items
# --------------------------------------------------------------------------- #
class PaneView(pg.GraphicsView):
    def __init__(self, pane):
        super().__init__()
        self.pane = pane

    def leaveEvent(self, ev):
        self.pane.on_leave()
        super().leaveEvent(ev)


class PaintItem(pg.ImageItem):
    """Base image that receives clicks, and drags while painting/erasing."""

    def __init__(self, pane):
        super().__init__(autoDownsample=True)
        self.pane = pane

    def hoverEvent(self, ev):
        if ev.isExit():
            return
        ev.acceptClicks(Qt.MouseButton.LeftButton)
        if self.pane.main.mode in ("paint", "erase"):
            ev.acceptDrags(Qt.MouseButton.LeftButton)

    def mouseClickEvent(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        ev.accept()
        self.pane.on_click(ev.pos(), ev.modifiers())

    def mouseDragEvent(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton or self.pane.main.mode not in ("paint", "erase"):
            ev.ignore()
            return
        ev.accept()
        if ev.isStart():
            self.pane.stroke_begin(ev.buttonDownPos())
        self.pane.stroke_to(ev.pos())
        if ev.isFinish():
            self.pane.stroke_end()


# --------------------------------------------------------------------------- #
# One pane
# --------------------------------------------------------------------------- #
class SlicePane(QtWidgets.QFrame):
    def __init__(self, main: "MainWindow", source):
        super().__init__()
        self.main = main
        self.source = source
        self.idx = -1
        self.view: dict | None = None
        self.lab_disp: np.ndarray | None = None
        self.rgba: np.ndarray | None = None
        self.disp_pos: dict = {}           # label -> display (row, col) of its centre of mass
        self.marker_target: tuple[int, int] | None = None
        self.lo_pct, self.hi_pct = 0.5, 99.5
        self._disp_cache: dict = {}
        self._last = None
        self._texts: list = []

        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setMinimumSize(260, 260)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(3)

        head = QtWidgets.QHBoxLayout()
        self.title_lbl = QtWidgets.QLabel(f"<b>{source.title}</b>")
        self.info_lbl = QtWidgets.QLabel()
        self.info_lbl.setStyleSheet("color: #9a9a9a")
        fit = QtWidgets.QToolButton()
        fit.setText("Fit")
        fit.setToolTip("Fit the image to the pane (F)")
        fit.clicked.connect(lambda: self.vb.autoRange(padding=0.02))
        head.addWidget(self.title_lbl)
        head.addSpacing(8)
        head.addWidget(self.info_lbl, 1)
        head.addWidget(fit)
        lay.addLayout(head)

        self.gv = PaneView(self)
        self.gv.setBackground("k")
        self.vb = pg.ViewBox(lockAspect=True, invertY=True, enableMenu=True)
        self.gv.setCentralItem(self.vb)
        self.img = PaintItem(self)
        self.overlay = pg.ImageItem()
        self.overlay.setZValue(10)
        self.vb.addItem(self.img)
        self.vb.addItem(self.overlay)
        pen = pg.mkPen("#ffd400", width=2)
        self.marker = pg.ScatterPlotItem(size=28, pen=pen, brush=pg.mkBrush(None), symbol="o", pxMode=True)
        self.marker.setZValue(20)
        self.vb.addItem(self.marker, ignoreBounds=True)
        self.brush_item = QtWidgets.QGraphicsEllipseItem()
        bpen = pg.mkPen("#ffffff", width=1)
        bpen.setCosmetic(True)
        self.brush_item.setPen(bpen)
        self.brush_item.setZValue(15)
        self.brush_item.hide()
        self.vb.addItem(self.brush_item, ignoreBounds=True)
        self.hover = pg.TextItem("", anchor=(-0.2, 1.2), fill=pg.mkBrush(0, 0, 0, 200))
        self.hover.setZValue(30)
        self.hover.hide()
        self.vb.addItem(self.hover, ignoreBounds=True)
        self.gv.scene().sigMouseMoved.connect(self.on_mouse_moved)
        lay.addWidget(self.gv, 1)

        nav = QtWidgets.QHBoxLayout()
        self.minus = QtWidgets.QToolButton()
        self.minus.setText("−")
        self.minus.setToolTip("Previous section (Left/Up)")
        self.plus = QtWidgets.QToolButton()
        self.plus.setText("+")
        self.plus.setToolTip("Next section (Right/Down)")
        self.slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.slider.setRange(0, max(0, len(source) - 1))
        self.goto = QtWidgets.QLineEdit()
        self.goto.setPlaceholderText(source.goto_hint)
        self.goto.setFixedWidth(84)
        self.goto.setToolTip("Type a section number and press Enter"
                             + (" (chunk:section also works)" if not source.is_template else ""))
        nav.addWidget(self.minus)
        nav.addWidget(self.slider, 1)
        nav.addWidget(self.plus)
        nav.addWidget(self.goto)
        lay.addLayout(nav)

        low = QtWidgets.QHBoxLayout()
        low.addWidget(QtWidgets.QLabel("Contrast"))
        self.lo_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.lo_slider.setRange(0, 500)
        self.hi_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.hi_slider.setRange(500, 1000)
        for s in (self.lo_slider, self.hi_slider):
            s.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.lo_slider.setToolTip("Black point (percentile of this section's intensities)")
        self.hi_slider.setToolTip("White point (percentile of this section's intensities)")
        auto = QtWidgets.QToolButton()
        auto.setText("Auto")
        auto.setToolTip("Reset contrast to the 0.5–99.5 percentile range")
        auto.clicked.connect(self.reset_contrast)
        low.addWidget(self.lo_slider, 1)
        low.addWidget(self.hi_slider, 1)
        low.addWidget(auto)
        lay.addLayout(low)

        # Messages (go-to results, warnings) float over the top-left of the image.
        self.note_lbl = QtWidgets.QLabel(self.gv)
        self.note_lbl.move(6, 6)
        self.note_lbl.hide()

        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(90)
        self._debounce.timeout.connect(lambda: self.set_index(self.slider.value()))
        self.slider.valueChanged.connect(self._on_slider)
        self.minus.clicked.connect(lambda: self.step(-1))
        self.plus.clicked.connect(lambda: self.step(1))
        self.goto.returnPressed.connect(self._on_goto)
        self.lo_slider.valueChanged.connect(self._on_contrast)
        self.hi_slider.valueChanged.connect(self._on_contrast)
        self.set_contrast(self.lo_pct, self.hi_pct, render=False)

    # -- navigation ---------------------------------------------------------- #
    def _on_slider(self, value):
        self.info_lbl.setText(f"{self.source.describe(value)} · {value + 1}/{len(self.source)}")
        self._debounce.start()

    def step(self, delta):
        self.set_index(self.idx + delta)

    def _on_goto(self):
        text = self.goto.text()
        if not text.strip():
            return
        idx, msg = self.source.find(text)
        if idx is None:
            self.note(msg, "error")
            return
        self.set_index(idx, force=True)
        if msg:
            self.note(msg, "warn")
        self.goto.clear()

    def set_index(self, idx, force=False):
        n = len(self.source)
        if n == 0:
            return
        idx = int(np.clip(idx, 0, n - 1))
        if idx == self.idx and not force:
            return
        self.idx = idx
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        self.note("")
        try:
            self.render()
        except Exception as exc:  # unreadable file: report, keep the GUI alive
            log.exception("Could not display %s %s", self.source.key, idx)
            self.note(f"cannot display: {exc}", "error")
        self.main.on_pane_moved(self)

    # -- contrast ------------------------------------------------------------ #
    def set_contrast(self, lo, hi, render=True):
        self.lo_pct, self.hi_pct = float(lo), float(hi)
        for s, v in ((self.lo_slider, lo), (self.hi_slider, hi)):
            s.blockSignals(True)
            s.setValue(int(round(v * 10)))
            s.blockSignals(False)
        if render and self.view is not None:
            self.img.setLevels(core.levels_from_sample(self.view["sample"], self.lo_pct, self.hi_pct))

    def _on_contrast(self, _v=None):
        self.set_contrast(self.lo_slider.value() / 10, self.hi_slider.value() / 10)

    def reset_contrast(self):
        self.set_contrast(0.5, 99.5)

    # -- rendering ----------------------------------------------------------- #
    def render(self):
        v = self.source.load(self.idx)
        key = (tuple(v["shape"]), v["kind"], tuple(v["flips"]))
        disp = self._disp_cache.get(key)
        if disp is None:
            disp = core.display_index(*key)
            self._disp_cache = {key: disp}
        image = np.take(v["native"], disp)
        sp_r, sp_c = core.display_spacing(v["spacing"], v["kind"])
        h, w = disp.shape
        first = self.view is None or self.view["dshape"] != (h, w)
        self.view = dict(v, disp=disp, dshape=(h, w), image=image, sp=(sp_r, sp_c))
        self.img.setImage(image, autoLevels=False,
                          levels=core.levels_from_sample(v["sample"], self.lo_pct, self.hi_pct))
        rect = QtCore.QRectF(0, 0, w * sp_c, h * sp_r)
        self.img.setRect(rect)
        self.overlay.setRect(rect)
        self.rebuild_overlay()
        self.update_info()
        if first:
            self.vb.autoRange(padding=0.02)
        for d in (1, -1, 2):
            self.source.prefetch(self.idx + d)

    def _colorize(self, lab):
        out = np.zeros(lab.shape + (4,), np.uint8)
        nz = np.flatnonzero(lab)
        if nz.size:
            out.reshape(-1, 4)[nz] = self.main.lut[color_index(lab.ravel()[nz])]
        return out

    def _overlay_rect(self):
        h, w = self.view["dshape"]
        return QtCore.QRectF(0, 0, w * self.view["sp"][1], h * self.view["sp"][0])

    def rebuild_overlay(self):
        if self.view is None:
            return
        self.hover.hide()  # the label under the cursor may have changed
        lab = self.source.labels(self.idx)
        if lab is None or not lab.any():
            self.lab_disp, self.rgba = None, None
            self.overlay.clear()
        else:
            self.lab_disp = np.take(lab, self.view["disp"])
            self.rgba = self._colorize(self.lab_disp)
            self.overlay.setImage(self.rgba, autoLevels=False, levels=(0, 255))
            self.overlay.setRect(self._overlay_rect())
        self.refresh_annotations()

    def update_overlay_region(self, r0, r1, c0, c1):
        """Re-colour only the edited rectangle (display coordinates)."""
        lab = self.source.labels(self.idx)
        if lab is None:
            self.rebuild_overlay()
            return
        if self.rgba is None:  # first label in this section: start from an empty overlay
            self.lab_disp = np.zeros(self.view["dshape"], core.LABEL_DTYPE)
            self.rgba = np.zeros(self.view["dshape"] + (4,), np.uint8)
            self.overlay.setImage(self.rgba, autoLevels=False, levels=(0, 255))
            self.overlay.setRect(self._overlay_rect())
        sub = np.take(lab, self.view["disp"][r0:r1, c0:c1])
        self.lab_disp[r0:r1, c0:c1] = sub
        self.rgba[r0:r1, c0:c1] = self._colorize(sub)
        self.overlay.updateImage()

    def _to_view(self, r, c):
        return self.img.mapToView(QtCore.QPointF(c + 0.5, r + 0.5))

    def refresh_annotations(self):
        """Number tags and the go-to marker, placed from the label statistics."""
        for t in self._texts:
            self.vb.removeItem(t)
        self._texts = []
        stats = self.source.slice_stats(self.idx) if self.view is not None else {}
        self.disp_pos = {}
        if stats:
            labs = list(stats)
            coms = np.array([stats[k].com for k in labs], float)
            rr, cc = core.native_to_display(coms[:, 0], coms[:, 1], self.view["shape"], self.view["kind"],
                                            self.view["flips"])
            self.disp_pos = {k: (float(r), float(c)) for k, r, c in zip(labs, rr, cc)}
        if self.main.show_numbers:
            font = QtGui.QFont()
            font.setBold(True)
            for lab, (r, c) in self.disp_pos.items():
                t = pg.TextItem(str(lab), color=label_qcolor(lab), anchor=(0.5, 1.35),
                                fill=pg.mkBrush(0, 0, 0, 150))
                t.setFont(font)
                t.setZValue(25)
                self.vb.addItem(t, ignoreBounds=True)
                t.setPos(self._to_view(r, c))
                self._texts.append(t)
        self._update_marker()
        labs = sorted(self.disp_pos)
        txt = ", ".join(map(str, labs[:12])) + (" …" if len(labs) > 12 else "")
        self.title_lbl.setText(f"<b>{self.source.title}</b>" + (f"  <span style='color:#bbb'>[{txt}]</span>"
                                                                  if labs else ""))

    def _update_marker(self):
        if self.marker_target and self.marker_target[0] == self.idx and self.marker_target[1] in self.disp_pos:
            p = self._to_view(*self.disp_pos[self.marker_target[1]])
            self.marker.setData([p.x()], [p.y()])
            self.marker.show()
            vr = self.vb.viewRect()
            if not vr.contains(p):
                self.vb.setRange(QtCore.QRectF(p.x() - vr.width() / 2, p.y() - vr.height() / 2,
                                               vr.width(), vr.height()), padding=0)
        else:
            self.marker.hide()

    def set_marker(self, label):
        self.marker_target = None if label is None else (self.idx, int(label))
        self._update_marker()

    def update_info(self):
        self.info_lbl.setText(f"{self.source.describe(self.idx)} · {self.idx + 1}/{len(self.source)}")

    def note(self, text, kind=""):
        if not text:
            self.note_lbl.hide()
            return
        self.note_lbl.setText(text)
        self.note_lbl.setToolTip(text)
        self.note_lbl.setStyleSheet(f"color: {NOTE_COLORS.get(kind, '#ddd')}; background: rgba(0,0,0,175);"
                                    "padding: 2px 6px; border-radius: 3px")
        self.note_lbl.setWordWrap(True)
        max_w = max(120, self.gv.width() - 12)
        width = min(max_w, self.note_lbl.fontMetrics().boundingRect(text).width() + 18)
        self.note_lbl.setFixedWidth(width)
        self.note_lbl.setFixedHeight(self.note_lbl.heightForWidth(width))
        self.note_lbl.show()
        self.note_lbl.raise_()

    # -- mouse ----------------------------------------------------------------- #
    def _inside(self, c, r):
        if self.view is None:
            return False
        h, w = self.view["dshape"]
        return 0 <= r < h and 0 <= c < w

    def _hover_tol(self):
        px = self.vb.viewPixelSize()
        sp_r, sp_c = self.view["sp"]
        return int(np.clip(math.ceil(6 * max(px[0] / sp_c, px[1] / sp_r)), 1, 80))

    def on_leave(self):
        self.hover.hide()
        self.brush_item.hide()

    def on_mouse_moved(self, scene_pos):
        if self.view is None:
            return
        self.main.active_pane = self
        if not self.vb.sceneBoundingRect().contains(scene_pos):
            self.on_leave()
            return
        p = self.img.mapFromScene(scene_pos)
        c, r = p.x(), p.y()
        self._update_brush(c, r)
        if not self._inside(c, r):
            self.hover.hide()
            self.main.set_hover_status("")
            return
        ri, ci = int(r), int(c)
        lab = core.label_near(self.lab_disp, ri, ci, self._hover_tol()) if self.lab_disp is not None else 0
        if lab:
            col = label_qcolor(lab).name()
            self.hover.setHtml(f"<span style='font-size:13pt; font-weight:bold; color:{col}'>{lab}</span>")
            self.hover.setPos(self.vb.mapSceneToView(scene_pos))
            self.hover.show()
        else:
            self.hover.hide()
        flat = int(self.view["disp"][ri, ci])
        u, v = np.unravel_index(flat, self.view["shape"])
        val = float(self.view["image"][ri, ci])
        if self.source.is_template:
            t = self.source.t
            ijk = [0, 0, 0]
            ijk[t.ax], ijk[t.inplane[0]], ijk[t.inplane[1]] = self.idx, int(u), int(v)
            w = t.world(ijk)
            where = f"voxel ({ijk[0]}, {ijk[1]}, {ijk[2]}) · ({w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}) mm"
        else:
            where = f"{self.source.describe(self.idx)} · pixel ({u}, {v})"
        self.main.set_hover_status(f"{self.source.key} · {where} · value {val:.4g}"
                                   + (f" · label {lab}" if lab else ""))

    def _update_brush(self, c, r):
        if self.main.mode not in ("paint", "erase") or not self._inside(c, r):
            self.brush_item.hide()
            return
        rad = self.main.brush_radius(self)
        rect = QtCore.QRectF(int(c) - rad, int(r) - rad, 2 * rad + 1, 2 * rad + 1)
        self.brush_item.setRect(self.img.mapRectToView(rect))
        self.brush_item.show()

    def on_click(self, pos, mods):
        c, r = pos.x(), pos.y()
        self.main.active_pane = self
        if not self._inside(c, r):
            return
        if mods & Qt.KeyboardModifier.ControlModifier:
            self.main.pick_label(self, c, r)
            return
        if self.main.mode not in ("paint", "erase"):
            return
        if (mods & Qt.KeyboardModifier.ShiftModifier) and self.main.mode == "paint":
            self.main.place_label(self, c, r)
            return
        self.main.begin_stroke(self)
        self.main.paint_segment(self, (c, r), (c, r))
        self.main.end_stroke()

    def stroke_begin(self, pos):
        self.main.begin_stroke(self)
        self._last = (pos.x(), pos.y())

    def stroke_to(self, pos):
        cur = (pos.x(), pos.y())
        if self._last is None:
            self._last = cur
        self.main.paint_segment(self, self._last, cur)
        self._last = cur
        self._update_brush(*cur)

    def stroke_end(self):
        self._last = None
        self.main.end_stroke()

    def state(self):
        return dict(index=self.idx, lo_pct=self.lo_pct, hi_pct=self.hi_pct)


# --------------------------------------------------------------------------- #
# Undo record
# --------------------------------------------------------------------------- #
class Stroke:
    _ids = itertools.count(1)

    def __init__(self, pane, desc, kind="paint"):
        self.id = next(Stroke._ids)
        self.pane = pane
        self.desc = desc
        self.kind = kind                        # "paint": only pane's own slice changed
        self.entries: list[tuple] = []          # (key, flat, old values, new value)

    def add(self, key, flat, old, value):
        self.entries.append((key, np.asarray(flat, np.int64), np.asarray(old), int(value)))

    def by_key(self):
        out: dict = {}
        for key, flat, old, value in self.entries:
            flats, labs = out.setdefault(key, ([], set()))
            flats.append(flat)
            labs.update(int(x) for x in np.unique(old))
            labs.add(value)
        return out


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, ds: core.Dataset, out_dir: Path, label_format: str, session_info: dict,
                 state: dict, autosave_min: float = 5.0, overwrite: bool = False,
                 version_tag: str | None = None, label_suffix: str = core.LABEL_SUFFIX):
        super().__init__()
        self.ds = ds
        self.stacks = ds.stacks
        self.template = ds.template
        self.out_dir = Path(out_dir).expanduser().resolve()
        self.label_format = label_format
        self.session_info = session_info
        self.overwrite = overwrite
        self.version_tag = version_tag
        self.label_suffix = label_suffix
        self._out_owned = False        # True once this session has written to out_dir
        self.qc_dir: Path | None = None
        self.mode = "navigate"
        self.undo_stack: list[Stroke] = []
        self._stroke: Stroke | None = None
        self._saved_top = None         # id of the newest edit at the last save (None = as loaded)
        self.dirty = False
        self.show_numbers = bool(state.get("show_numbers", True))
        self.alpha = int(state.get("overlay_alpha", 190))
        self.lut = self._make_lut(self.alpha)
        self.issues: list[core.Issue] = []
        self._load_warnings = [core.Issue("warning", w) for w in ds.warnings]
        self._icons: dict = {}

        self._update_title()
        self.resize(1750, 1020)
        self._build_toolbars(state)

        self.spanes = [SlicePane(self, StackSource(st)) for st in self.stacks]
        self.tpane = SlicePane(self, TemplateSource(self.template))
        self.panes = self.spanes + [self.tpane]
        self.pane_by_key = {p.source.key: p for p in self.panes}
        self.active_pane = self.spanes[0] if self.spanes else self.tpane

        grid_w = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(grid_w)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)
        ncol = 2 if len(self.spanes) > 1 else 1
        for i, p in enumerate(self.spanes):
            grid.addWidget(p, i // ncol, i % ncol)
        split = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(grid_w)
        split.addWidget(self.tpane)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 1)
        split.setSizes([900, 480])
        self.setCentralWidget(split)
        self._build_dock()
        self._build_menu()
        self._build_shortcuts()

        self.hover_lbl = QtWidgets.QLabel()
        self.statusBar().addPermanentWidget(self.hover_lbl)

        self._table_timer = QtCore.QTimer(self)
        self._table_timer.setSingleShot(True)
        self._table_timer.setInterval(150)
        self._table_timer.timeout.connect(self.refresh_table)

        # Initial positions: saved state, else the first labelled section / the middle.
        panes_state = state.get("panes", {})
        for p in self.panes:
            ps = panes_state.get(p.source.key, {})
            p.set_contrast(ps.get("lo_pct", 0.5), ps.get("hi_pct", 99.5), render=False)
            idx = ps.get("index")
            if idx is None:
                if p.source.is_template:
                    first = min(self.template.stats, default=None)
                    idx = self.template.com_slice(first) if first is not None else len(p.source) // 2
                else:
                    idx = min(p.source.stack.stats, default=len(p.source) // 2)
            p.set_index(idx, force=True)

        self.refresh_table()
        self.populate_issues(core.validate(self.stacks, self.template), switch=False)
        self.set_mode("navigate")
        self.statusBar().showMessage(
            "P paint · E erase · N navigate · Shift+click move landmark · Ctrl+click pick label · "
            "hover shows label numbers · Ctrl+Z undo · Ctrl+S save", 15000)

        self._autosave = QtCore.QTimer(self)
        if autosave_min and autosave_min > 0:
            self._autosave.setInterval(int(autosave_min * 60_000))
            self._autosave.timeout.connect(self._on_autosave)
            self._autosave.start()

    # -- UI construction ------------------------------------------------------- #
    def _make_lut(self, alpha):
        lut = PALETTE.copy()
        lut[1:, 3] = alpha
        return lut

    def _build_toolbars(self, state):
        tb = self.addToolBar("Edit")
        tb.setMovable(False)
        self.mode_group = QtGui.QActionGroup(self) if hasattr(QtGui, "QActionGroup") else QtWidgets.QActionGroup(self)
        self.mode_actions = {}
        for key, text, tip in (("navigate", "Navigate", "Pan/zoom only (N or Esc)"),
                               ("paint", "Paint", "Paint the current label (P). Shift+click moves the landmark."),
                               ("erase", "Erase", "Erase any label under the brush (E)")):
            act = QAction(text, self)
            act.setCheckable(True)
            act.setToolTip(tip)
            act.triggered.connect(lambda _c=False, k=key: self.set_mode(k))
            self.mode_group.addAction(act)
            tb.addAction(act)
            self.mode_actions[key] = act
        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel(" Label "))
        self.label_spin = QtWidgets.QSpinBox()
        self.label_spin.setRange(1, 2_147_483_647)
        self.label_spin.setValue(int(state.get("current_label", 1)))
        self.label_spin.setMinimumWidth(80)
        self.label_spin.setToolTip("Label value painted in every pane (shared)")
        tb.addWidget(self.label_spin)
        self.swatch = QtWidgets.QLabel()
        self.swatch.setFixedSize(18, 18)
        tb.addWidget(self.swatch)
        self.label_spin.valueChanged.connect(self._update_swatch)
        self._update_swatch()
        nf = QtWidgets.QToolButton()
        nf.setText("Next free")
        nf.setToolTip("Set the current label to one larger than any label in use")
        nf.clicked.connect(self.next_free_label)
        tb.addWidget(nf)
        self.protect_cb = QtWidgets.QCheckBox("Don't overwrite other labels")
        self.protect_cb.setChecked(bool(state.get("protect", True)))
        tb.addWidget(self.protect_cb)
        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel(" Brush radius: sections "))
        self.brush2d = QtWidgets.QSpinBox()
        self.brush2d.setRange(0, 200)
        self.brush2d.setValue(int(state.get("brush_section", 3)))
        self.brush2d.setSuffix(" px")
        tb.addWidget(self.brush2d)
        tb.addWidget(QtWidgets.QLabel(" template "))
        self.brush3d = QtWidgets.QSpinBox()
        self.brush3d.setRange(0, 50)
        self.brush3d.setValue(int(state.get("brush_template", 1)))
        self.brush3d.setSuffix(" vox")
        tb.addWidget(self.brush3d)
        tb.addSeparator()
        undo = QtWidgets.QToolButton()
        undo.setText("Undo")
        undo.clicked.connect(self.undo)
        tb.addWidget(undo)

        self.addToolBarBreak()
        tb2 = self.addToolBar("Go")
        tb2.setMovable(False)
        tb2.addWidget(QtWidgets.QLabel(" Go to label "))
        self.goto_label = QtWidgets.QLineEdit()
        self.goto_label.setFixedWidth(80)
        self.goto_label.setPlaceholderText("e.g. 12")
        self.goto_label.returnPressed.connect(self._on_goto_label)
        tb2.addWidget(self.goto_label)
        go = QtWidgets.QToolButton()
        go.setText("Go")
        go.clicked.connect(self._on_goto_label)
        tb2.addWidget(go)
        self.estimate_cb = QtWidgets.QCheckBox("Estimate section where missing")
        self.estimate_cb.setToolTip("For hemispheres without the label, jump to the section predicted from the "
                                    "landmarks they share with the template (linear fit of section number vs. "
                                    "template coronal position)")
        self.estimate_cb.setChecked(bool(state.get("estimate", False)))
        tb2.addWidget(self.estimate_cb)
        tb2.addSeparator()
        tb2.addWidget(QtWidgets.QLabel(" Label opacity "))
        self.alpha_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(0, 255)
        self.alpha_slider.setValue(self.alpha)
        self.alpha_slider.setFixedWidth(120)
        self.alpha_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.alpha_slider.valueChanged.connect(self._on_alpha)
        tb2.addWidget(self.alpha_slider)
        self.numbers_cb = QtWidgets.QCheckBox("Show label numbers")
        self.numbers_cb.setChecked(self.show_numbers)
        self.numbers_cb.toggled.connect(self._on_numbers)
        tb2.addWidget(self.numbers_cb)
        tb2.addSeparator()
        for text, slot, tip in (("Validate", lambda: self.populate_issues(core.validate(self.stacks, self.template)),
                                 "Check uniqueness and 2-D/3-D correspondence"),
                                ("Save", self.save, "Save all outputs (Ctrl+S)"),
                                ("QC figures…", self.export_qc, "Render per-label QC figures and a PDF")):
            b = QtWidgets.QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            tb2.addWidget(b)

    def _build_dock(self):
        dock = QtWidgets.QDockWidget("Landmarks", self)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
                         | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.tabs = QtWidgets.QTabWidget()
        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.cellClicked.connect(self._on_table_click)
        self.table.setToolTip("Click a row to go to that label")
        self.issues_list = QtWidgets.QListWidget()
        self.issues_list.itemActivated.connect(self._on_issue)
        self.issues_list.itemClicked.connect(self._on_issue)
        self.tabs.addTab(self.table, "Labels")
        self.tabs.addTab(self.issues_list, "Issues")
        dock.setWidget(self.tabs)
        dock.setMinimumWidth(480)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        for text, slot, sc in (("Save", self.save, "Ctrl+S"), ("QC figures…", self.export_qc, None),
                               ("Quit", self.close, "Ctrl+Q")):
            a = QAction(text, self)
            if sc:
                a.setShortcut(QtGui.QKeySequence(sc))
            a.triggered.connect(slot)
            m.addAction(a)
        e = self.menuBar().addMenu("&Edit")
        for text, slot, sc in (("Undo", self.undo, "Ctrl+Z"), ("Next free label", self.next_free_label, None)):
            a = QAction(text, self)
            if sc:
                a.setShortcut(QtGui.QKeySequence(sc))
            a.triggered.connect(slot)
            e.addAction(a)
        h = self.menuBar().addMenu("&Help")
        a = QAction("Mouse and keys", self)
        help_text = "Mouse (in a pane)" + __doc__.split("Mouse (in a pane)", 1)[1]
        a.triggered.connect(lambda: QtWidgets.QMessageBox.information(self, "Mouse and keys", help_text))
        h.addAction(a)

    def _build_shortcuts(self):
        def sc(keys, slot):
            s = QShortcut(QtGui.QKeySequence(keys), self)
            s.activated.connect(slot)
            return s
        sc("P", lambda: self.set_mode("paint"))
        sc("E", lambda: self.set_mode("erase"))
        sc("N", lambda: self.set_mode("navigate"))
        sc("Escape", lambda: self.set_mode("navigate"))
        for k in ("Left", "Up", "PgUp"):
            sc(k, lambda: self.active_pane.step(-1))
        for k in ("Right", "Down", "PgDown"):
            sc(k, lambda: self.active_pane.step(1))
        sc("[", lambda: self._brush_step(-1))
        sc("]", lambda: self._brush_step(1))
        sc("F", lambda: self.active_pane.vb.autoRange(padding=0.02))
        sc("Ctrl+G", lambda: (self.goto_label.setFocus(), self.goto_label.selectAll()))

    # -- small UI handlers ------------------------------------------------------- #
    def _update_swatch(self):
        self.swatch.setStyleSheet(f"background: {label_qcolor(self.label_spin.value()).name()};"
                                  "border: 1px solid #444; border-radius: 3px")

    def set_mode(self, mode):
        self.mode = mode
        self.mode_actions[mode].setChecked(True)
        cursor = Qt.CursorShape.CrossCursor if mode in ("paint", "erase") else Qt.CursorShape.ArrowCursor
        for p in self.panes:
            p.gv.setCursor(cursor)
            if mode == "navigate":
                p.brush_item.hide()

    def brush_radius(self, pane):
        return self.brush3d.value() if pane.source.is_template else self.brush2d.value()

    def _brush_step(self, d):
        box = self.brush3d if self.active_pane.source.is_template else self.brush2d
        box.setValue(box.value() + d)

    def _on_alpha(self, v):
        self.alpha = int(v)
        self.lut = self._make_lut(self.alpha)
        for p in self.panes:
            p.rebuild_overlay()

    def _on_numbers(self, on):
        self.show_numbers = bool(on)
        for p in self.panes:
            p.refresh_annotations()

    def set_hover_status(self, text):
        self.hover_lbl.setText(text)

    def flash(self, text, kind="", ms=8000):
        color = NOTE_COLORS.get(kind)
        self.statusBar().setStyleSheet(f"color: {color}" if kind else "")
        self.statusBar().showMessage(text, ms)

    def _update_title(self):
        self.setWindowTitle(f"Landmark labelling — saves to {self.out_dir}[*]")

    def update_dirty(self):
        """Unsaved = the newest edit differs from the one at the last save. Undoing
        back to the loaded (or last saved) state therefore counts as no change."""
        top = self.undo_stack[-1].id if self.undo_stack else None
        self.dirty = top != self._saved_top
        self.setWindowModified(self.dirty)

    def on_pane_moved(self, pane):
        pass

    def next_free_label(self):
        self.label_spin.setValue(core.next_free_label(self.stacks, self.template))
        self.flash(f"Current label set to {self.label_spin.value()} (unused everywhere)")

    # -- painting ------------------------------------------------------------- #
    def begin_stroke(self, pane, desc=None, kind="paint"):
        if self._stroke is None:
            what = "erase" if self.mode == "erase" else f"paint {self.label_spin.value()}"
            self._stroke = Stroke(pane, desc or f"{what} on {pane.source.key}", kind)

    def paint_segment(self, pane, p0, p1):
        if self._stroke is None or pane.view is None:
            return
        value = 0 if self.mode == "erase" else int(self.label_spin.value())
        protect = self.protect_cb.isChecked() and value != 0
        rad = self.brush_radius(pane)
        disp = pane.view["disp"]
        h, w = disp.shape
        target = pane.source.target(pane.idx, create=True)
        key = pane.source.target_key(pane.idx)
        (c0, r0), (c1, r1) = p0, p1
        dist = math.hypot(c1 - c0, r1 - r0)
        n = max(1, int(math.ceil(dist / max(0.5, rad * 0.5))))
        box = [h, 0, w, 0]
        blocked = 0
        for t in (np.linspace(0.0, 1.0, n + 1) if dist > 0 else (0.0,)):
            cc = int(math.floor(c0 + (c1 - c0) * t))
            rr = int(math.floor(r0 + (r1 - r0) * t))
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            ra, rb, ca, cb = max(0, rr - rad), min(h, rr + rad + 1), max(0, cc - rad), min(w, cc + rad + 1)
            yy, xx = np.ogrid[ra:rb, ca:cb]
            mask = (yy - rr) ** 2 + (xx - cc) ** 2 <= rad * rad
            tflat = pane.source.to_target_flat(pane.idx, disp[ra:rb, ca:cb][mask])
            cur = np.take(target, tflat)
            sel = cur != value
            if protect:
                blocked += int(np.count_nonzero(sel & (cur != 0)))
                sel &= cur == 0
            if not sel.any():
                continue
            np.put(target, tflat[sel], value)
            self._stroke.add(key, tflat[sel], cur[sel], value)
            box = [min(box[0], ra), max(box[1], rb), min(box[2], ca), max(box[3], cb)]
        if box[1] > box[0]:
            pane.update_overlay_region(*box)
        if blocked:
            self.flash("Other labels under the brush were left untouched "
                       "(untick \"Don't overwrite other labels\" to paint over them)", "warn", 4000)

    def end_stroke(self):
        rec, self._stroke = self._stroke, None
        if rec is None or not rec.entries:
            return
        self.undo_stack.append(rec)
        del self.undo_stack[:-300]
        self._apply_changes(rec.by_key(), own_pane=rec.pane if rec.kind == "paint" else None)
        self.update_dirty()

    def _target_for_key(self, key):
        if key[0] == core.TEMPLATE_KEY:
            return self.template.labels
        stack = self.pane_by_key[key[0]].source.stack
        return stack.get_labels(key[1], create=True)

    def _apply_changes(self, by_key, own_pane=None):
        """Refresh label statistics and the affected panes after edits.
        ``own_pane`` already shows the edit (painted incrementally)."""
        for key, (flats, labs) in by_key.items():
            flat = np.concatenate(flats)
            if key[0] == core.TEMPLATE_KEY:
                self.template.refresh_labels(labs, core.bbox_from_flat(flat, self.template.shape))
            else:
                self.pane_by_key[key[0]].source.stack.refresh(key[1])
        for p in self.panes:
            k = p.source.target_key(p.idx)
            if k in by_key:
                if p is own_pane:
                    p.refresh_annotations()
                else:
                    p.rebuild_overlay()
        self._table_timer.start()

    def undo(self):
        if self._stroke is not None:
            return
        if not self.undo_stack:
            self.flash("Nothing to undo")
            return
        rec = self.undo_stack.pop()
        by_key: dict = {}
        for key, flat, old, _value in reversed(rec.entries):
            tgt = self._target_for_key(key)
            before = np.take(tgt, flat)
            np.put(tgt, flat, old)
            flats, labs = by_key.setdefault(key, ([], set()))
            flats.append(flat)
            labs.update(int(x) for x in np.unique(before))
            labs.update(int(x) for x in np.unique(old))
        self._apply_changes(by_key)
        self.update_dirty()
        self.flash(f"Undone: {rec.desc}")

    def place_label(self, pane, c, r):
        """Shift+click: put the current label here and remove it everywhere else
        in this hemisphere (or in the template), keeping the label unique."""
        lab = int(self.label_spin.value())
        disp = pane.view["disp"]
        ri, ci = int(r), int(c)
        target = pane.source.target(pane.idx, create=True)
        spot = pane.source.to_target_flat(pane.idx, disp[ri:ri + 1, ci:ci + 1].ravel())
        occupant = int(np.take(target, spot)[0])
        if occupant not in (0, lab):
            self.flash(f"That spot is already label {occupant}; erase it first or click elsewhere", "warn")
            return
        self.begin_stroke(pane, desc=f"move label {lab} on {pane.source.key}", kind="place")
        moved_from = []
        if pane.source.is_template:
            flat = self.template.remove_label(lab)
            if flat.size:
                self._stroke.add((core.TEMPLATE_KEY,), flat, np.full(flat.size, lab), 0)
                moved_from.append("template")
        else:
            stack = pane.source.stack
            for idx, flat in stack.remove_label(lab):
                self._stroke.add((stack.id, idx), flat, np.full(flat.size, lab), 0)
                moved_from.append(stack.describe(idx, short=True))
        self.paint_segment(pane, (c, r), (c, r))
        self.end_stroke()
        self.flash(f"Label {lab} placed" + (f" (removed from {', '.join(moved_from)})" if moved_from else ""))

    def pick_label(self, pane, c, r):
        if pane.lab_disp is None:
            return
        lab = core.label_near(pane.lab_disp, int(r), int(c), pane._hover_tol())
        if lab:
            self.label_spin.setValue(lab)
            self.flash(f"Current label: {lab}")

    # -- go to label ------------------------------------------------------------ #
    def _on_goto_label(self):
        text = self.goto_label.text().strip()
        if not text:
            return
        try:
            lab = int(float(text))
        except ValueError:
            self.flash(f"'{text}' is not a label number", "error")
            return
        self.go_to_label(lab)

    def go_to_label(self, lab):
        st = self.template.stats.get(lab)
        if st is None:
            self.flash(f"Label {lab} does not exist in the 3D template.", "error")
            self.goto_label.setStyleSheet("background: #5a1f1f")
            QtCore.QTimer.singleShot(1200, lambda: self.goto_label.setStyleSheet(""))
            return
        com = np.rint(st.com).astype(int)
        self.tpane.set_index(int(com[self.template.ax]), force=True)
        self.tpane.set_marker(lab)
        self.tpane.note(f"label {lab}: centre of mass voxel ({com[0]}, {com[1]}, {com[2]})", "ok")
        self.label_spin.setValue(lab)
        found = []
        for pane in self.spanes:
            stack = pane.source.stack
            places = stack.locations().get(lab, [])
            if places:
                pane.set_index(places[0][0], force=True)
                pane.set_marker(lab)
                found.append(stack.id)
                if len(places) > 1:
                    pane.note(f"label {lab} is in {len(places)} sections: "
                              + ", ".join(stack.describe(i, short=True) for i, _ in places), "error")
                else:
                    pane.note(f"label {lab}", "ok")
                continue
            pane.set_marker(None)
            est = core.estimate_section(stack, self.template, lab) if self.estimate_cb.isChecked() else None
            if est is not None:
                pane.set_index(est[0], force=True)
                pane.note(f"label {lab} not placed · estimated ≈ section {est[1]:.0f} "
                          f"(fit to {est[2]} shared landmarks)", "warn")
            else:
                pane.note(f"label {lab} not on this hemisphere", "muted")
        self.flash(f"Label {lab}: template slice {com[self.template.ax]}; "
                   f"found in {len(found)}/{len(self.spanes)} hemispheres"
                   + (f" ({', '.join(found)})" if found else ""))
        self._select_table_row(lab)

    # -- tables ------------------------------------------------------------------ #
    def _icon(self, lab):
        ic = self._icons.get(lab)
        if ic is None:
            pm = QtGui.QPixmap(10, 10)
            pm.fill(label_qcolor(lab))
            ic = QtGui.QIcon(pm)
            self._icons[lab] = ic
        return ic

    def refresh_table(self):
        cov = core.coverage(self.stacks, self.template)
        keys = ["template"] + [s.id for s in self.stacks]
        headers = ["Label", "Template"] + [s.id for s in self.stacks]
        t = self.table
        t.setUpdatesEnabled(False)
        t.clear()
        t.setColumnCount(len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.setRowCount(len(cov))
        err_bg, err_fg, muted = QtGui.QColor("#f2c4c4"), QtGui.QColor("#5c0000"), QtGui.QColor("#8a8a8a")
        n_err = 0
        for row, (lab, cells) in enumerate(cov.items()):
            it = QtWidgets.QTableWidgetItem(str(lab))
            it.setIcon(self._icon(lab))
            it.setData(Qt.ItemDataRole.UserRole, lab)
            it.setData(Qt.ItemDataRole.DisplayRole, lab)
            t.setItem(row, 0, it)
            for col, key in enumerate(keys, start=1):
                cell = cells[key]
                item = QtWidgets.QTableWidgetItem(cell.text if cell.status != "absent" else "–")
                item.setData(Qt.ItemDataRole.UserRole, lab)
                if cell.status in core.ERROR_STATUSES:
                    item.setBackground(err_bg)
                    item.setForeground(err_fg)
                    item.setToolTip({"duplicate": "Label appears in more than one section of this hemisphere",
                                     "multiblob": "Label has more than one separate blob",
                                     "orphan": "Label has no counterpart in the 3D template",
                                     "missing": "Label exists on a hemisphere but not in the 3D template"
                                     }[cell.status])
                    n_err += 1
                elif cell.status == "absent":
                    item.setForeground(muted)
                t.setItem(row, col, item)
        t.resizeColumnsToContents()
        t.setUpdatesEnabled(True)
        self.tabs.setTabText(0, f"Labels ({len(cov)})" + (f" · {n_err} problems" if n_err else ""))

    def _select_table_row(self, lab):
        for row in range(self.table.rowCount()):
            it = self.table.item(row, 0)
            if it is not None and it.data(Qt.ItemDataRole.UserRole) == lab:
                self.table.blockSignals(True)
                self.table.selectRow(row)
                self.table.blockSignals(False)
                self.table.scrollToItem(it)
                return

    def _on_table_click(self, row, _col):
        it = self.table.item(row, 0)
        if it is not None:
            self.go_to_label(int(it.data(Qt.ItemDataRole.UserRole)))

    def populate_issues(self, issues, switch=True):
        self.issues = list(self._load_warnings) + list(issues)
        self.issues_list.clear()
        order = {"error": 0, "warning": 1, "info": 2}
        colors = {"error": "#e06c6c", "warning": "#e0a030", "info": "#9a9a9a"}
        for i, iss in sorted(enumerate(self.issues), key=lambda x: (order[x[1].severity], x[0])):
            item = QtWidgets.QListWidgetItem(f"{iss.severity.upper()}: {iss.message}")
            item.setForeground(QtGui.QColor(colors[iss.severity]))
            item.setData(Qt.ItemDataRole.UserRole, i)
            item.setToolTip(iss.message)
            self.issues_list.addItem(item)
        n_err = sum(i.severity == "error" for i in issues)
        self.tabs.setTabText(1, f"Issues ({n_err} errors)" if n_err else "Issues (none)")
        if switch:
            self.tabs.setCurrentIndex(1)
            self.flash(f"Validation: {n_err} error(s)" if n_err else "Validation passed: labels are unique and "
                       "every hemisphere label has a template counterpart", "error" if n_err else "ok")

    def _on_issue(self, item):
        iss = self.issues[int(item.data(Qt.ItemDataRole.UserRole))]
        if iss.template_slice is not None:
            self.tpane.set_index(iss.template_slice, force=True)
            self.tpane.set_marker(iss.label)
        if iss.specimen and iss.section_idx is not None and iss.specimen in self.pane_by_key:
            p = self.pane_by_key[iss.specimen]
            p.set_index(iss.section_idx, force=True)
            p.set_marker(iss.label)
            p.note(iss.message, "error" if iss.severity == "error" else "warn")
        if iss.label is not None:
            self.label_spin.setValue(iss.label)

    # -- saving / QC ---------------------------------------------------------- #
    def gui_state(self):
        return dict(panes={p.source.key: p.state() for p in self.panes},
                    current_label=int(self.label_spin.value()), brush_section=self.brush2d.value(),
                    brush_template=self.brush3d.value(), overlay_alpha=self.alpha,
                    show_numbers=self.show_numbers, protect=self.protect_cb.isChecked(),
                    estimate=self.estimate_cb.isChecked())

    def names(self):
        return core.output_names(self.out_dir, self.template.label_path, self.version_tag, self.label_suffix)

    def _ensure_writable(self, interactive):
        """New labels go to a new (or empty) folder. Writing into a folder that
        already holds files needs --overwrite or an explicit choice here."""
        while not self._out_owned:
            existing = core.folder_contents(self.out_dir)
            if not existing or self.overwrite:
                return True
            if not interactive:
                return False
            prev = core.previous_session(self.out_dir)
            what = (f"a session saved on {prev.get('saved')}" if prev else
                    f"{len(existing)} item(s), e.g. {', '.join(existing[:3])}")
            box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Icon.Warning, "Output folder not empty",
                                        f"{self.out_dir}\nalready contains {what}.\n\nSaving there would "
                                        "overwrite files in it. Choose a new folder for the new labels, or "
                                        "overwrite.", parent=self)
            choose = box.addButton("Choose another folder…", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
            over = box.addButton("Overwrite", QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
            box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
            box.setDefaultButton(choose)
            box.exec() if hasattr(box, "exec") else box.exec_()
            clicked = box.clickedButton()
            if clicked is over:
                return True
            if clicked is not choose:
                return False
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder for the new labels",
                                                                str(self.out_dir.parent))
            if not folder:
                return False
            self.out_dir = Path(folder).resolve()
            self._update_title()
        return True

    def save(self, quiet=False):
        """Write outputs if the annotations changed. Returns True when the state
        on disk is current (saved now, or nothing to save)."""
        if self._stroke is not None:
            return False
        if not self.dirty:
            if not quiet:
                self.flash("No changes to the annotations — nothing written."
                           + ("" if self._out_owned else " The input labels are current."))
            return True
        if not self._ensure_writable(interactive=not quiet):
            if quiet:
                self.flash(f"Autosave skipped: {self.out_dir} already contains files. Press Save to choose "
                           "another folder or overwrite.", "warn", 15000)
            return False
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            rep = core.save_session(self.names(), self.ds.df, self.ds.columns, self.stacks, self.template,
                                    self.label_format, input_csv_dir=Path(self.ds.csv_path).parent,
                                    session_info=self.session_info, gui_state=self.gui_state())
        except Exception as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            log.exception("Save failed")
            QtWidgets.QMessageBox.critical(self, "Save failed", f"{exc}\n\nNothing was lost from memory; "
                                           "fix the problem and save again.")
            return False
        QtWidgets.QApplication.restoreOverrideCursor()
        self._out_owned = True
        self._saved_top = self.undo_stack[-1].id if self.undo_stack else None
        self.update_dirty()
        self.populate_issues(rep.issues, switch=bool(rep.n_errors) and not quiet)
        self.flash(("Autosaved. " if quiet else "") + rep.summary(), "warn" if rep.n_errors else "ok", 12000)
        return True

    def _on_autosave(self):
        if self.dirty and self._stroke is None:
            self.save(quiet=True)

    def _qc_folder(self):
        """QC goes next to the saved labels; with nothing saved this session
        (so no output folder), ask where to put it instead of creating one."""
        if self._out_owned:
            return self.out_dir / "qc"
        if self.qc_dir is not None:
            return self.qc_dir
        start = str(Path(self.ds.csv_path).parent)
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Folder for QC figures (nothing has been saved, so there is no output folder yet)", start)
        if not folder:
            return None
        self.qc_dir = Path(folder)
        return self.qc_dir

    def export_qc(self):
        import landmark_qc
        qc_dir = self._qc_folder()
        if qc_dir is None:
            return
        dlg = QtWidgets.QProgressDialog("Rendering QC figures…", "Cancel", 0, 1, self)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(0)

        def progress(i, n, msg):
            dlg.setMaximum(max(n, 1))
            dlg.setValue(i)
            dlg.setLabelText(msg)
            QtWidgets.QApplication.processEvents()
            return not dlg.wasCanceled()

        try:
            out = landmark_qc.render_qc(qc_dir, self.stacks, self.template, progress=progress)
        except Exception as exc:
            dlg.close()
            log.exception("QC failed")
            QtWidgets.QMessageBox.critical(self, "QC failed", str(exc))
            return
        dlg.close()
        if out is not None:
            self.flash(f"QC written: {out}" + (" (unsaved edits included)" if self.dirty else ""), "ok", 15000)

    def closeEvent(self, ev):
        if self.dirty:
            ans = QtWidgets.QMessageBox.question(
                self, "Unsaved changes", "Save changes before closing?",
                QtWidgets.QMessageBox.StandardButton.Save | QtWidgets.QMessageBox.StandardButton.Discard
                | QtWidgets.QMessageBox.StandardButton.Cancel)
            if ans == QtWidgets.QMessageBox.StandardButton.Cancel:
                ev.ignore()
                return
            if ans == QtWidgets.QMessageBox.StandardButton.Save and not self.save():
                ev.ignore()
                return
        for st in self.stacks:
            st._pool.shutdown(wait=False, cancel_futures=True)
        ev.accept()


# --------------------------------------------------------------------------- #
# Start-up
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="section CSV (subject, hemisphere, section, autoradiograph, labels)")
    ap.add_argument("--template", help="3-D reference MRI (NIfTI)")
    ap.add_argument("--template-labels", help="3-D landmark label volume (NIfTI); created empty if missing")
    ap.add_argument("--out-dir", required=True,
                    help="new folder for the edited labels and the new CSV; created on the first save that has "
                         "changes (nothing is written if no annotation changes)")
    ap.add_argument("--specimens", nargs="+", help="specimen IDs to load, in pane order (e.g. 11530-L 11539-R)")
    ap.add_argument("--flip-lr", nargs="+", metavar="SPECIMEN",
                    help="show these specimens' sections mirrored left-right, e.g. a right hemisphere next to "
                         "left ones (display only: labels are saved in the files' original orientation); the "
                         "pane is titled <specimen>-flipped-lr")
    ap.add_argument("--coronal-axis", default="auto", help="template voxel axis that is coronal (auto/0/1/2)")
    ap.add_argument("--label-format", default="match", choices=core.LABEL_FORMATS,
                    help="file format of saved 2-D labels; 'match' = same as each autoradiograph "
                         "(NIfTI -> .nii.gz, PNG -> 16-bit PNG)")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow saving into an --out-dir that already contains files (default: the first "
                         "save asks you to choose another folder or confirm)")
    ap.add_argument("--version-tag", help="suffix for the 3-D label and CSV file names (default: the "
                                          "--out-dir folder name if it looks like v1.4, else none)")
    ap.add_argument("--label-suffix", default=core.LABEL_SUFFIX,
                    help="suffix of 2-D label files: <image name><suffix>.nii.gz (default: %(default)s)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--resume", action="store_true",
                   help="load the labels saved in --out-dir instead of --csv/--template-labels")
    g.add_argument("--fresh", action="store_true", help="load --csv/--template-labels even if --out-dir has a "
                                                        "saved session (don't ask)")
    ap.add_argument("--autosave-min", type=float, default=5.0, help="autosave interval in minutes (0 = off)")
    for role in ("sub", "hemi", "section", "raw", "labels"):
        ap.add_argument(f"--col-{role}", help=f"CSV column holding the {role} (default: auto-detect)")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args(argv)


def resolve_session(args, ask) -> dict:
    """Decide whether to resume from --out-dir; ``ask(text)`` returns
    'resume' | 'fresh' | 'cancel' when the user must choose."""
    out_dir = Path(args.out_dir).expanduser().resolve()
    overrides = {"sub": args.col_sub, "hemisphere": args.col_hemi, "section": args.col_section,
                 "raw": args.col_raw, "labels": args.col_labels}
    overrides = {k: v for k, v in overrides.items() if v}
    cfg = dict(out_dir=out_dir, csv=args.csv, template=args.template, template_labels=args.template_labels,
               specimens=args.specimens, coronal_axis=args.coronal_axis, overrides=overrides, resumed=False,
               flip_lr=args.flip_lr)
    prev = core.previous_session(out_dir)
    if prev is None:
        if args.resume:
            raise SystemExit(f"--resume: no saved session in {out_dir}")
        return cfg
    passed_prev_csv = args.csv and Path(args.csv).expanduser().resolve() == Path(prev["csv_path"])
    choice = "resume" if (args.resume or passed_prev_csv) else ("fresh" if args.fresh else None)
    if choice is None:
        choice = ask(f"{out_dir} already contains a saved session (last saved {prev.get('saved', '?')}).\n\n"
                     "Resume it (load its labels)? 'Start from inputs' loads the CSV and label volume you "
                     "passed instead. Either way, saving into this folder asks first unless --overwrite.")
    if choice == "cancel":
        raise SystemExit(0)
    if choice == "resume":
        cfg.update(csv=prev["csv_path"], template_labels=prev["template_labels_path"], resumed=True)
        cfg["template"] = args.template or prev.get("template")
        cfg["specimens"] = args.specimens or prev.get("specimens")
        cfg["flip_lr"] = args.flip_lr if args.flip_lr is not None else prev.get("flip_lr")
        if args.coronal_axis == "auto" and "coronal_axis" in prev:
            cfg["coronal_axis"] = prev["coronal_axis"]
        if not overrides and prev.get("columns"):
            cols = prev["columns"]
            cfg["overrides"] = {k: cols[k] for k in ("sub", "hemisphere", "section", "raw", "labels") if k in cols}
    return cfg


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("Landmark labelling")

    def ask(text):
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Icon.Question, "Saved session found", text)
        resume = box.addButton("Resume", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        fresh = box.addButton("Start from inputs", QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(resume)
        box.exec() if hasattr(box, "exec") else box.exec_()
        clicked = box.clickedButton()
        return "resume" if clicked is resume else ("fresh" if clicked is fresh else "cancel")

    try:
        cfg = resolve_session(args, ask)
        if not cfg["csv"] or not cfg["template"]:
            raise ValueError("--csv and --template are required (or --resume with a saved session)")
        prog = QtWidgets.QProgressDialog("Loading…", None, 0, 1)
        prog.setWindowTitle("Landmark labelling")
        prog.setMinimumDuration(300)

        def progress(msg, i, n):
            prog.setMaximum(max(n, 1))
            prog.setValue(i)
            prog.setLabelText(msg)
            app.processEvents()

        ds = core.load_dataset(cfg["csv"], cfg["template"], cfg["template_labels"], cfg["specimens"],
                               cfg["coronal_axis"], cfg["overrides"], progress, flip_lr=cfg["flip_lr"])
        prog.close()
    except SystemExit:
        raise
    except Exception as exc:
        log.debug(traceback.format_exc())
        QtWidgets.QMessageBox.critical(None, "Cannot start", str(exc))
        print(f"error: {exc}", file=sys.stderr)
        return 1

    session_info = dict(csv_input=str(Path(args.csv).resolve()) if args.csv else cfg["csv"],
                        template_labels_input=cfg["template_labels"], resumed=cfg["resumed"],
                        flip_lr=[st.id for st in ds.stacks if st.flip_lr])
    state = core.load_state(cfg["out_dir"])
    win = MainWindow(ds, cfg["out_dir"], args.label_format, session_info, state, args.autosave_min,
                     overwrite=args.overwrite, version_tag=args.version_tag, label_suffix=args.label_suffix)
    win.show()
    return app.exec() if hasattr(app, "exec") else app.exec_()


if __name__ == "__main__":
    sys.exit(main())
