"""Tests for the Qt-free core (run: python -m pytest test_landmark_core.py -q)."""
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import landmark_core as core

HERE = Path(__file__).resolve().parent
SPECIMENS = ["11530-L", "11539-L", "11539-R", "11543-L"]


@pytest.fixture(scope="session")
def data(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    subprocess.run([sys.executable, str(HERE / "make_synthetic_data.py"), str(out)], check=True,
                   capture_output=True)
    return out


def load(data, csv=None, labels=None):
    return core.load_dataset(str(csv or data / "sections.csv"), str(data / "template.nii.gz"),
                             str(labels or data / "template_landmarks.nii.gz"), SPECIMENS)


# -- display mapping ------------------------------------------------------- #
def test_display_index_raster_is_identity():
    disp = core.display_index((3, 4), "raster")
    assert np.array_equal(disp, np.arange(12).reshape(3, 4))


def test_display_index_nifti_shows_x_right_y_up():
    native = np.arange(12).reshape(4, 3)          # (x, y)
    shown = np.take(native, core.display_index(native.shape, "nifti"))
    assert shown.shape == (3, 4)
    assert shown[-1, 0] == native[0, 0]            # origin bottom-left
    assert shown[-1, -1] == native[-1, 0]          # x increases to the right
    assert shown[0, 0] == native[0, -1]            # y increases upward


def test_display_flips():
    disp = core.display_index((2, 3), "raster", (True, True))
    assert np.array_equal(np.take(np.arange(6).reshape(2, 3), disp), np.arange(6).reshape(2, 3)[::-1, ::-1])


@pytest.mark.parametrize("kind", ["raster", "nifti"])
@pytest.mark.parametrize("flips", [(False, False), (True, False), (False, True), (True, True)])
def test_native_to_display_inverts_display_index(kind, flips):
    shape = (7, 5)
    disp = core.display_index(shape, kind, flips)
    for flat in range(35):
        i, j = np.unravel_index(flat, shape)
        r, c = core.native_to_display(i, j, shape, kind, flips)
        assert disp[int(r), int(c)] == flat


# -- statistics -------------------------------------------------------------- #
def test_stats_components_and_com():
    a = np.zeros((20, 20), np.int32)
    a[2:4, 2:4] = 5
    a[10:12, 10:12] = 5                            # second blob of the same label
    a[15, 15] = 7
    st = core.compute_stats(a)
    assert st[5].n_components == 2 and st[5].count == 8
    assert st[7].com == (15.0, 15.0) and st[7].n_components == 1


def test_diagonal_pixels_are_one_blob():
    a = np.zeros((5, 5), np.int32)
    a[1, 1] = a[2, 2] = 3
    assert core.compute_stats(a)[3].n_components == 1


def test_label_near():
    a = np.zeros((10, 10), np.int32)
    a[5, 5] = 9
    assert core.label_near(a, 5, 7, 3) == 9 and core.label_near(a, 5, 9, 3) == 0


# -- csv / loading ------------------------------------------------------------ #
def test_columns_aliases():
    cols = core.resolve_columns(["Subject", "Hemisphere", "Section Number", "Autoradiograph", "Labels"])
    assert cols == {"sub": "Subject", "hemisphere": "Hemisphere", "section": "Section Number",
                    "raw": "Autoradiograph", "labels": "Labels"}
    cols = core.resolve_columns(["sub", "hemisphere", "sample", "raw", "landmark", "chunk", "acquisition"])
    assert cols["section"] == "sample" and cols["labels"] == "landmark" and cols["chunk"] == "chunk"


def test_load_orders_and_specimens(data):
    ds = load(data)
    assert [s.id for s in ds.stacks] == SPECIMENS
    for st in ds.stacks:
        nums = [s.number for s in st.sections]
        assert nums == sorted(nums)


def test_shape_mismatch_is_an_error(tmp_path, data):
    ds = load(data)
    s = ds.stacks[0].sections[0]
    bad = tmp_path / "bad.png"
    from PIL import Image
    Image.fromarray(np.zeros((5, 5), np.uint16)).save(bad)
    with pytest.raises(ValueError, match="shape"):
        core.read_labels(str(bad), ds.stacks[0].meta(0).shape)


# -- validation ----------------------------------------------------------------- #
def test_planted_errors_found(data):
    ds = load(data)
    errs = [i for i in core.validate(ds.stacks, ds.template) if i.severity == "error"]
    assert any(i.specimen == "11539-R" and "appears in 2 sections" in i.message for i in errs)
    assert any(i.specimen == "11543-L" and i.label == 99 and "no counterpart" in i.message for i in errs)
    assert len(errs) == 2


def test_template_multiblob_blocks_hemispheres_using_it(data):
    ds = load(data)
    lab = sorted(ds.stacks[0].all_labels())[0]
    far = np.argwhere(ds.template.labels == 0)[0]
    ds.template.labels[tuple(far)] = lab
    ds.template.stats = core.compute_stats(ds.template.labels)
    issues = core.validate(ds.stacks, ds.template)
    assert any(i.label == lab and "separate blobs" in i.message for i in issues)
    assert core.blocking_errors(issues, ds.stacks[0])


def test_estimate_section_close(data):
    ds = load(data)
    st = ds.stacks[0]
    errors = []
    for lab, places in st.locations().items():
        est = core.estimate_section(st, ds.template, lab)
        errors.append(abs(est[0] - places[0][0]))
    assert max(errors) <= 2


# -- saving ---------------------------------------------------------------------- #
def names_for(out_dir, data, tag=None):
    return core.output_names(out_dir, str(data / "template_landmarks.nii.gz"), tag)


def save(ds, out_dir, data, **kw):
    return core.save_session(names_for(out_dir, data), ds.df, ds.columns, ds.stacks, ds.template,
                             input_csv_dir=data, **kw)


def test_output_names_mirror_versioned_folder(tmp_path):
    n = core.output_names(tmp_path / "v1.4", "/x/v1.3/MEBRAINS_L_landmarks_v1.3.nii.gz")
    sec = core.Section(0, "11530-L", "11530", "L", 37, "/x/preprocessed/mr#rh#11530#L#musc#5768#37#L_crop.nii.gz")
    stack = core.SpecimenStack.__new__(core.SpecimenStack)
    stack.sub, stack.hemisphere = "11530", "L"
    assert n.section(sec, "nifti").name == "mr#rh#11530#L#musc#5768#37#L_crop_landmark.nii.gz"
    assert n.template().name == "MEBRAINS_L_landmarks_v1.4.nii.gz"
    assert n.hemi(stack).name == "11530_L_landmarks_v1.4.nii.gz"
    assert n.csv().name == "landmark_sections_v1.4.csv"
    assert core.output_names(tmp_path / "out", None).template().name == "template_landmarks.nii.gz"


def test_save_round_trip_and_hemisphere_files(tmp_path, data):
    ds = load(data)
    out = tmp_path / "v2.0"
    rep = save(ds, out, data)
    assert set(rep.hemi_written) == {"11530-L", "11539-L"}
    assert set(rep.hemi_skipped) == {"11539-R", "11543-L"}
    n = names_for(out, data)
    # the new CSV (bare label file names, raw paths rebased) loads back identically
    ds2 = load(data, n.csv(), n.template())
    assert np.array_equal(ds2.template.labels, ds.template.labels)
    for a, b in zip(ds.stacks, ds2.stacks):
        assert a.labels.keys() == b.labels.keys()
        assert all(np.array_equal(a.labels[k], b.labels[k]) for k in a.labels)
    for st in ds.stacks[:2]:
        img = nib.load(str(n.hemi(st)))
        assert set(np.unique(np.asanyarray(img.dataobj))) - {0} == st.all_labels()
        assert np.allclose(img.affine, ds.template.label_affine)


def test_brainbuilder_glob_finds_exactly_one_file(tmp_path, data):
    """BrainBuilder looks for '{raw_root}*.nii.gz' in the landmark directory."""
    ds = load(data)
    core.save_session(names_for(tmp_path, data), ds.df, ds.columns, ds.stacks, ds.template,
                      label_format="nifti", input_csv_dir=data)
    for st in ds.stacks:
        for idx in st.labels:
            root = core.strip_ext(st.sections[idx].raw)
            assert len(list(tmp_path.glob(f"{glob_escape(root)}*.nii.gz"))) == 1


def glob_escape(text):
    import glob
    return glob.escape(text)


def test_nifti_section_labels_keep_raw_geometry(tmp_path, data):
    ds = load(data)
    save(ds, tmp_path, data)
    st = ds.stacks[3]                                 # the NIfTI hemisphere
    idx = next(iter(st.labels))
    raw = nib.load(st.sections[idx].raw)
    out = nib.load(str(names_for(tmp_path, data).section(st.sections[idx], "nifti")))
    assert out.shape == raw.shape and np.allclose(out.affine, raw.affine)
    assert np.array_equal(np.asanyarray(out.dataobj), st.labels[idx])


def test_only_files_we_wrote_are_ever_deleted(tmp_path, data):
    ds = load(data)
    st = ds.stacks[0]
    n = names_for(tmp_path, data)
    labelled = sorted(st.labels)
    # a pre-existing file (not written by us) with a name we would produce for an empty section
    empty_idx = next(i for i in range(len(st)) if i not in st.labels)
    foreign = n.section(st.sections[empty_idx], "png")
    tmp_path.mkdir(exist_ok=True)
    foreign.write_bytes(b"not ours")
    save(ds, tmp_path, data)
    assert foreign.exists()                                   # never deleted
    ours = n.section(st.sections[labelled[0]], "png")
    assert ours.exists()
    st.labels[labelled[0]][:] = 0                             # erase a section we saved
    st.refresh(labelled[0])
    rep = save(ds, tmp_path, data)
    assert not ours.exists() and ours.name in rep.removed     # our own stale file is removed
    assert foreign.exists() and foreign.name in rep.kept_stale


def test_inputs_untouched(tmp_path, data):
    before = {p: p.stat().st_mtime_ns for p in data.rglob("*") if p.is_file()}
    ds = load(data)
    save(ds, tmp_path / "out", data)
    after = {p: p.stat().st_mtime_ns for p in data.rglob("*") if p.is_file()}
    assert before == after


def test_previous_session_found(tmp_path, data):
    ds = load(data)
    out = tmp_path / "v1.4"
    assert core.previous_session(out) is None
    save(ds, out, data)
    prev = core.previous_session(out)
    assert prev["csv_path"].endswith("landmark_sections_v1.4.csv")
    assert prev["template_labels_path"].endswith("template_landmarks_v1.4.nii.gz")


# -- csv builder ------------------------------------------------------------------- #
def test_make_section_csv(tmp_path):
    import make_section_csv
    listing = tmp_path / "files.txt"
    listing.write_text("preprocessed/mr#rh#11530#L#musc#5768#37#L_crop.nii.gz\n"
                       "preprocessed/mr#rh#11530#L#musc#5767#01#L_crop.nii.gz\n"
                       "preprocessed/mf#rh#11539#R#musc#5693#45#L_crop.nii.gz\n"
                       "v1.3/mr#rh#11530#L#musc#5767#01#L_crop_landmark.nii.gz\n"
                       "v1.3/MEBRAINS_L_landmarks_v1.3.nii.gz\n")
    out = tmp_path / "sections.csv"
    assert make_section_csv.main(["--root", str(tmp_path), "--file-list", str(listing),
                                  "--labels", "v1.3", "--out", str(out)]) == 0
    import pandas as pd
    df = pd.read_csv(out, dtype=str, keep_default_na=False)
    row = df[df.base == "mr#rh#11530#L#musc#5768#37#L_crop"].iloc[0]
    assert (row["sub"], row["hemisphere"], row["section"], row["ligand"]) == ("11530", "L", "37", "musc")
    assert row["labels"] == ""
    first = df[df.section == "1"].iloc[0]
    assert first["labels"] == "v1.3/mr#rh#11530#L#musc#5767#01#L_crop_landmark.nii.gz"
    assert core.resolve_columns(df.columns) ["section"] == "section"


# -- display flip ------------------------------------------------------------------ #
def test_flip_lr_is_display_only(tmp_path, data):
    ds = core.load_dataset(str(data / "sections.csv"), str(data / "template.nii.gz"),
                           str(data / "template_landmarks.nii.gz"), SPECIMENS, flip_lr=["11539-r"])
    flipped = {st.id: st for st in ds.stacks}
    assert flipped["11539-R"].display_name == "11539-R-flipped-lr"
    assert flipped["11530-L"].display_name == "11530-L"
    import landmark_qc
    st = flipped["11539-R"]
    lab, places = next(iter(st.locations().items()))
    cov = core.coverage(ds.stacks, ds.template)[lab][st.id]
    a = landmark_qc.section_panel(st, lab, cov)
    st.flip_lr = False
    b = landmark_qc.section_panel(st, lab, cov)
    assert np.array_equal(a["image"], b["image"][:, ::-1])
    assert a["pos"][0] == b["pos"][0] and a["pos"][1] == b["image"].shape[1] - 1 - b["pos"][1]
    assert a["title"].startswith("11539-R-flipped-lr")
    st.flip_lr = True
    # saved labels are identical to an unflipped session (original orientation on disk)
    save(ds, tmp_path / "f", data)
    ds_plain = load(data)
    save(ds_plain, tmp_path / "p", data)
    for name in sorted(p.name for p in (tmp_path / "p").glob("*_landmark.png")):
        from PIL import Image
        assert np.array_equal(np.asarray(Image.open(tmp_path / "f" / name)), np.asarray(Image.open(tmp_path / "p" / name)))


def test_flip_lr_unknown_specimen(data):
    with pytest.raises(ValueError, match="flip-lr"):
        core.load_dataset(str(data / "sections.csv"), str(data / "template.nii.gz"), None, SPECIMENS,
                          flip_lr=["11539-X"])
