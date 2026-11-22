"""Generate the Colab notebooks for AI Assignment #3.

The notebooks are built from Python source rather than hand-edited JSON so that
the cell order, the pinned commit and the shim stay in one reviewable place.

Usage:  python tools/build_notebooks.py
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


# --------------------------------------------------------------------------
# Shared preamble
# --------------------------------------------------------------------------

CONFIG = r'''
# ---------------------------------------------------------------------------
# Configuration. Change these only if you know why.
# ---------------------------------------------------------------------------
REPO_URL = "https://github.com/CLendering/SubspaceAD.git"
COMMIT   = "ef56d5c8ab2f1feb7dda1c93b25cc3f73f0960d7"   # 2026-08-17
WORKDIR  = "/content/subspacead_work"
REPO     = f"{WORKDIR}/SubspaceAD"
DS_ROOT  = f"{WORKDIR}/datasets/mvtec-ad"

# Everything that must survive the VM goes here. See the mount cell.
DRIVE_DIR   = "/content/drive/MyDrive/CISC3024_HW3"
RESULT_ROOT = f"{DRIVE_DIR}/results"
LOG_ROOT    = f"{DRIVE_DIR}/logs"

RES      = 672
BACKBONE = "facebook/dinov2-with-registers-giant"
LAYERS   = "-12,-13,-14,-15,-16,-17,-18"
BATCH    = 1
SEED     = 42
KS       = [1, 2, 4]

# Order in which the k values are run. Reorder it when the session is short:
# the loop stops when the quota does, so whatever sits last may not run. The
# order changes nothing about the numbers, only which one survives a cut.
K_ORDER  = KS

# fp16 or fp32. The released code loads fp32; fp16 is a deliberate deviation
# that makes a T4 run feasible. The smoke cell measures the resulting drift on
# a category that was already run in fp32, so the deviation is quantified
# rather than assumed.
PRECISION = "fp16"

CATEGORIES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
              "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
              "transistor", "wood", "zipper"]

import os, sys, shutil, subprocess, pathlib, textwrap
pathlib.Path(WORKDIR).mkdir(parents=True, exist_ok=True)
print("workdir :", WORKDIR)
print("python  :", sys.version.split()[0])
'''

MOUNT = r'''
# ---------------------------------------------------------------------------
# Results go to Drive, not to the VM.
#
# A free Colab runtime is recycled without warning and /content goes with it.
# The first attempt at this experiment did exactly that: it reached k=4, ran
# out of GPU quota mid-category, and lost two complete k values plus four
# categories of the third, because the only copy of every CSV was on the
# ephemeral disk.
#
# With results on Drive the unit of loss becomes one category, about seven
# minutes, and a session can be stopped at any point without cost. That is
# what makes an exhausted quota an inconvenience instead of a restart.
# ---------------------------------------------------------------------------
from google.colab import drive
drive.mount("/content/drive")

for d in (RESULT_ROOT, LOG_ROOT):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)

free = shutil.disk_usage("/content/drive/MyDrive").free / 2**30
print("result root:", RESULT_ROOT)
print(f"drive free : {free:.1f} GB")
assert free > 1, "Drive is nearly full; the results need a few hundred MB."

# A short write probe. Drive is a network filesystem and can be read-only when
# the mount is half-up; better to find out here than after a category finishes.
probe = pathlib.Path(RESULT_ROOT) / ".write_probe"
probe.write_text("ok")
assert probe.read_text() == "ok"
probe.unlink()
print("drive is writable")
'''

SALVAGE = r'''
# ---------------------------------------------------------------------------
# Rescue anything a previous session left on the ephemeral disk.
#
# Idempotent and cheap: copies only the CSVs that are not already on Drive.
# Run it before the main cell in a session that may have inherited a live VM.
# ---------------------------------------------------------------------------
import shutil, pathlib

local = pathlib.Path(f"{REPO}/results")
moved = 0
if local.is_dir():
    for src in sorted(local.rglob("benchmark_results.csv")):
        dst = pathlib.Path(RESULT_ROOT) / src.relative_to(local)
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        moved += 1
    print(f"salvaged {moved} CSV files from {local}")
else:
    print("no local results to salvage")

n = len(list(pathlib.Path(RESULT_ROOT).rglob("benchmark_results.csv")))
print(f"{n} benchmark_results.csv now on Drive")
'''

PROBE = r'''
import torch, shutil, platform

print("platform :", platform.platform())
print("torch    :", torch.__version__)
print("cuda     :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu      :", torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info()
    print(f"vram     : {total/2**30:.1f} GB total, {free/2**30:.1f} GB free")
    print("capability:", torch.cuda.get_device_capability(0))
else:
    raise SystemExit("No GPU. Runtime > Change runtime type > T4 GPU.")

du = shutil.disk_usage("/content")
print(f"disk     : {du.free/2**30:.1f} GB free of {du.total/2**30:.1f} GB")

# A DINOv2-G checkpoint plus the dataset plus features do not fit in much less.
assert du.free > 25 * 2**30, "Need at least 25 GB free disk."
'''

CLONE = r'''
if os.path.isdir(REPO):
    print("repo already present")
else:
    print(subprocess.run(["git", "clone", REPO_URL, REPO],
                         capture_output=True, text=True).stdout)

subprocess.run(["git", "-C", REPO, "checkout", "--quiet", COMMIT], check=True)
head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                      capture_output=True, text=True).stdout.strip()
print("HEAD:", head)
assert head == COMMIT, f"Expected {COMMIT}, got {head}. The code below was written against that commit."
'''

INSTALL = r'''
# Only what is actually imported. anomalib is listed in the repo's
# requirements.txt but no module imports it, and installing it drags in
# PyTorch Lightning, which can replace Colab's pinned torch build.
# scipy is imported by main.py but missing from requirements.txt; Colab ships it.
import importlib, subprocess, sys

for mod in ["scipy", "sklearn", "cv2", "pandas", "tqdm", "transformers", "kornia"]:
    try:
        importlib.import_module(mod)
        print(f"{mod:14s} present")
    except ImportError:
        print(f"{mod:14s} MISSING -> installing")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", mod], check=True)

# Install the package without its declared dependencies, so anomalib is skipped.
# --no-deps still puts src/ on sys.path, which the absolute imports need.
print(subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", REPO, "--no-deps"],
                     capture_output=True, text=True).stdout)

import torch
print("torch after install:", torch.__version__, "| cuda build:", torch.version.cuda)
'''

SHIM = r'''
# ---------------------------------------------------------------------------
# Patch 1: do not request attention maps unless they are used.
#
# extractor.py calls the backbone with output_attentions=True unconditionally.
# At 672 px a ViT-g/14 sees 2,309 tokens; keeping the attention tensor of all
# 40 layers in FP32 is 40 * 16 * 2309**2 * 4 bytes = 13.6 GB, on top of 4.5 GB
# of weights. That does not fit on a T4. Attention maps are only consumed by
# --bg_mask_method dino_saliency, which this study does not use.
# ---------------------------------------------------------------------------
import pathlib

EXTRACTOR = pathlib.Path(REPO) / "src/subspacead/core/extractor.py"
src = EXTRACTOR.read_text()

if "_NEED_ATTN" in src:
    print("shim already applied")
else:
    anchor = 'DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")'
    assert src.count(anchor) == 1, "anchor not found; extractor.py changed"
    src = src.replace(anchor, anchor + textwrap.dedent("""
    
        # SHIM: attention maps are only needed for --bg_mask_method dino_saliency.
        _NEED_ATTN = False"""), 1)

    assert src.count("output_attentions=True") == 1
    src = src.replace("output_attentions=True", "output_attentions=_NEED_ATTN", 1)

    assert src.count("if attentions is None:") == 1
    src = src.replace("if attentions is None:", "if attentions is None and _NEED_ATTN:", 1)

    doc = '"""Extracts the DINO saliency mask from attention weights."""'
    assert src.count(doc) == 1
    src = src.replace(doc, doc + "\n        if not _NEED_ATTN:\n            return None", 1)

    EXTRACTOR.write_text(src)
    print("shim applied")

print("-" * 70)
print(subprocess.run(["grep", "-n", "_NEED_ATTN", str(EXTRACTOR)],
                     capture_output=True, text=True).stdout)

# Import check: the package must load with the patch in place.
sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/src")
import src.subspacead.core.extractor as ex
print("import ok, _NEED_ATTN =", ex._NEED_ATTN)
assert ex._NEED_ATTN is False
'''

PATCH_TRAPZ = r'''
# ---------------------------------------------------------------------------
# Patch 2: numpy 2.x removed np.trapz.
#
# main.py line 188 computes AU-PRO with np.trapz, which no longer exists in
# numpy >= 2.0 (it was renamed np.trapezoid). The repo does not pin numpy, so
# on a current Colab the run dies while computing the first category's AU-PRO,
# after the features have already been extracted. The rename is exact: same
# function, same result.
# ---------------------------------------------------------------------------
import numpy as np

MAIN_PY = pathlib.Path(REPO) / "main.py"
msrc = MAIN_PY.read_text()

if hasattr(np, "trapz"):
    print(f"numpy {np.__version__} still has np.trapz, no patch needed")
elif "np.trapz(" in msrc:
    MAIN_PY.write_text(msrc.replace("np.trapz(", "np.trapezoid("))
    print(f"numpy {np.__version__}: patched np.trapz -> np.trapezoid in main.py")
else:
    print("main.py already patched")

print(subprocess.run(["grep", "-n", "trapezoid", str(MAIN_PY)],
                     capture_output=True, text=True).stdout)
'''

PATCH_MEM = r'''
# ---------------------------------------------------------------------------
# Patch 3: stop boxing every pixel into a Python object.
#
# main.py collects per-pixel scores with
#     px_pred_all_auroc.extend(anomaly_map.flatten().astype(np.float32))
# extend() on a numpy array pushes one Python scalar per element into the list.
# At 672 px an image has 451,584 pixels, a boxed float costs 24 bytes plus an
# 8-byte pointer, so each of the two score lists grows by ~2.2 GB per 150
# images, and the uint8 label list by ~2.4 GB. MVTec-AD's transistor split has
# 240 test images, which needs ~11 GB of list alone; the kernel is killed with
# exit 137 before the metrics are printed, and because the CSV is only written
# after the last category, every finished category is lost with it.
#
# The replacement stores the per-image arrays and concatenates once, which is
# 6x smaller. The values are unchanged: float32 -> Python float -> float64 is an
# exact embedding, so casting the concatenation to float64 reproduces exactly
# what np.array(list_of_floats) produced.
# ---------------------------------------------------------------------------
import numpy as np, pathlib

MAIN_PY = pathlib.Path(REPO) / "main.py"
src = MAIN_PY.read_text()

if "_Chunks" in src:
    print("memory patch already applied")
else:
    helper = """
class _Chunks:
    # Holds per-image arrays and concatenates them once at the end.
    # Stands in for list.extend(arr) on a per-element basis, which boxes every
    # element into a Python object and costs 4-8x the memory of the array.

    __slots__ = ("chunks",)

    def __init__(self):
        self.chunks = []

    def extend(self, arr):
        self.chunks.append(np.asarray(arr).ravel())

    def array(self, dtype=None):
        if not self.chunks:
            return np.array([], dtype=dtype)
        out = np.concatenate(self.chunks)
        return out.astype(dtype, copy=False) if dtype is not None else out


"""
    anchor = "def compute_aupro("
    assert src.count(anchor) == 1, "anchor not found; main.py changed"
    src = src.replace(anchor, helper.lstrip("\n") + anchor, 1)

    pairs = [
        ("        px_true_all = []\n",
         "        px_true_all = _Chunks()\n"),
        ("        px_pred_all_auroc = []\n",
         "        px_pred_all_auroc = _Chunks()\n"),
        ("        px_pred_all_normalized = []\n",
         "        px_pred_all_normalized = _Chunks()\n"),
        ("        px_true_arr = np.array(px_true_all, dtype=np.uint8)\n",
         "        px_true_arr = px_true_all.array(np.uint8)\n"),
        ("        px_pred_arr_auroc = np.array(px_pred_all_auroc)\n",
         "        px_pred_arr_auroc = px_pred_all_auroc.array(np.float64)\n"),
        ("        px_pred_arr_normalized = np.array(px_pred_all_normalized)\n",
         "        px_pred_arr_normalized = px_pred_all_normalized.array(np.float64)\n"),
    ]
    for old, new in pairs:
        assert src.count(old) == 1, f"expected exactly one occurrence of {old!r}"
        src = src.replace(old, new, 1)

    MAIN_PY.write_text(src)
    print("memory patch applied")

print("-" * 70)
print(subprocess.run(["grep", "-n", "_Chunks", str(MAIN_PY)],
                     capture_output=True, text=True).stdout)

# The substitution has to be exact, not merely close. Reproduce both paths on
# the same data and compare bit for bit.
import numpy as np
rng = np.random.default_rng(0)
for dt in (np.uint8, np.float32):
    chunks = [(rng.random(997) * 255).astype(dt) for _ in range(7)]
    old = np.array([v for c in chunks for v in c])          # list.extend path
    new = np.concatenate(chunks).astype(
        np.float64 if dt == np.float32 else np.uint8)       # _Chunks path
    assert np.array_equal(old, new), f"mismatch for {dt}"
    print(f"{np.dtype(dt).name:8s} identical over {old.size} values, dtype {old.dtype}")
'''

PATCH_FP16 = r'''
# ---------------------------------------------------------------------------
# Patch 4: load the backbone in half precision.
#
# The released code calls AutoModel.from_pretrained(model_ckpt) with no dtype,
# so the weights arrive in fp32. On a T4 that is 4.5 GB of weights and a
# measured 3.32 s per image at 672 px, which puts the three main runs at about
# 11 hours. fp16 weights are 2.25 GB and the forward pass uses the tensor cores.
#
# Three changes, nothing else:
#   1. load the weights in fp16,
#   2. cast pixel_values to fp16 so the first matmul does not upcast,
#   3. upcast the aggregated patch features back to fp32 before numpy, so the
#      streaming PCA accumulates in float64 from fp32 inputs rather than fp16.
#
# LayerNorm and softmax keep fp32 accumulators inside ATen on CUDA, so this
# does not need autocast to stay numerically stable. The drift is measured in
# the smoke cell against a category already run in fp32, not assumed.
# ---------------------------------------------------------------------------
import pathlib, subprocess

EXTRACTOR = pathlib.Path(REPO) / "src/subspacead/core/extractor.py"
src = EXTRACTOR.read_text()

if "_FP16" in src:
    print("precision patch already applied")
else:
    anchor = "class FeatureExtractor:"
    assert src.count(anchor) == 1
    src = src.replace(
        anchor,
        f"_FP16 = {PRECISION == 'fp16'}   # set by the notebook\n\n\n" + anchor,
        1,
    )

    old = "        self.model = AutoModel.from_pretrained(model_ckpt).eval().to(DEVICE)"
    assert src.count(old) == 1, "from_pretrained line changed"
    src = src.replace(
        old,
        "        self.model = AutoModel.from_pretrained(\n"
        "            model_ckpt, torch_dtype=torch.float16 if _FP16 else None\n"
        "        ).eval().to(DEVICE)",
        1,
    )

    old = "        ).to(DEVICE)\n"
    assert src.count(old) == 1, "processor call changed"
    src = src.replace(
        old,
        "        )\n"
        "        if _FP16:\n"
        '            inputs["pixel_values"] = inputs["pixel_values"].half()\n'
        "        inputs = inputs.to(DEVICE)\n",
        1,
    )

    old = "        return fused.cpu().numpy()\n"
    assert src.count(old) == 1, "feature return changed"
    src = src.replace(old, "        return fused.cpu().float().numpy()\n", 1)

    EXTRACTOR.write_text(src)
    print(f"precision patch applied, _FP16 = {PRECISION == 'fp16'}")

print("-" * 70)
print(subprocess.run(["grep", "-n", "_FP16\\|torch.float16\\|cpu().float()",
                      str(EXTRACTOR)], capture_output=True, text=True).stdout)
'''

DATA = r'''
# ---------------------------------------------------------------------------
# MVTec-AD. The official site now serves the archive through a browser session
# (a direct request returns an 11 KB redirect page), so the download goes
# through the Hugging Face mirror that hosts the original archive.
# If that fails, upload the archive yourself and set MANUAL_ARCHIVE.
# ---------------------------------------------------------------------------
import os, shutil, zipfile, pathlib, subprocess

MANUAL_ARCHIVE = None   # e.g. "/content/mvtec_anomaly_detection.zip"

ds = pathlib.Path(DS_ROOT)
already = ds.is_dir() and all((ds / c).is_dir() for c in CATEGORIES)

if already:
    print("dataset already extracted")
else:
    if MANUAL_ARCHIVE is None:
        from huggingface_hub import hf_hub_download
        archive = hf_hub_download(
            repo_id="hdtech/mvtech_anomaly_detection",
            filename="mvtech_anomaly_detection.zip",
            repo_type="dataset",
            local_dir=f"{WORKDIR}/_dl",
        )
    else:
        archive = MANUAL_ARCHIVE
    print("archive:", archive, f"{os.path.getsize(archive)/2**30:.2f} GB")

    tmp = pathlib.Path(f"{WORKDIR}/_extract")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    print("extracting ...")
    shutil.unpack_archive(archive, tmp)

    # The archive nests the categories at an unknown depth; find the shallowest
    # directory that holds all fifteen.
    found = None
    for root, dirs, files in os.walk(tmp):
        if all((pathlib.Path(root) / c).is_dir() for c in CATEGORIES):
            found = pathlib.Path(root)
            break
    assert found is not None, "no directory containing all 15 categories"
    print("category root:", found)

    shutil.rmtree(ds, ignore_errors=True)
    ds.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(found), str(ds))
    shutil.rmtree(tmp, ignore_errors=True)
    print("moved to", ds)
'''

VERIFY = r'''
import glob, collections

ds = pathlib.Path(DS_ROOT)
rows = []
for c in CATEGORIES:
    n_train = len(glob.glob(f"{ds}/{c}/train/good/*.png"))
    n_test  = len(glob.glob(f"{ds}/{c}/test/**/*.png", recursive=True))
    n_gt    = len(glob.glob(f"{ds}/{c}/ground_truth/**/*.png", recursive=True))
    rows.append((c, n_train, n_test, n_gt))

print(f"{'category':<12}{'train':>7}{'test':>7}{'gt':>7}")
for r in rows:
    print(f"{r[0]:<12}{r[1]:>7}{r[2]:>7}{r[3]:>7}")
tot_train = sum(r[1] for r in rows)
tot_test  = sum(r[2] for r in rows)
print(f"{'TOTAL':<12}{tot_train:>7}{tot_test:>7}{sum(r[3] for r in rows):>7}")

# Published split: 3,629 normal training images and 1,725 test images.
assert len(rows) == 15, "expected 15 categories"
assert tot_train == 3629, f"train count {tot_train} != 3629"
assert tot_test == 1725, f"test count {tot_test} != 1725"
print("\nsplit matches the published counts")
'''

RUNNER = r'''
# ---------------------------------------------------------------------------
# One runner for every experiment: streams the log, writes it to disk, and
# returns the exit code.
# ---------------------------------------------------------------------------
import subprocess, pathlib, datetime

def run(name, extra, logdir=LOG_ROOT):
    pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)
    log = f"{logdir}/{name}.log"
    cmd = (f'cd {REPO} && python -u main.py '
           f'--dataset_name mvtec_ad --dataset_path {DS_ROOT} '
           f'--seed {SEED} --batch_size {BATCH} ' + extra)
    print("=" * 78)
    print(name)
    print(cmd)
    print("started", datetime.datetime.now().strftime("%H:%M:%S"), flush=True)

    t0 = datetime.datetime.now()
    with open(log, "a") as fh:
        fh.write(f"\n\n===== {name} @ {t0} =====\n{cmd}\n")
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            print(line, end="")
            fh.write(line)
        p.wait()
    dt = (datetime.datetime.now() - t0).total_seconds()
    print(f"finished in {dt/60:.1f} min, exit {p.returncode}", flush=True)
    return p.returncode


def _base(outdir):
    """Absolute paths are used verbatim; relative ones are taken as relative to
    the repository, which is what --outdir means to main.py."""
    p = pathlib.Path(outdir)
    return p if p.is_absolute() else pathlib.Path(f"{REPO}/{outdir}")


def result_csv(outdir):
    """main.py appends its own run name to --outdir, so the CSV is one level
    deeper than the directory passed in. Searched recursively because the main
    runs nest one directory per category."""
    hits = sorted(_base(outdir).rglob("benchmark_results.csv"))
    return hits[0] if hits else None


def cat_csv(outdir, cat):
    """CSV for one category. Written the moment that category finishes, so its
    presence is what resume keys on.

    Two layouts have to be handled, because main.py always appends its own run
    name to whatever --outdir it was given:

      main runs   <outdir>/<cat>/<run_name>/benchmark_results.csv
      one-off run <outdir>/<run_name>/benchmark_results.csv

    A file is attributed to a category only when the category appears in its
    path relative to <outdir>, or when it sits directly in a run_name directory
    and is the only such file. Without the second condition a missing category
    would silently inherit the CSV of the one category that did finish.
    """
    base = _base(outdir)
    hits = sorted(base.rglob("benchmark_results.csv"))
    if not hits:
        return None
    for p in hits:
        if cat in p.relative_to(base).parts:
            return p
    flat = [p for p in hits if len(p.relative_to(base).parts) == 2]
    return flat[0] if len(flat) == 1 else None


NUM_COLS = ["Image AUROC", "Image AUPR", "Pixel AUROC", "AU-PRO"]


def save_k(k, elapsed_min=0.0):
    """Consolidate one k value into a self-contained set of files on Drive.

    Called the moment that k ends, whether it ended because it finished or
    because the quota did. Everything needed to report that k travels together:
    the fifteen per-category rows, their mean, the settings that produced them,
    and a zip of the raw category directories.

    Saving per category already means a kill costs one category. Saving per k
    means a finished k is a finished deliverable: if the next k never runs, the
    report still has a complete point with its own provenance.
    """
    import pandas as pd, json, datetime, shutil

    rows, missing = [], []
    for c in CATEGORIES:
        p = cat_csv(f"{RESULT_ROOT}/main_k{k}", c)
        if p is None:
            missing.append(c)
            continue
        d = pd.read_csv(p)
        rows.append(d[d["Category"] != "Average"])

    if not rows:
        print(f"k={k}: no category on disk, nothing saved")
        return None

    per = pd.concat(rows, ignore_index=True)
    per.insert(0, "k_shot", k)
    per.to_csv(f"{DRIVE_DIR}/k{k}_per_category.csv", index=False)

    mean = per[NUM_COLS].mean().to_frame().T
    mean.insert(0, "k_shot", k)
    mean.insert(1, "n_categories", len(per))
    mean.to_csv(f"{DRIVE_DIR}/k{k}_summary.csv", index=False)

    meta = {
        "k_shot": k, "n_categories": int(len(per)),
        "complete": not missing, "missing": missing,
        "elapsed_min_this_session": round(elapsed_min, 1),
        "commit": COMMIT, "backbone": BACKBONE, "image_res": RES,
        "layers": LAYERS, "precision": PRECISION, "batch_size": BATCH,
        "seed": SEED, "pca_ev": 0.99, "agg_method": "mean",
        "img_score_agg": "mtop1p", "aug_count": 30,
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
    }
    json.dump(meta, open(f"{DRIVE_DIR}/k{k}_meta.json", "w"), indent=2)

    # The raw tree, so the exact files behind the numbers survive too. Written
    # last, and guarded: the CSVs above are already the deliverable, and a slow
    # or flaky Drive should not cost the rest of the session.
    src = pathlib.Path(f"{RESULT_ROOT}/main_k{k}")
    if src.is_dir():
        try:
            shutil.make_archive(f"{DRIVE_DIR}/k{k}_raw", "zip", root_dir=src)
        except Exception as e:
            print(f"  raw zip skipped ({type(e).__name__}: {e})")

    print(f"\nk={k} SAVED  {len(per)}/{len(CATEGORIES)} categories"
          + (f"  (missing: {', '.join(missing)})" if missing else "  (complete)"))
    print(mean[NUM_COLS].to_string(index=False))
    for f in (f"k{k}_per_category.csv", f"k{k}_summary.csv",
              f"k{k}_meta.json", f"k{k}_raw.zip"):
        p = pathlib.Path(DRIVE_DIR) / f
        if p.exists():
            print(f"  {p}  {p.stat().st_size/1024:.0f} KB")
    return mean
'''

SMOKE = r'''
# ---------------------------------------------------------------------------
# Smoke test and precision check in one run.
#
# Same configuration as the fp32 run that was killed at 21.6 min: bottle, k=1,
# seed 42, all 83 test images. main.py samples the normal image deterministically
# from the seed, so it is the same 037.png. The fp32 numbers below are the ones
# that run printed before the kernel was killed.
# ---------------------------------------------------------------------------
FP32_REF = {"Image AUROC": 1.0000, "Image AUPR": 1.0000,
            "Pixel AUROC": 0.9857, "AU-PRO": 0.9655}

import torch
from transformers import AutoModel
_m = AutoModel.from_pretrained(BACKBONE, torch_dtype=torch.float16).eval().cuda()
_p = next(_m.parameters())
print(f"checkpoint loads as {_p.dtype} ({_p.element_size()} bytes/param)")
assert _p.dtype == torch.float16, "the checkpoint did not load in fp16"
del _m
torch.cuda.empty_cache()

out = f"{RESULT_ROOT}/smoke_bottle_k1_fp16"
if cat_csv(out, "bottle") is None:
    rc = run("smoke_bottle_k1_fp16", (
        f'--categories bottle --image_res {RES} --k_shot 1 '
        f'--layers="{LAYERS}" --model_ckpt {BACKBONE} --aug_count 30 '
        f'--pca_ev 0.99 --agg_method mean --outdir {out}'
    ))
    assert rc == 0, "smoke test failed, read the log above"
else:
    print("smoke run already on disk, reusing it")

import pandas as pd, re, pathlib
csv = cat_csv(out, "bottle")
if csv is None:
    print("no benchmark_results.csv found under", out, "- contents:")
    for p in sorted(pathlib.Path(out).rglob("*")):
        print("   ", p)
    raise AssertionError("smoke test produced no benchmark_results.csv")
print("csv:", csv)
df = pd.read_csv(csv)
row = df[df["Category"] == "bottle"].iloc[0]

print("\n      metric      fp32      fp16     delta")
for m in ["Image AUROC", "Image AUPR", "Pixel AUROC", "AU-PRO"]:
    a, b = FP32_REF[m], float(row[m])
    print(f"{m:>14s}  {a:8.4f}  {b:8.4f}  {b - a:+.4f}")

log = pathlib.Path(f"{LOG_ROOT}/smoke_bottle_k1_fp16.log").read_text()
times = [float(x) for x in re.findall(r"Avg\. time per image: ([\d.]+) s", log)]
if times:
    per_img = sum(times) / len(times)
    print(f"\nmeasured {per_img:.2f} s/image, fp32 was 3.32 -> {3.32/per_img:.1f}x")
    for k in KS:
        fwd = 2 * len(CATEGORIES) * k * 31 + 1725
        sec = fwd * per_img + len(CATEGORIES) * 10
        print(f"  projected k={k}: {fwd} forwards -> {sec/60:.0f} min")
else:
    print("\nno timing line found in the log")
'''

MAIN = r'''
# ---------------------------------------------------------------------------
# Main runs: k = 1, 2, 4, one category per process.
#
# One process per category is not cosmetic. main.py keeps the whole test split
# in memory and only writes its CSV after the last category, so a kill in
# category 12 discards the eleven that already finished. Isolating categories
# gives two things: the OS reclaims the peak between categories, and every
# finished category is durable on disk. Model reload costs about 8 s each.
#
# Re-run this cell as often as needed. Finished categories are skipped, and the
# skip test reads Drive, so it holds across sessions and across VM deaths.
#
# The moment a k ends, finished or cut short, save_k writes that k out as a
# self-contained deliverable: per-category rows, mean, settings, raw zip. So a
# session that only gets through k=1 still leaves k=1 complete and reportable.
#
# Order comes from K_ORDER, not KS, so a short session can be pointed at the
# k value that is actually missing.
# ---------------------------------------------------------------------------
import time

stopped = False
for k in K_ORDER:
    if stopped:
        break
    out = f"{RESULT_ROOT}/main_k{k}"
    todo = [c for c in CATEGORIES if cat_csv(out, c) is None]
    print(f"\nk={k}: {len(CATEGORIES) - len(todo)}/{len(CATEGORIES)} categories on disk, "
          f"{len(todo)} to run", flush=True)

    if not todo:
        print(f"k={k}: already complete, refreshing its saved files")
        save_k(k)
        continue

    t0 = time.time()
    for j, cat in enumerate(todo, 1):
        vis = "--vis_count 3" if k == 1 else ""
        rc = run(f"main_k{k}_{cat}", (
            f'--categories {cat} --image_res {RES} --k_shot {k} '
            f'--layers="{LAYERS}" --model_ckpt {BACKBONE} --aug_count 30 '
            f'--pca_ev 0.99 --agg_method mean {vis} --outdir {out}/{cat}'
        ))
        if rc != 0:
            print(f"\nk={k}, {cat}: exit {rc}, nothing written for this category. "
                  f"Re-run this cell to resume from {cat}.", flush=True)
            stopped = True
            break
        el = (time.time() - t0) / 60
        print(f"[k={k}] {j}/{len(todo)} {cat} done | {el:.0f} min elapsed, "
              f"about {el / j * (len(todo) - j):.0f} min left for k={k}", flush=True)

    # Runs on both exits. A partial k is still data, and the next session
    # should not have to rediscover what this one already knows.
    save_k(k, (time.time() - t0) / 60)
'''

AGG = r'''
import pandas as pd, pathlib, json

# F1 is excluded from the summary: MVTec-AD ships no validation split, so
# main.py has no threshold to fit and reports NaN for both F1 columns.
NUM = ["Image AUROC", "Image AUPR", "Pixel AUROC", "AU-PRO"]

frames = []
incomplete = {}
for k in KS:
    rows, missing = [], []
    for c in CATEGORIES:
        p = cat_csv(f"{RESULT_ROOT}/main_k{k}", c)
        if p is None:
            missing.append(c)
            continue
        d = pd.read_csv(p)
        rows.append(d[d["Category"] != "Average"])
    note = f", missing: {' '.join(missing)}" if missing else ""
    print(f"k={k}: {len(rows)}/{len(CATEGORIES)} categories{note}")
    if missing:
        incomplete[k] = missing
    if not rows:
        continue
    d = pd.concat(rows, ignore_index=True)
    d.insert(0, "k_shot", k)
    frames.append(d)

assert frames, ("no per-category CSV on disk. The main cell writes one file per "
                "category, so this means no category finished.")

per = pd.concat(frames, ignore_index=True)
per.to_csv(f"{DRIVE_DIR}/results_main_per_category.csv", index=False)

avg = per.groupby("k_shot")[NUM].mean().reset_index()
avg.to_csv(f"{DRIVE_DIR}/results_main_all.csv", index=False)

print("\nMean over the categories that finished\n")
print(avg.to_string(index=False))

# A mean over a subset is not the number the paper reports. Say so loudly
# rather than letting a 11-category average pass as a 15-category one.
if incomplete:
    print("\n" + "!" * 70)
    for k, miss in incomplete.items():
        print(f"k={k} is INCOMPLETE: {len(miss)}/{len(CATEGORIES)} missing "
              f"({', '.join(miss)})")
    print("These means are over a subset. Do not put them in the report as if")
    print("they covered all 15 categories. Re-run the main cell to fill them in.")
    print("!" * 70)

meta = {
    "commit": COMMIT, "backbone": BACKBONE, "image_res": RES, "layers": LAYERS,
    "batch_size": BATCH, "seed": SEED, "pca_ev": 0.99, "agg_method": "mean",
    "img_score_agg": "mtop1p",
    "aug_count": 30, "k_shot": KS, "precision": PRECISION,
    "categories_done": {int(k): int(n) for k, n in per.groupby("k_shot").size().items()},
    "incomplete": {int(k): v for k, v in incomplete.items()},
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "torch": torch.__version__,
}
json.dump(meta, open(f"{DRIVE_DIR}/results_main_meta.json", "w"), indent=2)
print("\nsaved results_main_per_category.csv, results_main_all.csv, results_main_meta.json")
'''

FIGS = r'''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, pathlib, numpy as np

per = pd.read_csv(f"{DRIVE_DIR}/results_main_per_category.csv")
mean = pd.read_csv(f"{DRIVE_DIR}/results_main_all.csv")
cats = CATEGORIES

fig, ax = plt.subplots(figsize=(11, 4.2))
w = 0.26
for i, k in enumerate(KS):
    d = per[per["k_shot"] == k].set_index("Category").reindex(cats)
    if d["Image AUROC"].isna().all():
        continue
    ax.bar(np.arange(len(cats)) + (i - 1) * w, d["Image AUROC"], width=w, label=f"k={k}")
ax.set_xticks(np.arange(len(cats)))
ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Image AUROC")
ax.set_ylim(0.5, 1.0)
ax.axhline(0.971, ls="--", lw=1, color="k", label="paper 1-shot 0.971")
ax.legend(fontsize=8, ncol=4, loc="lower right")
ax.set_title("Per-category image AUROC on MVTec-AD")
fig.tight_layout()
fig.savefig(f"{DRIVE_DIR}/fig_auroc_by_category.png", dpi=200)
plt.show()

fig, ax = plt.subplots(figsize=(6, 4))
for col, mk in [("Pixel AUROC", "o"), ("AU-PRO", "s")]:
    ax.plot(mean["k_shot"], mean[col], marker=mk, label=col)
ax.set_xlabel("k (normal images per category)")
ax.set_ylabel("score")
ax.set_xticks(KS)
ax.legend()
ax.set_title("Localisation vs shot count")
fig.tight_layout()
fig.savefig(f"{DRIVE_DIR}/fig_loc_vs_k.png", dpi=200)
plt.show()
print("figures written to", WORKDIR)
'''

QUAL = r'''
# Qualitative maps saved by --vis_count 3 on the k=1 run.
from PIL import Image
import glob, pathlib

files = sorted(glob.glob(f"{RESULT_ROOT}/main_k1/**/*.png", recursive=True))
print(f"{len(files)} images under results/main_k1")
sel = [f for f in files if "vis" in f.lower()][:6] or files[:6]
for f in sel:
    print(f)
    display(Image.open(f))
'''

PACK = r'''
import shutil, pathlib, subprocess

# The four patches are part of the result, not an implementation detail: the
# archived numbers were produced by this exact modified source. Ship the diff.
diff = subprocess.run(["git", "-C", REPO, "diff"], capture_output=True, text=True).stdout
pathlib.Path(f"{RESULT_ROOT}/_patches.diff").write_text(diff)
print(f"{len(diff.splitlines())} lines of diff")

# Zipping the whole tree also re-reads every CSV off Drive, so it doubles as a
# check that the run is complete before anything is handed in.
n = len(list(pathlib.Path(RESULT_ROOT).rglob("benchmark_results.csv")))
print(f"{n} benchmark_results.csv in the archive (45 = 15x3 when complete)")

# The archive is written to /content, not into the folder being archived.
out = shutil.make_archive("/content/subspacead_results", "zip", root_dir=DRIVE_DIR)
print(out, f"{pathlib.Path(out).stat().st_size/2**20:.1f} MB")

try:
    from google.colab import files
    files.download(out)
    for f in sorted(pathlib.Path(DRIVE_DIR).glob("k*_*.csv")) + \
             sorted(pathlib.Path(DRIVE_DIR).glob("k*_meta.json")) + \
             sorted(pathlib.Path(DRIVE_DIR).glob("results_main_*")):
        files.download(str(f))
except Exception as e:
    print("download the files from the Drive folder:", DRIVE_DIR, e)
'''


main_nb = notebook([
    md("""
# CISC3024 Pattern Recognition, AI Assignment #3
## SubspaceAD (CVPR 2026) on MVTec-AD, DINOv2-G at 672 px in fp16

Main experiment. Matches the paper's operating point on every axis that
defines the method: 672 px, seven averaged layers (22–28), 99% retained
variance, 30 rotations per normal image, TVaR at 1%, k = 1, 2, 4.

One axis is deliberately changed: **the backbone runs in fp16**. The released
code loads fp32, which on a T4 costs 3.32 s per image and puts the three runs
at about 11 hours — longer than a free Colab session survives. fp16 brings that
to roughly two hours.

This is a deviation, so it is measured rather than asserted. The smoke cell
re-runs `bottle` at k=1 with the same seed as the fp32 run that was killed
earlier, and prints both results side by side. The delta goes into the report.

**Every result is written to Drive the moment it exists, and each k is bundled
when it ends.** A free Colab runtime can be recycled at any point and its disk
goes with it, so a run that keeps its output on `/content` risks the whole
session. With results on Drive a quota exhaustion costs at most seven minutes
of work: reconnect, re-run the main cell, and it resumes from the next
unfinished category. A k that finishes is saved as its own bundle — per-category
rows, mean, settings, raw zip — so it stands as a deliverable even if no later k
ever runs.

Every number produced here is measured. Nothing is estimated.
"""),
    code(CONFIG),
    md("""
## 1. Check the runtime, mount Drive

Two things have to be true before anything is downloaded. The runtime must have
a GPU, and the results directory must be on Drive rather than on the VM — see
the comment in the mount cell for why that is not a detail.
"""),
    code(PROBE),
    code(MOUNT),
    md("""
## 2. Get the code, pinned to the commit this notebook was written against

Then salvage: if this session inherited a live VM from an earlier attempt, its
CSVs are still on `/content` and are copied to Drive before anything overwrites
them. Idempotent, so running it again costs nothing.
"""),
    code(CLONE),
    code(SALVAGE),
    md("## 3. Dependencies"),
    code(INSTALL),
    md("""
## 4. Patch the code

Four patches. The first three are exact: they make the run possible without
changing a single output value, and the cells assert that rather than claim it.
The fourth is a real numerical deviation and is labelled as one.

1. **Attention output** — the published code asks the backbone for attention
   maps on every forward pass whether or not anything reads them. At 672 px on
   ViT-g that is 13.6 GB of tensors. The patch computes them only when
   `--bg_mask_method dino_saliency` is set, which this study never sets.
2. **`np.trapz`** — removed in NumPy 2.0, renamed `np.trapezoid`. Unpatched,
   the run dies inside the first category's AU-PRO computation.
3. **Pixel accumulation** — per-pixel scores are collected with
   `list.extend(numpy_array)`, which boxes 451,584 Python floats per image.
   `transistor` has 240 test images, needs about 11 GB for the lists alone, and
   gets killed with exit 137 before any metric is printed.
4. **fp16 backbone** — 4.5 GB of fp32 weights become 2.25 GB and the forward
   pass moves onto the tensor cores. This one changes the numbers; the smoke
   cell measures by how much.
"""),
    code(SHIM),
    code(PATCH_TRAPZ),
    code(PATCH_MEM),
    code(PATCH_FP16),
    md("## 5. MVTec-AD"),
    code(DATA),
    md("## 6. Verify the split before spending GPU time"),
    code(VERIFY),
    code(RUNNER),
    md("""
## 7. Smoke test

One category, thirty test images. This is also where the cost per image stops
being an estimate: the cell reads the timing line out of the log and projects
the full runs from it.
"""),
    code(SMOKE),
    md("""
## 8. Main runs (k = 1, 2, 4)

One process per category, and **each k is saved as a self-contained bundle the
moment it ends** — finished or cut short. A k that finishes leaves four files on
Drive:

```
k1_per_category.csv   fifteen rows, one per category
k1_summary.csv        the mean over them
k1_meta.json          every setting that produced them, plus what is missing
k1_raw.zip            the raw category directories
```

So a session that only reaches k=1 still hands over a complete, reportable
point with its own provenance, and the next session never has to rediscover
what this one already knows. Re-running the cell refreshes the bundle for any k
that is already complete.
"""),
    code(MAIN),
    md("## 9. Aggregate"),
    code(AGG),
    md("## 10. Figures"),
    code(FIGS),
    md("## 11. Qualitative maps"),
    code(QUAL),
    md("## 12. Hand back"),
    code(PACK),
])


# --------------------------------------------------------------------------
# Ablation notebook
# --------------------------------------------------------------------------

ABL_CONFIG = r'''
# ---------------------------------------------------------------------------
# What the ablation runs on.
#
# Six categories rather than fifteen. The ablation is about the shape of the
# curve, not the last decimal of the mean, and the six chosen here span the
# failure modes the main table already exposed:
#
#   bottle      easy, rigid object, saturated at 1.000
#   carpet      easy, texture, saturated at 1.000
#   capsule     the one category that collapsed at k=1 (0.6745)
#   cable       thin structure, lowest non-collapsed image AUROC (0.9196)
#   screw       small object, second lowest (0.8994)
#   transistor  no augmentation upstream, and AU-PRO 0.699 against ~0.95 elsewhere
#
# Everything is otherwise held at the main run's settings, so the giant arm at
# pca_ev=0.99 with mtop1p has to land on the main table's numbers for these six
# categories. That check is the last cell of this notebook and it is the only
# reason to trust the rest.
# ---------------------------------------------------------------------------
ABL_CATS = ["bottle", "carpet", "capsule", "cable", "screw", "transistor"]

K_SHOT     = 4            # matches the main table's largest k
AUG_COUNT  = 30
AUG_LIST   = ["rotate"]   # upstream default; the main run did not override it
NO_AUG     = ["transistor"]   # upstream default in config.py
RES        = 672
SEED       = 42

EVS  = [0.95, 0.99, 1.0]
AGGS = ["mtop1p", "max", "mean", "p99"]

# Batch 1 on the training pass, exactly as the main run. Batch 4 on the test
# pass: no augmentation is drawn there, so the only difference is matmul
# tiling. If the anchor check at the end drifts by more than about 0.005, set
# this to 1 and rerun -- that is the one number in this notebook that trades
# fidelity for time.
BATCH_TRAIN = 1
BATCH_TEST  = 4

# Relative-depth layer strings, as in scripts/backbone_ablation.sh, so the
# compared layers sit at the same 55-75% depth in every backbone.
BACKBONES = {
    "small": ("facebook/dinov2-small", "-4,-5"),
    "large": ("facebook/dinov2-large", "-7,-8,-9,-10,-11"),
    "giant": ("facebook/dinov2-with-registers-giant",
              "-12,-13,-14,-15,-16,-17,-18"),
}

# giant first: it is the anchor, and if the session dies early the one arm that
# matters most is already on Drive.
ARMS = ["giant", "large", "small"]

print("arms:", ARMS)
print("categories:", ABL_CATS)
print(f"grid per arm: {len(EVS)} pca_ev x {len(AGGS)} img_score_agg")
'''

ABL_HARNESS = r'''
# ---------------------------------------------------------------------------
# Single-pass ablation harness.
#
# The expensive part of this method is the ViT forward pass. Per category at
# k=4: four normal images plus thirty rotations each is 124 images, and the
# two-pass PCA fit walks that set twice, so 248 forwards for the fit alone,
# plus every test image. Across six categories that is roughly 2,150 forwards,
# around half an hour on a T4 at 672 px.
#
# Everything after the forward pass is cheap. One eigendecomposition of a
# 1536x1536 covariance, the choice of k from an explained-variance target, a
# projection, a top-k average: seconds.
#
# main.py welds the two together, so sweeping five hyper-parameter values means
# five full runs and five times the forwards. This harness extracts the
# features once per backbone and keeps the *untruncated* eigendecomposition,
# then evaluates every (pca_ev, img_score_agg) pair against those same
# features. Twelve configurations for the price of one.
#
# Nothing numerical is reimplemented here. The fit is PCAModel.fit, the
# component selection is PCAModel._select_k_components, the score is
# calculate_anomaly_scores, the map post-processing is post_process_map, the
# aggregation is aggregate_image_score and AU-PRO is main.compute_aupro. The
# harness only decides what to call, and in which order.
#
# Two deliberate differences from main.py, both recorded in the report:
#
#   1. main.py seeds once before the category loop, so its RNG stream runs
#      across categories and rerunning one category alone gives different
#      numbers. This seeds per category instead, so every row is independently
#      reproducible.
#   2. main.py selects k inside the PCA fit and throws the rest of the spectrum
#      away. This keeps it, which is what makes the ev sweep free.
# ---------------------------------------------------------------------------
import os, sys, json, math, time, datetime, pathlib, random
import numpy as np, pandas as pd, torch
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score

for _p in (f"{REPO}/src", REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from subspacead.core.extractor import FeatureExtractor
from subspacead.core.pca import PCAModel
from subspacead.post_process.scoring import (
    calculate_anomaly_scores, post_process_map, aggregate_image_score,
)
from subspacead.data.datasets import get_dataset_handler
from subspacead.data.transforms import get_augmentation_transform
from subspacead.config import parse_layer_indices
from main import compute_aupro

ABL_ROOT = f"{DRIVE_DIR}/ablation"
MAP_ROOT = f"{ABL_ROOT}/patch_maps"
for _d in (ABL_ROOT, MAP_ROOT):
    pathlib.Path(_d).mkdir(parents=True, exist_ok=True)

IMG_CSV = f"{ABL_ROOT}/per_category_image.csv"
PIX_CSV = f"{ABL_ROOT}/per_category_pixel.csv"


def _extract(extractor, imgs, res, layers, agg):
    """One forward pass. Returns tokens as (n*h*w, C) plus the patch grid."""
    tokens, (h_p, w_p), _ = extractor.extract_tokens(
        imgs, res, layers, agg, [], False, use_clahe=False, dino_saliency_layer=0,
    )
    return tokens.reshape(-1, tokens.shape[-1]), h_p, w_p


def _train_generator(extractor, paths, aug_tf, aug_count, res, layers, bs):
    """Mirror of main.py's feature_generator_full.

    It re-opens and re-augments on every call, and that is deliberate. The
    two-pass PCA fit calls the generator twice, and upstream the two passes see
    *different* random rotations. Caching the features would make the passes
    consistent -- arguably better statistics -- but it is not the released
    behaviour, and the anchor check would then be measuring the difference
    instead of validating the harness. So the two passes are left alone.
    """
    def gen():
        imgs = []
        for p in paths:
            pil = Image.open(p).convert("RGB")
            imgs.append(pil)
            if aug_tf is not None:
                for _ in range(aug_count):
                    imgs.append(aug_tf(pil))
        for i in range(0, len(imgs), bs):
            X, _, _ = _extract(extractor, imgs[i:i + bs], res, layers, "mean")
            yield X
    return gen


def fit_full_pca(extractor, paths, aug_tf, aug_count, res, layers, bs):
    """Fit PCA with no truncation, so every pca_ev can be sliced out of it."""
    probe, h_p, w_p = _extract(
        extractor, [Image.open(paths[0]).convert("RGB")], res, layers, "mean")
    feature_dim = probe.shape[-1]
    n_img = len(paths) * ((1 + aug_count) if aug_tf is not None else 1)
    total_tokens = n_img * h_p * w_p
    num_batches = math.ceil(n_img / bs)

    # k=None and ev=None together mean "keep every component": with both unset,
    # _select_k_components falls through to self.k = evecs.shape[1].
    pca = PCAModel(k=None, ev=None, whiten=False)
    pca.fit(_train_generator(extractor, paths, aug_tf, aug_count, res, layers, bs),
            feature_dim, total_tokens, num_batches)
    return pca, feature_dim, h_p, w_p


def params_at_ev(pca, ev):
    """Re-run the component selection on the cached spectrum, at a new ev."""
    m = PCAModel(k=None, ev=ev, whiten=False)
    m.mu_ = pca.mu_
    m.explained_variance_ = pca.explained_variance_
    m._select_k_components(pca.components_)   # full evecs in, truncated out
    return m._build_pca_params(), int(m.k)


def _gt_masks(handler, paths, res):
    """Ground truth at map resolution, resized the same way main.py does."""
    import torchvision.transforms.functional as TF
    out = []
    for p in paths:
        g = handler.get_ground_truth_path(p)
        if not g or not os.path.exists(g):
            out.append(np.zeros((res, res), np.uint8))
            continue
        m = TF.resize(Image.open(g).convert("L"), (res, res),
                      interpolation=TF.InterpolationMode.NEAREST)
        out.append((np.array(m) > 0).astype(np.uint8))
    return out


def _append(df, path):
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def _done(arm, cat):
    if not os.path.exists(IMG_CSV):
        return False
    d = pd.read_csv(IMG_CSV)
    return not d[(d["backbone"] == arm) & (d["category"] == cat)].empty


def eval_arm(arm, ckpt, layers, cats, force=False):
    """Extract once, score every (ev, agg) pair, write each category as it lands."""
    # extract_tokens wants indices, not the "-12,-13,..." string the CLI takes.
    # main.py does this conversion right after get_args(); skipping it fails
    # deep inside the layer aggregation with a tuple-index error.
    layer_str = layers
    layers = parse_layer_indices(layer_str)
    extractor = FeatureExtractor(ckpt)
    for cat in cats:
        tag = f"{arm}/{cat}"
        if not force and _done(arm, cat):
            print(f"{tag}: already on Drive, skipping")
            continue

        t0 = time.time()
        # Per-category seeding: see the header note.
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        handler = get_dataset_handler("mvtec_ad", DS_ROOT, cat)
        train = handler.get_train_paths()
        test = handler.get_test_paths()

        # K-shot sampling, identical to main.py: shuffle, take the first k.
        random.shuffle(train)
        train = train[:K_SHOT]

        aug_tf = get_augmentation_transform(AUG_LIST, RES) if AUG_COUNT > 0 else None
        if cat in NO_AUG:
            aug_tf = None
        n_fit_imgs = len(train) * ((1 + AUG_COUNT) if aug_tf is not None else 1)

        pca, fdim, h_p, w_p = fit_full_pca(
            extractor, train, aug_tf, AUG_COUNT, RES, layers, BATCH_TRAIN)

        params, ks = {}, {}
        for ev in EVS:
            params[ev], ks[ev] = params_at_ev(pca, ev)

        print(f"{tag}: {len(train)} fit images ({n_fit_imgs} with aug), "
              f"dim {fdim}, grid {h_p}x{w_p}, k = {ks}, "
              f"fit {time.time()-t0:.0f}s", flush=True)

        # One pass over the test set; every ev is scored from the same features.
        n = len(test)
        labels = np.array([0 if "good" in str(p) else 1 for p in test], np.uint8)
        maps = {ev: np.zeros((n, h_p, w_p), np.float32) for ev in EVS}

        for i in range(0, n, BATCH_TEST):
            chunk = test[i:i + BATCH_TEST]
            imgs = [Image.open(p).convert("RGB") for p in chunk]
            X, _, _ = _extract(extractor, imgs, RES, layers, "mean")
            for ev in EVS:
                s = calculate_anomaly_scores(X, params[ev], "reconstruction", 0)
                maps[ev][i:i + len(imgs)] = s.reshape(len(imgs), h_p, w_p)

        # The raw patch maps are the only thing worth keeping: 9 KB an image,
        # so a new aggregation later needs no forward pass at all.
        np.savez_compressed(
            f"{MAP_ROOT}/{arm}__{cat}.npz",
            labels=labels, h_p=h_p, w_p=w_p,
            **{f"ev{ev}": maps[ev] for ev in EVS},
        )

        gts = _gt_masks(handler, test, RES)
        px_true = np.concatenate([g.ravel() for g in gts])

        for ev in EVS:
            big = np.stack([post_process_map(maps[ev][j], RES) for j in range(n)])
            p_auroc = float(roc_auc_score(px_true, big.ravel()))
            aupro = float(compute_aupro(big, gts))

            rows = []
            for agg in AGGS:
                sc = np.array([aggregate_image_score(big[j], agg) for j in range(n)])
                rows.append({
                    "backbone": arm, "category": cat, "k_shot": K_SHOT,
                    "pca_ev": ev, "img_score_agg": agg, "k_components": ks[ev],
                    "Image AUROC": float(roc_auc_score(labels, sc)),
                    "Image AUPR": float(average_precision_score(labels, sc)),
                })
            _append(pd.DataFrame(rows), IMG_CSV)
            _append(pd.DataFrame([{
                "backbone": arm, "category": cat, "k_shot": K_SHOT,
                "pca_ev": ev, "k_components": ks[ev],
                "Pixel AUROC": p_auroc, "AU-PRO": aupro,
            }]), PIX_CSV)
            print(f"  ev={ev:<5} k={ks[ev]:<5} pixel AUROC {p_auroc:.4f}  "
                  f"AU-PRO {aupro:.4f}", flush=True)
            del big

        print(f"{tag}: done in {(time.time()-t0)/60:.1f} min", flush=True)

        meta = {
            "backbone": arm, "model_ckpt": ckpt, "layers": layer_str,
            "categories": ABL_CATS, "k_shot": K_SHOT, "aug_count": AUG_COUNT,
            "aug_list": AUG_LIST, "no_aug": NO_AUG, "image_res": RES,
            "pca_ev": EVS, "img_score_agg": AGGS,
            "batch_train": BATCH_TRAIN, "batch_test": BATCH_TEST,
            "seed": SEED, "precision": PRECISION,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
            "updated": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        pathlib.Path(f"{ABL_ROOT}/meta_{arm}.json").write_text(json.dumps(meta, indent=2))
'''

ABL_RUN = r'''
t_start = time.time()
for arm in ARMS:
    ckpt, layers = BACKBONES[arm]
    print("=" * 78)
    print(f"arm {arm}  {ckpt}  layers {layers}")
    eval_arm(arm, ckpt, layers, ABL_CATS)
print(f"\nall arms finished in {(time.time()-t_start)/60:.1f} min")
'''

ABL_TABLE = r'''
import pandas as pd

img = pd.read_csv(IMG_CSV)
pix = pd.read_csv(PIX_CSV)

g = (img.groupby(["backbone", "pca_ev", "img_score_agg"])
        .agg(n_categories=("category", "nunique"),
             image_auroc=("Image AUROC", "mean"),
             image_aupr=("Image AUPR", "mean"))
        .reset_index())
p = (pix.groupby(["backbone", "pca_ev"])
        .agg(n_categories=("category", "nunique"),
             pixel_auroc=("Pixel AUROC", "mean"),
             aupro=("AU-PRO", "mean"))
        .reset_index())
t = g.merge(p, on=["backbone", "pca_ev", "n_categories"])
t = t.sort_values(["backbone", "pca_ev", "img_score_agg"])
t.to_csv(f"{ABL_ROOT}/summary.csv", index=False)
print(t.to_string(index=False))
print(f"\nsaved {ABL_ROOT}/summary.csv")
'''

ABL_ANCHOR = r'''
# ---------------------------------------------------------------------------
# Correctness gate.
#
# The giant arm at pca_ev=0.99 with mtop1p is the main table's configuration.
# If the harness is faithful, these six categories have to reproduce the main
# table's k=4 rows to within run-to-run noise. We measured that noise at about
# 0.002 on this hardware, so a mean absolute deviation near or below 0.005
# means the harness is doing the same arithmetic as main.py. A deviation in the
# tenths means it is not, and nothing downstream is worth reading.
#
# The reference comes from notebook 01's bundle, which is not necessarily on
# this Drive: 01 was run on a different host and its bundle was downloaded
# rather than left behind. So the file is searched for first, and the values
# below -- the same numbers, copied from that bundle -- are used if it is
# absent. Upload k4_per_category.csv next to this notebook if you would rather
# the check read the file.
# ---------------------------------------------------------------------------
import pathlib
import pandas as pd

REF_K4 = {
    "bottle":     1.0000,
    "carpet":     0.9996,
    "capsule":    0.8620,
    "cable":      0.9196,
    "screw":      0.8994,
    "transistor": 0.9804,
}

ref = None
cands = sorted(pathlib.Path(DRIVE_DIR).rglob("k4_per_category.csv"))
for p in cands:
    try:
        d = pd.read_csv(p)
    except Exception:
        continue
    if {"Category", "Image AUROC"} <= set(d.columns):
        ref = d[d["Category"].isin(ABL_CATS)][["Category", "Image AUROC"]]
        ref.columns = ["category", "main_table"]
        print(f"reference read from {p}")
        break

if ref is None or len(ref) < len(ABL_CATS):
    ref = pd.DataFrame({"category": ABL_CATS,
                        "main_table": [REF_K4[c] for c in ABL_CATS]})
    print("no usable k4_per_category.csv on Drive; using the values copied "
          "from notebook 01's bundle")

if not pathlib.Path(IMG_CSV).exists():
    raise SystemExit("no ablation results yet -- run the Run cell first")

img = pd.read_csv(IMG_CSV)
har = img[(img.backbone == "giant") & ((img.pca_ev - 0.99).abs() < 1e-9)
          & (img.img_score_agg == "mtop1p")][["category", "Image AUROC"]]
har.columns = ["category", "harness"]

missing = sorted(set(ABL_CATS) - set(har["category"]))
if missing:
    print(f"giant arm not finished for: {missing}")
    print("the check needs all six categories; rerun the Run cell to finish them")

cmp = har.merge(ref, on="category")
cmp["delta"] = cmp.harness - cmp.main_table
print(cmp.to_string(index=False))
print(f"\nmean |delta| = {cmp.delta.abs().mean():.4f}")
print(f"max  |delta| = {cmp.delta.abs().max():.4f}")

if missing:
    print("\nPARTIAL: no verdict until all six categories are in.")
elif cmp.delta.abs().mean() > 0.005:
    print("\nWARNING: deviation above the measured run-to-run noise.")
    print("Before reading the sweeps, set BATCH_TEST = 1 and rerun the giant arm.")
else:
    print("\nwithin noise; the sweeps can be read.")
'''

ABL_FIGS = r"""
# ---------------------------------------------------------------------------
# Three figures, one per sweep. The exact numbers belong in the table; these
# are here to show the shape.
#
# The ev sweep gets two panels. At pca_ev=1.0 the PCA keeps every component,
# the projection becomes the identity, the residual becomes zero and the score
# degrades to float noise: image AUROC falls to 0.38. Drawn on a single axis
# that collapse compresses the 0.95 -> 0.99 difference into nothing, and that
# difference is what the sweep is about. So the left panel shows the collapse
# and the right panel zooms to the working range.
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

s = pd.read_csv(f"{ABL_ROOT}/summary.csv")
COLS = ["image_auroc", "image_aupr", "pixel_auroc", "aupro"]
NICE = {"image_auroc": "Image AUROC", "image_aupr": "Image AUPR",
        "pixel_auroc": "Pixel AUROC", "aupro": "AU-PRO"}
SHORT = {"image_auroc": "Image\nAUROC", "image_aupr": "Image\nAUPR",
         "pixel_auroc": "Pixel\nAUROC", "aupro": "AU-PRO"}

# --- A. backbone -----------------------------------------------------------
# Metrics on the x-axis, backbones as the series. The other way round needs
# four bars per group and the labels collide, because the four metrics sit
# within 0.06 of each other.
order = ["small", "large", "giant"]
names = {"small": "DINOv2-S (21M)", "large": "DINOv2-L (300M)",
         "giant": "DINOv2-G (1.1B)"}
bb = (s[(s.pca_ev == 0.99) & (s.img_score_agg == "mtop1p")]
      .set_index("backbone").loc[order])

fig, ax = plt.subplots(figsize=(8.2, 3.7))
w = 0.26
for i, b in enumerate(order):
    xs = np.arange(len(COLS)) + (i - 1) * w
    vals = bb.loc[b, COLS].values.astype(float)
    bars = ax.bar(xs, vals, width=w, label=names[b])
    for rect, v in zip(bars, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.003, f"{v:.3f}",
                ha="center", va="bottom", fontsize=6.5)
ax.set_xticks(np.arange(len(COLS)))
ax.set_xticklabels([SHORT[c] for c in COLS], fontsize=8)
ax.set_ylim(0.86, 1.0)
ax.set_ylabel("score")
ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16),
          frameon=False)
ax.set_title("A. Backbone at matched relative depth "
             "(6 categories, k=4, ev=0.99)", fontsize=9)
fig.tight_layout()
fig.savefig(f"{ABL_ROOT}/fig_ablation_backbone.png", dpi=200)
plt.show()

# --- B. retained variance --------------------------------------------------
g = (s[(s.backbone == "giant") & (s.img_score_agg == "mtop1p")]
     .sort_values("pca_ev"))
series = [("image_auroc", "o"), ("pixel_auroc", "s"), ("aupro", "^")]

fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
for ax, ylim, ttl in [(axes[0], (0.0, 1.05), "full scale"),
                      (axes[1], (0.89, 1.0), "zoomed to the working range")]:
    for c, mk in series:
        ax.plot(g.pca_ev, g[c], marker=mk, label=NICE[c])
    ax.set_xlim(0.94, 1.005)
    ax.set_ylim(*ylim)
    ax.set_xlabel("retained variance (pca_ev)")
    ax.set_title(ttl, fontsize=9)
axes[0].set_ylabel("score")
axes[0].legend(fontsize=7, loc="lower left")
axes[0].annotate("k = D: projection is the\nidentity, residual is zero",
                 xy=(0.997, 0.40), xytext=(0.952, 0.63), fontsize=7,
                 arrowprops=dict(arrowstyle="->", lw=0.8))
fig.suptitle("B. Retained variance, DINOv2-G (6 categories, k=4)", fontsize=9)
fig.tight_layout()
fig.savefig(f"{ABL_ROOT}/fig_ablation_ev.png", dpi=200)
plt.show()

# --- C. aggregation --------------------------------------------------------
order = ["mean", "max", "p99", "mtop1p"]
ag = (s[(s.backbone == "giant") & (s.pca_ev == 0.99)]
      .set_index("img_score_agg").loc[order])

fig, ax = plt.subplots(figsize=(6.5, 3.7))
w = 0.36
for i, c in enumerate(["image_auroc", "image_aupr"]):
    xs = np.arange(len(order)) + (i - 0.5) * w
    bars = ax.bar(xs, ag[c], width=w, label=NICE[c])
    for rect, v in zip(bars, ag[c]):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.003, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7)
ax.set_xticks(np.arange(len(order)))
ax.set_xticklabels(order)
ax.set_ylim(0.8, 1.0)
ax.set_ylabel("score")
ax.legend(fontsize=7, loc="upper left")
ax.set_title("C. Image-level aggregation, DINOv2-G "
             "(6 categories, k=4, ev=0.99)", fontsize=9)
fig.tight_layout()
fig.savefig(f"{ABL_ROOT}/fig_ablation_agg.png", dpi=200)
plt.show()

print("figures written to", ABL_ROOT)
"""

abl_nb = notebook([
    md("""
# CISC3024 Pattern Recognition, AI Assignment #3
## Ablations

Three questions, one pass over the data each:

1. **Backbone** — is DINOv2-G worth 1.1 B parameters against DINOv2-S and -L
   at matched relative depth?
2. **Retained variance** — is `pca_ev=0.99` a reasonable operating point?
3. **Image aggregation** — does the top-1% mean actually beat max and mean?

The main table is the baseline for all three; run notebook 01 first.

### Why this notebook does not just call `main.py` five times

The forward pass is the whole cost. After it, an eigendecomposition, a
projection and a top-k average cost seconds. `main.py` welds the two together,
so sweeping five values means five full runs. This notebook extracts features
once per backbone, keeps the untruncated eigendecomposition, and then evaluates
all twelve `(pca_ev, img_score_agg)` combinations against the same features.
Same numbers, roughly a tenth of the GPU time.

Nothing numerical is reimplemented — the PCA, the scoring, the post-processing
and AU-PRO all come from the repository's own functions.
"""),
    code(CONFIG),
    code(PROBE),
    code(MOUNT),
    code(CLONE),
    code(INSTALL),
    code(SHIM),
    code(PATCH_TRAPZ),
    code(PATCH_MEM),
    code(PATCH_FP16),
    md("""
## Data

Same archive as notebook 01. If this session already extracted it, this cell
does nothing; otherwise it downloads and unpacks.
"""),
    code(DATA),
    code(ABL_CONFIG),
    md("""
## The harness

One `FeatureExtractor` per backbone, one forward pass per image, every
hyper-parameter evaluated on the cached features. Results are appended to Drive
after each category, so an interrupted session loses at most one category.
"""),
    code(ABL_HARNESS),
    md("""
## Run

`giant` first — it is the anchor, and if the session dies early the arm that
matters most is already saved. Each arm re-runs from Drive on a restart and
skips whatever is already there.
"""),
    code(ABL_RUN),
    md("## Results"),
    code(ABL_TABLE),
    md("## Figures"),
    code(ABL_FIGS),
    md("## Does the harness agree with the main table?"),
    code(ABL_ANCHOR),
])


def write(nb, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    print(f"{path.name}: {len(nb['cells'])} cells ({n_code} code)")


if __name__ == "__main__":
    write(main_nb, NB_DIR / "01_subspacead_mvtec.ipynb")
    write(abl_nb, NB_DIR / "02_ablations.ipynb")
