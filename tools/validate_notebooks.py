"""Static checks on the generated notebooks.

Run after tools/build_notebooks.py. It parses every code cell and replays the
extractor patch on a scratch copy of the vendored file, so a broken string
replacement is caught here instead of after a five minute model download.

Usage:  python tools/validate_notebooks.py
"""

import ast
import json
import pathlib
import shutil
import sys
import tempfile
import textwrap

ROOT = pathlib.Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
VENDOR = ROOT / "vendor/SubspaceAD/src/subspacead/core/extractor.py"

NOTEBOOKS = ["01_subspacead_mvtec.ipynb", "02_ablations.ipynb"]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_syntax(path, nb):
    errors = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        try:
            ast.parse(src)
        except SyntaxError as exc:
            errors.append(f"{path.name} cell {i}: {exc}")
    return errors


def find_shim_cell(nb):
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "EXTRACTOR" in src and "output_attentions=True" in src:
            return src
    return None


def replay_patch():
    """Apply the same four replacements the notebook applies, on a scratch copy."""
    if not VENDOR.exists():
        return ["vendored extractor.py missing; run: git clone into vendor/"]
    tmp = pathlib.Path(tempfile.mkdtemp())
    target = tmp / "extractor.py"
    shutil.copy(VENDOR, target)
    src = target.read_text(encoding="utf-8")

    anchor = 'DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")'
    if src.count(anchor) != 1:
        return [f"anchor count {src.count(anchor)}, expected 1"]
    src = src.replace(anchor, anchor + textwrap.dedent("""
    
        # SHIM: attention maps are only needed for --bg_mask_method dino_saliency.
        _NEED_ATTN = False"""), 1)

    if src.count("output_attentions=True") != 1:
        return ["output_attentions=True not unique"]
    src = src.replace("output_attentions=True", "output_attentions=_NEED_ATTN", 1)

    if src.count("if attentions is None:") != 1:
        return ["'if attentions is None:' not unique"]
    src = src.replace("if attentions is None:", "if attentions is None and _NEED_ATTN:", 1)

    doc = '"""Extracts the DINO saliency mask from attention weights."""'
    if src.count(doc) != 1:
        return ["saliency docstring not unique"]
    src = src.replace(doc, doc + "\n        if not _NEED_ATTN:\n            return None", 1)

    target.write_text(src, encoding="utf-8")

    problems = []
    try:
        ast.parse(src)
    except SyntaxError as exc:
        problems.append(f"patched extractor does not parse: {exc}")

    for marker in ["_NEED_ATTN = False", "output_attentions=_NEED_ATTN",
                   "if attentions is None and _NEED_ATTN", "if not _NEED_ATTN"]:
        if marker not in src:
            problems.append(f"marker missing after patch: {marker}")
    return problems


def main():
    problems = []
    for name in NOTEBOOKS:
        path = NB_DIR / name
        if not path.exists():
            problems.append(f"{name}: not generated")
            continue
        nb = load(path)
        problems += check_syntax(path, nb)
        shim = find_shim_cell(nb)
        if shim is None:
            problems.append(f"{name}: no extractor patch cell found")
        n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
        print(f"{name:34s} {len(nb['cells']):3d} cells ({n_code} code)")

    problems += replay_patch()

    if problems:
        print("\nFAILED")
        for p in problems:
            print("  -", p)
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
