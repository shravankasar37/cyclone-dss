"""Build notebooks/train_intensity_colab.ipynb from notebooks/src/train_intensity_colab.py.

The shared modules in ml/ are inlined as the first code cell so the notebook is
self-contained on Colab and uses exactly the same code as the backend.
Run: python notebooks/build_notebook.py
"""
import os
import sys
import re
import nbformat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME = sys.argv[1] if len(sys.argv) > 1 else "train_intensity_colab"
SRC = os.path.join(ROOT, "notebooks", "src", f"{NAME}.py")
DST = os.path.join(ROOT, "notebooks", f"{NAME}.ipynb")
SHARED = ["ml/cyclone_scale.py", "ml/adt_features.py", "ml/gridsat.py", "ml/nets.py"]

text = open(SRC, encoding="utf-8").read()
chunks = re.split(r"^# %%(.*)$", text, flags=re.M)[1:]
cells = []
for header, body in zip(chunks[::2], chunks[1::2]):
    body = body.strip("\n")
    if "[markdown]" in header:
        cells.append(nbformat.v4.new_markdown_cell(re.sub(r"^# ?", "", body, flags=re.M)))
    else:
        cells.append(nbformat.v4.new_code_cell(body))

shared_src = "# 0. Shared definitions (inlined from the repo's ml/ package - do not edit here)\n"
for p in SHARED:
    shared_src += f"\n# ---- {p} ----\n" + open(os.path.join(ROOT, p), encoding="utf-8").read()
cells.insert(1, nbformat.v4.new_code_cell(shared_src))

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata = {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
               "kernelspec": {"name": "python3", "display_name": "Python 3"},
               "language_info": {"name": "python"}}
nbformat.write(nb, DST)
print("wrote", DST, len(cells), "cells")
