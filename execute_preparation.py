"""Execute all preparation cells with this project's Python interpreter."""
from pathlib import Path
import os
import sys
import argparse

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/lab01-matplotlib")
os.environ.setdefault("JUPYTER_RUNTIME_DIR", "/private/tmp/lab01-jupyter")
os.environ.setdefault("IPYTHONDIR", "/private/tmp/lab01-ipython")

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

root = Path(__file__).resolve().parent
path = root / "lab01_preparation.ipynb"
km = KernelManager()
km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
nb = nbformat.read(path, as_version=4)
parser=argparse.ArgumentParser()
parser.add_argument('--fenics-only',action='store_true',help='Refresh only the FEniCSx report/plots using completed CLI runs')
args=parser.parse_args()
if args.fenics_only:
    indices=[i for i,c in enumerate(nb.cells) if c.cell_type=='code']
    target=next(i for i in indices if 'RECOMPUTE_FENICSX=False' in nb.cells[i].source)
    partial=nbformat.v4.new_notebook(cells=[nb.cells[indices[0]],nb.cells[target]])
    NotebookClient(partial,timeout=120,km=km,resources={"metadata":{"path":str(root)}}).execute()
    count=nb.cells[target].execution_count
    nb.cells[target]=partial.cells[-1]
    nb.cells[target].execution_count=count
else:
    NotebookClient(nb, timeout=120, km=km, resources={"metadata": {"path": str(root)}}).execute()
nbformat.write(nb, path)
print("Executed", 2 if args.fenics_only else sum(c.cell_type == "code" for c in nb.cells), "code cells")
