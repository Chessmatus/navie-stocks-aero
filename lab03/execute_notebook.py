"""Execute using the invoking project's Python, not a global Jupyter kernel."""
from pathlib import Path
import sys
import os
ROOT=Path(__file__).resolve().parents[1]
for key,name in [('MPLCONFIGDIR','mpl'),('IPYTHONDIR','ipython'),('JUPYTER_RUNTIME_DIR','jupyter')]:
    os.environ.setdefault(key,str(ROOT/'tmp/lab03'/name))
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


if __name__=='__main__':
    path=ROOT/'lab03/lab03_acoustic_scattering.ipynb'
    notebook=nbformat.read(path,as_version=4)
    km=KernelManager();km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    NotebookClient(notebook,timeout=300,km=km,resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(notebook,path)
    print('Executed',sum(c.cell_type=='code' for c in notebook.cells),'code cells')
