"""Execute with this interpreter: python -m lab02.execute_notebook."""
from pathlib import Path
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
for key,folder in [('MPLCONFIGDIR','mpl'),('IPYTHONDIR','ipython'),('JUPYTER_RUNTIME_DIR','jupyter')]:
    os.environ.setdefault(key,str(ROOT/'tmp/lab02'/folder))
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

if __name__=='__main__':
    path=ROOT/'lab02/lab02_microwave.ipynb'
    km=KernelManager()
    km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    notebook=nbformat.read(path,as_version=4)
    NotebookClient(notebook,timeout=300,km=km,resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(notebook,path)
    print('Executed',sum(c.cell_type=='code' for c in notebook.cells),'code cells')
