"""Load unchanged educational sources from the pinned student repository."""
from pathlib import Path
import sys

VENDOR = Path(__file__).parent / 'vendor' / 'student_maxwell'
sys.path.insert(0, str(VENDOR.resolve()))
try:
    from symbolic_core import ExpressionDAG
    from maxwell_symbolic import maxwell_model
    from inference import ParameterVector, fit_one_parameter, relative_l2
    from analytic_benchmark import plane_waves, benchmark
finally:
    sys.path.remove(str(VENDOR.resolve()))
