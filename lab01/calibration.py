"""Inverse-problem utilities; an actual Navier-Stokes solve is supplied by caller.

These functions do not implement a PDE solver and perform no data downloads.
"""
import numpy as np
from scipy.optimize import least_squares


def _velocity_pair(predicted, observed):
    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if predicted.shape != observed.shape or observed.ndim < 2 or observed.shape[-1] != 3:
        raise ValueError("Expected equal velocity arrays of shape (..., 3)")
    if not observed.size or not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("Velocities must be nonempty and finite; mask invalid points beforehand")
    return predicted, observed


def velocity_residual(predicted, observed, reference_velocity):
    """Flatten dimensionless component residuals for interior observations only."""
    predicted, observed = _velocity_pair(predicted, observed)
    if not np.isfinite(reference_velocity) or reference_velocity <= 0:
        raise ValueError("reference_velocity must be finite and positive")
    return ((predicted - observed) / reference_velocity).ravel()


def velocity_metrics(predicted, observed):
    """RMSE in m/s and relative L2 error; relative error is undefined for zero data."""
    predicted, observed = _velocity_pair(predicted, observed)
    error = predicted - observed
    axes = tuple(range(error.ndim - 1))
    denominator = np.linalg.norm(observed.ravel())
    return {
        "rmse_m_per_s": float(np.sqrt(np.mean(error**2))),
        "component_rmse_m_per_s": np.sqrt(np.mean(error**2, axis=axes)).tolist(),
        "relative_l2": float(np.linalg.norm(error.ravel()) / denominator) if denominator else float("nan"),
        "observation_vectors": observed.size // 3,
    }


def fit_viscosity(solve_at_measurements, observed, nu_reference, reference_velocity,
                  eta_bounds=(-np.inf, np.inf), max_nfev=40):
    """Fit nu=nu_reference*exp(eta) with TRF and a genuine PDE callback.

    solve_at_measurements(nu) must rerun the PDE with the same initial/boundary
    data and return H[u_nu] at the selected interior points/times, excluding
    the initial frame. Data selection and pressure handling belong to caller.
    Bounds in eta must contain zero, the initial reference value.
    """
    if not callable(solve_at_measurements):
        raise TypeError("Supply a callable implementing the PDE solve and observation operator")
    if not np.isfinite(nu_reference) or nu_reference <= 0:
        raise ValueError("nu_reference must be finite and positive")
    observed = np.asarray(observed, dtype=float)
    velocity_residual(observed, observed, reference_velocity)

    def residual(eta):
        nu = float(nu_reference * np.exp(eta[0]))
        if not np.isfinite(nu) or nu <= 0:
            raise ValueError("Viscosity left the finite positive range; tighten eta_bounds")
        return velocity_residual(solve_at_measurements(nu), observed, reference_velocity)

    result = least_squares(residual, x0=[0.0], bounds=eta_bounds, method="trf",
                           loss="linear", max_nfev=max_nfev)
    return float(nu_reference * np.exp(result.x[0])), result
