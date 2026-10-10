"""Exact 3D+t verification field; never treated as experimental measurements."""
import numpy as np
import sympy as sp


def travelling_abc_symbolic(kappa=0, drift=sp.Rational(7, 10)):
    """Periodic, translated Beltrami solution of dimensionless Navier–Stokes.

    curl(A)=A and Laplacian(A)=-A. A uniform drift transports the pattern;
    exp(-kappa*t) supplies viscous decay. kappa=0 also solves the selected
    inviscid vorticity equation. The pressure uses unit density.
    """
    x, y, z, t = sp.symbols("x y z t", real=True)
    phase = x-drift*t
    a = sp.Matrix([sp.sin(z)+sp.cos(y), sp.sin(phase)+sp.cos(z),
                   sp.sin(y)+sp.cos(phase)])
    amplitude = sp.exp(-kappa*t)
    vortex = amplitude*a
    return {"coordinates": (x, y, z, t), "u": sp.Matrix([drift, 0, 0])+vortex,
            "omega": vortex, "p": -vortex.dot(vortex)/2,
            "kappa": sp.sympify(kappa), "drift": sp.sympify(drift)}


def travelling_abc_field(x, y, z, t, kappa=0.0, drift=0.7):
    """Evaluate all three velocity components; last array axis is (u,v,w)."""
    if kappa < 0:
        raise ValueError("kappa must be nonnegative")
    x, y, z, t = np.broadcast_arrays(x, y, z, t)
    phase = x-drift*t
    amplitude = np.exp(-kappa*t)
    return np.stack([drift+amplitude*(np.sin(z)+np.cos(y)),
                     amplitude*(np.sin(phase)+np.cos(z)),
                     amplitude*(np.sin(y)+np.cos(phase))], axis=-1)


def plot_abc_3d(times=(0.0, 1.0, 2.0), kappa=0.0, n=5):
    """Show spatial vectors at several times with fixed axes and color scale."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    if len(times) == 0 or n < 2:
        raise ValueError("Supply at least one time and at least two points per axis")
    q = np.linspace(0.6, 2*np.pi-0.6, n)
    x, y, z = np.meshgrid(q, q, q, indexing="ij")
    fields = [travelling_abc_field(x, y, z, time, kappa=kappa) for time in times]
    speeds = [np.linalg.norm(field, axis=-1) for field in fields]
    norm = Normalize(vmin=min(v.min() for v in speeds), vmax=max(v.max() for v in speeds))
    fig = plt.figure(figsize=(4.6*len(times)+0.6, 4.8), layout="constrained")
    axes = []
    for i, (time, field, speed) in enumerate(zip(times, fields, speeds), 1):
        ax = fig.add_subplot(1, len(times), i, projection="3d")
        axes.append(ax)
        ax.quiver(x, y, z, field[..., 0], field[..., 1], field[..., 2],
                  length=0.22, normalize=False, arrow_length_ratio=0.28,
                  color="#386b8b", linewidth=0.65)
        ax.scatter(x, y, z, c=speed, cmap="viridis", norm=norm, s=10, depthshade=False)
        ax.set(xlabel="x*", ylabel="y*", zlabel="z*", title=f"t* = {time:g}",
               xlim=(0, 2*np.pi), ylim=(0, 2*np.pi), zlim=(0, 2*np.pi))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_ticks([0, np.pi, 2*np.pi], ["0", "π", "2π"])
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=24, azim=-56)
    fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=axes,
                 shrink=0.65, label="|u*| (цвет точек)", pad=0.02)
    fig.suptitle("Проверочное аналитическое решение в 3D + время; κ = "
                 f"{kappa:g}\nЭто не экспериментальный след за цилиндром", fontsize=12)
    return fig
