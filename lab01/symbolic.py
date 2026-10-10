"""Scalar SymPy representation of all three spatial components."""
import sympy as sp


def symbolic_equations():
    x, y, z, t = sp.symbols("x y z t", real=True)
    coords = (x, y, z)
    velocity = sp.Matrix([sp.Function(name)(x, y, z, t) for name in ("u", "v", "w")])
    p = sp.Function("p")(x, y, z, t)
    nu, rho = sp.symbols("nu rho", positive=True)

    def curl(a):
        return sp.Matrix([sp.diff(a[2], y)-sp.diff(a[1], z),
                          sp.diff(a[0], z)-sp.diff(a[2], x),
                          sp.diff(a[1], x)-sp.diff(a[0], y)])

    def lap(a):
        return a.applyfunc(lambda v: sum(sp.diff(v, c, 2) for c in coords))

    omega = curl(velocity)
    stretch = velocity.jacobian(coords) * omega
    transport = omega.jacobian(coords) * velocity
    momentum_rhs = -velocity.jacobian(coords)*velocity - sp.Matrix([sp.diff(p, c)/rho for c in coords]) + nu*lap(velocity)
    return {"coordinates": (*coords, t), "u": velocity, "p": p, "nu": nu, "rho": rho,
            "omega": omega, "divergence": sum(sp.diff(velocity[i], coords[i]) for i in range(3)),
            "momentum_rhs": momentum_rhs, "vorticity_rhs": stretch-transport+nu*lap(omega),
            "inviscid_rhs": stretch-transport}
