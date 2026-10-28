"""A low-loss TEM reduction of Maxwell; coefficients have physical units.

Convention E(x,t)=Re[E0 exp(-gamma*x) exp(+i*omega*t)].
The small-loss conductor term contributes equally to alpha and beta.
This is an effective propagation model, not a reconstruction of the full tuner.
"""
import numpy as np
from scipy.optimize import least_squares
from .upstream import ExpressionDAG

C = 299792458.0
F0 = 1e10
PARAMETERS = ('n', 'a', 'b')
NULL = np.array([1., 0., 0.])


def design(frequency):
    f = np.asarray(frequency, dtype=float)
    if f.ndim != 1 or not np.all(np.isfinite(f)) or np.any(f <= 0):
        raise ValueError('Need finite positive frequencies')
    u = f / F0
    zero = np.zeros_like(f)
    return np.stack((np.stack((zero, np.sqrt(u), u), -1),
                     np.stack((2*np.pi*f/C, np.sqrt(u), zero), -1)), 1)


def predict(frequency, theta):
    return design(frequency) @ np.asarray(theta)


def gamma(frequency, theta):
    p = predict(frequency, theta)
    return p[:, 0] + 1j*p[:, 1]


def field(x, time, frequency, theta):
    """Normalized E_y and H_z; exact source-free monochromatic Maxwell field.

    epsilon_eff(f) = c²(beta²-alpha²)/omega²,
    sigma(f)/epsilon0 = 2*c²*alpha*beta/omega. Dispersion is evaluated per tone.
    H normalization follows the student: e=sqrt(eps0)*E, h=sqrt(mu0)*H.
    """
    omega = 2*np.pi*frequency
    g = gamma(np.array([frequency]), theta)[0]
    e = np.exp(-g*np.asarray(x)[None, :] + 1j*omega*np.asarray(time)[:, None])
    h = C*g/(1j*omega)*e
    return e.real, h.real


def fit(f, y, scale, loss='linear', fixed=None):
    """Scaled least squares or SciPy TRF soft-L1; signed diagnostic coefficients.

    Signed a,b avoid a constrained bootstrap pileup at zero. A negative fitted
    coefficient is not interpreted as physical gain or negative conductivity.
    """
    x = design(f)
    y = np.asarray(y, dtype=float)
    s = np.broadcast_to(np.asarray(scale, dtype=float), y.shape)
    if y.shape != x.shape[:2] or np.any(s <= 0) or not np.all(np.isfinite(y)):
        raise ValueError('Incompatible or nonfinite data/scale')
    if loss not in ('linear', 'soft_l1'):
        raise ValueError('Unsupported loss')
    fixed = {} if fixed is None else dict(fixed)
    free = np.array([i for i in range(3) if i not in fixed], dtype=int)
    p = np.array([fixed.get(i, 0.) for i in range(3)])
    matrix = (x/s[..., None]).reshape(-1, 3)
    target = ((y-x@p)/s).ravel()
    columns = np.linalg.norm(matrix[:, free], axis=0)
    a = matrix[:, free] / columns
    if np.linalg.matrix_rank(a) < len(free):
        raise ValueError('Unidentifiable parameter subvector')
    start = np.linalg.lstsq(a, target, rcond=None)[0]
    if loss != 'linear':
        result = least_squares(lambda q: a@q-target, start, jac=lambda q: a.copy(),
                               loss=loss, f_scale=1., method='trf',
                               ftol=1e-10, xtol=1e-10, gtol=1e-10)
        if not result.success:
            raise RuntimeError(result.message)
        start = result.x
    p[free] = start/columns
    return p


def propagation_graph(include_extra=True):
    """Use the student's typed DAG, with alpha/beta as two observation roots."""
    g = ExpressionDAG()
    n = g.symbol('n', kind='parameter')
    a = g.symbol('a', kind='parameter')
    b = g.symbol('b', kind='parameter') if include_extra else None
    root_u = g.symbol('sqrt_f_over_f0')
    u = g.symbol('f_over_f0') if include_extra else None
    k0 = g.symbol('omega_over_c')
    skin = g.operation('mul', a, root_u)
    alpha = g.operation('add', skin, g.operation('mul', b, u)) if include_extra else skin
    phase = g.operation('mul', n, k0)
    beta = g.operation('add', phase, skin)
    g.set_root('alpha', alpha)
    g.set_root('beta', beta)
    return g


def expanded_maxwell_graph(include_extra=True):
    """Refine epsilon_r/gamma nodes of the student's lossy Maxwell RHS.

    Keeps its curl/scale/sub operations; replaces its material constants with
    the effective constitutive law of a tone. k0=omega/c is a known input.
    """
    g=propagation_graph(include_extra)
    alpha,beta=g.roots['alpha'],g.roots['beta']
    k0=next(i for i,node in enumerate(g.nodes) if node.name=='omega_over_c')
    square=lambda i:g.operation('mul',i,i)
    eps=g.operation('div',g.operation('sub',square(beta),square(alpha)),square(k0))
    sigma_over_eps0=g.operation('div',g.operation('mul',g.constant(2*C),
                                                g.operation('mul',alpha,beta)),k0)
    e=g.symbol('e','vector');h=g.symbol('h','vector');c=g.constant(C)
    propagation=g.operation('scale',g.operation('div',c,eps),g.operation('curl',h))
    loss=g.operation('scale',g.operation('div',sigma_over_eps0,eps),e)
    g.set_root('epsilon_r_effective',eps)
    g.set_root('sigma_over_epsilon0',sigma_over_eps0)
    g.set_root('e_rhs',g.operation('sub',propagation,loss))
    g.set_root('h_rhs',g.operation('neg',g.operation('scale',c,g.operation('curl',e))))
    return g


def material_node_map():
    from .upstream import maxwell_model
    original=maxwell_model(True)
    return {
        'original_material_nodes':{node.name:i for i,node in enumerate(original.nodes) if node.kind=='parameter'},
        'substitution':{'epsilon_r':'(beta^2-alpha^2)/k0^2','gamma':'2*c*alpha*beta/k0','mu_r':'1'},
        'parameters':{
            'n':{'null':1,'affected':['epsilon_r','gamma'],'observation_path':'k0*n -> beta'},
            'a':{'null':0,'affected':['epsilon_r','gamma'],'observation_path':'a*sqrt(u) -> alpha and beta'},
            'b':{'null':0,'affected':['epsilon_r','gamma'],'observation_path':'b*u -> alpha'}},
        'interpretation':'Refine original material nodes; curl couplings retained. Original gamma means sigma/epsilon0 [1/s], not propagation gamma [1/m].'
    }
