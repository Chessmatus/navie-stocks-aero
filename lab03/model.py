"""Pressure form of variable-density acoustics and candidate equation graphs.

For time-independent rho,K, eliminating particle velocity gives
p_tt = K div(rho**-1 grad(p)). This is the original acoustic system's
pressure equation; rho and K are known inputs, not inferred wall maps.
"""
from pathlib import Path
import json
import numpy as np
import networkx as nx
from scipy.ndimage import gaussian_filter, gaussian_filter1d

ROOT = Path(__file__).resolve().parent
TERMS = ('x_flux', 'y_flux', 'cross_flux', 'damping', 'reaction', 'cubic')
LABELS = (r'$K\partial_x(\rho^{-1}p_x)$', r'$K\partial_y(\rho^{-1}p_y)$',
          r'$K[\partial_x(\rho^{-1}p_y)+\partial_y(\rho^{-1}p_x)]/2$',
          r'$-p_t/T$', r'$-p/T^2$', r'$-p^3/(P^2T^2)$')


def bits(mask): return np.array([(int(mask)>>j)&1 for j in range(6)],dtype=bool)


def flux_operators(p, rho, bulk, dx):
    """Centered finite-volume flux differences, harmonic face mobility.

    Used to evaluate weak features; no derivatives of discontinuous density.
    Spatial axis ordering is (x,y), as in the Well metadata.
    """
    a = 1/np.asarray(rho,dtype=float)
    ax = 2*a[:-1]*a[1:]/(a[:-1]+a[1:])
    ay = 2*a[:,:-1]*a[:,1:]/(a[:,:-1]+a[:,1:])
    fx = ax*(p[...,1:,:]-p[...,:-1,:])/dx
    fy = ay*(p[...,:,1:]-p[...,:,:-1])/dx
    lx=np.zeros_like(p); ly=np.zeros_like(p)
    lx[...,1:-1,:]=(fx[...,1:,:]-fx[...,:-1,:])/dx
    ly[...,:,1:-1]=(fy[...,:,1:]-fy[...,:,:-1])/dx
    px=np.gradient(p,dx,axis=-2); py=np.gradient(p,dx,axis=-1)
    cross=.5*(np.gradient(a*py,dx,axis=-2)+np.gradient(a*px,dx,axis=-1))
    return bulk*lx,bulk*ly,bulk*cross


def well_features(path, sigma_t=1.4, sigma_x=2., stride=4, pressure_scale=None):
    """Smoothed space-time moment equations. Excludes filter boundaries.

    Convolution is applied to the complete flux before spatial smoothing,
    so density is never pulled outside a derivative. This is a discrete
    weak evaluation; truncation error is separately measured by rollouts.
    """
    with np.load(path) as d:
        p=d['p'].astype(float); rho=d['rho'].astype(float)
        bulk=float(np.median(rho*d['c']**2)); t=d['t'].astype(float)
    dt=float(np.median(np.diff(t))); dx=2/p.shape[-1]
    pressure_scale=float(np.max(np.abs(p[0]))) if pressure_scale is None else float(pressure_scale)
    sig=(sigma_t,sigma_x,sigma_x)
    lhs=gaussian_filter(p,sig,order=(2,0,0),mode='nearest')/dt**2
    lx,ly,cross=flux_operators(p,rho,bulk,dx)
    fields=[gaussian_filter(z,sig,mode='nearest') for z in (lx,ly,cross)]
    fields += [-gaussian_filter(p,sig,order=(1,0,0),mode='nearest')/dt,
               -gaussian_filter(p,sig,mode='nearest'),
               -gaussian_filter(p**3,sig,mode='nearest')/pressure_scale**2]
    margin_t=int(np.ceil(4*sigma_t))+1
    margin_x=int(np.ceil(4*sigma_x))+2
    sl=(slice(margin_t,-margin_t,2),slice(margin_x,-margin_x,stride),slice(margin_x,-margin_x,stride))
    X=np.stack([z[sl] for z in fields],axis=-1)
    y=lhs[sl]
    # Whole spatial frames are bootstrap blocks. No individual-pixel iid bootstrap.
    frame=np.broadcast_to(np.arange(X.shape[0])[:,None,None],y.shape)
    weights=np.ones(y.shape)
    return {'X':X.reshape(-1,6),'y':y.ravel(),'group':frame.ravel(),
            'shape':X.shape[:-1], 'dt_group':dt*2,
            'source':Path(path).name,'sigma_t':sigma_t,'sigma_x':sigma_x,
            'bulk':bulk,'dx':dx,'pressure_scale':pressure_scale,
            'time_range':[float(t[margin_t]),float(t[-margin_t-1])]}


def fit(X,y,mask=63):
    active=bits(mask)
    p=np.zeros(6)
    if active.any():
        scale=np.maximum(np.sqrt(np.mean(X[:,active]**2,axis=0)),1e-15)
        p[active]=np.linalg.lstsq(X[:,active]/scale,y,rcond=1e-10)[0]/scale
    return p


def relative_error(reference,prediction):
    return float(np.linalg.norm(prediction-reference)/max(np.linalg.norm(reference),1e-15))


def graph(mask, parameters=None):
    """Expression tree; repeated operands retain distinct occurrence IDs."""
    g=nx.DiGraph()
    g.add_node('sum',label='+',kind='operator')
    for j,on in enumerate(bits(mask)):
        if not on: continue
        key=f'term_{j}'
        g.add_node(key,label=TERMS[j],kind='operator')
        g.add_edge(key,'sum',position=j)
        theta=f'{key}_theta';g.add_node(theta,label='theta_'+str(j+1),kind='parameter')
        g.add_edge(theta,key,position=0)
        def derivative(axis,operand):return ('d/d'+axis,operand)
        inv=('1/rho','rho')
        expressions=[
            ('*','K',derivative('x',('*',inv,derivative('x','p')))),
            ('*','K',derivative('y',('*',inv,derivative('y','p')))),
            ('*','K/2',('+',derivative('x',('*',inv,derivative('y','p'))),
                             derivative('y',('*',inv,derivative('x','p'))))),
            ('neg',derivative('t','p')),('neg','p'),('neg',('*','P^-2',('cube','p')))]
        def add_expr(expr,parent,pos,path):
            label=expr[0] if isinstance(expr,tuple) else expr
            g.add_node(path,label=label,kind='operator' if isinstance(expr,tuple) else 'input')
            g.add_edge(path,parent,position=pos)
            if isinstance(expr,tuple):
                for k,child in enumerate(expr[1:]):add_expr(child,path,k,path+'_'+str(k))
        add_expr(expressions[j],key,1,key+'_expr')
    g.add_node('p_tt',label='p_tt',kind='output');g.add_edge('sum','p_tt',position=0)
    assert nx.is_directed_acyclic_graph(g)
    return g


def export_graph(mask,parameters,stem):
    g=graph(mask,parameters)
    order=list(nx.topological_sort(g))
    np.save(str(stem)+'_adjacency.npy',nx.to_numpy_array(g,nodelist=order,dtype=int))
    Path(str(stem)+'_graph.json').write_text(json.dumps({
        'mask':int(mask),'terms':[TERMS[j] for j in range(6) if bits(mask)[j]],
        'parameters':np.asarray(parameters).tolist(),'node_order':order,
        'nodes':[{'id':n,**g.nodes[n]} for n in order],
        'edges':[{'source':a,'target':b,**d} for a,b,d in g.edges(data=True)],
        'meaning':'Expression DAG, not a causal discovery proof. Time-unrolled solver is acyclic.'},indent=2))
