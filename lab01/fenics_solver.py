"""FEniCSx P2/P1 IPCS solver on a tetrahedral 3D fluid box.

Crank-Nicolson diffusion/convection with an extrapolated advecting velocity.
Velocity Dirichlet data on every face; homogeneous Neumann pressure increments
with a constant nullspace and zero integral mean. No added eddy viscosity.
Adapted formulation: jsdokken.com/dolfinx-tutorial/chapter2/ns_code2.html.
"""
from pathlib import Path
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx import fem,mesh,geometry
from dolfinx.fem import petsc as fp
from basix.ufl import element
import ufl

JIT={'cache_dir':str(Path(__file__).resolve().parents[1]/'.tools'/'fenics-jit')}


def ksp(matrix,kind,pc):
    solver=PETSc.KSP().create(matrix.comm)
    solver.setOperators(matrix);solver.setType(kind);solver.getPC().setType(pc)
    solver.setTolerances(rtol=1e-10,atol=1e-13,max_it=2000)
    solver.setErrorIfNotConverged(True)
    return solver


class IPCS:
    def __init__(self,bounds,shape,points,dt=.001):
        if MPI.COMM_WORLD.size!=1:raise ValueError('This comparison runner currently requires one MPI rank')
        self.bounds=bounds;self.shape=shape;self.dt=dt
        self.domain=mesh.create_box(MPI.COMM_WORLD,np.array(bounds).T,list(shape),cell_type=mesh.CellType.tetrahedron)
        domain=self.domain
        self.V=fem.functionspace(domain,element('Lagrange',domain.basix_cell(),2,shape=(3,)))
        self.Q=fem.functionspace(domain,element('Lagrange',domain.basix_cell(),1))
        self.coords=self.V.tabulate_dof_coordinates()
        self.normals=np.zeros_like(self.coords)
        for c,(lo,hi) in enumerate(bounds):
            self.normals[np.isclose(self.coords[:,c],lo,atol=1e-12,rtol=0),c]=-1
            self.normals[np.isclose(self.coords[:,c],hi,atol=1e-12,rtol=0),c]=1
        domain.topology.create_connectivity(domain.topology.dim-1,domain.topology.dim)
        facets=mesh.exterior_facet_indices(domain.topology)
        dofs=fem.locate_dofs_topological(self.V,domain.topology.dim-1,facets)
        self.g=fem.Function(self.V);self.bc=fem.dirichletbc(self.g,dofs)
        self.old=fem.Function(self.V);self.older=fem.Function(self.V)
        self.star=fem.Function(self.V);self.new=fem.Function(self.V)
        self.p=fem.Function(self.Q);self.phi=fem.Function(self.Q)
        self.wind=fem.Function(self.V)
        self.nu=fem.Constant(domain,PETSc.ScalarType(9e-7))
        self.k=fem.Constant(domain,PETSc.ScalarType(dt))
        u=ufl.TrialFunction(self.V);v=ufl.TestFunction(self.V)
        q=ufl.TestFunction(self.Q);phi=ufl.TrialFunction(self.Q)
        self.dx=ufl.Measure('dx',domain=domain,metadata={'quadrature_degree':5})
        dx=self.dx;mid=.5*(u+self.old)
        F=ufl.inner(u-self.old,v)*dx
        F+=self.k*ufl.inner(ufl.dot(ufl.grad(mid),self.wind),v)*dx
        F+=self.k*self.nu*ufl.inner(ufl.grad(mid),ufl.grad(v))*dx
        F-=self.k*self.p*ufl.div(v)*dx
        self.a1=fem.form(ufl.lhs(F),jit_options=JIT);self.L1=fem.form(ufl.rhs(F),jit_options=JIT)
        self.a2=fem.form(ufl.inner(ufl.grad(phi),ufl.grad(q))*dx,jit_options=JIT)
        self.L2=fem.form(-ufl.div(self.star)*q/self.k*dx,jit_options=JIT)
        self.a3=fem.form(ufl.inner(u,v)*dx,jit_options=JIT)
        self.L3=fem.form(ufl.inner(self.star-self.k*ufl.grad(self.phi),v)*dx,jit_options=JIT)
        self.A1=fp.create_matrix(self.a1)
        self.A2=fp.assemble_matrix(self.a2);self.A2.assemble()
        self.A3=fp.assemble_matrix(self.a3,bcs=[self.bc]);self.A3.assemble()
        self.null=PETSc.NullSpace().create(constant=True,comm=domain.comm)
        self.A2.setNullSpace(self.null);self.A2.setNearNullSpace(self.null)
        self.s1=ksp(self.A1,'bcgs','jacobi')
        self.s2=ksp(self.A2,'cg','gamg')
        self.s3=ksp(self.A3,'cg','jacobi')
        self.b1=fp.create_vector(self.V);self.b2=fp.create_vector(self.Q);self.b3=fp.create_vector(self.V)
        self.volume=float(np.prod([b-a for a,b in bounds]))
        self.area=float(2*sum(np.prod([b-a for j,(a,b) in enumerate(bounds) if j!=i]) for i in range(3)))
        n=ufl.FacetNormal(domain)
        self.flux=fem.form(ufl.dot(self.g,n)*ufl.ds,jit_options=JIT)
        self.pmean=fem.form(self.p*dx,jit_options=JIT)
        self.divnorm=fem.form(ufl.div(self.new)**2*dx,jit_options=JIT)
        self.energynorm=fem.form(ufl.inner(self.new,self.new)*dx,jit_options=JIT)
        self.points=np.asarray(points,dtype=float)
        tree=geometry.bb_tree(domain,domain.topology.dim)
        candidates=geometry.compute_collisions_points(tree,self.points)
        collisions=geometry.compute_colliding_cells(domain,candidates,self.points)
        self.cells=np.array([collisions.links(i)[0] if len(collisions.links(i)) else -1 for i in range(len(points))],dtype=np.int32)
        if np.any(self.cells<0):raise ValueError('Observation point outside finite-element domain')

    def boundary(self,provider,t):
        a=np.asarray(provider(self.coords,t),dtype=float)
        if a.shape!=self.coords.shape or not np.isfinite(a).all():raise ValueError('Invalid velocity provider')
        self.g.x.array[:]=a.ravel();self.g.x.scatter_forward()
        delta=float(fem.assemble_scalar(self.flux)/self.area)
        self.g.x.array[:]-=(delta*self.normals).ravel();self.g.x.scatter_forward()
        return delta,float(fem.assemble_scalar(self.flux))

    def rhs(self,b,L,a=None):
        with b.localForm() as local:local.set(0)
        fp.assemble_vector(b,L)
        if a is not None:fp.apply_lifting(b,[a],bcs=[[self.bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD,mode=PETSc.ScatterMode.REVERSE)
        if a is not None:fp.set_bc(b,[self.bc])

    def correct(self):
        self.rhs(self.b2,self.L2);self.null.remove(self.b2)
        self.s2.solve(self.b2,self.phi.x.petsc_vec);self.phi.x.scatter_forward()
        self.rhs(self.b3,self.L3,self.a3)
        self.s3.solve(self.b3,self.new.x.petsc_vec);self.new.x.scatter_forward()

    def solve(self,provider,times,nu,p0=None,verbose=False):
        times=np.asarray(times,dtype=float)
        if nu<=0 or np.any(np.diff(times)<=0):raise ValueError('Invalid viscosity/times')
        self.nu.value=nu;self.p.x.array[:]=0
        if p0 is not None:self.p.interpolate(lambda x:p0(x.T,float(times[0])))
        self.k.value=1.
        delta,flux=self.boundary(provider,float(times[0]))
        self.star.x.array[:]=self.g.x.array;self.correct()
        self.old.x.array[:]=self.new.x.array;self.older.x.array[:]=self.new.x.array
        self.p.x.array[:]-=fem.assemble_scalar(self.pmean)/self.volume
        history=[self.new.eval(self.points,self.cells).copy()]
        divergence=[float(np.sqrt(max(fem.assemble_scalar(self.divnorm),0)/self.volume))]
        shifts=[delta];fluxes=[flux];iterations=[];steps=0;start=time.perf_counter()
        for target in times[1:]:
            # Uniform subdivisions inside every measurement interval; no time extrapolation.
            t=float(times[len(history)-1]);nsteps=int(np.ceil((target-t)/self.dt-1e-10));step=(target-t)/nsteps
            self.k.value=step
            for substep in range(nsteps):
                next_t=t+(substep+1)*step
                delta,flux=self.boundary(provider,next_t)
                self.wind.x.array[:]=(self.old.x.array if steps==0 else 1.5*self.old.x.array-.5*self.older.x.array)
                self.wind.x.scatter_forward()
                self.A1.zeroEntries();fp.assemble_matrix(self.A1,self.a1,bcs=[self.bc]);self.A1.assemble()
                self.s1.setOperators(self.A1)
                self.rhs(self.b1,self.L1,self.a1)
                self.s1.solve(self.b1,self.star.x.petsc_vec);self.star.x.scatter_forward()
                self.correct()
                self.p.x.array[:]+=self.phi.x.array
                self.p.x.array[:]-=fem.assemble_scalar(self.pmean)/self.volume
                self.p.x.scatter_forward()
                self.older.x.array[:]=self.old.x.array;self.old.x.array[:]=self.new.x.array
                if not np.isfinite(self.old.x.array).all() or np.max(abs(self.old.x.array))>20:
                    raise FloatingPointError('Unstable finite-element velocity')
                iterations.append([self.s1.getIterationNumber(),self.s2.getIterationNumber(),self.s3.getIterationNumber()])
                steps+=1
            history.append(self.new.eval(self.points,self.cells).copy())
            divergence.append(float(np.sqrt(max(fem.assemble_scalar(self.divnorm),0)/self.volume)))
            shifts.append(delta);fluxes.append(flux)
            if verbose:print(f'FEniCSx t={target:.3f}, steps={steps}, div_RMS={divergence[-1]:.3g}',flush=True)
        return {'prediction':np.stack(history),'divergence_rms':np.array(divergence),
                'boundary_normal_shift':np.array(shifts),'boundary_flux':np.array(fluxes),
                'steps':steps,'seconds':time.perf_counter()-start,'iterations':np.array(iterations),
                'velocity_dofs':self.V.dofmap.index_map.size_global*self.V.dofmap.index_map_bs,
                'pressure_dofs':self.Q.dofmap.index_map.size_global,'dolfinx_version':dolfinx.__version__}

    def close(self):
        for obj in (self.s1,self.s2,self.s3,self.b1,self.b2,self.b3,self.A1,self.A2,self.A3,self.null):obj.destroy()


class ExperimentalProvider:
    """Same physical observations and space/time interpolation policy as MAC."""
    def __init__(self,measurements):
        from scipy.interpolate import RegularGridInterpolator
        self.times=measurements['times'];self.coordinates=None;self.sampled=None
        self.interpolators=[RegularGridInterpolator(tuple(measurements[k] for k in 'xyz'),v,bounds_error=True) for v in measurements['velocity']]

    def __call__(self,points,t):
        if self.coordinates is not points:
            self.sampled=np.stack([f(points) for f in self.interpolators]);self.coordinates=points
        if t<self.times[0]-1e-10 or t>self.times[-1]+1e-10:raise ValueError('Extrapolation forbidden')
        j=np.clip(np.searchsorted(self.times,t,side='right')-1,0,len(self.times)-2)
        alpha=np.clip((t-self.times[j])/(self.times[j+1]-self.times[j]),0,1)
        return (1-alpha)*self.sampled[j]+alpha*self.sampled[j+1]
