"""Read the verified experimental subset and interpolate it onto a 3D MAC grid."""
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator

BOUNDS = ((.020,.100),(-.024,.024),(-.024,.024))


def prepare_measurements(directory):
    directory=Path(directory)
    cache=directory/'measurement_crop.npz'
    if cache.exists():
        return dict(np.load(cache))
    files=sorted(directory.glob('B[0-9][0-9][0-9][0-9][0-9].txt'))
    if len(files)<2:raise FileNotFoundError('Fetch experimental frames first')
    velocities=[];frame_ids=[];axes=None;reference_coordinates=None
    for path in files:
        data=np.loadtxt(path,delimiter=',',skiprows=1)
        if data.ndim!=2 or data.shape[1]!=6 or not np.isfinite(data).all():
            raise ValueError(f'Invalid measured frame: {path}')
        if axes is None:
            reference_coordinates=data[:,:3].copy()
            full_axes=[np.unique(data[:,i])*.001 for i in range(3)]
            if np.prod([len(a) for a in full_axes])!=len(data):raise ValueError('Expected full rectilinear grid')
            indices=[np.searchsorted(a,data[:,i]*.001) for i,a in enumerate(full_axes)]
            crop=[]
            for axis,(lo,hi) in zip(full_axes,BOUNDS):
                crop.append(slice(max(np.searchsorted(axis,lo)-1,0),min(np.searchsorted(axis,hi)+2,len(axis))))
            axes=[a[c] for a,c in zip(full_axes,crop)]
        elif not np.array_equal(data[:,:3],reference_coordinates):
            raise ValueError('Experimental grid changed across frames')
        field=np.empty(tuple(len(a) for a in full_axes)+(3,))
        field[tuple(indices)]=data[:,3:]
        velocities.append(field[tuple(crop)])
        frame_ids.append(int(path.stem[1:]))
    result={'x':axes[0],'y':axes[1],'z':axes[2], 'velocity':np.stack(velocities),
            'frame_ids':np.array(frame_ids),'times':(np.array(frame_ids)-frame_ids[0])/500.,
            'absolute_frame_times':(np.array(frame_ids)-1)/500.}
    np.savez_compressed(cache,**result)
    return result


class MeasuredBoundary:
    """Trilinear spatial and piecewise-linear temporal interpolation; no extrapolation."""
    def __init__(self,grid,measurements):
        self.times=measurements['times']
        self.faces=[];self.ghosts=[];self.observed=[]
        axes=tuple(measurements[k] for k in 'xyz')
        def sample(interpolator,component=None,boundary=None):
            coords=grid.coordinates(component,boundary)
            points=np.stack(coords,axis=-1)
            return interpolator(points)
        for field in measurements['velocity']:
            interpolator=RegularGridInterpolator(axes,field,bounds_error=True)
            self.faces.append([sample(interpolator,c)[...,c] for c in range(3)])
            ghosts={}
            for c in range(3):
                for axis in range(3):
                    if c==axis:continue
                    for side in (0,1):
                        value=sample(interpolator,c,(axis,side))[...,c]
                        ghosts[(c,axis,side)]=np.squeeze(value,axis=axis)
            self.ghosts.append(ghosts)
            self.observed.append(sample(interpolator))
        self.observed=np.stack(self.observed)

    def __call__(self,t):
        if t<self.times[0]-1e-12 or t>self.times[-1]+1e-12:raise ValueError('Boundary extrapolation forbidden')
        j=min(max(np.searchsorted(self.times,t,side='right')-1,0),len(self.times)-2)
        alpha=np.clip((t-self.times[j])/(self.times[j+1]-self.times[j]),0,1)
        faces=[(1-alpha)*a+alpha*b for a,b in zip(self.faces[j],self.faces[j+1])]
        ghosts={k:(1-alpha)*self.ghosts[j][k]+alpha*self.ghosts[j+1][k] for k in self.ghosts[j]}
        return faces,ghosts


def interior(velocity):
    """Exclude the initial frame and two cell layers adjacent to the prescribed boundaries."""
    return velocity[1:,2:-2,2:-2,2:-2,:]


def observation_points():
    axes=(np.linspace(.035,.085,11),np.linspace(-.015,.015,7),np.linspace(-.015,.015,7))
    return np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)


def measured_observations(measurements,points):
    axes=tuple(measurements[k] for k in 'xyz')
    return np.stack([RegularGridInterpolator(axes,field,bounds_error=True)(points)
                     for field in measurements['velocity']])


def observe_solution(grid,velocity,points):
    return np.stack([RegularGridInterpolator(grid.centers,field,bounds_error=True)(points)
                     for field in velocity])
