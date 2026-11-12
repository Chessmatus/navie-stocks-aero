"""Read selected published HDF5 slabs using verified HTTP byte ranges.

The upstream chunks are 16 GB each. Only requested trajectories/time windows
are fetched. Download blocks are cached outside the versioned dataset.
"""
from pathlib import Path
import io
import json
import hashlib
import time
import requests
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
REVISION = '8df383a3223f40f7ce66fe77b4ff4d7006dbc272'
BASE = f'https://huggingface.co/datasets/polymathic-ai/acoustic_scattering_maze/resolve/{REVISION}'
CHUNK = {'train':0, 'valid':16, 'test':18}


class HTTPRangeFile(io.RawIOBase):
    """Seekable remote file; refuses unbounded HTTP responses."""
    def __init__(self, url, cache, size=15980298240, block=2**20):
        self.url, self.size, self.block = url, size, block
        self.pos = 0
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.downloaded = 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        self.pos = (0 if whence == 0 else self.pos if whence == 1 else self.size) + offset
        return self.pos

    def readinto(self, buffer):
        b = self.read(len(buffer)); buffer[:len(b)] = b
        return len(b)

    def read(self, count=-1):
        if count < 0: raise ValueError('An explicit bounded read is required')
        end = min(self.pos+count, self.size)
        data = []
        while self.pos < end:
            index, off = divmod(self.pos, self.block)
            path = self.cache / f'{index:08d}.bin'
            lo, hi = index*self.block, min((index+1)*self.block, self.size)-1
            if not path.exists():
                for attempt in range(5):
                    try:
                        # Range in query avoids stale CDN responses to a different Range header.
                        r = self.session.get(self.url+f'?download=true&range_start={lo}',
                            headers={'Range':f'bytes={lo}-{hi}', 'Accept-Encoding':'identity'},
                            timeout=90)
                        r.raise_for_status()
                        if r.status_code != 206 or len(r.content) != hi-lo+1:
                            raise IOError(f'Invalid bounded response: {r.status_code}, {len(r.content)}')
                        if not r.headers.get('Content-Range','').startswith(f'bytes {lo}-{hi}/'):
                            raise IOError('Content-Range mismatch')
                        path.write_bytes(r.content); self.downloaded += len(r.content)
                        break
                    except (requests.RequestException, IOError):
                        if attempt == 4: raise
                        time.sleep(1+attempt)
            block = path.read_bytes()
            take = min(end-self.pos, len(block)-off)
            if take <= 0: raise IOError('Invalid cache block')
            data.append(block[off:off+take]); self.pos += take
        return b''.join(data)


def inspect(split='train'):
    url = BASE+f'/data/{split}/acoustic_scattering_maze_chunk_{CHUNK[split]}.hdf5'
    with HTTPRangeFile(url, ROOT.parent/'tmp/lab03/http'/split) as stream:
        with h5py.File(stream, 'r') as f:
            print('ATTRIBUTES', dict(f.attrs), flush=True)
            f.visititems(lambda k,v: print(k, getattr(v,'shape',None),
                getattr(v,'chunks',None), dict(v.attrs), flush=True))


def fetch(split, sample=0, stop=101):
    out = ROOT/'data'/f'well_{split}_{sample}.npz'
    if out.exists():
        meta=json.loads(out.with_suffix('.json').read_text())
        if meta['stop']!=stop:raise ValueError('Existing slice has a different stop; use a separate output name')
        if hashlib.sha256(out.read_bytes()).hexdigest()!=meta['sha256']:raise IOError('Local slice hash mismatch')
        return out
    url = BASE+f'/data/{split}/acoustic_scattering_maze_chunk_{CHUNK[split]}.hdf5'
    with HTTPRangeFile(url, ROOT.parent/'tmp/lab03/http'/split) as stream:
        with h5py.File(stream, 'r') as f:
            # Filled against the public self-describing schema, not guessed axes.
            print('Preparing', split, flush=True)
            p = f['t0_fields/pressure'][sample,:stop]
            rho = f['t0_fields/density'][sample]
            c = f['t0_fields/speed_of_sound'][sample]
            t = f['dimensions/time'][:stop]
            x = f['dimensions/x'][:]; y = f['dimensions/y'][:]
            np.savez_compressed(out, p=p, rho=rho, c=c, t=t,
                exported_x=x, exported_y=y)
    record = {'url':url, 'revision':REVISION, 'sample':sample, 'stop':stop,
        'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
        'note':'Published simulation, 2D+time; spatial coordinates of solver are [-1,1]^2.'}
    out.with_suffix('.json').write_text(json.dumps(record,indent=2))
    print(out.name, p.shape, 'downloaded MB', stream.downloaded/1e6, flush=True)
    return out


def fetch_cno():
    import zipfile
    path=ROOT.parent/'tmp/lab03/WaveData_64x64_OUT.h5'
    if not path.exists():
        with HTTPRangeFile('https://zenodo.org/records/10406879/files/data.zip',
                ROOT.parent/'tmp/lab03/cno_http',size=2379635051) as stream:
            with zipfile.ZipFile(stream) as z:
                path.write_bytes(z.read('data/WaveData_64x64_OUT.h5'))
    from .analytic import archive_audit
    print(archive_audit())


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--split', default='train')
    parser.add_argument('--stop',type=int,default=101)
    parser.add_argument('--sample',type=int,default=0)
    parser.add_argument('--cno',action='store_true')
    args=parser.parse_args()
    if args.cno: fetch_cno()
    elif args.inspect: inspect(args.split)
    else: fetch(args.split,sample=args.sample,stop=args.stop)
