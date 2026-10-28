"""Run from repository root: python -m lab02.run [--bootstrap 400]."""
import argparse
from .experiment import run
from .plotting import make_all

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--bootstrap',type=int,default=1000)
    parser.add_argument('--rebuild-data',action='store_true')
    args=parser.parse_args()
    if args.bootstrap < 20:parser.error('Use at least 20 bootstrap replicates')
    report=run(args.bootstrap,rebuild_data=args.rebuild_data)
    make_all()
    for name,info in report['datasets'].items():
        print(name,info['parameters'],[(x['parameter'],x['decision']) for x in info['ablation']])
