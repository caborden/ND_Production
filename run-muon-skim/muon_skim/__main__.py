"""Run with python -m muon_skim; no notebook kernel or GPU is required."""
import argparse
from pathlib import Path

from .geometry import load_config
from .pipeline import discover, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='*', help='HDF5 files or recursively searched directories')
    parser.add_argument('--input-glob', action='append', default=[])
    parser.add_argument('--input-list', help='Paths relative to this list or absolute paths')
    parser.add_argument('--output-dir', required=True, help='New directory; existing directories are refused')
    parser.add_argument('--larnd-repo', required=True)
    parser.add_argument('--config', default=str(Path(__file__).resolve().parents[1] / 'configs/fsd_cube.json'))
    parser.add_argument('--min-length-cm', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--events-per-file', type=int)
    parser.add_argument('--prefix', default='muons')
    args = parser.parse_args()
    config = load_config(args.config)
    for key in ('min_length_cm', 'seed', 'events_per_file'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    paths = discover(args.inputs, args.input_glob, args.input_list)
    manifest = run(paths, args.output_dir, config, args.larnd_repo, args.prefix, args.config)
    print(f'Completed: {manifest["selected_events"]} events in {len(manifest["outputs"])} files')


if __name__ == '__main__':
    main()
