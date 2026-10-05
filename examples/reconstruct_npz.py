"""Apply the NumPy ECT/MPI adapter to an external operator/measurement NPZ."""
from pathlib import Path
import argparse
import json
import numpy as np
import mesa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    with np.load(args.input, allow_pickle=False) as data:
        operator = data['operator']
        measurements = data['measurements']
        shape = tuple(data['image_shape'])
    if measurements.ndim == 1:
        measurements = measurements[None, :]
    if measurements.ndim != 2:
        raise ValueError('measurements must have shape (m,) or (frames,m)')
    state = mesa.prepare(operator, shape, sigma=config['sigma'],
                         sensitivity_exponent=config['sensitivity_exponent'])
    outputs = [mesa.reconstruct(y, state, updates=config['updates'],
                               hill_power=config['hill_power']) for y in measurements]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, predictions=np.asarray(outputs), image_shape=shape)
    print(f'Saved {len(outputs)} independently reconstructed frames to {args.output}')


if __name__ == '__main__':
    main()
