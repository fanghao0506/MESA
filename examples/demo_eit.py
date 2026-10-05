"""Reconstruct three author-generated nonlinear EIT examples."""
from pathlib import Path
import argparse
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['torch', 'numpy'], default='torch')
    parser.add_argument('--device', default='cpu', help='PyTorch device: cpu or cuda')
    parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[1] / 'data/eit_examples.npz')
    parser.add_argument('--output', type=Path, default=Path('outputs/eit_demo'))
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    data = np.load(args.data, allow_pickle=False)
    shape = tuple(data['grid_shape'])
    indices = np.rint(data['points'][:, 3:5]).astype(int) - 1

    if args.backend == 'torch':
        from mesa import torch_backend
        state = torch_backend.prepare_operator(data['sensitivity'], shape,
            grid_indices=indices, sigma=1, sensitivity_exponent=1,
            updates=8, hill_power=1.5, selector='loocv', device=args.device)
        def reconstruct(y):
            return torch_backend.stages(y, state)
    else:
        import mesa
        # The sample EIT sensitivity uses an indexed/transposed grid, not row-major.
        order = np.ravel_multi_index(indices.T, shape)
        row_major = np.empty_like(data['sensitivity'])
        row_major[:, order] = data['sensitivity']
        state = mesa.prepare(row_major, shape, sigma=1, sensitivity_exponent=1)
        def reconstruct(y):
            fields, diagnostics = mesa.trajectory(y, state, checkpoints=(0, 8))
            output = mesa.readout(fields[8], y, state, power=1.5)
            return fields[0][order], fields[8][order], output[order], diagnostics

    records, stages = [], []
    for name, y in zip(data['case_names'], data['measurements']):
        initial, adapted, output, diagnostics = reconstruct(y)
        stages.append([initial, adapted, output])
        fitted = data['sensitivity'] @ output
        records.append(dict(case=str(name), **diagnostics,
            relative_measurement_residual=float(np.linalg.norm(fitted-y)/np.linalg.norm(y))))

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / 'reconstructions.npz',
                        stages=np.asarray(stages), case_names=data['case_names'])
    report = dict(backend=args.backend, device=args.device if args.backend == 'torch' else 'cpu',
                  cases=records, note='Three demonstration cases, not an aggregate benchmark.')
    (args.output / 'diagnostics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

    if not args.no_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 4, figsize=(8, 6), layout='constrained')
        for row, name in enumerate(data['case_names']):
            vectors = [data['truth'][row], *stages[row]]
            for col, vector in enumerate(vectors):
                grid = np.zeros(shape)
                grid[tuple(indices.T)] = vector
                peak = float(np.max(np.abs(grid)))
                axes[row, col].imshow(grid/max(peak, 1e-30), cmap='inferno', vmin=0, vmax=1, origin='lower')
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
                if row == 0:
                    axes[row, col].set_title(['Ground truth', 'Initial field', 'Adapted field', 'MESA output'][col], fontsize=10)
            axes[row, 0].set_ylabel(str(name), fontsize=9)
        fig.savefig(args.output / 'reconstructions.png', dpi=180)
        fig.savefig(args.output / 'reconstructions.pdf')
        plt.close(fig)


if __name__ == '__main__':
    main()
