"""Time the forward graph-aggregation operation at controlled matrix densities."""
import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import needle as ndl
from needle.sparse import CSRMatrix, fixed_dense_matmul, fixed_spmm
from apps.experiment import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/density_sweep.json'))
    args = parser.parse_args()
    rng = np.random.default_rng(17)
    n, width, repeats = 512, 16, 25
    x = ndl.Tensor(rng.normal(size=(n, width)).astype('float32'), device=ndl.cpu(), requires_grad=False)
    permutation = rng.permutation(n*n)
    results = []
    for density in (0.001, 0.01, 0.05, 0.2, 0.5, 1.0):
        # Nested patterns vary edge count at a fixed matrix and feature shape.
        positions = permutation[:round(n*n*density)]
        rows, cols = np.divmod(positions, n)
        values = rng.uniform(0.1, 1, size=len(rows)).astype('float32')
        matrix = CSRMatrix.from_coo(rows, cols, values, (n,n))
        host = np.zeros((n,n), np.float32)
        host[rows,cols] = values
        dense = ndl.Tensor(host, device=ndl.cpu(), requires_grad=False)
        paths = [(fixed_spmm, matrix), (fixed_dense_matmul, dense)]
        reference = paths[1][0](dense, x).numpy()
        np.testing.assert_allclose(fixed_spmm(matrix, x).numpy(), reference, rtol=1e-4, atol=1e-4)
        samples = [[], []]
        for iteration in range(3+repeats):
            for j in ((0,1) if iteration % 2 == 0 else (1,0)):
                op, graph = paths[j]
                start = time.perf_counter_ns()
                op(graph, x).realize_cached_data()
                elapsed = (time.perf_counter_ns() - start)/1e6
                if iteration >= 3:
                    samples[j].append(elapsed)
        record = dict(nnz=matrix.nnz, density=matrix.nnz/(n*n), csr_bytes=matrix.nbytes,
                      dense_bytes=host.nbytes, sparse_ms=statistics.median(samples[0]),
                      dense_ms=statistics.median(samples[1]), sparse_samples_ms=samples[0],
                      dense_samples_ms=samples[1])
        results.append(record)
        print(f"{density:6.1%}   CSR {record['sparse_ms']:.3f} ms   dense {record['dense_ms']:.3f} ms")
    save_json(args.output, dict(nodes=n, features=width, repeats=repeats, warmup=3, seed=17, results=results))


if __name__ == '__main__':
    main()
