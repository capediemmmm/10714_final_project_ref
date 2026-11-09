# Sparse Matrix Support and Graph Neural Networks in Needle

This project extends the HW4 Extra Needle implementation with native CPU sparse matrix multiplication and automatic differentiation. Its application is a two-layer GCN trained on the public Cora split. The English notebook, `final_report.ipynb`, contains the report, measured results and a copy of the source archive. It can be opened in Colab on a CPU runtime. The source ZIP also works independently.

## Reproduce locally

On macOS or Linux, use Python 3.10+ and a C++ compiler. From the project directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make
export PYTHONPATH="python:."
python -m pytest -q tests
python -m apps.verify_cora --output results/reproduced_correctness.json
python -m apps.experiment --epochs 100 --warmup 5 --seeds 7 19 41 --output results/reproduced_benchmark.json
python -m apps.density_sweep --output results/reproduced_sweep.json
```

For a smoke run, pass `--epochs 5 --warmup 1 --seeds 7`. The checkout includes the eight pinned dataset files. The training CLI checks their SHA-256 hashes before deserializing them. The experiment starts a fresh process for each dataset layout and seed. Run it when the CPU is idle to measure timing. The full runs here used macOS arm64, native CPU, Python 3.12.14, NumPy 2.5.4 and SciPy 1.18.1. SciPy reads the original serialized input files. Needle executes the model and its gradients.

## Operator

```python
import needle as ndl
from needle.sparse import CSRMatrix, spmm

A = CSRMatrix.from_coo([0, 1, 1], [1, 0, 2], [2., 3., 4.], (2, 3),
                       device=ndl.cpu(), requires_grad=True)
X = ndl.Tensor([[1.], [2.], [-1.]], device=ndl.cpu())
Y = spmm(A, X)                  # [[4.], [-1.]]
Y.backward(ndl.Tensor([[1.], [2.]], device=ndl.cpu(), requires_grad=False))
print(A.values.grad.numpy())    # [2., 2., -2.] in canonical CSR order
print(X.grad.numpy())           # [[6.], [2.], [8.]]
```

`from_coo` sorts host coordinates, adds duplicates and drops exact zeros. Direct CSR construction preserves the supplied values tensor, including its gradient history. Row pointers and columns are read-only int32 metadata. The C++ kernels implement the forward product, transposed product and sampled dot products for stored-value gradients. The fixed-graph APIs `fixed_spmm` and `fixed_dense_matmul` differentiate only the right operand. Both training paths use this contract.

## Results

Cora has 2,708 nodes and 13,264 normalized adjacency entries including self-loops. Train, validation, and test masks contain 140, 500, and 1,000 nodes. Each paired run used 100 Adam updates, 16 hidden features, dropout 0.5, learning rate 0.01 and weight decay 0.0005. Validation was evaluated every five epochs. The checkpoint with highest validation accuracy (earliest tie) was used for one test evaluation.

| Metric | Sparse Needle | Dense Needle |
|---|---:|---:|
| Test accuracy, seeds 7 / 19 / 41 | 81.3% / 79.9% / 78.1% | 81.3% / 79.9% / 78.1% |
| Mean test accuracy | 79.77% | 79.77% |
| Median time per epoch | 32.15 ms | 121.01 ms |
| Adjacency payload | 116,948 bytes | 29,333,056 bytes |
| Median peak process RSS | 141.70 MiB | 174.03 MiB |

Time is the median of the three run medians after removing the first five epochs per run. RSS is the median of three process high-water marks, including loading, training, and evaluation. The resulting 3.76x timing ratio and 250.82x adjacency storage ratio describe this implementation and machine. A separate seed-7 Cora check with dropout disabled matched every logit, masked loss, both weight gradients and both weight arrays after one Adam step exactly in float32. The independent test suite also checks nontrivial finite differences and one full step against NumPy.

The synthetic sweep varies density from 0.1% to 100% at a fixed 512-by-512 shape and 16 features. Each forward result was checked against dense Needle. CSR took 0.393 ms at 50% density and 0.796 ms at full density. Dense took about 0.733 ms in both cases. These are operator timings, separate from the full training experiment.

## Files

- `python/needle/sparse.py` and `src/ndarray_backend_cpu.cc` implement the new sparse representation, native kernels, and differentiation. Other Needle modules provide the homework baseline.
- `apps/gcn.py` implements data preparation, the model, masked loss and training.
- `tests/` contains operator and GCN checks. `apps/verify_cora.py` checks the complete real-data training step.
- `apps/experiment.py` and `apps/density_sweep.py` reproduce the paired and synthetic measurements, respectively.
- `results/` holds per-epoch losses, validation checkpoints, timings, memory, test accuracies, numerical errors, and the recorded test log.
- `data/cora/SOURCE.txt` identifies the dataset revision. `data/cora/SHA256SUMS` pins its inputs.

The notebook and presentation accompany the source archive. The archive contains the source, tests, input data, and measured results and builds the native extension on the target system. Distribute the source privately with the course report.

Model and data references: [Kipf and Welling, ICLR 2017](https://arxiv.org/abs/1609.02907), [original Cora files and public split](https://github.com/tkipf/gcn/tree/39a4089fe72ad9f055ed6fdb9746abdcfebc4d81/gcn/data), [course project](https://dlsyscourse.org/project/).
