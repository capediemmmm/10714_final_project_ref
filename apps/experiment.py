"""Paired Cora training runs. Execute from the project root with PYTHONPATH=python:."""
import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import needle as ndl
from apps.gcn import GCN, evaluate, load_cora, make_adjacency, prepare_inputs, train_epoch


def check_data(path):
    for line in (path / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split()
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Cora checksum mismatch: {name}')


def save_json(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def worker(args):
    import scipy
    check_data(args.data)
    graph = load_cora(args.data)
    features = graph.features.shape[1]
    classes = int(graph.labels.max()) + 1
    labels, val_mask, test_mask = graph.labels, graph.val_mask, graph.test_mask
    device = ndl.cpu()
    config = dict(hidden=args.hidden, epochs=args.epochs, lr=args.lr, weight_decay=args.decay,
                  dropout=args.dropout, warmup=args.warmup, validation_interval=args.validate_every)
    info = dict(nodes=len(labels), features=features, classes=classes, nonzeros=len(graph.values),
                train=int(graph.train_mask.sum()), validation=int(val_mask.sum()), test=int(test_mask.sum()))
    adj = make_adjacency(graph, args.layout, device=device)
    adjacency_bytes = adj.nbytes if args.layout == 'sparse' else adj.shape[0] * adj.shape[1] * 4
    inputs = prepare_inputs(graph, device=device)
    del graph
    seed = args.seeds[0]
    np.random.seed(seed)
    model = GCN(features, args.hidden, classes, dropout=args.dropout, device=device)
    optimizer = ndl.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    history = []
    best_accuracy = -1
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        loss = train_epoch(model, adj, inputs, optimizer)
        elapsed = time.perf_counter() - start
        record = dict(epoch=epoch, loss=float(loss), seconds=elapsed)
        if epoch % args.validate_every == 0 or epoch == args.epochs:
            accuracy = float(evaluate(model, adj, inputs, labels, val_mask))
            record['val_accuracy'] = accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch
                best_weights = [p.numpy().copy() for p in model.parameters()]
        history.append(record)
    for param, weight in zip(model.parameters(), best_weights):
        param.data = ndl.Tensor(weight, device=device, requires_grad=False)
    test_accuracy = float(evaluate(model, adj, inputs, labels, test_mask))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(layout=args.layout, seed=seed, config=config, graph=info, adjacency_bytes=adjacency_bytes,
                peak_rss_bytes=int(peak if sys.platform == 'darwin' else peak * 1024),
                median_epoch_s=statistics.median(e['seconds'] for e in history[args.warmup:]),
                selected_epoch=best_epoch, validation_accuracy=best_accuracy,
                test_accuracy=test_accuracy, history=history,
                environment=dict(platform=platform.platform(), architecture=platform.machine(),
                                 python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                 backend=str(device), thread_settings={k: os.environ.get(k) for k in
                                 ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/cora'))
    parser.add_argument('--output', type=Path, default=Path('results/benchmark.json'))
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 19, 41])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--validate-every', type=int, default=5)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--decay', type=float, default=0.0005)
    parser.add_argument('--layout', choices=['sparse','dense'], help=argparse.SUPPRESS)
    args=parser.parse_args()
    if not 0 <= args.warmup < args.epochs or args.validate_every < 1:
        parser.error('need 0 <= warmup < epochs and validate-every >= 1')
    if args.layout:
        print(json.dumps(worker(args), allow_nan=False))
        return
    runs=[]
    for i,seed in enumerate(args.seeds):
        # Alternate order across pairs to reduce systematic warm-machine effects.
        layouts = ('sparse','dense') if i % 2 == 0 else ('dense','sparse')
        for layout in layouts:
            cmd=[sys.executable,'-m','apps.experiment','--data',str(args.data),'--seeds',str(seed),
                 '--layout',layout,'--epochs',str(args.epochs),'--warmup',str(args.warmup),
                 '--validate-every',str(args.validate_every),'--hidden',str(args.hidden),
                 '--dropout',str(args.dropout),'--lr',str(args.lr),'--decay',str(args.decay)]
            env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
            finished=subprocess.run(cmd,env=env,check=True,stdout=subprocess.PIPE,text=True)
            # The homework backend prints its selected implementation on import.
            row=json.loads(finished.stdout.splitlines()[-1])
            runs.append(row)
            print(f"{layout:6s} seed {seed} | {row['median_epoch_s']:.4f} s/epoch | "
                  f"val {row['validation_accuracy']:.3f} | test {row['test_accuracy']:.3f}", flush=True)
    save_json(args.output, dict(runs=runs))


if __name__ == '__main__':
    main()
