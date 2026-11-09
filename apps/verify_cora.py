"""Compare one actual full-graph training step with identical weights and no dropout."""
import argparse
from pathlib import Path

import numpy as np

import needle as ndl
from apps.experiment import check_data, save_json
from apps.gcn import GCN, load_cora, make_adjacency, masked_cross_entropy, prepare_inputs


def one_step(graph, layout, seed):
    device = ndl.cpu()
    np.random.seed(seed)
    model = GCN(graph.features.shape[1], 16, graph.num_classes, dropout=0, device=device)
    model.eval()
    adj = make_adjacency(graph, layout, device=device)
    features, labels, train_mask = prepare_inputs(graph, device=device)
    logits = model(features, adj)
    loss = masked_cross_entropy(logits, labels, train_mask)
    optimizer = ndl.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss.backward()
    measured = {
        'logits': logits.numpy().copy(),
        'loss': loss.numpy().copy(),
        'weight1_gradient': model.W1.grad.numpy().copy(),
        'weight2_gradient': model.W2.grad.numpy().copy(),
    }
    optimizer.step()
    measured.update(weight1_after_step=model.W1.numpy().copy(), weight2_after_step=model.W2.numpy().copy())
    return measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/cora'))
    parser.add_argument('--output', type=Path, default=Path('results/correctness.json'))
    args = parser.parse_args()
    check_data(args.data)
    graph = load_cora(args.data)
    seed = 7
    dense = one_step(graph, 'dense', seed)
    sparse = one_step(graph, 'sparse', seed)
    errors = {}
    for name, reference in dense.items():
        np.testing.assert_allclose(sparse[name], reference, rtol=1e-4, atol=1e-5, err_msg=name)
        errors[name] = float(np.abs(sparse[name] - reference).max())
    result = dict(seed=seed, dataset='Cora', dropout=0, optimizer='Adam',
                  atol=1e-5, rtol=1e-4, dense_loss=float(dense['loss'].item()),
                  sparse_loss=float(sparse['loss'].item()), max_abs_error=errors)
    save_json(args.output, result)
    for name, error in errors.items():
        print(f'{name:22s} {error:.8g}')


if __name__ == '__main__':
    main()
