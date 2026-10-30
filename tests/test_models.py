import torch

from tinysleep.models import CONFIGS, TinySleepCNN, cost, receptive_field, stack_context


def test_param_count_matches_cost_model():
    for name in CONFIGS:
        for K in (0, 4):
            m = TinySleepCNN(name, context=K)
            n_bn_extra = sum(b.weight.numel() for b in m.modules() if isinstance(b, torch.nn.BatchNorm1d))
            # after folding, each BN (gamma, beta) becomes one bias per channel
            assert sum(p.numel() for p in m.parameters()) - n_bn_extra == cost(name, K)["params"]


def test_receptive_field_formula():
    assert receptive_field([(1, 3, 1, 1)]) == 3
    assert receptive_field([(1, 3, 1, 2), (1, 3, 1, 1)]) == 3 + 1 + 2 * 2  # pool adds 1, next conv 2*j


def test_forward_shapes():
    m = TinySleepCNN("xs", context=2).eval()
    E = m.encoder(torch.zeros(7, 3000))
    assert E.shape == (7, m.encoder.dim)
    assert m.classify(stack_context(E, 2)).shape == (7, 5)


def test_stack_context_is_causal_and_edge_padded():
    E = torch.arange(4.0)[:, None]
    assert stack_context(E, 2).tolist() == [[0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 2, 1]]
