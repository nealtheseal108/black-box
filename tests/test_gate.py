from src.trading.types import Signal
from src.trading.gate import passes_gate, net_edge


def sig(prob, price): return Signal.from_quote("X", prob, price, "t")


def test_gate_rejects_below_threshold():
    assert not passes_gate(sig(0.57, 0.55), threshold=0.08, fee=0.0)  # edge 0.02 < 0.08


def test_gate_rejects_when_fees_eat_the_edge():
    # gross edge 0.09 > 0.08 threshold, but fee 0.04 leaves net 0.05 < 0.06 floor
    assert not passes_gate(sig(0.64, 0.55), threshold=0.08, fee=0.04)


def test_gate_passes_when_gross_and_net_clear():
    assert passes_gate(sig(0.65, 0.55), threshold=0.08, fee=0.02)  # edge .10, net .08


def test_net_edge_subtracts_fee_from_abs_edge():
    assert abs(net_edge(sig(0.65, 0.55), fee=0.02) - 0.08) < 1e-9
