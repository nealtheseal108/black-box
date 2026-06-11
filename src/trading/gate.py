"""Signal gate — INTERFACES.md §3 G3.

Fire only when abs(edge) > threshold AND net edge after fees > G3_NET_EDGE_FLOOR.
An edge that vanishes after fees is not an edge.
"""

G3_NET_EDGE_FLOOR = 0.06


def net_edge(signal, fee: float) -> float:
    """Return the gross absolute edge minus the fee."""
    return abs(signal.edge) - fee


def passes_gate(signal, threshold: float, fee: float) -> bool:
    """Return True iff the signal clears both the threshold and the net-of-fees floor."""
    return abs(signal.edge) > threshold and net_edge(signal, fee) > G3_NET_EDGE_FLOOR
