from src.trading.markets import MarketLink
from src.trading.client import PaperKalshiClient
from src.live.episode_log import EpisodeLogger
from src.trading.executor import TradeExecutor


class _FakeQuotes:
    def __init__(self, table):
        self._table = table     # {canonical: {venue: price}}
    def quotes_for(self, canonical):
        return self._table.get(canonical, {})


def _mapping():
    return {
        "rate cut": MarketLink("rate cut", "K-RC", "P-RC"),
        "trump": MarketLink("trump", "K-TR", "P-TR"),
    }


def _executor(quotes_table, tmp_path):
    return TradeExecutor(
        mapping=_mapping(),
        quotes=_FakeQuotes(quotes_table),
        client=PaperKalshiClient(log_path=tmp_path / "paper.jsonl"),
        episode_log=EpisodeLogger(tmp_path / "episodes.jsonl"),
        bankroll=None, threshold=0.08, fee=0.02,
    )


def test_routes_to_higher_net_edge_venue(tmp_path):
    ex = _executor({"rate cut": {"kalshi": 0.60, "polymarket": 0.70}}, tmp_path)
    opened = ex.evaluate({"rate cut": 0.9})
    assert len(opened) == 1
    assert opened[0].venue == "kalshi"
    assert opened[0].order.side == "yes"


def test_skips_term_below_gate(tmp_path):
    ex = _executor({"rate cut": {"kalshi": 0.60}}, tmp_path)
    assert ex.evaluate({"rate cut": 0.62}) == []


def test_one_shot_does_not_retrade(tmp_path):
    ex = _executor({"rate cut": {"kalshi": 0.60}}, tmp_path)
    first = ex.evaluate({"rate cut": 0.9})
    second = ex.evaluate({"rate cut": 0.95})
    assert len(first) == 1 and second == []


def test_refuses_unmapped_term(tmp_path):
    ex = _executor({"soft landing": {"kalshi": 0.10}}, tmp_path)
    assert ex.evaluate({"soft landing": 0.9}) == []


def test_settle_logs_episode_with_pnl_sign(tmp_path):
    import json
    ep = tmp_path / "episodes.jsonl"
    ex = TradeExecutor(mapping=_mapping(), quotes=_FakeQuotes({"rate cut": {"kalshi": 0.60}}),
                       client=PaperKalshiClient(log_path=tmp_path / "p.jsonl"),
                       episode_log=EpisodeLogger(ep), bankroll=None, threshold=0.08, fee=0.02)
    ex.evaluate({"rate cut": 0.9})
    ex.settle({"rate cut": 1})
    rec = json.loads(ep.read_text().strip().splitlines()[0])
    assert rec["reward"] > 0
    assert rec["action"]["venue"] == "kalshi"
