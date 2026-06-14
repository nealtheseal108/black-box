from __future__ import annotations

from src.trading.markets import MarketLink
from live_run import replay_hearing_with_trading


class _FakeQuotes:
    def __init__(self, mapping):
        self._mapping = mapping

    def quotes_for(self, canonical):
        return {"kalshi": 0.30} if canonical in self._mapping else {}


def test_replay_with_trading_places_and_settles(tmp_path):
    mapping = {
        "inflation": MarketLink("inflation", "K-INF", "P-INF"),
        "rate cut": MarketLink("rate cut", "K-RC", "P-RC"),
        "soft landing": MarketLink("soft landing", "K-SL", "P-SL"),
    }
    result = replay_hearing_with_trading(
        quotes=_FakeQuotes(mapping), mapping=mapping,
        paper_log=tmp_path / "paper.jsonl", episode_log=tmp_path / "ep.jsonl",
    )
    assert result["n_trades"] >= 2
    assert result["n_episodes"] == result["n_trades"]
    assert "total_pnl" in result
