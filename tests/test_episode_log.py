import json
from src.live.episode_log import EpisodeLogger


def test_logs_state_action_reward_as_jsonl(tmp_path):
    path = tmp_path / "episodes.jsonl"
    logger = EpisodeLogger(path)
    logger.log(state={"rate cut": 0.6}, action={"ticker": "X", "size": 2}, reward=1.0)
    logger.log(state={"rate cut": 1.0}, action={"ticker": "X", "size": 0}, reward=0.0)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["state"] == {"rate cut": 0.6}
    assert first["action"] == {"ticker": "X", "size": 2}
    assert first["reward"] == 1.0
    assert "ts" in first
