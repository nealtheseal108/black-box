from src.live.streamtext import ReplayStream, StreamTextClient


def test_replay_stream_yields_chunks_covering_full_text():
    text = " ".join(str(i) for i in range(50))   # 50 words
    chunks = list(ReplayStream(text, chunk_words=20).stream())
    assert len(chunks) == 3                        # 20 + 20 + 10
    assert " ".join(chunks).split() == text.split()


def test_streamtext_client_parses_json_and_advances_cursor():
    pages = iter([
        {"content": "the fed will ", "lastPosition": 13},
        {"content": "cut rates", "lastPosition": 22},
        {"content": "", "lastPosition": 22},
    ])
    def fake_fetch(event, last):
        return next(pages)
    client = StreamTextClient(event_id="CFI-FRB", fetch=fake_fetch, poll_idle_limit=1)
    deltas = list(client.stream())
    assert deltas == ["the fed will ", "cut rates"]
