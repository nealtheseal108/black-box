# Phrase → Market Signal Map (Brief §4.1)

Seed mapping of Warsh rhetorical phrases to trading signal axes.
C2 lift scores refine this; high-lift bigrams/trigrams surface new candidates.

| Phrase | Axis | Weight | Market Implication |
|---|---|---|---|
| inflation is a choice | hawkish | 0.9 | Rate hike / hold more likely; fade dovish markets |
| without excuse or equivocation | hawkish | 1.0 | Strongest hawkish signal; strong conviction language |
| foursquare within its role | independence | 0.8 | Fed defending mandate vs. political pressure; fade political-override bets |
| forays far afield | hawkish | 0.6 | Criticism of scope creep; hawkish/credibility signal |
| misallocation of capital | qt | 0.7 | Balance sheet / QT concern; supports faster runoff bets |
| bloated balance sheet | qt | 0.8 | Direct QT advocacy; supports runoff / rate higher |
| regime change | independence | 0.6 | Institutional concern; uncertainty premium signal |
| fiscal dominance | independence | 0.9 | Fed independence at risk from fiscal; hawkish/independence signal |
| stronger not hotter | dovish | 0.7 | Supply-side optimism; dovish framing; against premature hiking |

## Axis Definitions

- **hawkish** — language indicating preference for tighter monetary policy, higher rates, or credibility defense
- **dovish** — language indicating tolerance for accommodation, supply-side framing, or growth priority
- **independence** — language defending central bank independence from fiscal or political interference
- **qt** — language signaling concern about balance sheet size; preference for quantitative tightening

## Usage

These phrases are encoded in `src/warsh/lexicon.py` as `SIGNAL_LEXICON`.
`score_diction(text)` aggregates axis weights for any input text.
`phrase_signals(text)` returns matched entries with phrase, axis, and weight.

C2 lift scoring (`src/warsh/fingerprints.py`) identifies additional high-frequency
Warsh phrases vs. baseline English. Top candidates should be reviewed and added to
`SIGNAL_LEXICON` with appropriate axis assignments.
