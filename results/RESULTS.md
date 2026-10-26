# Results

SkillSpan test set: 3,569 sentences from 65 job postings. Span F1 in %; strict = exact boundaries,
loose = any overlap. Intervals: 95 % bootstrap over postings. The last column is a paired bootstrap
against the best system (difference in strict micro F1, and its interval).

| System | Strict F1 (micro) | 95 % interval | Skill | Knowledge | Loose F1 | vs best |
|---|---:|---|---:|---:|---:|---|
| Lexicon (no learning) | **18.6** | 15.9–21.7 | 12.3 | 21.2 | 30.0 | — |

## Settings

- **Lexicon (no learning)**: entries = 3768
