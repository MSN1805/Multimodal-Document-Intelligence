# Evaluation Report — Multimodal Document Intelligence

## Evaluation Version
V2.2

## Dataset
- Document: `EJ1172284.pdf`
- Total questions: 18
- Answerable questions: 16
- Unanswerable questions: 2

## Retrieval Performance

| Metric | Result |
|---|---:|
| Hit@1 | 87.50% |
| Recall@3 | 93.75% |
| Recall@5 | 93.75% |
| MRR | 88.54% |

## Performance

| Measurement | Result |
|---|---:|
| Document processing time | 6.395 s |
| Average retrieval time | 0.0309 s/query |
| Chunks | 54 |
| Vectors | 54 |
| Extracted tables | 33 |

## Remaining Challenging Questions

### Q04 — Most frequently used mobile device
- Expected page: 5
- Retrieved pages: 4, 4, 8, 7, 8

### Q09 — Most frequent usage location
- Expected page: 7
- Retrieved pages: 4, 4, 8, 7, 8

## Evaluation Notes

The reported retrieval metrics are calculated using the 16 answerable questions.
Q15 and Q16 are explicitly unanswerable benchmark questions and are excluded
from normal retrieval-success metrics.

The full RAG evaluation was skipped because local LLM inference adds
significant execution time.

## Conclusion

V2.2 provides strong retrieval performance on the current benchmark, with
relevant evidence appearing within the top three results for most answerable
questions. The remaining errors are concentrated in highly specific factual
retrieval cases.
