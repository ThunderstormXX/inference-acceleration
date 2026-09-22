# Causal n-gram feasibility on saved Qwen trajectories

This document describes CPU-only replay; **its simulated rounds are not measured inference speedup**. Subsequent GPU experiments are recorded separately in the [local experiment log](local-speed-experiments-2026-09-22.md).

11 unique trajectories, 22528 output tokens. Each prompt is available at the start; only already committed output is added. The first generated token is treated as the prefill seed and excluded from decode positions.

Rows 100–109 come from the earlier confidence collection; row 110 uses its AR-before output. All were previously inspected, so this is development material, not an unseen test set.

The proposer chooses the longest suffix (up to 16 tokens), then the latest earlier occurrence with a known continuation. It never reads another chain or extrapolates beyond its committed history. Later true tokens are used only to label the already-issued proposal.

## Every-position diagnostic

Overlapping positions are dependent; the same tokens can contribute to several candidate windows. Coverage is the fraction of decode positions with a proposal. Accuracy is the first-token accuracy conditional on a proposal; accepted length counts only the uninterrupted correct prefix.

| Minimum suffix | Max draft | Coverage | First correct | Mean accepted prefix | All offered accepted |
|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 64.0% | 60.6% | 0.606 | 60.6% |
| 2 | 2 | 64.0% | 60.6% | 1.006 | 40.4% |
| 2 | 4 | 64.0% | 60.6% | 1.471 | 19.4% |
| 4 | 1 | 35.3% | 66.7% | 0.667 | 66.7% |
| 4 | 2 | 35.3% | 66.7% | 1.114 | 45.5% |
| 4 | 4 | 35.3% | 66.7% | 1.686 | 25.4% |
| 8 | 1 | 11.2% | 75.4% | 0.754 | 75.4% |
| 8 | 2 | 11.2% | 75.4% | 1.319 | 58.6% |
| 8 | 4 | 11.2% | 75.4% | 2.133 | 38.5% |

## Simulated causal rounds

After each lookup the replay commits the matching prefix plus one correction/bonus token, capped at the budget. No match commits one AR token. These counts describe a hypothetical exact verifier; its batched numerical behavior, GDN rollback, GPU cost and Python/Metal synchronization are unmeasured. **Tokens per simulated round is not a speedup**, since wider rounds cost more.

| Minimum suffix | Max draft | Rounds | Draft rounds | Tokens / round | First correct | CPU lookup µs/query |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 17205 | 9095 | 1.309 | 58.4% | 3.59 |
| 2 | 2 | 15678 | 7568 | 1.436 | 55.2% | 3.66 |
| 2 | 4 | 14469 | 6359 | 1.556 | 53.9% | 3.66 |
| 4 | 1 | 19324 | 4747 | 1.165 | 67.3% | 3.38 |
| 4 | 2 | 18363 | 3786 | 1.226 | 67.4% | 3.39 |
| 4 | 4 | 17725 | 3148 | 1.270 | 65.9% | 3.47 |
| 8 | 1 | 21441 | 1451 | 1.050 | 74.2% | 2.68 |
| 8 | 2 | 21102 | 1112 | 1.067 | 73.7% | 2.72 |
| 8 | 4 | 20847 | 857 | 1.080 | 72.8% | 2.75 |

CPU figures time `propose` only in one replay pass, excluding indexing updates, token transport and all GPU work. They are descriptive and are not subtracted from or converted into GPU savings.

The JSON includes per-chain metrics, accepted-prefix histograms and bounded examples of full/partial/zero acceptance. No best hyperparameter is selected from these labels.

## Reproduction

```bash
bash scripts/analysis/ngram_feasibility.sh --bundle docs/results/ngram-feasibility-inputs-2026-09-22.json.gz
```

The bundle contains only prompt/output token IDs and original source hashes. Run without `--bundle` to read the original local experiment artifacts. `--widths 1 2 4 --min-matches 2 4 8 --max-match 16` are the defaults.
