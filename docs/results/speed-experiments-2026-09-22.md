# Local speed experiments

Experimental measurements, separated by protocol; no cross-experiment ranking.

Numbers are mean ± sample SD across measured requests. Repeated prompts are dependent: request count is **not** a count of independent tasks. SD describes these runs, not uncertainty over a task population. Missing SD for a single observation is not zero variance. References/brackets are excluded from means.

A failed/unknown parity, sleep event, incomplete run or unresolved source warning prevents a speedup claim. Reported rates remain visible for diagnosis. An absolute AR start/end bracket drift above 5% also makes the experiment diagnostic, without invalidating its raw measurements. Even a valid comparison is empirical for these prompts.

## 20260922-affine-confirm

Family: `kernel_sweep`; status: **completed**. Output budget: 128; requested repeats: 2. Trace enabled: False.

DFlash block size: 3.

- First output token belongs to prefill; decode is N-1 tokens.
- Identical target weights; fresh cache per request; warm paths before timing.
- Variant order reverses on alternate repetitions; stock AR brackets entire sweep.
- DFlash draft remains resident during AR if any DFlash variant requested.
- Token parity is empirical for these prompts, not a universal mathematical guarantee.

| Variant | Decode tok/s | Paired ratio vs AR | Runs | Unique prompts | Observed repeats | Parity | Awake | Status |
|---|---:|---:|---:|---:|---|---|---|---|
| ar | 36.403 ± 0.304 | — | 6 | 3 | [0, 1] | PASS | PASS | control |
| ar-target | 34.880 ± 0.093 | 0.958 ± 0.007 | 6 | 3 | [0, 1] | PASS | PASS | validated experimental comparison |
| dflash | 35.567 ± 1.436 | 0.977 ± 0.046 | 6 | 3 | [0, 1] | PASS | PASS | validated experimental comparison |
| dflash-all-argmax | 40.161 ± 1.512 | 1.104 ± 0.049 | 6 | 3 | [0, 1] | PASS | PASS | validated experimental comparison |
| dflash-target | 39.932 ± 1.670 | 1.097 ± 0.053 | 6 | 3 | [0, 1] | PASS | PASS | validated experimental comparison |
| dflash-target-argmax | 39.994 ± 1.492 | 1.099 ± 0.049 | 6 | 3 | [0, 1] | PASS | PASS | validated experimental comparison |

- AR drift bracket: `{"after_over_before": 1.0035805491416703, "after_tok_s": 36.52472075655789, "before_tok_s": 36.39440878741252, "prompt_index": 0, "source": "raw AR bracket measurements"}`.

## 20260922-affine-scout

Family: `kernel_sweep`; status: **completed**. Output budget: 64; requested repeats: 1. Trace enabled: False.

DFlash block size: 3.

- First output token belongs to prefill; decode is N-1 tokens.
- Identical target weights; fresh cache per request; warm paths before timing.
- Variant order reverses on alternate repetitions; stock AR brackets entire sweep.
- DFlash draft remains resident during AR if any DFlash variant requested.
- Token parity is empirical for these prompts, not a universal mathematical guarantee.

| Variant | Decode tok/s | Paired ratio vs AR | Runs | Unique prompts | Observed repeats | Parity | Awake | Status |
|---|---:|---:|---:|---:|---|---|---|---|
| ar | 36.343 (SD n/a) | — | 1 | 1 | [0] | PASS | PASS | control |
| ar-argmax | 36.339 (SD n/a) | 1.000 (SD n/a) | 1 | 1 | [0] | PASS | PASS | validated experimental comparison |
| ar-head | 36.309 (SD n/a) | 0.999 (SD n/a) | 1 | 1 | [0] | PASS | PASS | validated experimental comparison |
| ar-target | 34.944 (SD n/a) | 0.961 (SD n/a) | 1 | 1 | [0] | PASS | PASS | validated experimental comparison |
| dflash | 37.091 (SD n/a) | 1.021 (SD n/a) | 1 | 1 | [0] | PASS | PASS | validated experimental comparison |
| dflash-all | 38.007 (SD n/a) | 1.046 (SD n/a) | 1 | 1 | [0] | PASS | PASS | validated experimental comparison |
| dflash-all-argmax | 41.729 (SD n/a) | 1.148 (SD n/a) | 1 | 1 | [0] | PASS | PASS | validated experimental comparison |
| dflash-argmax | 37.413 (SD n/a) | 1.029 (SD n/a) | 1 | 1 | [0] | PASS | PASS | validated experimental comparison |
| dflash-head | 37.855 (SD n/a) | 1.042 (SD n/a) | 1 | 1 | [0] | PASS | PASS | validated experimental comparison |
| dflash-target | 41.361 (SD n/a) | 1.138 (SD n/a) | 1 | 1 | [0] | PASS | PASS | validated experimental comparison |

- AR drift bracket: `{"after_over_before": 0.9924912267167156, "after_tok_s": 36.282149002737654, "before_tok_s": 36.556644558726745, "prompt_index": 0, "source": "raw AR bracket measurements"}`.

## 20260922-mtp-k3-confirm

Family: `mtp_sweep`; status: **completed**. Output budget: 256; requested repeats: 2. Trace enabled: False.

- Fixed-budget greedy; EOS ignored; first token in prefill, N-1 decode tokens.
- One target object; draft weights remain resident for AR; fresh request caches.
- Every K has a neighboring matched AR; pair order and depth order reverse across repetitions.
- All methods warmed for 16 tokens; start/end AR drift bracket excluded from averages.
- Timings include final cache materialization; text decoding and host snapshots are outside decode.
- Stdev describes this small repeated-prompt diagnostic, not independent task-population uncertainty.

| Variant | Decode tok/s | Paired ratio vs AR | Runs | Unique prompts | Observed repeats | Parity | Awake | Status |
|---|---:|---:|---:|---:|---|---|---|---|
| ar_k1 | 31.209 ± 0.134 | — | 6 | 3 | [0, 1] | PASS | PASS | control |
| mtp_k3 | 38.373 ± 0.544 | 1.230 ± 0.022 | 6 | 3 | [0, 1] | PASS | PASS | validated experimental comparison |

- AR drift bracket: `{"after_over_before": 1.0060412349553256, "after_tok_s": 31.348712427750435, "before_tok_s": 31.160464738945326, "prompt_index": 0, "source": "raw AR bracket measurements"}`.

## 20260922-mtp-scout

Family: `mtp_sweep`; status: **completed**. Output budget: 128; requested repeats: 1. Trace enabled: False.

- Fixed-budget greedy; EOS ignored; first token in prefill, N-1 decode tokens.
- One target object; draft weights remain resident for AR; fresh request caches.
- Every K has a neighboring matched AR; pair order and depth order reverse across repetitions.
- All methods warmed for 16 tokens; start/end AR drift bracket excluded from averages.
- Timings include final cache materialization; text decoding and host snapshots are outside decode.
- Stdev describes this small repeated-prompt diagnostic, not independent task-population uncertainty.

| Variant | Decode tok/s | Paired ratio vs AR | Runs | Unique prompts | Observed repeats | Parity | Awake | Status |
|---|---:|---:|---:|---:|---|---|---|---|
| ar_k1 | 36.226 ± 0.141 | — | 4 | 1 | [0] | PASS | PASS | control |
| mtp_k2 | 38.987 (SD n/a) | 1.073 (SD n/a) | 1 | 1 | [0] | PASS | PASS | validated experimental comparison |
| mtp_k3 | 43.521 (SD n/a) | 1.206 (SD n/a) | 1 | 1 | [0] | PASS | PASS | validated experimental comparison |
| mtp_k4 | 10.895 (SD n/a) | 0.301 (SD n/a) | 1 | 1 | [0] | PASS | PASS | validated experimental comparison |
| mtp_k5 | 7.428 (SD n/a) | 0.204 (SD n/a) | 1 | 1 | [0] | PASS | PASS | validated experimental comparison |

- AR drift bracket: `{"after_over_before": 1.0083953200139204, "after_tok_s": 36.36403592530187, "before_tok_s": 36.06128985683897, "prompt_index": 0, "source": "raw AR bracket measurements"}`.

## 20260922-mtp-stream3-scout

Family: `mtp_sweep`; status: **completed**. Output budget: 128; requested repeats: 1. Trace enabled: False.

- Fixed-budget greedy; EOS ignored; first token in prefill, N-1 decode tokens.
- One target object; draft weights remain resident for AR; fresh request caches.
- Every K has a neighboring matched AR; pair order and depth order reverse across repetitions.
- All methods warmed for 16 tokens; start/end AR drift bracket excluded from averages.
- Timings include final cache materialization; text decoding and host snapshots are outside decode.
- Stdev describes this small repeated-prompt diagnostic, not independent task-population uncertainty.

| Variant | Decode tok/s | Paired ratio vs AR | Runs | Unique prompts | Observed repeats | Parity | Awake | Status |
|---|---:|---:|---:|---:|---|---|---|---|
| ar_k1 | 36.335 ± 0.168 | — | 3 | 1 | [0] | PASS | PASS | control |
| mtp_k3 | 24.825 (SD n/a) | 0.681 (SD n/a) | 1 | 1 | [0] | PASS | PASS | validated experimental comparison |
| mtp_k4 | 23.414 (SD n/a) | 0.643 (SD n/a) | 1 | 1 | [0] | PASS | PASS | validated experimental comparison |
| mtp_k5 | 22.052 (SD n/a) | 0.610 (SD n/a) | 1 | 1 | [0] | PASS | PASS | validated experimental comparison |

- AR drift bracket: `{"after_over_before": 0.9866938832160033, "after_tok_s": 35.94080704768057, "before_tok_s": 36.425488856316896, "prompt_index": 0, "source": "raw AR bracket measurements"}`.

## 20260922-mtp-tiled2-scout

Family: `mtp_sweep`; status: **completed**. Output budget: 128; requested repeats: 1. Trace enabled: False.

- Fixed-budget greedy; EOS ignored; first token in prefill, N-1 decode tokens.
- One target object; draft weights remain resident for AR; fresh request caches.
- Every K has a neighboring matched AR; pair order and depth order reverse across repetitions.
- All methods warmed for 16 tokens; start/end AR drift bracket excluded from averages.
- Timings include final cache materialization; text decoding and host snapshots are outside decode.
- Stdev describes this small repeated-prompt diagnostic, not independent task-population uncertainty.

| Variant | Decode tok/s | Paired ratio vs AR | Runs | Unique prompts | Observed repeats | Parity | Awake | Status |
|---|---:|---:|---:|---:|---|---|---|---|
| ar_k1 | 30.468 ± 0.074 | — | 4 | 1 | [0] | PASS | PASS | control |
| mtp_k2 | 31.689 (SD n/a) | 1.041 (SD n/a) | 1 | 1 | [0] | PASS | PASS | validated experimental comparison |
| mtp_k3 | 18.163 (SD n/a) | 0.594 (SD n/a) | 1 | 1 | [0] | PASS | PASS | validated experimental comparison |
| mtp_k4 | 17.539 (SD n/a) | 0.576 (SD n/a) | 1 | 1 | [0] | PASS | PASS | validated experimental comparison |
| mtp_k5 | 16.510 (SD n/a) | 0.543 (SD n/a) | 1 | 1 | [0] | PASS | PASS | validated experimental comparison |

- AR drift bracket: `{"after_over_before": 1.0112722669043177, "after_tok_s": 30.517894674983197, "before_tok_s": 30.17772332312033, "prompt_index": 0, "source": "raw AR bracket measurements"}`.

## 20260922-mtp-vocab-scout

Family: `mtp_sweep`; status: **completed**. Output budget: 128; requested repeats: 2. Trace enabled: False.

- Fixed-budget greedy; EOS ignored; first token in prefill, N-1 decode tokens.
- One target object; draft weights remain resident for AR; fresh request caches.
- Every K has a neighboring matched AR; pair order and depth order reverse across repetitions.
- All methods warmed for 16 tokens; start/end AR drift bracket excluded from averages.
- Timings include final cache materialization; text decoding and host snapshots are outside decode.
- Stdev describes this small repeated-prompt diagnostic, not independent task-population uncertainty.

| Variant | Decode tok/s | Paired ratio vs AR | Runs | Unique prompts | Observed repeats | Parity | Awake | Status |
|---|---:|---:|---:|---:|---|---|---|---|
| ar_k1 | 36.575 ± 0.158 | — | 4 | 1 | [0, 1] | PASS | PASS | control |
| mtp-stock_k3 | 43.928 ± 1.579 | 1.203 ± 0.050 | 2 | 1 | [0, 1] | PASS | PASS | validated experimental comparison |
| mtp_k3 | 37.986 ± 0.252 | 1.037 ± 0.010 | 2 | 1 | [0, 1] | PASS | PASS | validated experimental comparison |

Direct draft comparison: shortlist MTP versus stock full-vocabulary draft, matched by prompt, repeat and block size. This ratio is separate from each method's neighboring AR control.

| Block size | Shortlist / stock ratio | Pairs | Unique prompts | Parity | Status |
|---:|---:|---:|---:|---|---|
| 3 | 0.865 ± 0.025 | 2 | 1 | PASS | validated experimental comparison |

- AR drift bracket: `{"after_over_before": 1.00391500026661, "after_tok_s": 36.66489917423125, "before_tok_s": 36.521915863887024, "prompt_index": 0, "source": "raw AR bracket measurements"}`.

## 20260922-mtp-vocab32k-confirm

Family: `mtp_sweep`; status: **completed**. Output budget: 256; requested repeats: 2. Trace enabled: False.

- Fixed-budget greedy; EOS ignored; first token in prefill, N-1 decode tokens.
- One target object; draft weights remain resident for AR; fresh request caches.
- Every K has a neighboring matched AR; pair order and depth order reverse across repetitions.
- All methods warmed for 16 tokens; start/end AR drift bracket excluded from averages.
- Timings include final cache materialization; text decoding and host snapshots are outside decode.
- Stdev describes this small repeated-prompt diagnostic, not independent task-population uncertainty.

| Variant | Decode tok/s | Paired ratio vs AR | Runs | Unique prompts | Observed repeats | Parity | Awake | Status |
|---|---:|---:|---:|---:|---|---|---|---|
| ar_k1 | 33.092 ± 3.704 | — | 12 | 3 | [0, 1] | PASS | PASS | diagnostic only: AR bracket drift -20.2% exceeds 5%; diagnostic only; source validation warnings |
| mtp-stock_k3 | 39.923 ± 3.586 | 1.209 ± 0.039 | 6 | 3 | [0, 1] | PASS | PASS | diagnostic only: AR bracket drift -20.2% exceeds 5%; diagnostic only; source validation warnings |
| mtp_k3 | 40.619 ± 3.879 | 1.232 ± 0.069 | 6 | 3 | [0, 1] | PASS | PASS | diagnostic only: AR bracket drift -20.2% exceeds 5%; diagnostic only; source validation warnings |

Direct draft comparison: shortlist MTP versus stock full-vocabulary draft, matched by prompt, repeat and block size. This ratio is separate from each method's neighboring AR control.

| Block size | Shortlist / stock ratio | Pairs | Unique prompts | Parity | Status |
|---:|---:|---:|---:|---|---|
| 3 | 1.018 ± 0.033 | 6 | 3 | PASS | diagnostic only |

- Source warning: AR bracket drift -20.2% exceeds 5%; diagnostic only
- AR drift bracket: `{"after_over_before": 0.798446105892078, "after_tok_s": 29.194081772909797, "before_tok_s": 36.56362221253768, "prompt_index": 0, "source": "raw AR bracket measurements"}`.

## 20260922-mtp-vocab32k-recheck

Family: `mtp_sweep`; status: **completed**. Output budget: 256; requested repeats: 2. Trace enabled: False.

- Fixed-budget greedy; EOS ignored; first token in prefill, N-1 decode tokens.
- One target object; draft weights remain resident for AR; fresh request caches.
- Every K has a neighboring matched AR; pair order and depth order reverse across repetitions.
- All methods warmed for 16 tokens; start/end AR drift bracket excluded from averages.
- Timings include final cache materialization; text decoding and host snapshots are outside decode.
- Stdev describes this small repeated-prompt diagnostic, not independent task-population uncertainty.

| Variant | Decode tok/s | Paired ratio vs AR | Runs | Unique prompts | Observed repeats | Parity | Awake | Status |
|---|---:|---:|---:|---:|---|---|---|---|
| ar_k1 | 30.401 ± 0.136 | — | 12 | 3 | [0, 1] | PASS | PASS | control |
| mtp-stock_k3 | 37.363 ± 0.262 | 1.229 ± 0.010 | 6 | 3 | [0, 1] | PASS | PASS | validated experimental comparison |
| mtp_k3 | 38.025 ± 1.451 | 1.250 ± 0.049 | 6 | 3 | [0, 1] | PASS | PASS | validated experimental comparison |

Direct draft comparison: shortlist MTP versus stock full-vocabulary draft, matched by prompt, repeat and block size. This ratio is separate from each method's neighboring AR control.

| Block size | Shortlist / stock ratio | Pairs | Unique prompts | Parity | Status |
|---:|---:|---:|---:|---|---|
| 3 | 1.018 ± 0.040 | 6 | 3 | PASS | validated experimental comparison |

- AR drift bracket: `{"after_over_before": 0.9682707347316344, "after_tok_s": 29.511446396602295, "before_tok_s": 30.4785070311783, "prompt_index": 0, "source": "raw AR bracket measurements"}`.

## 20260922-mtp-vocab32k-scout

Family: `mtp_sweep`; status: **completed**. Output budget: 128; requested repeats: 2. Trace enabled: False.

- Fixed-budget greedy; EOS ignored; first token in prefill, N-1 decode tokens.
- One target object; draft weights remain resident for AR; fresh request caches.
- Every K has a neighboring matched AR; pair order and depth order reverse across repetitions.
- All methods warmed for 16 tokens; start/end AR drift bracket excluded from averages.
- Timings include final cache materialization; text decoding and host snapshots are outside decode.
- Stdev describes this small repeated-prompt diagnostic, not independent task-population uncertainty.

| Variant | Decode tok/s | Paired ratio vs AR | Runs | Unique prompts | Observed repeats | Parity | Awake | Status |
|---|---:|---:|---:|---:|---|---|---|---|
| ar_k1 | 36.608 ± 0.102 | — | 4 | 1 | [0, 1] | PASS | PASS | control |
| mtp-stock_k3 | 43.750 ± 1.950 | 1.194 ± 0.049 | 2 | 1 | [0, 1] | PASS | PASS | validated experimental comparison |
| mtp_k3 | 47.673 ± 0.027 | 1.303 ± 0.004 | 2 | 1 | [0, 1] | PASS | PASS | validated experimental comparison |

Direct draft comparison: shortlist MTP versus stock full-vocabulary draft, matched by prompt, repeat and block size. This ratio is separate from each method's neighboring AR control.

| Block size | Shortlist / stock ratio | Pairs | Unique prompts | Parity | Status |
|---:|---:|---:|---:|---|---|
| 3 | 1.091 ± 0.048 | 2 | 1 | PASS | validated experimental comparison |

- AR drift bracket: `{"after_over_before": 1.0013683911413718, "after_tok_s": 36.63980573352046, "before_tok_s": 36.58973666200704, "prompt_index": 0, "source": "raw AR bracket measurements"}`.

## 20260922-mtp-vocab32k-trace512

Family: `mtp_sweep`; status: **completed**. Output budget: 512; requested repeats: 1. Trace enabled: True.

- Fixed-budget greedy; EOS ignored; first token in prefill, N-1 decode tokens.
- One target object; draft weights remain resident for AR; fresh request caches.
- Every K has a neighboring matched AR; pair order and depth order reverse across repetitions.
- All methods warmed for 16 tokens; start/end AR drift bracket excluded from averages.
- Timings include final cache materialization; text decoding and host snapshots are outside decode.
- Stdev describes this small repeated-prompt diagnostic, not independent task-population uncertainty.

| Variant | Decode tok/s | Paired ratio vs AR | Runs | Unique prompts | Observed repeats | Parity | Awake | Status |
|---|---:|---:|---:|---:|---|---|---|---|
| ar_k1 | 29.560 ± 0.208 | — | 2 | 1 | [0] | PASS | PASS | diagnostic only: AR bracket drift +16.9% exceeds 5%; diagnostic only; source validation warnings |
| mtp-stock_k3 | 34.082 (SD n/a) | 1.159 (SD n/a) | 1 | 1 | [0] | PASS | PASS | diagnostic only: AR bracket drift +16.9% exceeds 5%; diagnostic only; source validation warnings |
| mtp_k3 | 38.209 (SD n/a) | 1.286 (SD n/a) | 1 | 1 | [0] | PASS | PASS | diagnostic only: AR bracket drift +16.9% exceeds 5%; diagnostic only; source validation warnings |

Direct draft comparison: shortlist MTP versus stock full-vocabulary draft, matched by prompt, repeat and block size. This ratio is separate from each method's neighboring AR control.

| Block size | Shortlist / stock ratio | Pairs | Unique prompts | Parity | Status |
|---:|---:|---:|---:|---|---|
| 3 | 1.121 (SD n/a) | 1 | 1 | PASS | diagnostic only |

- Source warning: AR bracket drift +16.9% exceeds 5%; diagnostic only
- AR drift bracket: `{"after_over_before": 1.1691670731607982, "after_tok_s": 34.906768146503964, "before_tok_s": 29.85609922466843, "prompt_index": 0, "source": "raw AR bracket measurements"}`.

## 20260922-ngram-scout

Family: `ngram_sweep`; status: **completed**. Output budget: 128; requested repeats: 1. Trace enabled: False.

N-gram proposal widths: [1, 2, 4]; minimum suffixes: [2, 4]. Each variant is paired with its neighboring AR.

- Fixed-budget greedy; first token in prefill; EOS ignored.
- Same loaded target and no neural draft; fresh request caches.
- Longest suffix from current prompt and committed output only; no teacher answers.
- Transactional stock MTP verifier commits accepted+1 inputs; no direct hybrid-cache trimming.
- Each retrieval configuration has a neighboring AR; reverse pair order on repeat 2.
- Discarded explicit verifier warmups cover every width before measured generation.
- Per-round times are host observations; no extra per-operation GPU synchronization.

| Variant | Decode tok/s | Paired ratio vs AR | Runs | Unique prompts | Observed repeats | Parity | Awake | Status |
|---|---:|---:|---:|---:|---|---|---|---|
| ar | 32.595 ± 2.428 | — | 6 | 1 | [0] | PASS | PASS | diagnostic only: AR bracket drift +6.4% exceeds 5%; diagnostic only; source validation warnings |
| ngram-w1-m2 | 28.310 (SD n/a) | 0.985 (SD n/a) | 1 | 1 | [0] | PASS | PASS | diagnostic only: AR bracket drift +6.4% exceeds 5%; diagnostic only; source validation warnings |
| ngram-w1-m4 | 31.771 (SD n/a) | 0.913 (SD n/a) | 1 | 1 | [0] | PASS | PASS | diagnostic only: AR bracket drift +6.4% exceeds 5%; diagnostic only; source validation warnings |
| ngram-w2-m2 | 30.780 (SD n/a) | 0.916 (SD n/a) | 1 | 1 | [0] | PASS | PASS | diagnostic only: AR bracket drift +6.4% exceeds 5%; diagnostic only; source validation warnings |
| ngram-w2-m4 | 32.514 (SD n/a) | 0.938 (SD n/a) | 1 | 1 | [0] | PASS | PASS | diagnostic only: AR bracket drift +6.4% exceeds 5%; diagnostic only; source validation warnings |
| ngram-w4-m2 | 8.514 (SD n/a) | 0.257 (SD n/a) | 1 | 1 | [0] | PASS | PASS | diagnostic only: AR bracket drift +6.4% exceeds 5%; diagnostic only; source validation warnings |
| ngram-w4-m4 | 11.263 (SD n/a) | 0.368 (SD n/a) | 1 | 1 | [0] | PASS | PASS | diagnostic only: AR bracket drift +6.4% exceeds 5%; diagnostic only; source validation warnings |

- Source warning: AR bracket drift +6.4% exceeds 5%; diagnostic only
- AR drift bracket: `{"after_over_before": 1.0640510335410522, "after_tok_s": 30.619059213261817, "before_tok_s": 28.775931086091557, "prompt_index": 0, "source": "raw AR bracket measurements"}`.

## 20260922-ngram-traced-confirm

Family: `ngram_sweep`; status: **completed**. Output budget: 256; requested repeats: 2. Trace enabled: True.

N-gram proposal widths: [1, 2]; minimum suffixes: [2]. Each variant is paired with its neighboring AR.

- Fixed-budget greedy; first token in prefill; EOS ignored.
- Same loaded target and no neural draft; fresh request caches.
- Longest suffix from current prompt and committed output only; no teacher answers.
- Transactional stock MTP verifier commits accepted+1 inputs; no direct hybrid-cache trimming.
- Each retrieval configuration has a neighboring AR; reverse pair order on repeat 2.
- Discarded explicit verifier warmups cover every width before measured generation.
- Per-round times are host observations; no extra per-operation GPU synchronization.

| Variant | Decode tok/s | Paired ratio vs AR | Runs | Unique prompts | Observed repeats | Parity | Awake | Status |
|---|---:|---:|---:|---:|---|---|---|---|
| ar | 33.397 ± 3.786 | — | 4 | 1 | [0, 1] | PASS | PASS | diagnostic only: AR bracket drift -20.9% exceeds 5%; diagnostic only; source validation warnings |
| ngram-w1-m2 | 31.679 ± 4.963 | 0.950 ± 0.014 | 2 | 1 | [0, 1] | PASS | PASS | diagnostic only: AR bracket drift -20.9% exceeds 5%; diagnostic only; source validation warnings |
| ngram-w2-m2 | 33.157 ± 2.951 | 0.993 ± 0.046 | 2 | 1 | [0, 1] | PASS | PASS | diagnostic only: AR bracket drift -20.9% exceeds 5%; diagnostic only; source validation warnings |

- Source warning: AR bracket drift -20.9% exceeds 5%; diagnostic only
- AR drift bracket: `{"after_over_before": 0.7909739359206909, "after_tok_s": 28.89973686783232, "before_tok_s": 36.53690160370851, "prompt_index": 0, "source": "raw AR bracket measurements"}`.

## 20260922-zmlx-scout

Family: `kernel_sweep`; status: **unavailable**. Output budget: 64; requested repeats: 2. Trace enabled: False.

DFlash block size: 3.

- First output token belongs to prefill; decode is N-1 tokens.
- Identical target weights; fresh cache per request; warm paths before timing.
- Variant order reverses on alternate repetitions; stock AR brackets entire sweep.
- DFlash draft remains resident during AR if any DFlash variant requested.
- Token parity is empirical for these prompts, not a universal mathematical guarantee.

| Variant | Decode tok/s | Paired ratio vs AR | Runs | Unique prompts | Observed repeats | Parity | Awake | Status |
|---|---:|---:|---:|---:|---|---|---|---|
| ar | — | — | 0 | unknown | [] | unknown | unknown | diagnostic only: experiment status unavailable; experiment contains errors; source validation warnings; no raw measurements; token parity unknown; sleep status unknown; run-wide parity not confirmed; run-wide awake status not confirmed |
| zmlx-both | — | — | 0 | unknown | [] | unknown | unknown | diagnostic only: experiment status unavailable; experiment contains errors; source validation warnings; no raw measurements; token parity unknown; sleep status unknown; run-wide parity not confirmed; run-wide awake status not confirmed; not all measurements have an exact matched AR pair |
| zmlx-deltanet | — | — | 0 | unknown | [] | unknown | unknown | diagnostic only: experiment status unavailable; experiment contains errors; source validation warnings; no raw measurements; token parity unknown; sleep status unknown; run-wide parity not confirmed; run-wide awake status not confirmed; not all measurements have an exact matched AR pair |
| zmlx-swiglu | — | — | 0 | unknown | [] | unknown | unknown | diagnostic only: experiment status unavailable; experiment contains errors; source validation warnings; no raw measurements; token parity unknown; sleep status unknown; run-wide parity not confirmed; run-wide awake status not confirmed; not all measurements have an exact matched AR pair |

- Source warning: completed raw run count does not match the requested protocol
- Recorded errors: 4; details preserved in JSON.

## 20260922-zmlx-scout-v2

Family: `kernel_sweep`; status: **completed**. Output budget: 64; requested repeats: 2. Trace enabled: False.

DFlash block size: 3.

- First output token belongs to prefill; decode is N-1 tokens.
- Identical target weights; fresh cache per request; warm paths before timing.
- Variant order reverses on alternate repetitions; stock AR brackets entire sweep.
- DFlash draft remains resident during AR if any DFlash variant requested.
- Token parity is empirical for these prompts, not a universal mathematical guarantee.

| Variant | Decode tok/s | Paired ratio vs AR | Runs | Unique prompts | Observed repeats | Parity | Awake | Status |
|---|---:|---:|---:|---:|---|---|---|---|
| ar | 36.716 ± 0.066 | — | 2 | 1 | [0, 1] | PASS | PASS | control |
| zmlx-both | 36.588 ± 0.021 | 0.997 ± 0.002 | 2 | 1 | [0, 1] | PASS | PASS | validated experimental comparison |
| zmlx-deltanet | 36.784 ± 0.058 | 1.002 ± 0.003 | 2 | 1 | [0, 1] | PASS | PASS | validated experimental comparison |
| zmlx-swiglu | 36.694 ± 0.015 | 0.999 ± 0.001 | 2 | 1 | [0, 1] | PASS | PASS | validated experimental comparison |

- AR drift bracket: `{"after_over_before": 1.0187981563722641, "after_tok_s": 36.83543647675097, "before_tok_s": 36.15577457257537, "prompt_index": 0, "source": "raw AR bracket measurements"}`.
