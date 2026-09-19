# Speculative decoding: пять запросов × 128 токенов

У каждой семьи свой target-only baseline; между MLX-LM/DFlash и MLX-VLM/MTP baseline не подменяются.

| Семья / конфигурация | Prefill mean ± SD | Decode mean ± SD | Decode × mean | Parity |
|---|---:|---:|---:|---|
| DFlash baseline | 164.27 ± 29.69 | 30.16 ± 0.18 | 1.000 | reference |
| dflash-k2-q4-128 | 157.15 ± 27.27 | 29.31 ± 1.94 | 0.972 | passed |
| dflash-k3-q4-128 | 169.10 ± 34.12 | 30.91 ± 1.87 | 1.025 | passed |
| dflash-k5-q4-128 | 150.08 ± 26.85 | 24.95 ± 1.79 | 0.827 | passed |
| dflash-k3-q8-128 | 156.11 ± 27.33 | 28.43 ± 1.11 | 0.943 | passed |
| MTP baseline | 160.21 ± 33.49 | 29.13 ± 1.95 | 1.000 | reference |
| mtp-k2-128 | 148.38 ± 26.10 | 31.29 ± 1.27 | 1.074 | passed |
| mtp-k3-128 | 164.84 ± 32.84 | 38.15 ± 2.75 | 1.310 | passed |

Все speculative-конфигурации совпали со своим baseline по 640 выходным IDs. Это проверка данной выборки. Для MTP сохраняется ограничение: оптимизированный quantized projection/argmax и обычный baseline отличаются реализацией; отдельной ablation этого фактора не было.

Пять разных prompts, SD ddof=1, работа от батареи, wired memory и caffeinate di. Фиксации частот и температуры нет; SD не является доверительным интервалом. Для выбора MTP дополнительно выполнена [серия 512 токенов с baseline до и после](mtp-512.md).

[JSON с исходными per-sample числами, acceptance, parity и SHA-256](speculative-128.json).
