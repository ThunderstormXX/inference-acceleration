# Кандидаты на ускорение Qwen3.5-9B на M2 Pro

Проверка первоисточников: **22 сентября 2026**. Target проекта: текущие MLX 4-bit веса Qwen3.5-9B; Apple M2 Pro, 19 GPU cores, 16 GB. Цель — скорость одной генерации, а не aggregate throughput сервера. Это дополнение к [общему roadmap](beyond-mtp-local-roadmap-2026-09-22.md): ниже акцент на доступном коде и аппаратных ограничениях. Обзор не является исчерпывающим каталогом всех публикаций.

**В этой заметке внешние цифры — измерения авторов источников, не результаты на нашем Mac.** Порядок экспериментов — наша оценка применимости. Наличие функции, checkpoint или обещания exactness не заменяет локальную проверку всех output IDs.

## Приоритетный набор экспериментов

| Приоритет | Что пробовать | Внешнее измерение и требования | Сохранение текущего target |
|---|---|---|---|
| 1 | ZMLX: отдельно fused DeltaNet, SwiGLU, затем совместно | Автор: +7.5% Qwen3.5-9B-4bit на M4 Max 36 GB. Stock MLX ≥0.30, Python ≥3.10, macOS ≥14 | Веса те же; нужна наша проверка numerics |
| 2 | Аудит MTP: повторно использовать hidden последней позиции verifier для следующего цикла | mlx-node/MTPLX показали цену лишнего full-target forward; выигрыш зависит от модели и scheduling | Веса те же; сначала установить, нет ли этого уже в нашем runtime |
| 3 | CPU n-gram lookup + существующий target verifier, сначала 1 proposal | Не требует дополнительной сети; есть hybrid-GDN реализации и измерения на других Mac | Target тот же; корректный GDN rollback обязателен |
| 4 | Full-vocabulary fused argmax и small-M affine projections | Есть код MLX-VLM; small-M quantized matmul отдельно подтверждён как проблема MLX | Quantized веса те же; reduction order может менять near-ties |
| 5 | EAGLE3 с полным словарём target при проверке | Есть trained draft для Qwen3.5-9B; достоверного 9B/M2 Pro speedup не найдено | Target тот же, новая draft-сеть |
| 6 | gmlx / llama.cpp native MTP как отдельный runtime+GGUF эксперимент | gmlx: Q6_K, M5 Max128GB, ≈70→112 tok/s. M1/M2 не были фокусом tuning | Другой quant checkpoint: отдельная speed–quality ветка |
| 7 | mlxcel на локальных MLX SafeTensors | Native Rust/MLX runtime без конвертации; свежие text decode medians близки к MLX-LM | Веса можно сохранить, output parity ещё надо проверить |
| 8 | Weight/KV quantization sweep при контроле качества | KV compression интереснее на длинном контексте; малые bitwidth не гарантируют быстрый kernel | Меняется численная модель/cache; это не exact ускорение |

### 1. ZMLX: простая ablation текущего AR

[ZMLX](https://github.com/Hmbown/ZMLX) сообщает token-identical greedy outputs в собственных тестах. Патч сворачивает несколько операций после GDN convolution и dense SwiGLU. Это небольшое вмешательство без новой модели.

Важное уточнение по [DeltaNet source](https://github.com/Hmbown/ZMLX/blob/main/src/zmlx/patch/patterns/deltanet.py): `S > 1` возвращает исходный forward. Поэтому конкретно этот GDN patch ускоряет **S=1**, но не является готовой оптимизацией multi-token verification. Quantized input projections тоже остаются отдельными вызовами; их fusion реализован только для dense weights. В коде есть FP32-state и resync knobs, так что короткого совпадения недостаточно для вывода о длинных цепочках. [Протокол benchmark](https://github.com/Hmbown/ZMLX/blob/main/docs/BENCHMARKS.md).

### 2. Устранить лишнюю работу между MTP-циклами

[mlx-node performance log](https://github.com/mlx-node/mlx-node/blob/main/docs/perf.md) сравнивает два runtime на одинаковом Qwen3.6-27B affine-int4 checkpoint и M5 Max. При практически одинаковом полном acceptance MTPLX получает 56.43 tok/s против 40.18: его следующий цикл использует уже вычисленный `verify_hidden[last]` и bonus token, а второй runtime дополнительно выполнял одиночный full-target forward.

Это **не прогноз +40% для нас**. В том же источнике chaining на M1–M4 выключен по умолчанию из-за lazy-slice scheduling и acceptance regression в проверенной конфигурации. Сначала считаем target forwards на нашем полном принятии и выясняем, существует ли лишний вызов вообще. Тот же log документирует неудачные small-M попытки: forced GEMM и multi-row qmv оказались медленнее stock. Переписывать kernel имеет смысл после ablation на наших формах тензоров.

### 3. Драфт из уже доступного текста

В [MLX-LM issue1497 от 08.07.2026](https://github.com/ml-explore/mlx-lm/issues/1497) описан n-gram draft с checkpoint/rollback recurrent state. Автор на Qwen3.6-27B, M2 Ultra и **экспериментальной mixed 2-bit quant** получил 44.6→52.1 tok/s с одним proposal и hit rate44%; два proposals дали34.4. Статус closed у issue не доказывает, что код вошёл в нашу установленную версию. Его предположение о переносимости speedup между quant formats здесь не считается доказанным.

[OptiQ](https://mlx-optiq.pages.dev/docs/speculative) уже сочетает n-gram с MTP и адаптирует длину по acceptance и фактической стоимости verification. Авторские live coding-agent turns дали1.62× на **другой MoE модели/M4**, а chat/research1.11×. Это аргумент за workload-specific проверку, не готовая цифра для9B.

У нас выполнен [CPU causal replay](ngram-feasibility-2026-09-22.md) на11 сохранённых цепочках по2048 токенов. Только собственный prompt и уже сгенерированный префикс; ни внешнего корпуса, ни будущего ответа в индексе. При минимальном совпадении2 токена предложение существует на64.0% decode-позиций, первый токен совпадает в60.6% этих случаев. Width1/2/4 даёт средний принятый префикс0.606/1.006/1.471 токена на возможность. **Это feasibility, не live speedup.** После CPU replay выполнены live-прогоны с GPU verifier и rollback: устойчивого выигрыша пока нет, а drift AR делает обе серии диагностическими. Подробности в [локальном журнале экспериментов](local-speed-experiments-2026-09-22.md). Все11 задач уже просматривались и являются development material.

### 4. Вероятная цена quantized verifier

[MLX issue4265 от 15.08.2026](https://github.com/ml-explore/mlx/issues/4265) содержит воспроизводимый microbenchmark на M3 Ultra: для формы `(M,5120) @ (17408,5120)^T`, affine int4/group64, M1 стоит0.041ms, M4 —0.134ms, M8 —0.252ms. Эти цифры не являются нашим замером. Собственный scalar kernel автора улучшил масштабирование по M, но оказался в2.6–3.3 раза медленнее по абсолютной задержке.

Наша первая проверка — уже существующие `optimized_affine_linear` и `optimized_affine_argmax` в [MLX-VLM](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/quantized_verifier.py). Argmax должен оставаться по **всему словарю**. Draft head, target head и projections надо включать отдельно, сравнивая также оптимизированный AR. Перенос prefill GEMM или weight reuse сам по себе не обещает выигрыша.

### 5–8. Другие готовые пути

- [Qwen3.5-9B-Eagle3](https://huggingface.co/yuyijiong/Qwen3.5-9B-Eagle3): готовые веса сокращают стоимость старта. Перед timing надо решить описанный в [roadmap](beyond-mtp-local-roadmap-2026-09-22.md) риск hot-vocabulary verification и проверить recurrent state на отказах.
- [gmlx](https://github.com/asher/gmlx/blob/main/docs/performance.md): интересен как отдельная реализация native MTP. Авторские9B Q6_K ≈70→112 tok/s измерены наM5 Max с512-token prompt. Код дляM1/M2 использует стандартные kernels и не был benchmarked против llama.cpp авторами. Переход на GGUF меняет квантовку; наши MLX token IDs не служат автоматическим эталоном его numerics.
- [mlxcel](https://github.com/lablup/mlxcel): загружает существующие MLX SafeTensors. Важна свежая методика: README пересмотрел старые громкие результаты; сентябрьские text decode medians около parity. Старый9B1.38× не следует приводить как подтверждённый выигрыш текущего релиза. Нужен короткий собственный запуск, без ожидания обязательного ускорения.
- [gmlx cache measurements](https://github.com/asher/gmlx/blob/main/docs/performance.md): у hybrid моделей KV растёт лишь в attention-слоях, и автор рекомендует KV quant прежде всего когда ограничивает длинный контекст. Для наших коротких prompts+2048 output это менее убедительная первая ставка. Изменение KV precision оцениваем вместе с quality/argmax agreement.

## Свежие работы, полезные для дальнейшего исследования

**[TreeWY, 21.08.2026](https://arxiv.org/abs/2608.20961)** заменяет snapshots состояния каждого узла draft-tree компактными pseudo-values и восстанавливает только принятую ветку через tree-structured WY. Замеры наQwen3.5-35B/397B касаются прежде всего serving, ограниченного памятью; вне этого режима бывает несколько процентов потери. Готового Metal/9B пути не найдено. У нас K3 и небольшая цена rollback, поэтому это не первый оптимизируемый участок.

**[Uno + transactional DeltaNet](https://github.com/johnathonkillaly/uno-deltanet-mlx)** — независимый опыт с frozen backbone и noise-row LoRA. НаQwen3.5-4B Base BF16/M4 Max128GB AR49.68→54.05tok/s приK2; K4 —51.02. Сохранение промежуточных recurrent states устраняет replay forward. Для9B готового адаптера не найдено. Заявленная cacheless exactness не означает полного совпадения cached AR; автор это разделяет. Наш existing rollback уже недорогой, перенос одного commit-механизма вряд ли решит основную проблему.

**[Alloy: Gated DeltaNet on Apple Silicon, 21.07.2026](https://www.prysmlabs.ai/blog/gated-deltanet-apple-silicon)** показывает fused recurrence и chunked prefill. Авторские1.54× относятся к **prefill Qwen3.5-0.8B Q8 наM4 Max**, decode от переключения этого пути не ускоряется. Verifier использует serial recurrence, потому что chunked triangular formulation переставляет арифметику и может поменять near-tie argmax. Это полезное ограничение для попытки перенести prefill kernel в speculation.

**[Lossless but Not Free, 19.07.2026](https://arxiv.org/abs/2607.17283)** исследует speculative decoding на consumer Apple Silicon сQwen2.5 draft/target. Лучший режим1.61×, три из пяти замедлились. Существенны draft-step latency и scaling verifier по ширине. Greedy exactness real-model leg проверена наfp32 CPU, а не доказана для quantized Metal; переносить её на наш runtime нельзя.

Также сохраняют ценность уже разобранные [DFlash2](https://inco.ai/blog/dflash2/), [LibraSpec](https://arxiv.org/abs/2608.08721), [Quantize the Target, Quantize the Drafter](https://arxiv.org/abs/2607.04244). Их наличие не означает готового9B/M2 checkpoint или ускорения именно наших weights.

## Что уже проверено на нашем Mac

Кандидаты 1–4 проверены практически: ZMLX, chained MTP audit, n-gram draft, affine kernels и fused argmax. Дополнительно проверены ширина MTP, streamed/tiled dispatch и уменьшение словаря только draft-головы. Положительные и отрицательные результаты, ограничения и команды сохранены в [журнале экспериментов](local-speed-experiments-2026-09-22.md). EAGLE3, gmlx, mlxcel и новые quant checkpoint в эту серию не входят.

## Критерий продолжения

Короткие interleaved/ABBA эксперименты на одинаковом target, prompt IDs и output budget; prefill отдельно; сохраняя output IDs и timing. Новый optimized AR — обязательный контроль для каждой kernel-оптимизации. Для speculation считаем полезные committed tokens, draft/verify/rollback и память. Меняющие target квантовки живут в отдельной таблице качества и скорости. Длинная серия100×2048 пока не является выполненным результатом.
