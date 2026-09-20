# Ускорение LLM inference: обзор на 19 сентября 2026 года

Основное окно обзора — **19.09.2025–19.09.2026**. Для papers ниже указана дата **arXiv v1**, а существенные обновления — отдельно; год конференции, новая версия статьи или интеграция в runtime не считаются датой изобретения метода. Это выборочный обзор первичных источников, а не исчерпывающий поиск или независимое воспроизведение всех результатов. Источники после даты отсечения не используются.

Задача проекта существенно уже общей задачи serving: **Qwen3.5-9B-4bit, Apple M2 Pro, 16 GB unified memory, MLX, один запрос одновременно**. Наиболее близкий практический следующий шаг — адаптивная длина уже работающего native MTP, с измерением полной цены confidence gate. Замена target на diffusion-модель, CUDA kernels и многосерверный KV store требуют другой постановки. Ни один приведённый серверный speedup не является прогнозом скорости на этом Mac.

Развёрнутые DFlash/MTP, точные revisions, версии библиотек и исторические короткие измерения описаны в [предыдущем исследовании](speculative-decoding-research.md). Здесь они не объявляются новыми результатами. Прерванная длинная серия `mtp-long2048`, в которой машина действительно спала и менялся источник питания, исключена из выводов; её локальные данные сохранены для аудита. Новый обзор не устанавливает пакеты и не запускает GPU.

## Что именно можно ускорять

| Уровень | Что уменьшается | Типичный предел | Что сохраняется |
|---|---|---|---|
| Speculative decoding / MTP | Число дорогих target-шагов на полезный токен | Цена draft и проверки, acceptance, rollback | Target может остаться прежним; нужна корректная проверка |
| Адаптивная глубина / ширина | Бесполезная проверка маловероятных продолжений | Цена confidence и управления, различия нагрузки | Зависит от причинности политики и verifier |
| Новая diffusion / semi-AR модель | Последовательная зависимость между позициями | Число denoising-шагов и качество | Обычно меняются модель или её распределение |
| Kernel fusion, layout, precision | Передачи памяти, промежуточные массивы, запуски kernels | Конкретные GPU ISA и memory hierarchy | Точная математика либо приближённая арифметика — это разные случаи |
| KV compression / sparse attention | Чтение длинного cache | Длина контекста, стоимость отбора/dequantization | Часто меняется вычисление target; нужна оценка качества |
| Prefix reuse, batching, serving | Повторный prefill, простои, transfers | Повторяемость prefixes, concurrency, сеть | При корректном cache возможно сохранить вычисление, но меняется workload |

**Три разных смысла “без потерь”.** Корректный speculative sampler может сохранять распределение target. Совпадение greedy token IDs — более узкая проверка конкретных входов и арифметики. Сходный benchmark score или pass@1 не означает ни первого, ни второго. При квантизации и изменении размера GEMM возможны различия округления даже у алгоритмически корректного verifier. Поэтому ниже слово «проверяемый» не заменяет локальную проверку parity.

## Методы из основного окна

В столбце «Mac» **готово** означает наличие применённой в проекте комбинации, **кандидат** — переносимую идею без подтверждённого готового пути для наших весов, **другая задача** — изменение модели, аппаратуры или режима. Если точная модель GPU не подтверждена проверенными фрагментами первоисточника, это явно указано; характеристика не угадывается.

### Draft, verifier и адаптивное управление

| Метод и дата | Механизм / bottleneck | Веса и обучение | Сохранение результата | Аппаратура / предпосылка источника | Применимость к нашему Mac |
|---|---|---|---|---|---|
| **SpecDiff-2**, v1 **01.11.2025**, v2 04.11.2025 ([paper](https://arxiv.org/abs/2511.00606), [детали](https://arxiv.org/html/2511.00606v2)) | Одношаговый diffusion draft вместо последовательной генерации proposals; согласование train/inference | Обучаемый aligned draft; target остаётся verifier | Проверяемая speculation, не замена target diffusion-моделью | A100 80 GB; пары DiffuLLaMA/Llama2 и DiffuCoder/Qwen2.5 с совместимым tokenizer | Кандидат; готовый exact Qwen3.5-9B MLX путь здесь не подтверждён |
| **DART**, v1 **27.01.2026** ([paper](https://arxiv.org/abs/2601.19278), [эксперименты](https://arxiv.org/html/2601.19278v1)) | Параллельные proposals из target features, n-gram continuity и дерево проверки | Специально обученный draft; нужен tree verifier | Target проверяет предложения | Qwen3; H20-3e 141 GB, дополнительная проверка A100 40 GB | Кандидат; перенос дерева и отката hybrid cache существенен |
| **DFlash**, v1 **05.02.2026**, v2 **28.05.2026** ([paper](https://arxiv.org/abs/2602.06036), [runtime](https://github.com/z-lab/dflash)) | Небольшой block-diffusion draft использует target features; параллельная генерация блока | Отдельный обученный draft, target не переобучается | Проверка target; локальная численная parity проверяется отдельно | В paper преимущественно H200; SGLang-эксперимент Llama на B200 | **Готово**: официальный MLX backend и Qwen3.5-9B draft; quantized runtime рекомендует `block_size <= 5` |
| **Learning to Draft (LTD)**, arXiv v1 **02.03.2026** ([paper](https://arxiv.org/abs/2603.01639), [training](https://arxiv.org/html/2603.01639v1)) | Две RL-политики выбирают размер/глубину speculation по throughput reward | Обучение дополнительных политик; target не требуется переобучать | В рамках проверяемого speculative runtime | Одна A100; профиль реальной цены важнее одной acceptance | Кандидат: сначала дешёвый gate, затем RL, только если его сложность окупается |
| **SpecBound**, v1 **14.04.2026** ([paper](https://arxiv.org/abs/2604.12247), [метод и training](https://arxiv.org/html/2604.12247v1)) | Self-draft из ранних слоёв; calibrated confidence ограничивает ширину/глубину | Base LLM фиксирован, **дополнительные LM heads обучаются**; не полностью training-free | Авторы заявляют эквивалентную проверку target | H800; дополнительный RTX 3090; training heads на 4×H800 | Кандидат; нет подтверждённого готового Qwen3.5 hybrid-cache MLX пути |
| **Native Qwen3.5 MTP в MLX-VLM**, доступная реализация/checkpoint **2026** ([веса](https://huggingface.co/mlx-community/Qwen3.5-9B-MTP-4bit), [runtime](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/speculative/drafters/qwen3_5_mtp/README.md)) | Маленькая MTP-голова переиспользует target features/embedding/head | Веса головы выделены из исходного Qwen3.5-9B; локального training нет | Специализированный verifier и rollback; greedy parity проверяется | MLX / Apple Silicon; **2026 — дата доступной интеграции, не происхождения MTP** | **Готово**; веса головы ≈136.9 MB в закреплённом snapshot; естественная база для gate |
| **DSpark**, v1 **06.07.2026** ([paper](https://arxiv.org/abs/2607.05147), [scheduler](https://arxiv.org/html/2607.05147v1)) | Semi-AR draft; calibrated prefix-survival confidence + профиль пропускной способности задают длину проверки | Обучаемые draft и confidence head | Lossless scheduler требует причинности; paper отдельно разбирает leakage | Production DeepSeek-V4; точный GPU в проверенных разделах не установлен | Кандидат как принцип scheduling; готовых весов/MLX для нашего target здесь не установлено |
| **AdaFlash**, v1 **21.07.2026**, v2 **14.09.2026** ([paper](https://arxiv.org/abs/2607.19223), [официальный repo](https://github.com/ZinYY/AdaFlash)) | On-policy distillation diffusion draft + adaptive length head; variable-length batching | Обновляются draft и length head, target фиксирован; требуется training pipeline | Target verification; не «пропуск проверки по уверенности» | SGLang GPU serving, точная GPU здесь не установлена; отдельно Ascend 910C | Кандидат; опубликованный Qwen3-8B draft не является Qwen3.5-9B; training/runtime заметно сложнее gate |
| **AngelSpec / DFly**, v1 **28.07.2026**, v2 29.07.2026 ([paper](https://arxiv.org/abs/2607.25852)) | Разная специализация MTP и block-diffusion под chat и code/math; cost-aware verification | Обучение специализированных drafters; иной target family | Проверяемая speculation | Hy3, concurrency 4–64; точная GPU здесь не установлена | Кандидат как принцип выбора drafter по workload; готовый exact MLX checkpoint не подтверждён |
| **Verifier Skipping / Prefix Scheduling**, v1 **14.08.2026** ([paper](https://arxiv.org/abs/2608.14787)) | По confidence часть draft-prefix принимается **без target** | Политики по raw confidence либо обученным оценкам | **Lossy**: одинаковый наблюдаемый pass@1 не доказывает parity | HumanEval, DiffuCoder-7B + Qwen3-32B; метрика включает число verifier calls | Другая постановка: не использовать для цели сохранить greedy output target |

У DFlash наличие diffusion drafter не делает target diffusion-моделью: autoregressive target остаётся окончательным арбитром. SpecDiff-2 уже описывал одношаговое diffusion drafting в ноябре 2025 года, поэтому DFlash не следует называть первым методом такого типа. В нашем случае важное преимущество DFlash — **реальный официальный MLX путь**, а не перенос paper speedup на M2. [SpecDiff-2](https://arxiv.org/abs/2511.00606), [DFlash](https://github.com/z-lab/dflash).

DSpark особенно близок к идее gate: он использует произведения **calibrated conditional acceptance probabilities** и измеренную цену проверки. Это отличается от произведения двух максимальных softmax-probabilities draft. AdaFlash предсказывает полезную длину обученной головой и адаптирует draft online; это ещё более дорогой механизм, а не подтверждение, что простой порог всегда ускорит inference. [DSpark, §3.2 и §5.2](https://arxiv.org/html/2607.05147v1), [AdaFlash, §3](https://arxiv.org/html/2607.19223v2).

### Изменение модели, kernels, cache и serving

| Метод и дата | Механизм / bottleneck | Веса и обучение | Сохранение результата | Аппаратура / предпосылка источника | Применимость к нашему Mac |
|---|---|---|---|---|---|
| **Fast-dLLM v2**, v1 **30.09.2025** ([paper](https://arxiv.org/abs/2509.26328), [repo NVIDIA](https://github.com/NVlabs/Fast-dLLM)) | AR→block-diffusion, parallel commits, block/sub-block cache | **Преобразование и fine-tuning target**, около 1B training tokens | Сравнение качества, **не эквивалентность прежнему AR output** | A100/H100, Qwen2.5-7B как AR reference | Другая модель/задача; не флаг ускорения exact Qwen3.5-9B-4bit |
| **VecInfer**, v1 **07.10.2025** ([paper](https://arxiv.org/abs/2510.06175), [calibration/kernels](https://arxiv.org/html/2510.06175v1)) | Outlier suppression + vector quantization KV; fused dequantization/attention | LLM не fine-tune; calibration и K-means codebook необходимы | Приближённый KV, возможна потеря качества | H100, kernels также A100; длинные контексты порядка 64–192K | Кандидат для иной long-context нагрузки; CUDA codec не готовый MLX путь |
| **SPEQ**, v1 **21.10.2025** ([paper](https://arxiv.org/abs/2510.18525)) | Quantized draft из части битов target weights; shared storage и аппаратное исполнение | Без дополнительного draft training/storage; algorithm–hardware co-design | Speculative verification full target | Специальный reconfigurable processing-element array | Другая аппаратура; существующее MLX affine 4-bit представление не даёт этот механизм автоматически |
| **LycheeDecode**, v1 **04.02.2026** ([paper](https://arxiv.org/abs/2602.04541), [эксперименты](https://arxiv.org/html/2602.04541v1)) | Retrieval heads выбирают токены, sparse heads переиспользуют выбор; меньше KV reads | Обучаемый выбор heads через HardKuma / distillation | Sparse approximation, не exact full attention | Head training A100 80 GB; TileLang kernels A800, 16–128K | Кандидат для long context; Qwen3.5 hybrid target не равен исследованному Qwen3 |
| **FlashAttention-4**, arXiv v1 **05.03.2026** ([paper](https://arxiv.org/abs/2603.05451), [официальная интеграция](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/)) | Пайплайнинг exact attention под дисбаланс tensor cores / softmax / memory | Веса не меняются, training не нужен | Та же attention-операция, но floating-point порядок может отличаться | B200 BF16 в основных kernel comparisons; CuTe-DSL / CUDA | Другая аппаратура; CUDA kernel не запускается на Metal; prefill и длинный attention полезнее короткого decode |
| **vLLM × Mooncake Store**, интеграция **06.05.2026** ([официальный отчёт](https://vllm.ai/blog/2026-05-06-mooncake-store)) | Distributed prefix KV reuse, prefill/decode disaggregation, asynchronous transfer | Веса не меняются; нужен повторяемый prefix и serving infrastructure | Cache может сохранять вычисление при корректных ключах/состоянии | Kimi-2.5 NVFP4: сравнение на 12 GB200, scaling до 60 GB200 | Другая задача; локальный prefix reuse полезен в диалоге, но не ускоряет изолированный свежий decode сам по себе |
| **SSV**, v1 **19.05.2026**, v2 20.05.2026 ([paper](https://arxiv.org/abs/2605.19893)) | Совместная оптимизация sparse attention и speculative verification, reuse/fusion | Требуется NSA-compatible attention/runtime; profile-guided orchestration | Сравнение внутри выбранного precision/sparsity режима; не обещание full-attention parity | H100; baseline — autoregressive **NSA**, не произвольный dense AR | Кандидат только после существенного изменения attention/backend; старое название v1 — SpecSA |
| **Fast-dLLM++**, v1 **01.06.2026**, v2 **15.06.2026** ([paper](https://arxiv.org/abs/2606.02955), [repo](https://github.com/Ringo-Star/FastdLLM_plusplus)) | Fréchet profile использует неоднородные confidence при выборе parallel commits | Training-free **для уже diffusion-модели** | Comparable quality; не exact AR target distribution | H100 80 GB, LLaDA-8B-Instruct; дополнительный Dream-7B | Другая модель; не продолжение нумерации Fast-dLLM v2 и не drop-in Qwen3.5 |

Здесь нельзя сравнивать столбики speedup разных papers как единую таблицу лидеров. Например, FA4 сообщает до **1.3× относительно cuDNN 9.13 и 2.7× относительно Triton на B200 BF16** — это kernel comparison. Mooncake показывает **3.8× serving throughput** на trace с повторяемыми prefixes и **12 GB200**, одновременно резко меняя cache hit rate. SSV сравнивается с **AR NSA**. Эти знаменатели отличаются от нашего batch=1 decode и друг от друга. [FA4](https://arxiv.org/abs/2603.05451), [Mooncake](https://vllm.ai/blog/2026-05-06-mooncake-store), [SSV](https://arxiv.org/abs/2605.19893).

На 16 GB важен также объём временных массивов и resident memory: дополнительная модель или большой verify block способны проиграть, даже увеличив число принятых токенов. Это локальная инженерная гипотеза, которую проверяют профилем памяти и времени, а не самостоятельный новый метод из таблицы. Для Qwen3.5 откат после частичного принятия обязан восстановить **attention KV и convolution/recurrent state**; см. [специализированный verifier MLX-VLM](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/qwen3_5/speculative_verifier.py).

## Более ранний background — не открытия последнего года

- **Классический speculative decoding:** arXiv v1 **30.11.2022**, v2 **18.05.2023**. Маленький draft и корректная проверка большим target; основа аргумента о сохранении распределения. [Leviathan et al.](https://arxiv.org/abs/2211.17192).
- **EAGLE-3:** v1 **03.03.2025**, v3 **23.04.2025**. Обучаемый draft с multi-layer target features и training-time test. Существование Qwen3.5 checkpoint в 2026 году не делает сам EAGLE-3 методом из основного окна. [Paper](https://arxiv.org/abs/2503.01840), [исследованный checkpoint](https://huggingface.co/yuyijiong/Qwen3.5-9B-Eagle3).
- **Первый Fast-dLLM:** v1 **28.05.2025**, v3 **03.07.2025**. Cache и parallel decoding для diffusion LLM; его нужно отличать от v2 и Fréchet-profile работы выше. [Paper](https://arxiv.org/abs/2505.22618).
- **Prefix caching, batching, weight quantization, KV paging, MTP как семейство** не изобретены в 2026 году. В таблице отмечены конкретные новые работы или датированная интеграция, а не переименованы старые техники. Дата доступного MLX checkpoint также не раскрывает точную программу обучения MTP: из неё нельзя вывести training horizon, frozen/joint stages или состав teacher-data.

## Проверяемый план confidence gate `p1 * p2`

Это **план проверки эксперимента**, а не заявление об уже полученном ускорении; наличие реализации gate само по себе не заменяет измерение. Начальная область — **greedy decoding**, один prompt, текущий native MTP. В проектной договорённости `K=3` соответствует максимум двум draft-предложениям и одному target-токену за полностью удачный цикл; до переноса порога нужно сверить именно эту семантику в runtime.

### 1. Зафиксировать смысл confidence

Пусть `d1`, `d2` — два proposed IDs, а `q` — распределение **draft**, доступное до target verification:

```text
p1 = q(d1 | prefix)
p2 = q(d2 | prefix, d1)
score = p1 * p2
```

При таком условном последовательном определении это draft-вероятность конкретной пары; предположение независимости не требуется. Если оба значения получены как параллельные marginal confidences без conditioning на первый proposal, произведение — только эвристика, не совместная вероятность.

**`score` не равен вероятности принять пару target.** Для greedy target важно совпадение argmax, а не только уверенность draft; для stochastic acceptance важны также target/draft likelihood ratios. Калибровка нужна по событиям `A1 = принят первый draft`, `A1∩A2 = принят весь префикс`. Если отдельно оценены `s1=P(A1)` и `s2=P(A2 | A1)`, тогда `S1=s1`, `S2=s1*s2` — survival probabilities. При обычном цикле без EOS/output cap ожидаемое число committed tokens приближённо равно `1 + S1 + S2`; это не превращает raw `p1*p2` в calibrated survival. Направление подтверждается [DSpark](https://arxiv.org/html/2607.05147v1), но точная калибровка для нашего MTP ещё должна быть измерена.

Оптимизировать следует полезные токены за время, например `E[committed] / E[draft + verify + rollback + gate]`, а не acceptance отдельно. Очень короткий draft может иметь идеальную acceptance и всё равно быть медленнее. Цена зависит от verify width и текущего memory/kernel режима.

### 2. Поставить gate в точку, где ещё можно сэкономить

- Если `p1*p2` вычисляется **после двух proposals**, цена их генерации уже уплачена. Gate может уменьшить последующую проверку; нельзя приписывать ему экономию второго draft forward.
- Для экономии второго proposal нужен отдельный ранний decision после `p1`, с корректным продолжением состояния. Это другая политика, её следует сравнивать отдельно.
- Уже использованный fused argmax не предоставляет нормированную вероятность бесплатно. Нужны winning logit и `logsumexp` по словарю либо эквивалентный kernel; top-1/top-2 margin — иной score. Нельзя вычислить softmax-probability только из ID победителя.
- GPU→host `.item()`/чтение confidence может синхронизировать очередь. Стоимость reduction, materialization logits, host branch и дополнительного cache handling должна входить в decode time. При `K=3` эта цена способна превысить всю экономию.

Для сохранения greedy output gate выбирает только **длину проверяемого префикса**. Ни один draft ID не становится окончательным без проверки target; rollback остаётся специализированным для Qwen3.5. Это отличается от lossy verifier skipping из таблицы.

### 3. Не допустить leakage

Online features должны быть доступны **до** решения: draft logits/confidence, уже committed prefix, прежняя статистика. Target logits текущей проверки, фактический accepted count и будущие teacher tokens могут быть **labels для calibration**, но не входом действующего gate. Нельзя обучить/подобрать порог на тех же prompts и затем назвать их результат независимой проверкой.

Для **stochastic sampling** нужна отдельная проверка причинности. Даже при последующем verifier ретроспективный выбор длины по будущим sampled proposals может нарушить обычное доказательство сохранения распределения. DSpark специально разбирает это ограничение и причинный scheduler; его аргумент нельзя заменить фразой «target всё равно проверяет». Поэтому первый эксперимент ограничивается greedy parity, а stochastic режим требует корректного stopping rule и самостоятельного обоснования. [DSpark, §5.2 и Appendix A](https://arxiv.org/html/2607.05147v1).

### 4. Разделить calibration, replay и реальный тест

1. На отдельной calibration-выборке собрать draft confidence **до verification**, фактические prefix acceptance labels, затраты каждого этапа, выбранную глубину и конфигурацию kernels. Деление делать по prompts, не по correlated rounds одного prompt. Существующие IDs/acceptance без сохранённых logits не позволяют восстановить `p1*p2` задним числом.
2. Заранее определить небольшую сетку порогов или простую калибровку и выбрать её только на calibration/validation. Зафиксировать правило и расходы на сбор features до итогового holdout.
3. Offline replay позволяет оценить связь score с acceptance и потенциально лишней проверкой. Он **не измеряет counterfactual tok/s**: новая длина меняет следующие round boundaries, число bonus tokens, cache state, kernel shapes и host overhead. Нельзя вычесть времена rejected tokens из старого trace и объявить это реальным ускорением.
4. Реально сравнить на одинаковых holdout prompts: фиксированный K2, фиксированный K3, gate; добавить контроль K3 с тем же вычислением confidence, но без изменения решений. Этот контроль отделяет цену feature collection от выигрыша policy. Если наблюдения собраны только после селективного gate, отсутствующие длинные proposals дают selection bias; полноценный shadow trace фиксированного K3 полезен для calibration, но не заменяет timing нового режима.

### 5. Измерять результат, который нужен пользователю

Новый gate сравнивается со своим **одинаково instrumented** baseline. Нельзя сопоставлять per-token clocks / proposal host reads с историческими uninstrumented числами и приписывать всю разницу policy. Предпочтителен заранее заданный парный порядок блоков `A0 B0 | B1 A1 | ...`, отдельные warmup/model loads явно учитываются. Питание, сон, thermal snapshots и invalidation rule задаются до серии; доказанный sleep interval исключает блок/серию по этому правилу, а не потому, что число не понравилось.

В итоговом отчёте нужны: prefill и decode отдельно; полезные committed tokens, не все proposals; pooled throughput `sum(tokens)/sum(seconds)` и mean/SD скоростей по prompts; paired speedup distribution; end-to-end `prefill + decode`; weighted `accepted/drafted`, отдельно emitted draft tokens; все mismatched IDs и первый mismatch. SD между prompts — **не доверительный интервал**. Для интервала по paired effect нужна, например, повторная выборка на уровне prompts, а не отдельных correlated tokens.

Если EOS игнорируется ради фиксированных 2048 output tokens, tok/s не равно скорости получения законченного ответа. Отдельно показываются доля prompts с первым EOS в пределах budget, его output index и время commit-event с этим EOS, когда trace это позволяет. Не прошедшая parity конфигурация остаётся в отчёте, но не называется lossless. Любое утверждение об ускорении gate требует завершённого парного эксперимента: уверенность draft сама по себе такого результата не доказывает.

## Приоритет для следующего изменения проекта

1. **Оставить native MTP K3 и K2 контрольными точками.** Сначала измерить минимальную цену получения confidence и проверить правильность места gate / rollback; отдельную training pipeline пока не добавлять.
2. **Начать с проверяемого confidence gate на greedy режиме, затем добавить cost-aware выбор.** Полезная заимствуемая идея DSpark/LTD — учитывать стоимость проверки, а не максимизировать acceptance. Отрицательный speedup с сохранённой parity — полноценный результат.
3. **DFlash оставить самостоятельным конкурентом** с тем же target snapshot, workload и memory policy. Готовый официальный MLX backend полезнее обещаний переноса CUDA kernel; новый draft требует отдельного подтверждения совместимости и памяти.
4. **KV compression / sparse attention / diffusion target изучать отдельными экспериментами качества и нагрузки.** Они могут быть важны для значительно более длинного контекста или другой модели, но меняют исходный контракт сравнения.

Этот приоритет — инженерный вывод для имеющегося проекта. Он не ранжирует глобальный SOTA и не обещает, что adaptive gate обязательно обгонит небольшой фиксированный MTP-блок.

Практическая проверка raw `p1 × p2` с отдельными calibration/held-out цепочками описана в [MTP confidence experiment](mtp-confidence-experiment.md). Выбор порога по balanced accuracy сам по себе не калибрует вероятности acceptance и не оптимизирует tok/s.
