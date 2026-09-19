# Qwen3.5 inference lab · Apple Silicon

[Главная страница с анимацией](../README.md). Команды ниже выполняются из корня репозитория. Публикуемые отчёты находятся в `docs/results/`; подробные сырые результаты создаются локально в `artifacts/`.

Воспроизводимое сравнение prefill/decode для **mlx-community/Qwen3.5-9B-4bit** на 100 записях **ThunderstormXXL/deepscaler-teacher-sft-vllm-official-40k**. Каждый исполняемый сценарий имеет `.sh` для окружения и одноимённый `.py` для запуска Python-классов.

Целевая машина текущего эксперимента: Apple M2 Pro, 19 GPU cores, 16 GB unified memory, macOS 15.6.1. Используется текстовая часть модели.

## Структура

```text
configs/                        # описание стандартного протокола
src/inference_lab/
  core/                         # общие параметры, окружение и I/O
  models/
    download.py                 # ModelDownloader: target snapshot + контрольные суммы
    draft.py                    # закреплённый DFlash checkpoint
    mtp.py                      # закреплённая native MTP голова
  environments/speculative.py   # отдельное окружение DFlash из lockfile
  visualization/
    trace.py                    # реальные события генерации и MTP verification
    render.py                   # воспроизведение записанных событий в GIF
  data/
    download.py                 # DatasetSliceDownloader: ровно 100 полных строк
    prompts.py                  # PromptDataset: общие token IDs для всех движков
  backends/
    base.py                     # интерфейс InferenceBackend и фабрика
    apple/
      mlx_backend.py            # MLX-LM, Metal
      mlx_vlm_backend.py        # MLX-VLM, текстовая часть
      vllm_backend.py           # официальный vLLM + vllm-metal
      speculative/
        dflash_backend.py       # официальный DFlash, наблюдение за acceptance/rollback
        mtp_backend.py          # native MTP со штатным verifier MLX-VLM
      transformers/
        backend.py              # Transformers/PyTorch MPS + MetalLinear
        metal_runtime.py        # совместимые Metal shaders для macOS 15
  benchmarking/
    runner.py                   # BenchmarkRunner: прогрев, фазы, GPU lock
    metrics.py                  # arithmetic mean / aggregate / median
    report.py                   # BenchmarkReport: Markdown, JSON, CSV
    validation.py               # FinalSeriesValidator: аудит исходных замеров
    speculative.py              # параметры и запуск пары MLX-LM / DFlash
    mtp.py                      # параметры и запуск пары MLX-VLM / MTP
    speculative_report.py       # парное сравнение скорости, acceptance и token IDs
  diagnostics/host.py           # HostDiagnostics: питание, память, thermal state
scripts/
  setup/environment.{sh,py}
  setup/transformers.{sh,py}
  setup/speculative.{sh,py}     # отдельный runtime для DFlash
  setup/doctor.{sh,py}
  setup/validate_metal.{sh,py}
  setup/telemetry.{sh,py}
  setup/{macmon,process_policy}.{sh,py}
  download/{model,dataset}.{sh,py}
  download/draft.{sh,py}        # закреплённые DFlash draft weights
  download/mtp.{sh,py}          # отдельная native MTP голова (~137 MB)
  benchmark/{transformers,vllm,mlx,mlx_vlm}.{sh,py}
  benchmark/awake.{sh,py}        # временные idle-sleep assertions для одного backend
  benchmark/recheck.{sh,py}      # повтор первых 5 примеров всеми четырьмя движками
  benchmark/speculative.{sh,py} # DFlash и сопоставимый обычный decode
  benchmark/mtp.{sh,py}         # native MTP и сопоставимый MLX-VLM decode
  report/{compare,validate}.{sh,py}
  report/recheck.{sh,py}         # сравнение повтора с исходными замерами
  report/speculative.{sh,py}     # ускорение, acceptance и совпадение output token IDs
  demo/{trace,render}.{sh,py}    # запись генерации и создание анимации
  run_all.{sh,py}
models/qwen3.5-9b-mlx-4bit/      # исходный HF snapshot (~6 GB)
data/raw/                       # 100 полных строк JSONL + manifest
artifacts/
  logs/                         # логи команд
  runs/<UTC>-<backend>/         # summary, prompts, samples, ошибки
  reports/                      # итоговая таблица
tests/                          # проверки метрик и границ фаз без GPU
```

## GIF: обычная генерация и MTP рядом

[В исходном темпе](assets/mtp-real-time.gif) · [Замедление 0.25× для наблюдения за draft-пакетами](assets/mtp-0.25x.gif)

Записаны реальные события одного промпта, без искусственного изменения относительной скорости. Золотым показаны предложения, зелёным — подтверждённый вывод. На выбранном фрагменте рассуждения MTP вывел 128 токенов за 3.914 с, baseline — за 4.169 с; все token IDs совпали. Это иллюстрация выбранного запроса с наблюдением за промежуточными событиями, а не замена бенчмарка. [Воспроизведение и результаты остальных проб](generation-demo.md).

## Подготовка

Нужен `uv`, macOS 15+ и Apple Silicon. Установка ограничена каталогом проекта; глобальный Python не меняется.

```bash
bash scripts/setup/environment.sh
bash scripts/setup/transformers.sh
bash scripts/setup/doctor.sh
bash scripts/setup/environment.sh --backend vllm
```

| Задача | Окружение | Зависимость от native ABI |
|---|---|---|
| Загрузка данных, MLX-LM, MLX-VLM, отчёты | `.venv`, Python 3.13 | Общий MLX runtime |
| Transformers MPS | `.venv-transformers`, Python 3.13 | Torch 2.10.0; на macOS 15 исходные MLX shaders компилируются через Torch |
| Официальный vLLM Metal | `.venv-vllm`, Python 3.12 | Mac wheels cp312 и закреплённая плагином версия MLX |
| Официальный DFlash | `.venv-speculative`, Python 3.13 | MLX 0.32.2, MLX-LM 0.31.3; `requirements-speculative.lock` |

Установка использует неизменяемые `requirements-mac.lock`, `requirements-transformers.lock` и `requirements-vllm.lock`, Python 3.13.5 / 3.12.8. Повторный setup синхронизирует окружение с lockfile и проверяет совместимость зависимостей. `setup/transformers.sh` подготавливает исходники Metal kernels без запуска GPU. На macOS 15 готовая библиотека Hugging Face несовместима с версией Metal; адаптер компилирует исходные affine shaders MLX 0.32.2 через `torch.mps.compile_shader`. MLX runtime в этом backend не импортируется. Отдельные окружения позволяют использовать согласованные версии; benchmark `.sh` автоматически выбирает нужный Python.

## Отдельные задачи

Команды можно запускать из любой рабочей директории, указывая путь к `.sh`. Ниже команды относительно корня проекта.

```bash
# 1. Исходная модель; готовые файлы повторно не скачиваются
bash scripts/download/model.sh

# 2. Только первые 100 строк train через Hugging Face rows API
bash scripts/download/dataset.sh

# 3. Transformers / PyTorch MPS
bash scripts/benchmark/transformers.sh

# 4. vLLM через официальный Metal plugin
bash scripts/benchmark/vllm.sh

# 5. Альтернативные реализации
bash scripts/benchmark/mlx.sh
bash scripts/benchmark/mlx_vlm.sh

# 6. Сводный отчёт
bash scripts/report/compare.sh

# 7. Проверка четырёх завершённых прогонов итоговой серии awake
bash scripts/report/validate.sh
```

Полная последовательность: `bash scripts/run_all.sh`; если веса и данные уже проверены, `bash scripts/run_all.sh --skip-downloads`. Ошибка одного backend сохраняется и не мешает запуску остальных.

Повтор итоговой серии при временном запрете простоя системы/экрана и с wired memory для двух MLX backend:

```bash
bash scripts/run_all.sh --skip-downloads --awake --mlx-wired-memory --label awake
# Один backend в тех же условиях
bash scripts/benchmark/awake.sh --backend mlx --wired-memory --label awake
```

Обёртка использует `caffeinate -di` только на время процесса, освобождает assertions при завершении и не меняет постоянные настройки питания. Она не симулирует пользовательский ввод (`-u` не используется). Режим сохраняется в `environment.caffeinate_assertions`. `run_all.sh --wait-for-gpu` позволяет поставить следующую последовательность после уже работающего benchmark.

По умолчанию модель закреплена на ревизии `8b2b98c00a6b4d291155e4890773ca8f769aee53`, датасет — на `6f0fadbf6b495cdea0e4bd83c4b64c7e36a992a3`. Проверяются хеши скачанных файлов и сохранённой выборки. Dataset Viewer отдаёт текущую ревизию: если она изменится, скрипт явно остановится, сохранив существующую выборку, вместо незаметной подмены данных.

## Параметры нагрузки

Стандарт: 100 условий задач → Qwen chat template → по 128 новых токенов рассуждения, greedy, batch size 1. Полные teacher chains сохранены в JSONL, но в режиме `problem` не передаются на вход. Это измерение скорости, без проверки правильности математических решений.

```bash
# Короткая проверка перед полным запуском
bash scripts/benchmark/mlx.sh --count 1 --max-new-tokens 8 --label smoke

# Продолжение 100 эталонных цепочек: 512 teacher-токенов в prompt
bash scripts/benchmark/mlx.sh --prompt-mode chain-prefix --chain-prefix-tokens 512

# Более длинный decode
bash scripts/benchmark/mlx.sh --max-new-tokens 512

# Эксперимент с 8-bit attention KV-cache
bash scripts/benchmark/mlx.sh --kv-bits 8 --label kv8

# Отдельный контроль политики активности macOS
bash scripts/benchmark/mlx.sh --count 3 --user-initiated --label activity-control

# Политика памяти штатного MLX-LM: recommended wired limit на время запроса
bash scripts/benchmark/mlx.sh --wired-memory --label wired
bash scripts/benchmark/mlx_vlm.sh --wired-memory --label wired
```

Четыре базовых benchmark `.sh` принимают одинаковые CLI-параметры (`--help`); speculative-сценарии имеют отдельные параметры, описанные ниже. KV-квантизация доступна для MLX backend; неподдерживаемый режим явно отвергается. По умолчанию prompt длиннее 2048 токенов вызывает ошибку; увеличить лимит можно через `--max-prompt-tokens`. Скрытого обрезания данных нет. vLLM настроен на общий контекст 4096 токенов.

Опция `--user-initiated` удерживает `NSActivityUserInitiated` только на время конечного запуска и обязательно завершает activity при выходе. По умолчанию она выключена. Это подсказка macOS о характере работы, а не фиксация частоты GPU; режим дисплея, QoS и постоянные настройки питания не меняются. Флаг и завершение activity записываются в результат.

`--wired-memory` доступен для MLX/MLX-VLM; парные speculative-протоколы MTP и DFlash включают эту политику обязательно. Он повторяет политику штатного генератора MLX-LM: временно устанавливает рекомендованный Metal wired limit вокруг каждого запроса, синхронизирует GPU и восстанавливает прежний лимит. vLLM Metal делает это внутри плагина. У PyTorch 2.10 MPS соответствующего публичного API нет; `set_per_process_memory_fraction` задаёт лимит выделений и не заменяет wiring.

## Что именно сравнивается

Исходные веса имеют **MLX affine 4-bit, group size 64**. Они не являются готовым PyTorch checkpoint с обычным `quantization_config.quant_method`. Transformers adapter переносит те же упакованные linear weights в официальный `MetalLinear`, восстанавливает имена/представления RMSNorm и Conv1D и распаковывает embedding в BF16. Это отдельный адаптер совместимости; результат нельзя приписывать прямому стандартному `from_pretrained`.

vLLM запускается через официальный **vllm-metal**, а не через CUDA. MLX-LM и MLX-VLM используют общий MLX runtime, но свои реализации/загрузчики модели. Дополнительные варианты и источники описаны в [обзоре](framework-research.md).

Prefill считает входные токены и включает получение первого выходного токена. Decode считает последующие `N−1` токенов. Загрузка/токенизация/прогрев исключены. В vLLM измеряется время шагов движка вместе с scheduler; это явно записано в метаданных. Подробности: [методика](benchmark-methodology.md).

## Результаты и проверка

[Сводная таблица](results/frameworks.md) генерируется только из фактических `summary.json`. Каждый запуск сохраняет `samples.jsonl` с временами, числом токенов и сгенерированным текстом. Ошибки сохраняются в `error.txt`. Для сопоставимости проверяйте одинаковые хеши prompt, режим и длину генерации; smoke-запуски не заменяют 100-примерный benchmark.

`report/validate.sh` проверяет последние четыре полных запуска серии `awake`: одинаковые хеши модели, датасета и prompt, 100 индексов, фактические token IDs и длительности обеих фаз. Все агрегаты пересчитываются из `samples.jsonl`; результат сохраняется в `artifacts/reports/final-validation.json`. Неполная серия или несовпадение завершают проверку с ошибкой.

Для повторной проверки скорости на первых пяти примерах всех четырёх движков:

```bash
bash scripts/benchmark/recheck.sh
# Только пересчитать сравнение из сохранённых замеров
bash scripts/report/recheck.sh
```

Повтор использует 128 новых токенов, один прогрев, `caffeinate -di` и wired memory для MLX-LM/MLX-VLM. Каждый запуск сохраняет отдельные данные и логи. [Отчёт повторной проверки](results/frameworks.md) сравнивает новые пять примеров с теми же первыми пятью из исходной серии; статистики всех 100 приведены отдельно. Выборочная дисперсия и стандартное отклонение рассчитаны по скоростям отдельных запросов с делителем `n−1`; `mean ± SD` описывает разброс между запросами, а не доверительный интервал. Для одного запроса выборочная дисперсия не определена (`null`).

```bash
.venv/bin/python -m pytest -q
# Отдельная проверка арифметики переноса MLX-весов в Transformers, без GPU
PYTHONPATH=src .venv-transformers/bin/python -m unittest discover -s tests -p test_transformers_adapter.py -v
# Численная проверка матричных Metal-ядер на GPU, без загрузки модели
bash scripts/setup/validate_metal.sh
# Питание, thermalState, память и фоновые процессы; без GPU-вычислений и sudo
bash scripts/setup/telemetry.sh
```

`artifacts/smoke-parity.json` фиксирует совпадение первых восьми токенов Transformers и MLX на одном одинаковом prompt; это короткая проверка переноса весов, а не оценка качества на всём датасете. Численная проверка kernels сохраняет ошибки FP32/FP16/BF16 в `artifacts/metal-kernel-validation.json`.

## Speculative decoding: DFlash

Используется официальный [Z-Lab DFlash](https://github.com/z-lab/dflash), закреплённый Git commit и отдельное окружение `.venv-speculative`. Target остаётся исходной Qwen3.5-9B-4bit. [Draft-модель](https://huggingface.co/z-lab/Qwen3.5-9B-DFlash) скачивается отдельно (~2.58 GB BF16), проверяется по SHA256 и квантуется в памяти перед загрузкой target; файлы target не меняются. Сравнение кандидатов и ограничения — в [обзоре](speculative-decoding-research.md).

```bash
bash scripts/setup/speculative.sh
bash scripts/download/draft.sh

# Обычный greedy decode в том же окружении: 5 примеров × 128 токенов
bash scripts/benchmark/speculative.sh --mode baseline

# DFlash: блок из 1 опорного и максимум 2 draft-токенов
bash scripts/benchmark/speculative.sh --block-size 3 --draft-bits 4

# Проверка большего блока
bash scripts/benchmark/speculative.sh --block-size 5 --draft-bits 4
```

Каждый benchmark выводит каталог со своим `summary.json`, `prompts.json` и `samples.jsonl`. Фазы разделены так же, как в обычном тесте: первый токен относится к prefill, оставшиеся `N−1` — к decode. В decode включены draft, target verification, откаты рекуррентного состояния и завершение всех отложенных GPU-вычислений. Детокенизация текста выполняется после замера. `caffeinate -di` действует только во время запуска.

Acceptance — отношение принятых draft-токенов к предложенным и проверенным draft-токенам; bonus-токен основной модели исключён. Учёт основан на фактических откатах upstream, включая последний блок при ограничении длины. Название алгоритма само по себе не подтверждает совпадение конкретных GPU-вычислений: отчёт отдельно сравнивает все сгенерированные token IDs с обычным greedy decode.

Для отчёта передайте каталоги нужных запусков явно:

```bash
bash scripts/report/speculative.sh \
  --baseline artifacts/runs/<baseline-run> \
  --speculative artifacts/runs/<dflash-run>
```

Загрузка модели и draft, а также один прогрев исключены из фазовых замеров. Для оценки выигрыша используются свежие baseline и DFlash из одного окружения, с одинаковыми входами и длиной генерации; прежние замеры других движков не подставляются вместо baseline.

## Native MTP

Второй вариант использует отдельную [Qwen3.5-9B-MTP-4bit голову](https://huggingface.co/mlx-community/Qwen3.5-9B-MTP-4bit) и штатный speculative verifier MLX-VLM 0.7.1 в основном окружении `.venv`. Голова занимает ~137 MB и использует embedding/output head основной модели. Исходный checkpoint target не содержит MTP weights, поэтому эта загрузка обязательна.

```bash
bash scripts/download/mtp.sh
bash scripts/benchmark/mtp.sh --mode baseline
bash scripts/benchmark/mtp.sh  # блок 3 по умолчанию: лучший из проверенных MTP 2/3

# Проверка на более длинном ответе: 5 примеров × 512 токенов
bash scripts/benchmark/mtp.sh --mode baseline --max-new-tokens 512
bash scripts/benchmark/mtp.sh --max-new-tokens 512
```

Для этой пары baseline — обычный MLX-VLM с тем же target. Между MLX-LM/DFlash и MLX-VLM/MTP реализации отличаются; отчёт отвергает смешивание этих пар. В MTP prefill сохраняются hidden states всех частей prompt, а их обработка draft-головой включена в decode. Используются штатные операции и rollback MLX-VLM. Отчёт запускается той же парой `scripts/report/speculative.{sh,py}` с явными каталогами MTP baseline и кандидатов; отдельный каталог задаётся `--output artifacts/reports/mtp`.

### Полная серия 100 × 2048 с записью событий

Это отдельный эксперимент с более длинным выводом и включённым наблюдением за генерацией. На момент добавления этого раздела серия ещё выполняется; ускорение и полный паритет для всех 100 строк пока не установлены. Результаты прежних пяти запросов ниже не заменяют её итогов.

После подготовки основного окружения, target, датасета и MTP-головы:

```bash
bash scripts/benchmark/mtp_series.sh \
  --count 100 --max-new-tokens 2048 \
  --chunk-size 5 --block-size 3 --trace-generation \
  --label mtp-long2048 --output artifacts/series/mtp-long2048
```

`--chunk-size 5` означает пять строк датасета в одном процессе, а `--block-size 3` — опорный токен и до двух предложений MTP внутри одного запроса. Серия выполняет 20 пар: `A₀ B₀ | B₁ A₁ | A₂ B₂…`, где A — обычный MLX-VLM, B — native MTP. Первая пара использует строки 0–4, следующая — 5–9, и так до 99. Оба метода получают каждую строку ровно один раз; исходные индексы и dataset SHA сохраняются. Это чередование ABBA уменьшает связь метода с порядком выполнения, но не фиксирует температуру, частоты и питание.

Всего запускаются 40 отдельных процессов. Каждый загружает модель, выполняет один полный прогрев на 2048 токенов и затем измеряет пять запросов со свежим cache. Загрузка и 40 дополнительных прогревов не входят в фазовые скорости; это не один непрерывный запуск с единственной загрузкой. Для каждого процесса действуют одинаковые `caffeinate -di` и wired memory.

Ход серии и попытки сохраняются в `artifacts/series/mtp-long2048/manifest.json`, а измерения — в отдельных каталогах `artifacts/runs/`. После остановки продолжите **ту же конфигурацию** с `--resume`:

```bash
bash scripts/benchmark/mtp_series.sh \
  --count 100 --max-new-tokens 2048 \
  --chunk-size 5 --block-size 3 --trace-generation \
  --label mtp-long2048 --output artifacts/series/mtp-long2048 --resume
```

Продолжение сверяет параметры, хеши измерительного кода, входов и runtime, а также проверяет уже завершённые процессы. Оно не запускает второй экземпляр поверх работающей серии. Незавершённый процесс повторяется целиком в новой попытке; прежние частичные файлы сохраняются, их строки не смешиваются с новой попыткой. Изменение закреплённых входов или кода требует отдельной серии. По умолчанию перед следующим процессом серия останавливается со статусом `paused_battery`, если аккумулятор разряжается и заряд ниже 20% (`--min-battery-percent`); продолжение также выполняется через `--resume`.

После полного покрытия индексов 0–99 создаются объединённые результаты. Их точные каталоги находятся в `manifest.json` → `merged_runs`, сравнительный JSON — в `comparison_path`. Объединение сохраняет происхождение строк и хеши исходных файлов. `status=completed` означает завершение измерений; отдельно проверяйте `parity_status`, который может быть `failed`. При полном выполнении каждый метод выдаёт 204 800 токенов, из них 204 700 относятся к decode. Совпадение проверяется по всем выходным ID, а не только по видимому фрагменту или тексту до EOS.

В этой серии `--trace-generation` включён по умолчанию. Каждая строка `samples.jsonl` содержит `generation_trace`: исходные token IDs, события `draft`/`commit`, временные метки, результаты проверки предложений и EOS IDs. После проверки строка записывается с `flush`/`fsync`. Baseline фиксирует токены при уже существующих чтениях `.item()` без дополнительных GPU-синхронизаций; MTP дополнительно читает готовые предложения на CPU перед verification. Эти чтения, вызовы часов и запись событий в память входят в измеренное время. Построение Unicode-текста для анимации и запись файлов выполняются вне фаз. Поэтому сравниваются два запуска с одной политикой наблюдения; эту серию нельзя подставлять вместо прежних замеров без событий.

Фиксированный бюджет игнорирует EOS: если ответ закончился раньше, генерация продолжается до 2048 токенов. Скорость по полному бюджету не является временем решения задачи. Время первого EOS можно определить отдельно по фактическому событию `commit`, содержащему этот токен. Для пакета MTP это время готовности пакета, без интерполяции внутри него. Отсутствие EOS означает только, что он не появился в записанном бюджете.

### GIF любой сохранённой траектории

После завершения серии выберите исходный индекс строки 0–99. Новый инференс не выполняется; используются сохранённые события и локальные файлы токенизатора, веса моделей не загружаются:

```bash
bash scripts/demo/from_logs.sh \
  --series artifacts/series/mtp-long2048 --index 42 \
  --rates 1 --fps 20 --output artifacts/demos/sample-042
```

Команда создаёт `trace.json`, GIF, кадры preview/final и `render-manifest.json`. Длинный текст прокручивается; обе дорожки используют одну шкалу воспроизведения и свои реальные времена от начала prefill. Если выходы различаются, показываются обе фактические последовательности с отметкой расхождения, без утверждения о точном совпадении и без отношения скоростей для одинакового текста.

`--series` требует уже объединённых результатов. Для готовой пары процессов до завершения всей серии можно передать `--baseline` и `--mtp` с их каталогами вместо `--series`; выбранный `--index` должен присутствовать в обоих. `--trace-only` создаёт только обогащённый текстом trace, а `--max-visible-tokens 256` ограничивает видимый префикс, сохраняя полную запись. `--rates 1 0.25` создаёт обычное и одинаково замедленное в четыре раза воспроизведение. Допустимые FPS для точных длительностей кадров GIF: 1, 2, 4, 5, 10, 20, 25, 50 и 100.

Показ заканчивается перед первым EOS каждой дорожки либо на заданном пределе префикса; это не обрезает сырые 2048 token IDs. Время окончания видимого текста и время первого EOS — разные показатели. Из старых средних tok/s без событий восстановить реальные моменты появления токенов нельзя. Подробности записи и визуализации: [демонстрация генерации](generation-demo.md).

В сравнении оценивается весь путь исполнения: обычный MLX-VLM против MTP с его специализированным verifier и fused quantized argmax. Даже при совпадении token IDs итоговую разницу времени нельзя приписать только уменьшению числа проходов target; отдельного эксперимента, изолирующего вклад этих kernels, здесь нет.

### Локальные результаты speculative decoding

На M2 Pro из проверенных вариантов выбран **native MTP, блок 3**. В [серии 5 × 512 токенов](results/mtp-512.md) получены следующие средние и выборочные SD:

| Режим | Prefill, tok/s | Decode, tok/s |
|---|---:|---:|
| MLX-VLM baseline до MTP | 178.61 ± 29.77 | 34.81 ± 0.47 |
| Native MTP, блок 3 | 181.64 ± 31.79 | 41.58 ± 0.96 |
| MLX-VLM baseline после MTP | 151.10 ± 25.33 | 30.30 ± 4.39 |

Относительно более быстрого baseline MTP дал **+19.4% decode**. Все **2560 выходных token IDs** совпали с обоими baseline; принято **84.57%** draft-токенов. Peak allocated memory MLX выросла с **5.48 до 5.62 GB**. [Контроль после MTP](results/mtp-512.md) показывает снижение скорости baseline; поэтому больший выигрыш относительно него не выбран основной оценкой. Все эти запуски прошли от батареи, частоты и температура не фиксировались; SD описывает разброс пяти запросов, а не доверительный интервал.

[Короткая серия MTP, 5 × 128](results/speculative-128.md): блок 2 — 31.29 ± 1.27, блок 3 — 38.15 ± 2.75 против 29.13 ± 1.95 tok/s baseline. [Официальный DFlash](results/speculative-128.md) тоже установлен и проверен: лучший из вариантов 2/3/5 и draft 4/8-bit дал 30.91 ± 1.87 против 30.16 ± 0.18 tok/s, всего +2.5%. Все выходные token IDs в этих коротких проверках совпали. Исходные времена, дисперсии, счётчики и версии сохранены в JSON/CSV рядом с отчётами.
