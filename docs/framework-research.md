# Инференс Qwen3.5-9B на Apple Silicon: проверка совместимости

Проверено 18 сентября 2026 года. Здесь приведены сведения из первичных источников и отмеченные локальные проверки совместимости. Числа скорости следует брать только из файлов завершённых прогонов проекта.

## Что представляет собой выбранная модель

`mlx-community/Qwen3.5-9B-4bit` — конвертация `Qwen/Qwen3.5-9B`, выполненная MLX-VLM 0.3.12. Репозиторий содержит и языковую, и визуальную части. Размер опубликованных весов около 5,95 GB. В `config.json` записаны affine-квантизация 4 bit, группы 64 и гибридная архитектура: три слоя linear attention на один full attention. Формат safetensors сам по себе не делает упакованные MLX-веса совместимыми с любым движком. Автоматически сгенерированные примеры «Use this model» на Hugging Face не являются проверкой запуска на Mac. [Карточка модели](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit), [конфигурация](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit/raw/main/config.json).

## Практические варианты

| Движок | Выбранные MLX-веса | Подход к проверке |
|---|---|---|
| MLX-LM | Да, языковая часть | Основная точка отсчёта; прямые синхронизированные вызовы модели |
| MLX-VLM | Да, включая vision | Текстовый тест возможен; отдельный путь исполнения и загрузки |
| Transformers + PyTorch MPS | Через адаптер формата | Родной `MetalLinear` читает те же affine-веса; архитектура и cache остаются Transformers |
| vLLM + vllm-metal | Да, поддерживается семейство Qwen3.5 | Установить согласованные Mac wheels; отключить prefix cache |
| vllm-mlx | Да, отдельный проект поверх MLX | Отдельно подписывать результаты; это не автоматически upstream vLLM |
| llama.cpp / Metal | Нужен GGUF | Сравнение другого представления весов; отметить квантизацию и источник GGUF |

### MLX-LM и MLX-VLM

MLX-LM поддерживает текстовый Qwen3.5 начиная с 0.30.7. Текущая реализация умеет читать checkpoint с `text_config`, исключать vision-веса и создавать одновременно recurrent state и KV-cache. Это позволяет измерять исходный MLX checkpoint без повторного квантования. Версию 0.31.0 не следует выбирать в качестве старого «быстрого baseline»: она была отозвана, поэтому для воспроизводимости нужно записывать реально установленную актуальную версию. [Релизы MLX-LM](https://github.com/ml-explore/mlx-lm/releases), [реализация Qwen3.5](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py), [сообщение сопровождаемых разработчиками о 0.31.0](https://github.com/ml-explore/mlx-lm/issues/1425).

Официальная карточка выбранного checkpoint предлагает MLX-VLM. Для задачи с текстовыми цепочками загрузка vision-части не обязательна. MLX-VLM отдельно нормализует имена весов, переставляет оси convolution и меняет convention RMSNorm. Эти преобразования нельзя игнорировать при переносе весов в другой runtime. [Конверсия Qwen3.5 MLX-VLM](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/qwen3_5/qwen3_5.py).

### Transformers на MPS

Современный Transformers имеет `MetalConfig` и `MetalLinear`: affine-квантизация 2/4/8 bit, упакованные uint32, отдельные scales и qbiases, fused dequantization/matmul через Metal kernels. Нужен пакет `kernels`; ядро загружается с Hugging Face при первом использовании. Следовательно, утверждение «4-bit Transformers принципиально невозможен на Mac» устарело. [Документация Metal quantization](https://huggingface.co/docs/transformers/main/quantization/metal).

Прямой `from_pretrained` для этого MLX checkpoint не является готовым совместимым путём: отличаются quantization metadata, имена affine biases и conventions некоторых слоёв. Адаптер проекта использует родной Transformers `Qwen3_5ForCausalLM`, подменяет только quantized linear modules на официальный `MetalLinear`, загружает исходные packed weights, разворачивает embedding в BF16 и обращает MLX-преобразования RMSNorm/Conv1d. Он проверяет формы и полноту всех языковых весов. Такой путь необходимо проверить коротким прогоном и сравнением вывода с MLX перед полным benchmark; наличие класса и успешная установка не доказывают корректность или скорость. [MetalLinear](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/metal_quantization.py), [архитектура Transformers Qwen3.5](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py).

Для этого пути создано отдельное окружение `.venv-transformers`: `bash scripts/setup/transformers.sh`. Оно закрепляет PyTorch 2.10.0 и Transformers 5.17.0, поскольку опубликованные native kernels имеют отдельные ABI-сборки для Torch 2.8, 2.9 и 2.10. Самый новый доступный Torch не обязательно совместим с таким готовым бинарным ядром. Точные разрешённые версии сохраняются в `requirements-transformers.lock`. [Опубликованные сборки Metal kernel](https://huggingface.co/kernels-community/mlx-quantization-metal-kernels/tree/main/build). Локальная проверка также выявила ограничение Transformers 5.17: `kernels` должен быть в диапазоне `[0.16, 0.17)`. Поэтому закреплён `kernels==0.16.0`; первоначальный отказ с 0.17.1 сохранён в логе установки.

**Реально обнаруженное ограничение macOS 15.6.1:** готовая библиотека успешно импортируется, но первый GPU-вызов завершается ошибкой `This library is using language version 4.0 which is not supported on this OS`. Результат сохранён в `artifacts/runs/20260918T160424.543902Z-transformers-smoke/error.txt`. Проверка самого раннего доступного бинарного выпуска от февраля 2026 также обнаружила Metal 4.0 в заголовке shader library. Импорт библиотеки не подтверждает совместимость её GPU-кода.

Решение проекта для этой ОС: компиляция неизменённых affine `qmv`/`qmm` templates из headers **MLX 0.32.2** через **`torch.mps.compile_shader`**. Компилятор текущей ОС создаёт совместимый Metal-код; полный Xcode и обновление macOS не нужны. Torch выделяет все tensors и управляет GPU-вызовами, а MLX runtime не импортируется. Это Transformers с пользовательским адаптером quantized kernels, а не стандартный запуск `from_pretrained`. В метаданных каждого запуска сохранены версия исходников и SHA256 развёрнутого shader source. [Исходные quantized shaders MLX](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/quantized.h), [PyTorch compile_shader](https://docs.pytorch.org/docs/stable/generated/torch.mps.compile_shader.html).

Короткий фактический запуск Transformers с этим адаптером завершился успешно: первые восемь greedy token IDs полностью совпали с MLX на одном prompt из 133 токенов. Проверка записана в `artifacts/smoke-parity.json`; она не доказывает совпадения всех длинных ответов. Повторяемая задача `bash scripts/setup/validate_metal.sh` отдельно сравнивает prefill/decode kernels FP32/FP16/BF16 с независимым CPU matmul, включая неполные tiles. Qwen recurrent attention использует reference PyTorch implementations; optional CUDA/Triton-пакеты здесь не дают автоматического ускорения MPS.

### vLLM с vllm-metal

`vllm-metal` находится в организации vllm-project и подключает вычисления MLX/Metal к upstream vLLM: API, scheduler и block manager остаются vLLM. Требуются Apple Silicon и macOS 15+. Семейство Qwen3.5 поддержано как гибрид SDPA/GDN. Ускорения M5 NAX относятся к M5; их нельзя переносить на M2 Pro. [Архитектура проекта](https://github.com/vllm-project/vllm-metal), [матрица моделей](https://github.com/vllm-project/vllm-metal/blob/main/docs/supported_models.md).

Проверенный стабильный выпуск — `vllm-metal==0.29.0` от 11 сентября 2026, согласованный с `vllm==0.29.0`. Готовые wheels требуют native Python 3.12. MLX закреплён на 0.32.1; требуются Transformers >=5.10.4 и huggingface_hub >=1.28.0. Для гибридного GDN prefix cache остаётся экспериментальным с известными расхождениями вывода; для исходного benchmark его нужно отключить. [Релиз 0.29.0](https://github.com/vllm-project/vllm-metal/releases/tag/v0.29.0).

В проектном окружении Python 3.12 можно установить согласованную пару, не затрагивая системный Python:

```sh
uv pip install --python .venv-vllm/bin/python \
  'https://github.com/vllm-project/vllm/releases/download/v0.29.0/vllm-0.29.0%2Bcpu-cp312-cp312-macosx_11_0_arm64.whl' \
  'https://github.com/vllm-project/vllm-metal/releases/download/v0.29.0/vllm_metal-0.29.0-cp312-cp312-macosx_15_0_arm64.whl'
```

URL upstream wheel составлен тем же способом, что в штатном установщике; реальная успешность скачивания и импорта фиксируется логом setup. Нельзя устанавливать эти cp312 wheels в основное окружение cp313. [Установщик](https://github.com/vllm-project/vllm-metal/blob/main/install.sh).

### vllm-mlx

Это отдельный MLX server с собственными механизмами batching. В зависимостях текущего проекта `vllm` — optional extra, поэтому запуск команды `vllm-mlx` сам по себе нельзя представлять как измерение upstream vLLM. Текущая версия в исходниках — 0.4.1; минимум MLX-LM 0.31.3 и MLX-VLM 0.6.5, причём последняя граница связана с исправлением двойной обработки Qwen3.5 weights. Для сравнения одиночного запроса его выигрыш над прямым MLX требует измерения; преимущество continuous batching относится прежде всего к суммарной пропускной способности нескольких запросов. [Метаданные проекта](https://github.com/waybarrios/vllm-mlx/blob/main/pyproject.toml), [руководство batching](https://github.com/waybarrios/vllm-mlx/blob/main/docs/guides/continuous-batching.md).

### llama.cpp / Metal

llama.cpp — полезный независимый кандидат с Metal backend и GGUF-весами. HTTP server возвращает внутренние timings отдельно для prompt evaluation и generation; так можно сравнивать фазы без подмены prefill сетевым TTFT. Для данного исследования необходимо отдельно скачать GGUF-конверсию той же Qwen3.5-9B, зафиксировать её ревизию/квант и не объявлять, что использованы байт-в-байт те же MLX-веса. Встроенный `llama-bench` полезен для синтетических длин, но сам по себе не выполняет требование о конкретных 100 цепочках датасета. [Сборка Metal](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md), [API и timings сервера](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

## Правила сравнения

1. Одни и те же 100 сохранённых строк, tokenizer/chat template, входные token IDs, ограничение длины и число новых токенов. Записывать, какая часть цепочки была входом: задача, задача с префиксом решения или вся цепочка.
2. Batch size 1 и concurrency 1 для основной таблицы. Throughput при concurrency >1 — отдельный эксперимент.
3. Один раз загрузить веса и сделать warmup вне измерения; новый cache/state на каждый запрос. Отключить reuse prefix cache.
4. Prefill заканчивается после первого токена и синхронизации GPU. При N выходных токенах обычный autoregressive decode выполняет N−1 отдельных forward calls; для speculative decoding N−1 — число полезных выходных токенов, а число verification rounds записывается отдельно. Отдельно сохранять абсолютное время и число токенов.
5. Показывать как среднее скоростей по строкам, так и `sum(tokens)/sum(seconds)`: это разные агрегаты. HTTP TTFT называть TTFT, если нет внутренних данных о фазах.
6. Фиксированная генерация с игнорированием EOS измеряет скорость вычислений, а не длительность решения задачи и не качество рассуждений. Короткое ограничение вывода не должно выдаваться за генерацию полных RL-цепочек.
7. На Mac 16 GB запускать движки последовательно. Отмечать swap, фоновую GPU-нагрузку, модель чипа, версии пакетов и реальные причины пропуска движка.

Практический порядок: прямой MLX-LM → Transformers Metal с проверкой адаптера → vLLM Metal → MLX-VLM/vllm-mlx → GGUF llama.cpp. Назвать победителя можно только после сопоставимых локальных измерений.
