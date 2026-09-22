# Speculative decoding для Qwen3.5-9B на Apple Silicon

Обновлено 19 сентября 2026 года. Целевая машина проекта — M2 Pro с 16 GB unified memory; целевые веса — существующая `mlx-community/Qwen3.5-9B-4bit`. Документ объединяет первоначальное исследование первичных источников и последующую локальную проверку. DFlash и native MTP развёрнуты, их draft-веса скачаны; завершены серии 5 × 128 и сравнение MTP K3 с предшествующим baseline на 5 × 512 выходных токенов. Внешние сведения о методах ниже отделены от измерений на этой машине.

## Состояние реализации

1. **Официальный DFlash от Z-Lab, локальный MLX backend — реализован и измерен.** Проверены блоки 2/3/5 с 4-bit draft и блок 3 с 8-bit draft. Запуск: [scripts/benchmark/speculative.sh](../scripts/benchmark/speculative.sh) с сопровождающим Python-скриптом.
2. **Нативный MTP через `mlx-vlm` 0.7.1 — реализован и измерен.** Проверены блоки 2 и 3 с отдельной 4-bit MTP-головой размером 136.9 MB. Запуск: [scripts/benchmark/mtp.sh](../scripts/benchmark/mtp.sh) с сопровождающим Python-скриптом.
3. **EAGLE3 — исследовательский кандидат без локальной реализации.** Checkpoint для нужного target существует, но готовность именно этой комбинации на MLX здесь не подтверждена.

Нельзя называть результаты доказательством «SOTA на M2 Pro» или переносить цифры CUDA/M5 из README на эту машину. DFlash — современный метод с официальным MLX runtime; лучший вариант для конкретной нагрузки определяется одинаковыми локальными измерениями. DFlash2 в текущем README представлен для других моделей, и наличие DFlash2 само по себе не означает поддержку Qwen3.5-9B.

## Выполненная проверка: 5 prompts × 128 токенов

Скорости ниже — decode mean ± выборочное SD по пяти разным prompts, в tok/s. Первый выходной токен относится к prefill; decode учитывает 127 полезных токенов каждого запроса. У каждой семьи собственный target-only baseline: MLX-LM для DFlash и MLX-VLM для MTP. Между семьями baseline не подменяется.

| Runtime и режим | Decode mean ± SD, tok/s | Отношение mean к своему baseline |
|---|---:|---:|
| MLX-LM baseline | 30.16 ± 0.18 | 1.000 |
| DFlash K2, draft 4-bit | 29.31 ± 1.94 | 0.972 |
| DFlash K3, draft 4-bit | 30.91 ± 1.87 | 1.025 |
| DFlash K5, draft 4-bit | 24.95 ± 1.79 | 0.827 |
| DFlash K3, draft 8-bit | 28.43 ± 1.11 | 0.943 |
| MLX-VLM baseline | 29.13 ± 1.95 | 1.000 |
| MTP K2 | 31.29 ± 1.27 | 1.074 |
| MTP K3 | 38.15 ± 2.75 | 1.310 |

Источники чисел, prefill, сырые прогоны и их SHA-256: [отчёт DFlash](results/speculative-128.md) и [отчёт MTP](results/speculative-128.md), с JSON/CSV рядом. Все перечисленные speculative-конфигурации совпали со своим baseline по всем 640 выходным token IDs. Это проверка данной выборки, а не доказательство эквивалентности для произвольных входов. У MTP K3 принято 406 из 463 предложенных draft-токенов: weighted acceptance 87.69%.

У DFlash K3 4-bit наблюдаемая прибавка составляет только 2.5%, крупного ускорения серия не показала. [Повторный MLX-LM baseline после серии](results/speculative-128.md) дал 29.01 ± 0.18 tok/s против исходных 30.16 ± 0.18; поэтому небольшую разницу нельзя уверенно отделить от дрейфа условий. MTP K3 в серии 128 токенов показал +31.0% к своему baseline.

Все завершённые серии выше выполнялись от батареи, с `caffeinate -di` и wired memory. Снимки питания включены в отчёты; эти меры не фиксируют температуру, частоты и остальные условия. SD между пятью разными prompts не является доверительным интервалом и не измеряет дисперсию повторных запусков одного prompt. В обоих семействах зафиксированы `mlx==0.32.2`, `mlx-metal==0.32.2`, `mlx-lm==0.31.3`; у пары MTP дополнительно совпадает `mlx-vlm==0.7.1`.

## Дополнительная проверка: 5 prompts × 512 токенов

Сравнение с baseline **до** MTP завершено: MLX-VLM дал **34.81 ± 0.47 tok/s**, MTP K3 — **41.58 ± 0.96 tok/s**, отношение средних **1.194× (+19.4%)**. Все **2560 выходных token IDs совпали**; принято **1606 / 1899 draft-токенов (84.57%)**. Подробности и исходные файлы: [отчёт MTP 512](results/mtp-512.md). Эта серия также выполнена от батареи с `caffeinate -di` и wired memory.

[Контрольный baseline после MTP](results/mtp-512.md) завершён: **30.30 ± 4.39 tok/s**, при тех же 2560 выходных token IDs. Отношение к нему составляет 1.372×, но заметное снижение и разброс baseline показывают изменение условий между запусками. Основной ориентир — более осторожные **+19.4% относительно быстрого baseline до серии**, а не гарантированный выигрыш на любом запросе. Peak allocated memory MLX: 5.48 GB baseline / 5.62 GB MTP; это память аллокатора MLX, не общая память процесса или системы.

**Для этого проекта рекомендуется native MTP K3 среди проверенных вариантов:** прибавка decode наблюдается при 128 и 512 токенах, с совпадением выходных IDs на обеих выборках. Проектный benchmark теперь использует `block_size=3` по умолчанию. Это локальный выбор для проверенной модели и нагрузки, а не утверждение о глобальном SOTA; значение по умолчанию в upstream checkpoint остаётся **2**.

## 1. Официальный DFlash

[Официальный репозиторий Z-Lab](https://github.com/z-lab/dflash) описывает локальный MLX backend для Qwen3, Qwen3.5, Qwen3.6 и Gemma 4. Текущий способ установки — `dflash[local]`. Документация отдельно рекомендует **`block_size <= 5`, если target или draft квантизован**, из-за производительности MLX quantized matmul при большем размере проверки.

[PyPI metadata](https://pypi.org/pypi/dflash/json) на дату проверки: `dflash==0.1.0`, Python >= 3.10, MIT. Для macOS arm64 extra `local` закрепляет `mlx==0.32.0`, `mlx-lm==0.31.3`, `huggingface-hub==1.27.0`. Это сведения опубликованного пакета. В проекте DFlash развёрнут отдельно в `.venv-speculative` из закреплённых исходников; фактически измеренная пара использует MLX 0.32.2, как указано выше и в metadata прогонов. Уже измеренные основные окружения не заменяются зависимостями wheel.

Для проектного адаптера выбран официальный source commit [`07ebd93db9f472af339b644bb70221ad8428328a`](https://github.com/z-lab/dflash/tree/07ebd93db9f472af339b644bb70221ad8428328a). Его локальную загрузку и фактические версии окружения фиксирует реализация benchmark; зависимости опубликованного wheel выше приведены как справка, а не как утверждение о версиях выполненного прогона.

Справочный пример интерфейса официального runtime. Это **не команды, по которым получены локальные числа**: для них использован проектный парный `.sh`/`.py` benchmark.

```sh
python -m pip install 'dflash[local]==0.1.0'
dflash generate mlx \
  --model ./models/qwen3.5-9b-mlx-4bit \
  --draft z-lab/Qwen3.5-9B-DFlash \
  --draft-bits 4 \
  --block-size 5 \
  'Solve the problem step by step.'
```

В проекте используется Python-адаптер с отдельными таймерами prefill/decode и raw token IDs. Путь к target указывает на уже скачанный snapshot проекта. Для самостоятельного использования upstream CLI следует сверить его с закреплённым commit: ветка `main` и опубликованный wheel могут различаться.

### Draft checkpoint

[z-lab/Qwen3.5-9B-DFlash](https://huggingface.co/z-lab/Qwen3.5-9B-DFlash), [машиночитаемое описание файлов](https://huggingface.co/api/models/z-lab/Qwen3.5-9B-DFlash?blobs=true):

| Поле | Значение |
|---|---|
| Revision | `5fc3b3d474760f18c516db87d84c37edbfd3ede6` |
| Последнее изменение | 2026-06-19 |
| Лицензия весов | Apache-2.0 |
| Target при обучении | `Qwen/Qwen3.5-9B` |
| Параметры draft | 1,291,904,512, BF16 |
| `model.safetensors` | 2,583,816,465 bytes |
| SHA-256 весов | `0a42274b32554f48de1faa0d42824e9c2ceda649c30ae0a731cddf410dd698c7` |

Это совместное переобучение Z-Lab и Modal с длиной последовательности 40k токенов и sliding-window attention. Draft имеет шесть слоёв, hidden size 4096 и использует признаки восьми слоёв target. Значение `block_size: 16` в конфигурации обучения **не отменяет** рекомендацию runtime `<= 5` для квантизованного inference.

Draft не является самостоятельной языковой моделью: предложения проверяет target. BF16 checkpoint обучался для исходной модели; фактическая доля принятия с имеющимся 4-bit target теперь приведена в локальном отчёте DFlash. `--draft-bits 4` уменьшает объём постоянных весов draft в памяти, но исходное скачивание остаётся BF16, а преобразование может требовать временную память. Завершённые прогоны подтверждают работоспособность проверенных настроек на этой машине; по одному размеру файла нельзя гарантировать работу других длин контекста, поскольку учитываются target, draft, cache, промежуточные массивы и память других процессов.

## 2. MTP через stock MLX-VLM

[mlx-community/Qwen3.5-9B-MTP-4bit](https://huggingface.co/mlx-community/Qwen3.5-9B-MTP-4bit), [API файлов](https://huggingface.co/api/models/mlx-community/Qwen3.5-9B-MTP-4bit?blobs=true) — **отдельные веса MTP-головы**, а не ещё одна полная 9B-модель.

| Поле | Значение |
|---|---|
| Revision | `222dfd2c23fc9518d7b817e4f8e0cb0571787489` |
| Последнее изменение | 2026-06-01 |
| Лицензия весов | Apache-2.0 |
| `model.safetensors` | 136,884,332 bytes |
| SHA-256 весов | `ff2f9298bb78a0f015a9998fecda6b5d40c2995d9ecd55e65a206ee5e6d6f897` |
| Формат | `qwen3_5_mtp`, affine 4-bit, group size 64 |
| Размер блока по умолчанию в upstream checkpoint | 2 |

Полный snapshot также содержит tokenizer и сопутствующие файлы, поэтому 136.9 MB — размер именно весов, а не всей загрузки. Не следует интерпретировать количество упакованных U32-элементов в API как точное количество исходных параметров.

Первичные источники runtime: [инструкция MTP](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/speculative/drafters/qwen3_5_mtp/README.md), [реестр draft-моделей](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/speculative/drafters/__init__.py), [Qwen3.5 target](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/qwen3_5/language.py), [специализированный verifier](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/qwen3_5/speculative_verifier.py). Первоначальная проверка установленного `mlx-vlm==0.7.1` подтвердила сопоставление `qwen3_5_mtp -> mtp` и методы speculative rollback у Qwen3.5; затем этот stock runtime использован в завершённых GPU-прогонах через проектный адаптер. Лицензия runtime — MIT, Python >= 3.10; [PyPI metadata](https://pypi.org/pypi/mlx-vlm/json) указывает MLX >= 0.32.2.

Справочный upstream CLI для MTP. Локальные измерения выше выполнены парным проектным benchmark, а не этим CLI-примером:

```sh
python -m mlx_vlm.generate \
  --model ./models/qwen3.5-9b-mlx-4bit \
  --draft-model mlx-community/Qwen3.5-9B-MTP-4bit \
  --draft-kind mtp \
  --draft-block-size 2 \
  --prompt 'Solve the problem step by step.' \
  --max-tokens 128 \
  --temperature 0 \
  --enable-thinking
```

Голова использует embedding/output head target. Работа с нашим exact 4-bit snapshot и совпадение greedy token IDs подтверждены локально для пяти prompts × 128 токенов при K2/K3 и пяти prompts × 512 токенов при K3. Это более узкое утверждение, чем общая совместимость любой конфигурации. Пример model card с другим уровнем квантизации target сам по себе такой проверкой не является.

Не путать эту готовую реализацию MLX-VLM с [MLX-LM PR #990](https://github.com/ml-explore/mlx-lm/pull/990): на дату проверки PR остаётся открытым, не merged. Наличие PR не означает, что команда `mlx_lm --mtp` доступна в stock MLX-LM.

## 3. EAGLE3: checkpoint есть, интеграция не подтверждена

[yuyijiong/Qwen3.5-9B-Eagle3](https://huggingface.co/yuyijiong/Qwen3.5-9B-Eagle3), [API файлов](https://huggingface.co/api/models/yuyijiong/Qwen3.5-9B-Eagle3?blobs=true): Apache-2.0; revision `714ce5fc15d0f67c2cda74ca5d692865c49f5374`; файл весов 947,153,480 bytes, SHA-256 `ed6476c11742118c17883bf7546b80356674e222a48c319b5d2ade15d450f262`. Это BF16 draft с собственным словарём 50k и отображением на словарь target.

Model card показывает использование SGLang/GPU, что не доказывает готовность конкретного checkpoint для MLX. В MLX-VLM есть общий интерфейс EAGLE3 и соответствующие Qwen3.5 capture hooks, но формат checkpoint, vocabulary mapping и parity этой комбинации здесь не проверялись. Поэтому setup-команду и локальные цифры для неё не приводим.

## Почему обычный маленький draft не подключается автоматически

Qwen3.5 сочетает attention KV cache с рекуррентным состоянием linear attention. После частичного принятия speculative блока нужно откатить **и KV cache, и convolution/SSM state**. Простое удаление лишних KV-токенов не восстанавливает состояние модели.

В установленном `mlx-lm==0.31.3` классический `speculative_generate_step` проверяет `can_trim_prompt_cache` и отклоняет неподдерживаемый cache. См. [generate.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py) и [cache.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py). Поэтому произвольная пара «Qwen3.5-0.8B draft + Qwen3.5-9B target» не является готовой заменой без verifier с поддержкой гибридного состояния. Удалять защитную проверку нельзя. Официальный DFlash и MTP в MLX-VLM следует использовать через их специализированные пути проверки и rollback.

## Протокол и дальнейшие проверки

- Сохранить exact target revision/hash, draft revision/hash, версии runtime и исходный commit. Target, tokenizer, входные token IDs, greedy режим и лимит генерации должны совпадать у baseline и speculative.
- Сначала проверить несколько коротких greedy outputs на совпадение token IDs. Квантизованная батчевая проверка может иметь отличия округления относительно однотокенной генерации; заявлять точную parity можно только в границах реально проверенных случаев.
- Использовать target-only baseline того же runtime с теми же memory/awake policies; именно так построены обе завершённые серии. Сравнение с ранее измеренным другим runtime показывать отдельно.
- Измерять prefill и decode раздельно с синхронизацией MLX. В принятом протоколе первый выходной токен относится к prefill, decode содержит `N - 1` полезных выходных токенов. Draft-предложения, отвергнутые токены и повторная работа в числитель tok/s не входят.
- Сохранять число предложенных/принятых токенов, число verification cycles, среднее принятие за цикл, peak memory и абсолютные tok/s вместе с отношением скоростей.
- Завершённая короткая серия использует одинаковые пять prompts, batch size 1 и 128 новых токенов, warmup и свежий cache каждого запроса. Более длинные генерации сравнивать отдельной серией с одинаковой длиной у обоих методов; контроль baseline до и после помогает обнаружить дрейф условий. Доступные снимки питания сохранять вместе с результатами.
- Результаты всех проверенных блоков DFlash и MTP сохранены, включая замедление. Локальный выбор MTP K3 опирается на завершённые сравнения при 128 и 512 токенах; дополнительный baseline после длинной серии проверяет дрейф условий. Разброс между разными prompts не является доверительным интервалом или дисперсией повторных прогонов одного prompt.

Другие порты, например [bstnxbt/dflash-mlx](https://github.com/bstnxbt/dflash-mlx), существуют, но при наличии официального MLX runtime не являются первым выбором. Их опубликованные результаты на другой машине и другой точности не переносятся на наш target; одинаковое имя CLI `dflash` также требует изоляции окружений при возможном последующем сравнении.
