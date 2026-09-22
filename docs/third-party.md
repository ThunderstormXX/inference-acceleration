# Сторонние компоненты и происхождение данных

Документ описывает upstream-компоненты и не назначает лицензию собственному коду проекта. Весовые файлы, установленные зависимости, кеши и исходный датасет не включены в Git.

| Компонент | Использование | Лицензия / источник |
|---|---|---|
| [MLX](https://github.com/ml-explore/mlx) | GPU runtime; quantized shader headers берутся из установленного пакета | [MIT, Apple](https://github.com/ml-explore/mlx/blob/main/LICENSE) |
| [MLX-LM](https://github.com/ml-explore/mlx-lm) | Target loading и decode | [MIT](https://github.com/ml-explore/mlx-lm/blob/main/LICENSE) |
| [MLX-VLM](https://github.com/Blaizzy/mlx-vlm) | Qwen3.5 target и native MTP, измерена версия 0.7.1 | [MIT, Prince Canuma](https://github.com/Blaizzy/mlx-vlm/blob/main/LICENSE) |
| [ZMLX](https://github.com/Hmbown/ZMLX/tree/d8cd0d88d4299ca6821d706f9d5b4a3520d688f5) | Experimental scoped DeltaNet/SwiGLU fusion; pinned source checkout, not vendored | [MIT, Hunter Bown](https://github.com/Hmbown/ZMLX/blob/d8cd0d88d4299ca6821d706f9d5b4a3520d688f5/LICENSE) |
| [DFlash](https://github.com/z-lab/dflash/tree/07ebd93db9f472af339b644bb70221ad8428328a) | Официальный MLX runtime, закреплённый commit | [MIT, Z Lab](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/LICENSE) |
| [Transformers](https://github.com/huggingface/transformers) | Target architecture и MetalLinear | [Apache-2.0](https://github.com/huggingface/transformers/blob/main/LICENSE) |
| [PyTorch](https://github.com/pytorch/pytorch) | MPS tensors и runtime Metal compilation | [BSD-style и сопутствующие notices](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| [vLLM](https://github.com/vllm-project/vllm) / [vllm-metal](https://github.com/vllm-project/vllm-metal) | vLLM с Metal backend | [vLLM Apache-2.0](https://github.com/vllm-project/vllm/blob/main/LICENSE), [vllm-metal LICENSE](https://github.com/vllm-project/vllm-metal/blob/main/LICENSE) |

При распространении копий сторонних исходников или существенных фрагментов следует сохранять их copyright/license notices. В проекте upstream runtime/headers устанавливаются как зависимости; опубликованные результаты содержат provenance, а не копии весов.

## Модели и данные

- Target: [mlx-community/Qwen3.5-9B-4bit](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit), revision 8b2b98c00a6b4d291155e4890773ca8f769aee53; производная [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B). Карточки указывают Apache-2.0.
- Native MTP: [mlx-community/Qwen3.5-9B-MTP-4bit](https://huggingface.co/mlx-community/Qwen3.5-9B-MTP-4bit), revision 222dfd2c23fc9518d7b817e4f8e0cb0571787489; выделенная и квантизованная MTP-голова Qwen3.5-9B, Apache-2.0.
- DFlash draft: [z-lab/Qwen3.5-9B-DFlash](https://huggingface.co/z-lab/Qwen3.5-9B-DFlash), revision 5fc3b3d474760f18c516db87d84c37edbfd3ede6, Apache-2.0. Лицензия весов отличается от MIT-лицензии runtime.
- Dataset: [ThunderstormXXL/deepscaler-teacher-sft-vllm-official-40k](https://huggingface.co/datasets/ThunderstormXXL/deepscaler-teacher-sft-vllm-official-40k), revision 6f0fadbf6b495cdea0e4bd83c4b64c7e36a992a3; карточка указывает MIT. Загрузчик получает первые 100 полных строк. Полные teacher responses не публикуются в results bundles; сохранены идентификаторы, хеши и измерения.

## Публичный пример

[mtp-trace.json](examples/mtp-trace.json) — очищенная копия реальной инструментированной генерации: один выбранный пример, а не независимый throughput-бенчмарк. [Provenance](examples/mtp-trace.provenance.json) отдельно хранит SHA-256 исходного и опубликованного файлов. Выходные IDs, текст, события и тайминги сохранены; пути относительные, идентификаторы процессов удалены.
