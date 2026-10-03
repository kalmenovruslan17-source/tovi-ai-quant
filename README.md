# tovi-ai-quant

Production-код GPU-пода Tovi (MiniMax H3 / FastH3 на FastVideo) плюс 4-битная квантизация DiT для удешевления генерации.

| файл | что это |
|---|---|
| `main.py` | API мини-аппа: очередь задач, проксирует генерацию в GPU-воркеры. Не менялся. |
| `gpu_worker.py` | GPU-воркер (`GPU_WORKER_MODE=t2va` → FastH3 на :8091, `ref2va` → MiniMax H3 на :8092). Добавлен `QUANT_MODE`; по умолчанию поведение прежнее. |
| `tovi_quant/` | квантизация: режимы, интеграция SVDQuant/nunchaku в FastVideo, `doctor` для проверки на поде |
| `scripts/bench_quant.py` | сквозное сравнение режимов через настоящий `gpu_worker.py`: время, VRAM, $ за видео, PSNR/SSIM/аудио против bf16, side-by-side ролики |
| `scripts/install_quant.sh` | сборка nunchaku в venv пода |
| `Dockerfile` | образ воркера поверх CUDA-образа FastVideo |

## Важные выводы до запуска

1. **На RTX PRO 6000 (Blackwell, sm_120) SVDQuant работает в NVFP4, а не в INT4.** nunchaku сам выбирает `fp4` на sm_120/121, а INT4-ядра рассчитаны на GPU до Blackwell. NVFP4 — те же 4 бита (W4A4, SVDQuant + low-rank ветка), но с fp8-скейлами на группу из 16 элементов, и качество у него лучше, чем у INT4. Режим называется `svdq`; точность выбирается автоматически (`SVDQ_PRECISION=auto`), так что на других GPU тот же код будет работать в INT4.
2. **Готового SVDQuant-чекпойнта для MiniMax H3 / FastH3 нет, и deepcompressor эту архитектуру не поддерживает.** Поэтому веса конвертируются при старте воркера из bf16-чекпойнта: SVD rank-32 ветка, RTN-квантизация остатка в 4 бита и, опционально, SmoothQuant-сглаживание по калибровочной статистике (`QUANT_MODE=calib`). Это упрощённый SVDQuant (без GPTQ и подбора поворотов, как в deepcompressor), поэтому качество обязательно проверяем бенчмарком. Результат конвертации кэшируется в `/workspace/quant_cache`, повторный старт его просто загружает.
3. **В FastVideo уже есть свой NVFP4 для FFN-слоёв MiniMax H3** (через flashinfer). Он доступен как `QUANT_MODE=fp4` и работает без nunchaku, поэтому служит базовой точкой для сравнения.
4. **t2va-воркер сейчас, по всей видимости, работает с layerwise CPU-offload DiT.** В `gpu_worker.py` для t2va не задан `OffloadConfig`, а дефолт FastVideo — `dit_layerwise=True`, то есть веса DiT подкачиваются через PCIe на каждом шаге. Вероятно, так сделано, чтобы оба воркера поместились на одну GPU. По конфигу FastVideo линейные слои блоков H3 — около 17 млрд параметров, это примерно 35 ГБ в bf16 и примерно 10 ГБ в 4 битах. В режиме `svdq` упакованные веса всегда лежат на GPU, а `T2VA_DIT_OFFLOAD=0` отключает offload полностью. Реальные цифры покажет бенчмарк.
5. Ускорение получится меньше «×3 на GEMM» из статей про nunchaku: внимание (VSA), VAE-декод и текст-энкодер остаются в 16 битах. Экономия денег = (время генерации bf16 / время генерации quant) плюс возможность держать обе модели на одной GPU без offload. Обе величины меряет `bench_quant.py`.

## Режимы `QUANT_MODE`

| режим | что квантуется | зависимости | статус |
|---|---|---|---|
| `bf16` (по умолчанию) | ничего, production как есть | — | без изменений |
| `fp4` | FFN (`fc_in`, `fc_out`) всех блоков, нативный NVFP4 FastVideo, RTN | flashinfer (уже есть в FastVideo) | готов к замеру |
| `svdq` | линейные слои блоков из `SVDQ_LAYERS`, SVDQuant W4A4 на ядрах nunchaku | nunchaku (`scripts/install_quant.sh`) | готов к замеру |
| `calib` | ничего; bf16-прогон с записью max\|x\| по входным каналам для сглаживания | — | готов |

Переменные окружения (читаются в процессе воркера при старте):

| переменная | по умолчанию | смысл |
|---|---|---|
| `QUANT_MODE` | `bf16` | см. выше |
| `SVDQ_PRECISION` | `auto` | `auto` → `nvfp4` на sm_120/121, иначе `int4` |
| `SVDQ_RANK` | `32` | ранг 16-битной low-rank ветки (кратен 16) |
| `SVDQ_LAYERS` | `all` | `all` (attn q/k/v/out + FFN), `ffn`, `attn` или своё регулярное выражение по префиксу слоя |
| `SVDQ_SMOOTH_ALPHA` | `0.5` | сила SmoothQuant-миграции |
| `QUANT_CACHE_DIR` | `/workspace/quant_cache` | где лежат кэш упакованных весов и калибровка |
| `SVDQ_CALIB_PATH` | `<cache>/<модель>-calib.pt`, если файл есть | статистика активаций; пустое значение отключает сглаживание |
| `SVDQ_CACHE_PATH` | `<cache>/svdq-<модель>-<точность>-r<ранг>-<слои>-<smooth>.safetensors` | пустое значение отключает кэш |
| `T2VA_DIT_OFFLOAD` | `1` | `0` — держать DiT t2va целиком на GPU |

Пример: `QUANT_MODE=svdq GPU_WORKER_MODE=t2va /workspace/venv_max/bin/python gpu_worker.py`. В лог при старте пишется строка `quant: ...`, в ответ `/health` и `/generate` добавлено поле `quant_mode`.

## Как это устроено

- `tovi_quant/quant_math.py` — математика на чистом torch: сглаживание, SVD-разложение, NVFP4/INT4-квантизация, fake-quant модель ядра. Покрыта тестами на CPU.
- `tovi_quant/nunchaku_backend.py` — упаковка в layout ядер nunchaku и вызов `quantize_w4a4_act_fuse_lora` + `gemm_w4a4`, тех же функций, что вызывает `SVDQW4A4Linear` из nunchaku. Упаковка побитово сверена с эталонным конвертером deepcompressor (`convert_to_nunchaku_w4x4y16_linear_weight`) для nvfp4 и int4. Корневой `__init__` nunchaku не импортируется, чтобы его зависимости diffusers/transformers не конфликтовали с FastVideo.
- `tovi_quant/fastvideo_svdq.py` — регистрирует в FastVideo методы квантизации `svdq` и `svdq_calib` через штатный `register_quantization_config`. Конфиг передаётся в spawn-воркер FastVideo через pickle. Конвертация выполняется в хуке `fsdp_load._maybe_quantize_model`, сразу после загрузки bf16-весов на GPU и до включения layerwise offload. bf16-веса квантованных слоёв освобождаются.

Проверено на FastVideo `0cc41a2` (2026-10-02) и nunchaku `302e0e9`. Если версия FastVideo на поде старше, `doctor` сразу скажет, чего не хватает.

## Проверка на GPU: что запустить

Каждый шаг — отдельное включение пода. Все команды выполняются из корня этого репозитория на поде, в venv воркера. **Во время шагов 2–3 production-воркеры должны быть остановлены**: память меряется по всей GPU, и им не хватит VRAM.

**Шаг 1 — установка и проверка ядер (~20–30 мин, модели не загружаются).**
```bash
git clone -b claude/vibrant-edison-3gywd7 https://github.com/kalmenovruslan17-source/tovi-ai-quant /workspace/tovi-ai-quant
cd /workspace/tovi-ai-quant
PYTHON_BIN=/workspace/venv_max/bin/python bash scripts/install_quant.sh
/workspace/venv_max/bin/python -m tovi_quant.doctor --json /workspace/quant_doctor.json
```
Пришлите вывод `doctor` (или `quant_doctor.json`). В нём будут: версии, ошибка SVDQ и FP4 относительно bf16 на реальных размерах слоёв H3, время GEMM в каждом режиме и `kernel_vs_model` — проверка, что упаковка совпадает с установленными ядрами (должно быть порядка 1e-2 или меньше).

**Шаг 2 — сквозное сравнение (~1–1.5 ч).**
```bash
/workspace/venv_max/bin/python scripts/bench_quant.py --worker t2va --modes bf16,fp4,svdq \
    --gpu-price-per-hour <цена пода $/ч> --out /workspace/quant_bench/t2va
```
Пришлите `report.md` и посмотрите глазами `side_by_side/*.mp4` (слева bf16, справа квантованная версия, звук квантованной). Первый старт `svdq` дольше из-за конвертации, дальше веса берутся из кэша. Чтобы проверить вариант без offload, добавьте `--env T2VA_DIT_OFFLOAD=0`.

**Шаг 3 (если на шаге 2 качество `svdq` хуже допустимого) — калибровка и повтор.**
```bash
/workspace/venv_max/bin/python scripts/bench_quant.py --worker t2va --modes calib \
    --prompts my_calib_prompts.txt --out /workspace/quant_bench/calib   # 10–20 разнообразных промптов
/workspace/venv_max/bin/python scripts/bench_quant.py --worker t2va --modes bf16,svdq --out /workspace/quant_bench/t2va_smooth
```
После калибровочного прогона `svdq` автоматически подхватывает `/workspace/quant_cache/fasth3-calib.pt` и строит отдельный кэш. Для ref2va то же самое, только с `--worker ref2va --reference-image <путь к фото>`.

## Выкатка

После бенчмарка остаётся задать `QUANT_MODE` (и, если нужно, `T2VA_DIT_OFFLOAD=0`) в окружении процесса воркера. Откат — `QUANT_MODE=bf16` или просто убрать переменную. `main.py` менять не нужно.

## Ограничения

- `svdq` не поддерживает LoRA (`lora_path`), FSDP-инференс и пока не проверялся с `torch.compile`; в production ничего из этого не используется.
- q/k/v квантуются тремя отдельными GEMM, и вход для каждой квантуется заново. Слияние в один QKV-GEMM — следующая оптимизация, если профиль покажет, что это заметно.
- Конвертация на старте добавляет несколько минут только при первом запуске с новыми настройками, дальше работает кэш. Кэш инвалидируется при смене ранга, точности, набора слоёв или калибровки.

## Тесты

```bash
pip install torch safetensors numpy pytest
pytest            # математика; интеграционные тесты FastVideo пропускаются, если fastvideo не установлен
```
С `PYTHONPATH=<FastVideo>:<nunchaku>` дополнительно проверяется склейка с реальным FastVideo: регистрация конфига, хук загрузчика, конвертация, кэш, калибровка. CUDA-ядра в этих тестах заменены fake-quant моделью, а на GPU их проверяет `doctor`.
