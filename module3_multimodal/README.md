# Module 3: Multimodal Foundation Models

**Goal:** Understand how encoders and decoders work in (multimodal) foundation models, by building a tiny one yourself, looking inside two well-known multimodal models, and comparing their operational scale (890 parameters vs. 151M and 224M) and energy cost.

## Notebook

[`01_encoder_decoder.ipynb`](01_encoder_decoder.ipynb) has three parts:

| Part | Model | What you see |
| :--- | :--- | :--- |
| A | Tiny autoencoder (890 parameters, trained in the notebook) on Swiss daily load profiles | An encoder compresses 24 hourly values to 2 numbers, a decoder rebuilds them. The 2-D embedding separates working days from weekends and winter from summer without being told. The decoder generates new profiles. |
| B | [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) on EuroSAT satellite images | Separate image and text encoders map into one shared 512-number space. Zero-shot land-use classification, prompt sensitivity, UES-class F1 scores (*Residential*, *Industrial*, *Highway*), and a linear probe on frozen image embeddings. |
| C | [BLIP base captioning](https://huggingface.co/Salesforce/blip-image-captioning-base) | An image encoder plus a text decoder that writes a caption token by token, and how it fails outside its training domain. |

Every step prints tensor shapes and is profiled for VRAM, runtime and GPU energy.

## Data and models

| Item | Source | Licence | Disk |
| :--- | :--- | :--- | ---: |
| Swiss hourly load 2016–2020 (same file as Module 1) | [Open Power System Data – Time series](https://data.open-power-system-data.org/time_series/2020-10-06) | See the OPSD page; ENTSO-E data terms apply | 1.5 MB |
| EuroSAT RGB, test split (5,400 images, 64 × 64 px, 10 classes) | [EuroSAT](https://github.com/phelber/EuroSAT) via [blanchon/EuroSAT_RGB](https://huggingface.co/datasets/blanchon/EuroSAT_RGB) | MIT | 34 MB |
| CLIP ViT-B/32 | OpenAI | MIT | 1,214 MB |
| BLIP image captioning base | Salesforce | BSD-3-Clause | 1,981 MB |

All are downloaded automatically on the first run (load data to `data/cache/`, images and models to the Hugging Face cache).

## Results

| Experiment | Overall Accuracy / RMSE | Macro F1 | Residential F1 | Industrial F1 | Highway F1 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Autoencoder reconstruction RMSE (unseen years 2019–2020) | 3.5% of daily mean | – | – | – | – |
| CLIP zero-shot, best template `"{}"` (random guess: 10%) | 46.2% | 0.395 | 0.697 | 0.521 | 0.510 |
| **CLIP linear probe** (logistic regression on frozen embeddings) | **87.6%** | **0.873** | **0.927** | **0.939** | **0.846** |
| BLIP captions of satellite tiles | Mostly wrong ("a satellite image of the moon"); decoder stays within its everyday-photo training domain. | – | – | – | – |

## Model cards

| Model | Params (M) | Precision | Disk (MB) | Peak VRAM (GB) | Min Hardware | Licence | Limits |
| :--- | ---: | :--- | ---: | ---: | :--- | :--- | :--- |
| Load autoencoder (tiny) | 0.00 (890) | `float32` | – | 0.02 | `CPU-ok` | MIT | 24-h shape only |
| CLIP ViT-B/32 | 151.3 | `float32` | 1,214.0 | 1.41 | `GPU <= 4 GB` | MIT | 77 text tokens, 224×224 px crop |
| BLIP captioning base | 224.0 | `float32` | 1,980.5 | 1.42 | `GPU <= 4 GB` | BSD-3-Clause | Everyday-photo domain, 384×384 px |

## Measured runtime & energy

Measured on 2026-11-19, same workstation as Module 1 (NVIDIA Quadro RTX 6000, PyTorch 2.11.0+cu128, `transformers 5.23.0`), repeat run with data and models cached.

| Step | Time (s) | Peak VRAM (GB) | Avg Power (W) | GPU Energy (J) | mJ / item |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1. Load electricity demand | 0.0 | 0.00 | 42.1 | 2.5 | – |
| 2. Train autoencoder (3,000 steps) | 11.1 | 0.02 | 37.9 | 422.1 | 140.7 / step |
| 3. Load EuroSAT images (5,400 tiles) | 2.0 | 0.02 | 25.4 | 51.9 | 9.6 / image |
| 4. Load CLIP | 2.0 | 0.62 | 25.8 | 53.3 | – |
| 5. CLIP: encode 5,400 images (CPU resize + GPU ViT) | 24.7 | 1.41 | 109.0 | 2,697.6 | 499.6 / image |
| 6. CLIP: zero-shot classification (3 templates) | 0.1 | 0.63 | 98.4 | 8.7 | 0.5 / pred |
| 7. Linear probe on frozen embeddings | 0.4 | 0.02 | 81.2 | 34.8 | 12.9 / image |
| 8. Load BLIP | 2.4 | 0.92 | 30.0 | 73.4 | – |
| 9. BLIP: caption 10 images (×2 prompts = 20 captions) | 1.1 | 1.42 | 113.1 | 119.0 | 5,952.1 / caption |
| **Total (profiled steps, NVML GPU)** | **44.0** | **1.42** | **78.7** | **3,463.3** | – |

Notice the energy contrast: encoding a satellite tile once with CLIP costs **~0.50 J/image** (and is heavily CPU-resize-bound at 22.6% GPU utilisation), whereas autoregressive text generation with BLIP costs **~5.95 J/caption** (~12× more GPU energy per image).

| Telemetry Scope | Duration (s) | GPU (Wh) | CPU + RAM (Wh, modeled) | Total (Wh) | CO₂ CH 30 g/kWh (mg) | CO₂ EU 230 g/kWh (mg) | CO₂ US 380 g/kWh (mg) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Profiled steps (GPU only, NVML 50 ms) | 44.0 | 0.962 | – | 0.962 | 28.9 | 221.3 | 365.6 |
| Whole notebook (CodeCarbon offline) | 53.9 | 0.924 | 1.185 | 2.110 | 63.3 | 485.2 | 801.7 |

Structured machine-readable results are saved to [`benchmarks/results/module3_multimodal_01_encoder_decoder.json`](../benchmarks/results/module3_multimodal_01_encoder_decoder.json) and `data/cache/module3_multimodal_runtime.csv`.
