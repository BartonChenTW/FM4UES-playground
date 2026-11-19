# Module 3: Multimodal Foundation Models

**Goal:** Understand how encoders and decoders work in (multimodal) foundation models, by building a tiny one yourself, looking inside two well-known multimodal models, and comparing their operational scale (890 parameters vs. 151M and 224M) and energy cost.

## Notebooks

1. **[`01_encoder_decoder.ipynb`](01_encoder_decoder.ipynb)** — foundational walkthrough (`890` params vs. `151 M` and `224 M`):
   - **Part A:** Tiny autoencoder on Swiss daily load profiles (`24 -> 2 -> 24`, `3.5%` unseen RMSE).
   - **Part B:** [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) zero-shot (`46.2%`) vs. linear probe (`87.6%`) on EuroSAT, plus UES-class F1 (*Residential*, *Industrial*, *Highway*).
   - **Part C:** [BLIP base captioning](https://huggingface.co/Salesforce/blip-image-captioning-base) failure audit outside its everyday-photo domain.
2. **[`02_geospatial_energy_fms.ipynb`](02_geospatial_energy_fms.ipynb)** — geospatial & EO backbones (install `pip install -r requirements-extras/multimodal.txt`):
   - Compares **CLIP ViT-B/32**, **SigLIP 2 Base** (`google/siglip2-base-patch16-224`), **RemoteCLIP ViT-B/32** (`chendelong/RemoteCLIP`), and **SatlasPretrain Swin-V2-Base** (`torchgeo/swin_v2_b_sentinel2_rgb_satlas`, `[0, 1]` Sentinel-2 RGB normalization) on EuroSAT (zero-shot, 50/50 linear probe, `5/10/50/150`-shot curves) and **20 cm IGN BD ORTHO rooftop PV detection** (`Cyrille37/solar-panels-IGN-bdortho`, `286` train / `50` val).
3. **[`03_vlm_and_multimodal_fusion.ipynb`](03_vlm_and_multimodal_fusion.ipynb)** — modern local VLMs & cross-modal fusion (also requires `requirements-extras/multimodal.txt`):
   - **Part A:** Audits **BLIP base** vs. **SmolVLM-500M-Instruct** (`fp16`, `1.50 GB` VRAM) vs. **Qwen2.5-VL-3B-Instruct** (`fp16`, `7.72 GB` VRAM) on structured JSON rooftop PV inspection (`20` IGN 20 cm roofs) and synthetic EPC/utility bill extraction (`12` cards = `60` exact fields).
   - **Part B:** Tests strictly leakage-free 5-fold CV cross-modal fusion of **zoom-19 Esri Chicago aerial footprints** (`~0.22 m/px`, `600` buildings) + Module 2 tabular features across `Ridge`, `LightGBM`, and `TabPFN v2`.

Every step prints tensor shapes or structured tables and is profiled for VRAM, runtime and GPU energy.

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

Structured machine-readable results are saved to [`benchmarks/results/module3_multimodal_01_encoder_decoder.json`](../benchmarks/results/module3_multimodal_01_encoder_decoder.json), [`benchmarks/results/module3_multimodal_02_geospatial_energy_fms.json`](../benchmarks/results/module3_multimodal_02_geospatial_energy_fms.json), [`benchmarks/results/module3_multimodal_03_vlm_and_multimodal_fusion.json`](../benchmarks/results/module3_multimodal_03_vlm_and_multimodal_fusion.json), and `data/cache/module3_multimodal_runtime.csv`.

## Geospatial & EO backbones (`02_geospatial_energy_fms.ipynb`)

| Backbone | Params (M) | Peak VRAM (GB) | EuroSAT ZS Acc | EuroSAT Probe Acc | 5-shot / 50-shot | Residential / Industrial / Highway F1 | IGN PV Val AUC | IGN PV Val F1 | Encode mJ/img |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CLIP ViT-B/32 | 151.3 | 1.01 | 46.3% | 87.5% | 63.9% / 82.6% | 0.927 / 0.939 / 0.846 | 0.949 | 0.870 | 303.1 |
| SigLIP 2 Base (16/224) | 375.2 | 2.52 | 46.1% | 89.3% | 70.6% / 84.8% | 0.921 / 0.943 / 0.845 | **0.962** | 0.875 | 949.8 |
| RemoteCLIP ViT-B/32 | 151.3 | 0.94 | 32.9% | 92.8% | 78.3% / 89.3% | 0.974 / **0.983** / 0.901 | 0.918 | 0.857 | 367.8 |
| **SatlasPretrain Swin-V2-B** | 86.9 | 1.63 | – | **97.1%** | **87.7% / 94.4%** | **0.982** / 0.979 / **0.951** | 0.941 | **0.880** | 1,362.3 |

Run energy (`02_geospatial_energy_fms.ipynb`): **4.9153 Wh** NVML GPU (`17,695.2 J`, `91.9 s`) / **9.2050 Wh** CodeCarbon offline (`137.6 s`). Notice `[0, 1]` Sentinel-2 RGB scaling is essential for SatlasPretrain (`94.0%` vs `75.2%` with ImageNet normalization in 250-tile smoke checks).

## Local VLM audit & aerial–tabular fusion (`03_vlm_and_multimodal_fusion.ipynb`)

### Part A: Rooftop PV inspection (20 IGN 20 cm roofs) & EPC bill OCR (12 cards = 60 fields)

| VLM (Precision) | Params (M) | Peak VRAM (GB) | PV Acc | PV F1 | Bare-Roof FP Halluc. | A2 JSON Valid | EPC Field Exact-Match | Energy (J/roof \| J/card) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BLIP base (`fp32`) | 224.0 | 0.97 | 70.0% | 0.571 | 0.0% | 0.0% | 0.0% | 14.3 / 12.2 |
| SmolVLM-500M-Instruct (`fp16`) | 507.5 | 1.50 | 55.0% | 0.690 | **90.0%** | 91.7% | 76.7% | 198.9 / 289.8 |
| **Qwen2.5-VL-3B-Instruct** (`fp16`) | 3,754.6 | 7.72 | **80.0%** | **0.750** | **0.0%** | **100.0%** | **100.0%** | 192.8 / 474.2 |

Notice the ~300–500× energy and accuracy gap: a frozen EO linear probe hits **0.880 F1** on IGN 20 cm PV in ~0.3–1.4 J/img, whereas generative VLMs spend **193–199 J/roof** (and SmolVLM hallucinates panels on 90% of bare roofs), making VLMs suited for structured document/bill OCR (`100%` exact across 60 fields with `Qwen2.5-VL-3B`) rather than high-throughput binary roof screening.

### Part B: Cross-modal fusion (600 Chicago buildings, Esri zoom-19 tiles + Module 2 features, 5-fold CV)

| Modality | Ridge MAE / R² | LightGBM MAE / R² | TabPFN v2 MAE / R² |
| :--- | ---: | ---: | ---: |
| Tabular only | 28.86 / 0.051 | 29.83 / 0.081 | **26.96 / 0.182** |
| Aerial only (`SigLIP 2` PCA-16) | 29.88 / 0.017 | 32.52 / -0.073 | 29.41 / 0.012 |
| Aerial only (`RemoteCLIP` PCA-16) | 30.44 / 0.019 | 33.02 / -0.094 | 29.40 / 0.019 |
| Fused (`Tab + SigLIP 2`) | 28.85 / **0.094** | 30.27 / 0.039 | 27.11 / 0.173 |
| Fused (`Tab + RemoteCLIP`) | 28.81 / 0.056 | 30.45 / 0.042 | 27.22 / **0.182** |

Overhead imagery adds weak neighborhood/massing signal only for linear Ridge (`R² 0.051 -> 0.094`), while strong tabular learners (`TabPFN v2`) gain **zero lift** — confirming that EUI variance above the tabular ceiling reflects indoor HVAC, schedules, and occupancy invisible from above. Notebook 03 run energy: **5.9677 Wh** NVML GPU (`21,483.6 J`, `230.7 s`) / **14.9507 Wh** CodeCarbon offline (`285.5 s`).

