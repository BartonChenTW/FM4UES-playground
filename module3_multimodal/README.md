# Module 3: Multimodal Foundation Models

**Goal:** Understand how encoders and decoders work in (multimodal) foundation models, by building a tiny one yourself and by looking inside two well-known multimodal models.

## Notebook

[`01_encoder_decoder.ipynb`](01_encoder_decoder.ipynb) has three parts:

| Part | Model | What you see |
| :--- | :--- | :--- |
| A | Tiny autoencoder (890 parameters, trained in the notebook) on Swiss daily load profiles | An encoder compresses 24 hourly values to 2 numbers, a decoder rebuilds them. The 2-D embedding separates working days from weekends and winter from summer without being told. The decoder generates new profiles. |
| B | [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) on EuroSAT satellite images | Separate image and text encoders map into one shared 512-number space. Zero-shot land-use classification, the effect of prompt wording, and a linear probe on the frozen image embeddings. |
| C | [BLIP base captioning](https://huggingface.co/Salesforce/blip-image-captioning-base) | An image encoder plus a text decoder that writes a caption token by token, and how it fails outside its training domain. |

Every step prints the tensor shapes, so you can follow the data through each model.

## Data and models

| Item | Source | Licence | Disk |
| :--- | :--- | :--- | ---: |
| Swiss hourly load 2016–2020 (same file as Module 1) | [Open Power System Data – Time series](https://data.open-power-system-data.org/time_series/2020-10-06) | See the OPSD page; ENTSO-E data terms apply | 1.5 MB |
| EuroSAT RGB, test split (5,400 images, 64 x 64 px, 10 classes) | [EuroSAT](https://github.com/phelber/EuroSAT) via [blanchon/EuroSAT_RGB](https://huggingface.co/datasets/blanchon/EuroSAT_RGB) | MIT | 34 MB |
| CLIP ViT-B/32 | OpenAI | MIT | 1.2 GB |
| BLIP image captioning base | Salesforce | BSD-3-Clause | 1.9 GB |

All are downloaded automatically on the first run (load data to `data/cache/`, images and models to the Hugging Face cache).

## Results

| Experiment | Result |
| :--- | :--- |
| Autoencoder reconstruction error on 2019–2020 (unseen years) | 3.5% of the average load |
| CLIP zero-shot accuracy, 10 classes (random guess: 10%) | 46.2% with plain class names; 38.3–40.9% with "a satellite photo of ..." templates |
| Linear probe on frozen CLIP image embeddings | 87.6% |
| BLIP captions of satellite tiles | Mostly wrong ("a satellite image of the moon"). The decoder only knows its training domain (everyday photos). |

## Measured runtime

Measured on 2026-11-18, same workstation as Module 1 (NVIDIA Quadro RTX 6000, PyTorch 2.11.0+cu128, transformers 5.23.0), repeat run with data and models cached. All times are in seconds.

| Step | GPU |
| :--- | ---: |
| 1. Load electricity demand | 0.1 |
| 2. Train autoencoder (3,000 steps) | 11.4 |
| 3. Load EuroSAT images | 1.9 |
| 4. Load CLIP | 2.4 |
| 5. CLIP: encode 5,400 images (mostly image resizing on the CPU) | 21.3 |
| 6. CLIP: zero-shot classification (3 prompt templates) | 0.1 |
| 7. Linear probe | 0.4 |
| 8. Load BLIP | 2.2 |
| 9. BLIP: caption 10 images, twice | 0.8 |
| **Total of timed steps** | **40.6** |
| Whole notebook, start to finish | about 63 |

The first run also downloads about 3 GB (CLIP, BLIP, images), which adds a few minutes depending on your network. The notebook saves its timing table to `data/cache/module3_runtime.csv`.
