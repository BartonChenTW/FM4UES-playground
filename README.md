# FM4UES-playground [![Version v0.2.0](https://img.shields.io/badge/version-v0.2.0-0b6e69.svg)](https://github.com/BartonChenTW/FM4UES-playground/releases/tag/v0.2.0)

**Version:** [`v0.2.0`](https://github.com/BartonChenTW/FM4UES-playground/releases/tag/v0.2.0) · [`CHANGELOG.md`](CHANGELOG.md) · **Review status:** pending author pass in [`docs/to_review.md`](docs/to_review.md)

A hands-on learning playground for exploring **foundation models (FMs)** in the **urban energy systems (UES)** domain. Designed for learners in UES who are new to foundation models and Python, each module introduces core concepts step by step with simple, well-commented notebooks that run on a local workstation, preferably with an NVIDIA GPU.

**Website:** a guided walkthrough of all modules for energy system modellers, with figures and a practical playbook, is at
**https://bartonchentw.github.io/FM4UES-playground/** (source in [`docs/`](docs/)).

## Modules

| Module | Topic | Notebooks | Status |
| :--- | :--- | :--- | :--- |
| [Module 1: Time-Series](module1_timeseries/README.md) | Day-ahead demand & solar PV forecasting (Chronos-Bolt, Chronos-2, TimesFM 2.5, TiRex vs. LightGBM/Ridge) | [01_demand_forecasting.ipynb](module1_timeseries/01_demand_forecasting.ipynb), [02_tsfm_benchmark_energy.ipynb](module1_timeseries/02_tsfm_benchmark_energy.ipynb) | Ready |
| [Module 2: Tabular](module2_tabular/README.md) | Building EUI across CV, temporal & spatial splits (TabPFN v2, TabICLv2 vs. LightGBM/CatBoost/XGBoost) | [01_tabular_intro.ipynb](module2_tabular/01_tabular_intro.ipynb), [02_tabular_fms_comparison.ipynb](module2_tabular/02_tabular_fms_comparison.ipynb) | Ready |
| [Module 3: Multimodal](module3_multimodal/README.md) | Autoencoder, geospatial EO backbones (CLIP, SigLIP 2, RemoteCLIP, SatlasPretrain), local VLMs (SmolVLM-500M, Qwen2.5-VL-3B) & aerial–tabular fusion | [01_encoder_decoder.ipynb](module3_multimodal/01_encoder_decoder.ipynb), [02_geospatial_energy_fms.ipynb](module3_multimodal/02_geospatial_energy_fms.ipynb), [03_vlm_and_multimodal_fusion.ipynb](module3_multimodal/03_vlm_and_multimodal_fusion.ipynb) | Ready |
| [Module 4: GridFM / Power Grids](module4_grid/README.md) | Physics-aware GENCO graph FMs on IEEE 30-bus & 118-bus AC Power Flow and AC-OPF contingencies | [01_gridfm_genco.ipynb](module4_grid/01_gridfm_genco.ipynb) | Ready |

Each core `01_*.ipynb` notebook runs in about 1–2 minutes on a GPU once data and models are cached (measured runtimes, memory, and energy are in each module's README, exported to [`benchmarks/results/`](benchmarks/results/) via [`fm4ues/`](fm4ues/), and aggregated into [`benchmarks/`](benchmarks/) CSVs and the [Benchmarks Hub](docs/benchmarks.html) via [`scripts/build_benchmark_summary.py`](scripts/build_benchmark_summary.py)). Use [`scripts/run_notebooks.py`](scripts/run_notebooks.py) (`--audit-only`, `--check-urls`, `--all`) to reproduce or verify all 8 notebooks. Optional dependencies for advanced benchmark notebooks and Module 4 live in [`requirements-extras/`](requirements-extras/) (`tsfm.txt`, `tabular.txt`, `gridfm.txt`, `multimodal.txt`).

## How to run (local workstation)

**Requirements:** Windows or Linux, [Miniconda](https://docs.conda.io/en/latest/miniconda.html), and an internet connection on first run (data and model weights are downloaded and cached). An NVIDIA GPU with a recent driver (version 570 or newer) is recommended. Everything also runs on CPU, more slowly.

1. Create the conda environment (installs PyTorch with CUDA 12.8 and all packages, about 5 GB on disk):

   ```bash
   conda env create -f environment.yml
   conda activate fm4ues
   ```

2. Register the environment as a Jupyter kernel:

   ```bash
   python -m ipykernel install --user --name fm4ues --display-name "Python (fm4ues)"
   ```

3. Check that the GPU is visible (should print `True` and your GPU name):

   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

4. Open a notebook in VS Code or JupyterLab, select the **Python (fm4ues)** kernel and run all cells.

Downloaded data (a few MB, plus ~15–20 MB cached parquet partitions for Module 4 and ~10 MB zoom-19 aerial tiles for Module 3) is stored in `data/cache/` or `HF_HOME` (ignored by git). Model weights across all four modules (~5 GB for core notebooks; ~16 GB when including optional `requirements-extras/` models such as `Qwen2.5-VL-3B-Instruct`, `SigLIP 2 Base`, and `GENCO`) are stored in the Hugging Face cache (`~/.cache/huggingface`),
except TabPFN, which uses its own cache folder.

### Keeping everything on a data drive (optional)

If your system drive is small, put the environment and the model cache on another drive (example: `F:\Barton`):

```bash
conda env create -f environment.yml -p F:\Barton\envs\fm4ues
conda activate F:\Barton\envs\fm4ues
conda env config vars set HF_HOME=F:\Barton\cache\huggingface TABPFN_MODEL_CACHE_DIR=F:\Barton\cache\tabpfn   # model downloads go here
python -m ipykernel install --user --name fm4ues --display-name "Python (fm4ues)"
```

Jupyter starts the kernel without activating the conda environment, so also add the model caches to the kernel: open the `kernel.json` file in the folder printed by the last command and add
`"env": {"HF_HOME": "F:\\Barton\\cache\\huggingface", "TABPFN_MODEL_CACHE_DIR": "F:\\Barton\\cache\\tabpfn"}`.

## Contributing

Contributions, bug fixes, and beginner-friendly improvements are welcome! Please open an issue to discuss major changes, keep code simple and well commented for UES learners, and avoid committing large datasets.

**Notebook outputs:** finished example notebooks are committed **with their outputs** (plots, tables, runtime summary), so readers can see the results without running anything. Before committing, re-run the notebook top to bottom in a fresh kernel (e.g. `jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb`) so the outputs match the code, and make sure no local paths, tokens or progress-bar noise appear in them. Placeholder or work-in-progress notebooks are committed without outputs.
