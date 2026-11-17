# FM4UES-playground

A hands-on learning playground for exploring **foundation models (FMs)** in the **urban energy systems (UES)** domain. Designed for learners in UES who are new to foundation models and Python, each module introduces core concepts step by step with simple, well-commented notebooks that run on a local workstation, preferably with an NVIDIA GPU.

## Modules

| Module | Topic | Notebook | Status |
| :--- | :--- | :--- | :--- |
| [Module 1: Time-Series](module1_timeseries/README.md) | Day-ahead electricity demand forecasting with time-series FMs (Chronos-Bolt, Chronos-2) | [01_demand_forecasting.ipynb](module1_timeseries/01_demand_forecasting.ipynb) | Ready |
| [Module 2: Tabular](module2_tabular/README.md) | Tabular FMs on public building energy data | [01_tabular_intro.ipynb](module2_tabular/01_tabular_intro.ipynb) | Placeholder |
| [Module 3: Multimodal](module3_multimodal/README.md) | How encoders and decoders work in multimodal FMs | [01_encoder_decoder.ipynb](module3_multimodal/01_encoder_decoder.ipynb) | Placeholder |

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

Downloaded data is stored in `data/cache/` (ignored by git). Model weights (about 1.3 GB for Module 1) are stored in the Hugging Face cache (`~/.cache/huggingface`).

### Keeping everything on a data drive (optional)

If your system drive is small, put the environment and the model cache on another drive (example: `F:\Barton`):

```bash
conda env create -f environment.yml -p F:\Barton\envs\fm4ues
conda activate F:\Barton\envs\fm4ues
conda env config vars set HF_HOME=F:\Barton\cache\huggingface   # model downloads go here
python -m ipykernel install --user --name fm4ues --display-name "Python (fm4ues)"
```

Jupyter starts the kernel without activating the conda environment, so also add the model cache to the kernel: open the `kernel.json` file in the folder printed by the last command and add `"env": {"HF_HOME": "F:\\Barton\\cache\\huggingface"}`.

## Contributing

Contributions, bug fixes, and beginner-friendly improvements are welcome! Please open an issue to discuss major changes, keep code simple and well commented for UES learners, and avoid committing large datasets.

**Notebook outputs:** finished example notebooks are committed **with their outputs** (plots, tables, runtime summary), so readers can see the results without running anything. Before committing, re-run the notebook top to bottom in a fresh kernel (e.g. `jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb`) so the outputs match the code, and make sure no local paths, tokens or progress-bar noise appear in them. Placeholder or work-in-progress notebooks are committed without outputs.
