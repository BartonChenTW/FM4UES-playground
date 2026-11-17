<!-- Note: Replace the placeholder "YOUR_USER" with your GitHub username in the Colab badge URLs below and in the notebook setup cells. -->

# FM4UES-playground

A Colab-first learning playground for exploring **foundation models (FMs)** in the **urban energy systems (UES)** domain. Designed for learners in UES who are new to foundation models and Python, each module introduces core concepts step by step with simple, well-commented notebooks that run directly in Google Colab with minimal setup.

## Modules

| Module | Topic | Notebook | Open in Colab |
| :--- | :--- | :--- | :--- |
| [Module 1: Time-Series](module1_timeseries/README.md) | Demand and weather forecasting with time-series FMs | [01_demand_forecasting.ipynb](module1_timeseries/01_demand_forecasting.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BartonChenTW/FM4UES-playground/blob/main/module1_timeseries/01_demand_forecasting.ipynb) |
| [Module 2: Tabular](module2_tabular/README.md) | Tabular FMs on public building energy data | [01_tabular_intro.ipynb](module2_tabular/01_tabular_intro.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BartonChenTW/FM4UES-playground/blob/main/module2_tabular/01_tabular_intro.ipynb) |
| [Module 3: Multimodal](module3_multimodal/README.md) | How encoders and decoders work in multimodal FMs | [01_encoder_decoder.ipynb](module3_multimodal/01_encoder_decoder.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BartonChenTW/FM4UES-playground/blob/main/module3_multimodal/01_encoder_decoder.ipynb) |

## How to run

1. Click an **Open in Colab** badge in the table above to open the notebook in Google Colab.
2. Enable a GPU (`Runtime` -> `Change runtime type` -> `GPU`) **only if the notebook states that it needs one**; most introductory cells work on CPU.
3. Run the **setup cell** at the top of the notebook first to clone the repository (when in Colab) and install dependencies from `requirements.txt`.

## Contributing

Contributions, bug fixes, and beginner-friendly improvements are welcome! Please open an issue to discuss major changes, keep code simple and well commented for UES learners, and avoid committing large datasets or notebook outputs.
