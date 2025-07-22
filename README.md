# STAGED

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Spatio Temporal Agent-Based Graph Evolution Dynamics (STAGED)

## 🛠️ Installation

This project uses [uv](https://docs.astral.sh/uv/) to manage dependencies. To set up the project locally:

1. **Install dependencies**:

    ```bash
    uv sync  # Creates a virtual environment and installs dependencies
    ```

2. **Activate the virtual environment**:

    ```bash
    source .venv/bin/activate
    ```
---
## Example of a main run

    ```bash
    python src/main.py --mode train --config src/config/ode_config.yaml
    ```

## Inference run

```bash
python3 src/inference.py --checkpoint_path results/checkpoints/checkpoints_20250722_193041/best_model.pt --config src/config/ode_config.yaml
```

## To use jupyter notebook the following command might be necessary:

```bash
uv run python -m ipykernel install --user --name staged --display-name "Python (staged)"
``` 

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         staged and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── staged   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes staged a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

Open a Jupyter notebook in the notebooks/ folder. You can start by creating a new notebook and doing some exploratory data analysis. 

The naming scheme looks like this:

0.01-pjb-data-source-1.ipynb

0.01 - Helps leep work in chronological order. The structure is PHASE.NOTEBOOK. NOTEBOOK is just the Nth notebook in that phase to be created. For phases of the project, we generally use a scheme like the following, but you are welcome to design your own conventions:

0 - Data exploration - often just for exploratory work
1 - Data cleaning and feature creation - often writes data to data/processed or data/interim
2 - Visualizations - often writes publication-ready viz to reports
3 - Modeling - training machine learning models
4 - Publication - Notebooks that get turned directly into reports

pjb - Your initials; this is helpful for knowing who created the notebook and prevents collisions from people working in the same notebook.

data-source-1 - A description of what the notebook cover
