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
├── Makefile              <- Makefile with convenience commands like `make data` or `make train`
│
├── README.md             <- The top-level README for developers using this project.
│
├── docs                  <- A default mkdocs project; see www.mkdocs.org for details
│
├── notebooks             <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── scripts               <- Slurm scripts for running code on HPC
│
├── tests                 <- Experiments to make sure code is working
│
└── src                   <- Source code for use in this project.
    │
    ├── __init__.py                   <- Makes staged a Python module
    │
    ├── main.py                       <- CLI for training STAGED
    │
    ├── inference.py                  <- Access point for making predictions of cell dynamics
    │
    ├── data                          <- A folder for data-related scripts
    │   └── data_processor.py         <- Defines DataProcessor class, which turns raw cell position and gene expression data
    │
    ├── evaluation                    <- Currently empty
    │
    ├── models                        <- Directory for the core STAGED scripts                
    │   └── staged.py                 <- Where the actual STAGED object is defined
    │
    ├── trainer                       <- Contains scripts for training STAGED based on real-world or simulated data
    │
    └── utils                         <- Contains various utility scripts
        ├── graph_constructor.py      <- A script to manage and update the GRN of each cell over time
        └── graph_data_processor.py   <- Turns individual cell graphs into torch-friendly batched format

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
