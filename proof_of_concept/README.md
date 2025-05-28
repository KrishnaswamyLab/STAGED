# STAGED: Spatiotemporal Analysis of Gene Expression Dynamics

A proof-of-concept implementation of the STAGED algorithm for predicting gene expression trajectories using spatial information and gene regulatory networks (GRNs), with Neural ODE integration.

## Features

- **Multiple Prediction Modes**: Traditional next-step, k-step ahead, and Neural ODE prediction
- **Modular Data Generation**: Factory pattern for different synthetic data types
- **Clean Training Interface**: Command-line interface with comprehensive parameter control
- **Visualization Suite**: Automatic plotting and result saving

## Quick Start

```bash
# Neural ODE training with oscillatory data
python main.py --mode ode --data oscillatory --iterations 50

# Next-step prediction with hex grid data  
python main.py --mode one_step --data hex_grid --iterations 100

# K-step prediction with sinusoidal data
python main.py --mode k_step --data sinusoidal --k_steps 3 --iterations 75
```

## Project Structure

```
proof_of_concept/
├── main.py                    # Main training interface
├── models/
│   ├── staged.py             # Core STAGED model
│   ├── training.py           # Training logic and configurations
│   └── inference_processor.py # Inference and Neural ODE integration
├── utils/
│   ├── data_factory.py       # Data generation for different types
│   ├── visualization.py      # Plotting and result saving
│   └── graph_constructor.py  # Graph construction utilities
├── tests/                    # Comprehensive test suite
│   ├── test_training_ode.py  # Neural ODE training tests
│   ├── test_training_next_step.py # Traditional training tests
│   └── temporal_data_generator.py # Test data generators
└── requirements.txt          # Dependencies
```

## Neural ODE Integration

STAGED now supports Neural ODE prediction where the model learns derivatives that are integrated over time using `torchdiffeq.odeint`:

- **Continuous predictions**: Evaluate at any time points
- **Interpolated history**: Access lagged values at non-discrete times
- **Multiple ODE methods**: Euler, RK4, DoPri5, Adams

## Available Data Types

1. **`oscillatory`** - Realistic gene expression with regulatory interactions
2. **`damped_oscillator`** - Physics-based harmonic oscillators for ODE testing  
3. **`sinusoidal`** - Simple sinusoidal patterns for quick testing
4. **`hex_grid`** - Hexagonal spatial arrangement test data
5. **`square_grid`** - Square grid spatial arrangement test data

## Training Modes

### 1. Neural ODE (`--mode ode`)
```bash
python main.py --mode ode --data oscillatory --eval_times "0.0,0.5,1.0,1.5"
```

### 2. Next-Step (`--mode one_step`)
```bash
python main.py --mode one_step --data hex_grid --iterations 100
```

### 3. K-Step (`--mode k_step`)
```bash
python main.py --mode k_step --data sinusoidal --k_steps 5
```

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Key requirements:
- PyTorch 1.8+
- PyTorch Geometric
- torchdiffeq (for Neural ODE)
- NetworkX
- NumPy, Matplotlib

## Testing

Run the test suite:

```bash
# Test Neural ODE functionality
python -m pytest tests/test_training_ode.py -v

# Test traditional training
python -m pytest tests/test_training_next_step.py -v

# Run all tests
python -m pytest tests/ -v
```

---

# Training Interface Documentation

## Parameter Reference

### Required Parameters
- `--mode`: Prediction mode (ode, one_step, k_step)
- `--data`: Data type to use

### Training Parameters
- `--iterations`: Number of training iterations (default: 50)
- `--lr`: Learning rate (default: 0.01)
- `--batch_size`: Batch size (default: 4)

### Model Architecture
- `--hidden_dim`: Hidden dimension size (default: 64)
- `--gat_layers`: Number of GAT layers (default: 1)
- `--mlp_layers`: Number of MLP layers (default: 3)
- `--dropout`: Dropout rate (default: 0.1)

### Time Lag Parameters (Delta)
- `--delta_gl`: Gene-ligand time lag (default: 1)
- `--delta_lr`: Ligand-receptor time lag (default: 2)
- `--delta_rg`: Receptor-gene time lag (default: 1)
- `--delta_gg`: Gene-gene time lag (default: 0, automatically set to 0 for ODE mode)

### Mode-Specific Parameters

**For ODE mode:**
- `--eval_times`: Comma-separated evaluation times (e.g., "0.0,0.5,1.0")
- `--ode_method`: Integration method (euler, rk4, dopri5, adams)

**For K-step mode:**
- `--k_steps`: Number of steps to predict ahead

### Output Options
- `--save_dir`: Directory to save results and plots (optional)
- `--device`: Device to use (auto, cpu, cuda)
- `--seed`: Random seed for reproducibility (default: 42)

## Complete Example

```bash
python main.py \
  --mode ode \
  --data oscillatory \
  --iterations 100 \
  --lr 0.005 \
  --batch_size 8 \
  --hidden_dim 128 \
  --gat_layers 2 \
  --mlp_layers 4 \
  --dropout 0.15 \
  --delta_gl 1 \
  --delta_lr 2 \
  --delta_rg 1 \
  --eval_times "0.0,0.2,0.4,0.6,0.8,1.0" \
  --ode_method dopri5 \
  --save_dir results/my_experiment \
  --device auto \
  --seed 123
```

## Output

### Console Output
- Training configuration summary
- Progress during training
- Final results (initial loss, final loss, loss reduction)

### Saved Results (if `--save_dir` specified)
- `training_results.json`: Complete configuration and results
- `model.pth`: Trained model state dict
- `training_plot.png`: Loss curves (linear and log scale)

## Example Workflows

### Quick Testing
```bash
# Quick ODE test (10 iterations)
python main.py --mode ode --data sinusoidal --iterations 10 --eval_times "0.0,1.0"

# Quick next-step test
python main.py --mode one_step --data hex_grid --iterations 10
```

### Mode Comparison
```bash
# Test same data with different modes
python main.py --mode one_step --data oscillatory --iterations 50
python main.py --mode ode --data oscillatory --iterations 50 --eval_times "0.0,0.4,0.8"
python main.py --mode k_step --data oscillatory --iterations 50 --k_steps 3
```

### Production Training
```bash
# Full training with result saving
python main.py \
  --mode ode \
  --data oscillatory \
  --iterations 200 \
  --lr 0.003 \
  --hidden_dim 256 \
  --gat_layers 2 \
  --eval_times "0.0,0.1,0.2,0.3,0.4,0.5" \
  --save_dir results/production_run
```

## Tips

1. **For ODE mode**: Start with fewer evaluation times and simpler data
2. **For debugging**: Use `--iterations 5` and `--data sinusoidal`
3. **For production**: Increase `--hidden_dim`, `--gat_layers`, and `--iterations`
4. **For reproducibility**: Always specify `--seed`
5. **For GPU training**: Use `--device cuda` (if available)

---

## Model Architecture

STAGED uses Graph Attention Networks (GAT) to model:
- Cell-type-specific gene regulatory networks
- Ligand-receptor interactions between cells  
- Spatial proximity effects
- Temporal dynamics with configurable time lags

The Neural ODE extension allows continuous-time prediction by learning expression derivatives rather than discrete next-step values.

## Citation

If you use this code in your research, please cite:

```
@article{STAGED2023,
  title={STAGED: Spatiotemporal Analysis of Gene Expression Dynamics},
  author={Your Name},
  journal={Journal Name},
  year={2023}
}
```

## License

MIT License - see LICENSE file for details. 