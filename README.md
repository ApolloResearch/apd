# SPD - Stochastic Parameter Decomposition
Code used in the paper [Stochastic Parameter Decomposition (forthcoming)](TODO)

Weights and Bias [report](https://wandb.ai/apollo-interp/spd-tms/reports/SPD-paper-report--VmlldzoxMzE0NzE1OA?accessToken=h7mgfw2k37gfyssr65fn2960w9yz53jge7uz9oxugbbl28di4a0xep5kpgbf4z8g) accompanying the paper.

## Installation
From the root of the repository, run one of

```bash
make install-dev  # To install the package, dev requirements and pre-commit hooks
make install  # To just install the package (runs `pip install -e .`)
```

## Usage
Place your wandb information in a .env file. You can use the .env.example file as an example.

The repository consists of several `experiments`, each of which containing scripts to run SPD,
analyse results, and optionally a train a target model:
- `spd/experiments/tms` - Toy model of superposition
- `spd/experiments/resid_mlp` - Toy model of compressed computation and toy model of distributed
  representations
- `spd/experiments/lm` - Language model loaded from huggingface.

### Run DPD
SPD can be run by executing any of the `*_decomposition.py` scripts defined in the experiment
subdirectories. A config file is required for each experiment, which can be found in the same
directory. For example:
```bash
python spd/experiments/tms/tms_decomposition.py spd/experiments/tms/tms_config.yaml
```
will run SPD on TMS with the config file `tms_config.yaml` (which is the main config file used
for the TMS experiments in the paper).

Wandb sweep files are also provided in the experiment subdirectories, and can be run with e.g.:
```bash
wandb sweep spd/experiments/tms/tms_sweep_config.yaml
```

All experiments call the `optimize` function in `spd/run_spd.py`, which contains the main SPD logic.

## Development

Suggested extensions and settings for VSCode/Cursor are provided in `.vscode/`. To use the suggested
settings, copy `.vscode/settings-example.json` to `.vscode/settings.json`.

There are various `make` commands that may be helpful

```bash
make check  # Run pre-commit on all files (i.e. pyright, ruff linter, and ruff formatter)
make type  # Run pyright on all files
make format  # Run ruff linter and formatter on all files
make test  # Run tests that aren't marked `slow`
make test-all  # Run all tests
```