# Gaussian Workflow Manager

A workflow generator for multi-stage Gaussian DFT calculations on HPC clusters. It reads a YAML configuration, generates all Gaussian input files (`.com`), SLURM submission scripts, and management utilities (submit, status, cancel, extract) — then you submit to your scheduler.

**It does not run calculations itself.** It creates everything you need so that `sbatch` does.

## Features

- **Conformer search** via CREST, RDKit, or OpenBabel (with automatic fallback)
- **Multi-stage DFT** with dependency management (ground state → excited states → redox)
- **SLURM integration** with job arrays, dependencies, and real-time status monitoring
- **Parallel conformer optimization** using multiprocessing (4–8× speedup with RDKit)
- **YAML-driven** — one file controls everything; swap configs for different workflows
- **Portable** — supports both `module load` and manual Gaussian path setup

## Installation

### Requirements

- Python ≥ 3.7
- Conda (Anaconda, Miniconda, or Miniforge) — recommended for managing dependencies
- Access to a SLURM cluster with Gaussian 16 installed
- `bc` — standard Linux calculator, used in generated scripts for energy unit conversion (pre-installed on virtually all Linux systems; verify with `which bc`)

### Step 1: Clone the repository

```bash
git clone https://github.com/lopezgroup-northeastern/gaussian-workflow-manager.git
cd gaussian-workflow-manager
```

### Step 2: Create a conda environment and install everything

The fastest way — installs all dependencies in one shot:

```bash
conda create -n gwm python=3.10 -y
conda activate gwm
conda install -c conda-forge rdkit openbabel xtb crest -y
pip install pyyaml rich
```

That's it. You now have everything needed for any workflow configuration (CREST, RDKit, or OpenBabel conformer search).

### Step 3: Verify the installation

```bash
conda activate gwm
python -c "import yaml; print('PyYAML:', yaml.__version__)"
python -c "from rdkit import Chem; print('RDKit: OK')"
python -c "import rich; print('Rich:', rich.__version__)"
obabel --version
crest --version
xtb --version
which bc
```

All six should print without errors. If any fails, see the detailed breakdown below.

### Minimal install (if you only need basic workflows)

If you don't need CREST conformer search and don't need RDKit:

```bash
conda create -n gwm python=3.10 -y
conda activate gwm
pip install pyyaml
```

This is enough to generate workflows with `conformer_search.enabled: false`. You add packages later as needed.

### Detailed dependency breakdown

| Package | Required? | What it does | Install with |
|---------|-----------|--------------|--------------|
| **PyYAML** | **Yes** | Reads/writes YAML configuration files | `pip install pyyaml` |
| **RDKit** | Recommended | Fast conformer generation (ETKDGv3 + UFF) with parallel optimization | `conda install -c conda-forge rdkit` |
| **OpenBabel** | Recommended | InChIKey generation (molecule naming) and fallback conformer generator | `conda install -c conda-forge openbabel` or `module load OpenBabel` on cluster |
| **CREST** | Optional | Thorough conformer search via xTB metadynamics — most accurate but slowest | `conda install -c conda-forge crest` |
| **xTB** | Optional | Semi-empirical quantum chemistry — required by CREST | `conda install -c conda-forge xtb` |
| **rich** | Optional | Colored progress bars, formatted tables, and pretty log output | `pip install rich` |
| **tqdm** | Optional | Simple progress bars (fallback if rich is not installed; you do not need both) | `pip install tqdm` |

**Not needed:** numpy, pandas, scipy, matplotlib — the script uses only the Python standard library plus the packages above.

**On the cluster (not in your conda env):**

| Tool | What it does | How to get it |
|------|-------------|---------------|
| **Gaussian 16** (`g16`) | Runs the DFT calculations | `module load gaussian/g16` or set `gaussian_root` in YAML |
| **formchk** | Generates formatted checkpoint files (`.fchk`) | Comes with Gaussian 16 |
| **SLURM** (`sbatch`, `squeue`, `scancel`) | Job scheduler — submits and manages jobs | Provided by your cluster |
| **bc** | Arithmetic in generated bash scripts (HOMO/LUMO → eV) | Pre-installed on Linux; check with `which bc` |

### Installing individual packages

If the all-in-one install fails for one package, install them separately:

```bash
conda activate gwm

# Required
pip install pyyaml

# RDKit (if conda-forge fails, try: mamba install -c conda-forge rdkit)
conda install -c conda-forge rdkit -y

# OpenBabel
conda install -c conda-forge openbabel -y
# OR if your cluster has it as a module:
module load OpenBabel/2.4.1    # check your cluster's module name

# CREST + xTB (both needed for CREST conformer search)
conda install -c conda-forge xtb crest -y
# OR install from source: https://github.com/crest-lab/crest
#                          https://github.com/grimme-lab/xtb

# Rich (pretty console output)
pip install rich
```

## Quick Start

### 1. Generate a template config

```bash
conda activate gwm
python flow.py --template
```

This creates `workflow_config.yaml` with all available options documented in comments. Open it and edit at minimum:

```yaml
slurm:
  gaussian_module: "gaussian/g16"   # Your cluster's Gaussian module name
  # gaussian_root: "/path/to/g16"   # Or set this if your cluster has no modules
  email: "you@university.edu"       # Your email for SLURM notifications
```

To find your cluster's Gaussian module name:

```bash
module avail gaussian
# Example output: gaussian/g16
```

### 2. Add your molecules

Place your input geometries as `.xyz` files in the directory specified by `project.xyz_dir` (default: `xyz/`):

```bash
mkdir xyz
cp your_molecules/*.xyz xyz/
```

### 3. Generate workflows

```bash
python flow.py workflow_config.yaml
```

This reads all `.xyz` files, generates InChIKeys, and creates a complete directory tree with Gaussian inputs, SLURM scripts, and management utilities.

### 4. Transfer to cluster and submit

If you generated workflows on your local machine, transfer to your cluster:

```bash
scp -r workflow_run/ user@cluster:/path/to/your/project/
```

Then on the cluster:

```bash
cd workflow_run
./workflow.sh submit          # Submit all molecules
./workflow.sh status          # Check progress
./workflow.sh extract         # Collect results as CSV
```

> **Tip:** Running `flow.py` directly on the cluster (recommended) skips the transfer step entirely.

## Gaussian Environment Setup

The generated SLURM scripts need to find Gaussian. Two options in the YAML config:

**Option A — Module system (recommended, portable):**

```yaml
slurm:
  gaussian_module: "gaussian/g16"
  gaussian_root: null
```

The generated scripts will contain `module load gaussian/g16`. To find your module name, run `module avail gaussian` on your cluster.

**Option B — Manual path:**

```yaml
slurm:
  gaussian_module: null
  gaussian_root: "/shared/software/gaussian/g16"
```

The generated scripts will `export g16root=...` and `source` the Gaussian profile directly.

## Example Configurations

Pre-built YAML configs are in [`examples/`](examples/). Copy one, edit the `slurm` section to match your cluster, and run:

```bash
cp examples/tddft_absorption.yaml workflow_config.yaml
vim workflow_config.yaml    # Edit gaussian_module, email, partition
python flow.py workflow_config.yaml
```

| File | What it does | Conformers | Calculations |
|------|-------------|------------|--------------|
| `verde_full_workflow.yaml` | Complete photophysics pipeline | CREST | S0, S1, T1, TD-DFT, cation, anion |
| `crest_conformers_only.yaml` | Conformer search + single opt | CREST | S0 vacuum only |
| `opt_freq_basic.yaml` | Ground state thermochemistry | None | S0 vacuum + S0 solvent (opt+freq) |
| `tddft_absorption.yaml` | UV-Vis absorption prediction | RDKit | S0 vacuum → S0 solvent → TD-DFT |
| `redox_potentials.yaml` | Electrochemical potentials | None | S0, cation, anion (vacuum + solvent) |
| `quick_test.yaml` | Verify cluster setup (PM6, 10 min) | None | S0 vacuum → S0 solvent |

## Command Line Reference

```
python flow.py [config_file] [--template] [--VERDE] [--force|-y]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `config_file` | `workflow_config.yaml` | YAML configuration file |
| `--template` | — | Generate template config and exit |
| `--VERDE` | — | Generate VERDE preset config and run immediately |
| `--force`, `-y` | — | Overwrite output directory without prompting |

```bash
# Generate template, edit, then run
python flow.py --template
vim workflow_config.yaml
python flow.py

# VERDE preset (generates config + runs in one step)
python flow.py --VERDE

# Custom config, overwrite existing output
python flow.py my_config.yaml --force
```

## Workflow Management

After generating workflows, use the master script:

```bash
cd workflow_run

./workflow.sh submit              # Submit all molecules
./workflow.sh submit MOLECULE_ID  # Submit one molecule
./workflow.sh status              # Compact batch status
./workflow.sh status all -v       # Verbose per-molecule detail
./workflow.sh status MOLECULE_ID  # Detailed status for one molecule
./workflow.sh cancel all          # Cancel everything
./workflow.sh cancel MOL_ID calc  # Cancel specific calculation
./workflow.sh extract             # Extract energies to CSV
./workflow.sh clean               # Interactive cleanup
```

Per-molecule scripts are also available:

```bash
cd workflow_run/MOLECULE_ID
./submit.sh       # Submit this molecule's workflow
./status.sh       # Detailed status with energies
./cancel.sh all   # Cancel all jobs for this molecule
```

## Configuration Reference

The YAML config has five sections. The `--template` output documents every field with comments. Key points:

**`conformer_search.backend`** — `"crest"` (most thorough), `"rdkit"` (fast), or `"openbabel"` (fallback). With `allow_fallback: true`, if the chosen backend fails, others are tried automatically.

**`calculations`** — Each entry is a calculation stage. The `depends_on` field creates SLURM dependencies so stages run in the right order. You can add, remove, or reorder stages freely — the dependency graph is resolved automatically.

**Charge/multiplicity** — Set independently per stage. Typical patterns:

| State | Charge | Multiplicity |
|-------|--------|-------------|
| Neutral singlet (S0) | 0 | 1 |
| Triplet (T1) | 0 | 3 |
| Radical cation (oxidation) | +1 | 2 |
| Radical anion (reduction) | −1 | 2 |

**`conformer_search.gaussian.charge`** should match `s0_opt_freq_vac` (both describe the same electronic state).

## Troubleshooting

**"No XYZ files found"** — Make sure `project.xyz_dir` in your YAML points to a directory containing `.xyz` files.

**"Gaussian root not found"** — Set `slurm.gaussian_module` to your cluster's Gaussian module name (`module avail gaussian` to check), or set `slurm.gaussian_root` to the correct path.

**Jobs fail immediately** — Check the `.err` and `.out` files in the calculation directory. Common causes: wrong partition name, Gaussian not in PATH, insufficient memory.

**"RDKit not available"** — Install with `conda install -c conda-forge rdkit`, or switch to `backend: "crest"` or `backend: "openbabel"`.

**"CREST not found"** — Install with `conda install -c conda-forge crest xtb`, or switch to `backend: "rdkit"`.

**"bc: command not found" in extract** — Install `bc` with your system package manager (`sudo apt install bc` on Debian/Ubuntu, `sudo yum install bc` on CentOS/RHEL). This is only needed for `./workflow.sh extract`.

**General debugging:**

```bash
# Check the workflow generation log
cat workflow_run/workflow_*.log

# Run the built-in diagnostic
cd workflow_run && ./diagnose.sh

# Check a specific job
cat workflow_run/MOLECULE_ID/CALC_NAME/mol_*.log
cat workflow_run/MOLECULE_ID/CALC_NAME/*.err
```

## License

MIT — see [LICENSE](LICENSE).

