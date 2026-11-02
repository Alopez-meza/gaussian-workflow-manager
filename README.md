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

## Quick Start

### 1. Install dependencies

```bash
pip install pyyaml
# Optional (recommended for conformer search):
conda install -c conda-forge rdkit
# Optional (for rich console output):
pip install rich
```

OpenBabel (`obabel`) should be in PATH or loadable via module for InChIKey generation.

### 2. Generate a template config

```bash
python flow.py --template
```

This creates `workflow_config.yaml`. Edit it to match your cluster:

```yaml
slurm:
  gaussian_module: "gaussian/g16"   # Your cluster's Gaussian module
  # gaussian_root: "/path/to/g16"   # Or set this if no module system
  email: "you@university.edu"
```

### 3. Add your molecules

```bash
mkdir xyz
cp your_molecules/*.xyz xyz/
```

### 4. Generate and submit

```bash
python flow.py workflow_config.yaml
cd workflow_run
./workflow.sh submit          # Submit all molecules
./workflow.sh status          # Check progress
./workflow.sh extract         # Collect results as CSV
```

## Gaussian Environment Setup

The generated SLURM scripts need to find Gaussian. You have two options in the YAML config:

**Option A — Module system (recommended, portable):**

```yaml
slurm:
  gaussian_module: "gaussian/g16"
  gaussian_root: null
```

The generated scripts will contain `module load gaussian/g16`, which works on any cluster with an environment modules system.

**Option B — Manual path:**

```yaml
slurm:
  gaussian_module: null
  gaussian_root: "/path/to/gaussian/g16root"
```

The generated scripts will `export g16root=...` and `source` the Gaussian profile directly.

## Example Configurations

Pre-built YAML configs are in [`examples/`](examples/):

| File | Description |
|------|-------------|
| `verde_full_workflow.yaml` | Complete photophysics: S0, S1, T1, TD-DFT, cation, anion with CREST |
| `crest_conformers_only.yaml` | CREST conformer search + single vacuum opt (no excited states) |
| `opt_freq_basic.yaml` | Ground state opt + freq in vacuum and solvent (no conformers) |
| `tddft_absorption.yaml` | Opt → TD-DFT absorption with RDKit conformers |
| `redox_potentials.yaml` | Neutral + cation + anion for electrochemical potentials |
| `quick_test.yaml` | PM6 test run with 10-minute time limits |

Copy one and edit:

```bash
cp examples/tddft_absorption.yaml workflow_config.yaml
# Edit gaussian_module, email, partition, etc.
python flow.py workflow_config.yaml
```

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

### Examples

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
- Neutral singlet: charge `0`, multiplicity `1`
- Triplet: same charge, multiplicity `3`
- Radical cation: charge `+1`, multiplicity `2`
- Radical anion: charge `-1`, multiplicity `2`

**`conformer_search.gaussian.charge`** should match `s0_opt_freq_vac` (both describe the same electronic state).

## Troubleshooting

1. **Check the log file**: `workflow_TIMESTAMP.log` in the output directory
2. **Run diagnostics**: `./diagnose.sh` (generated in the output directory)
3. **Check individual jobs**: `MOLECULE_ID/CALC_NAME/mol_*.log` and `*.out`/`*.err`
4. **Verify Gaussian access**: `module load gaussian/g16 && which g16`

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use this tool in your research, please cite:

> Gaussian Workflow Manager — A. López-Meza, Lopez Group, Northeastern University.
> https://github.com/lopezgroup-northeastern/gaussian-workflow-manager
