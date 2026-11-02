#!/usr/bin/env python3
"""
================================================================================
                    GAUSSIAN WORKFLOW MANAGER
================================================================================

Complete single-file solution for managing multi-stage Gaussian calculations
with conformer search (CREST / RDKit / OpenBabel), SLURM integration, and YAML.

Author:       Andrés López-Meza
Institution:  Northeastern University — Lopez Group
Repository:   https://github.com/lopezgroup-northeastern/gaussian-workflow-manager
Version:      2.1
Date:         November 2025

Features:
  - Conformer search: default template uses CREST; backends configurable in YAML
  - Multi-stage DFT calculations (charge/mult per stage come from workflow_config.yaml)
  - SLURM job management with dependencies
  - Parallel UFF optimization when using RDKit (multi-worker speedup)
  - Real-time status monitoring (cached squeue in generated scripts)
  - YAML-based configuration (--template vs --VERDE presets documented in-file)
  - Tuned for VERDE DB-style workflows (cation / cat / an branches in template notes)

If you use this workflow in your research, please cite:
  "Gaussian Workflow Manager — A. López-Meza, Lopez Group, Northeastern University"
  https://github.com/lopezgroup-northeastern/gaussian-workflow-manager

================================================================================
"""

import os
import sys
import yaml
import shutil
import subprocess
import argparse
import copy
import math
import re
import logging
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from multiprocessing import Pool, cpu_count

TQDM_AVAILABLE = False
# Try to import rich for beautiful progress bars and console output
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, TaskID
    from rich.logging import RichHandler
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Fallback to tqdm if rich is not available
    try:
        from tqdm import tqdm
        TQDM_AVAILABLE = True
    except ImportError:
        TQDM_AVAILABLE = False


class WorkflowLogger:
    """Robust logging system for Gaussian Workflow Manager"""
    
    def __init__(self, log_file: Optional[Path] = None, verbose: bool = True):
        """Initialize logger with file and console handlers"""
        self.logger = logging.getLogger('GaussianWorkflow')
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()  # Remove any existing handlers
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        simple_formatter = logging.Formatter('%(message)s')
        
        # Console handler
        if RICH_AVAILABLE and verbose:
            # Use RichHandler for beautiful console output
            console_handler = RichHandler(
                console=Console(stderr=True),
                show_time=True,
                show_path=False,
                rich_tracebacks=True,
                markup=True
            )
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(simple_formatter)
        else:
            # Standard console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(simple_formatter)
        
        self.logger.addHandler(console_handler)
        
        # File handler (if log file specified)
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
            file_handler.setLevel(logging.DEBUG)  # Log everything to file
            file_handler.setFormatter(detailed_formatter)
            self.logger.addHandler(file_handler)
            self.log_file = log_file
        else:
            self.log_file = None
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)
    
    def success(self, message: str):
        """Log success message (info level with special formatting)"""
        if RICH_AVAILABLE:
            self.logger.info(f"[green][OK][/green] {message}")
        else:
            self.logger.info(f"[OK] {message}")
    
    def section(self, title: str, level: int = 0):
        """Log section header"""
        if level == 0:
            separator = "=" * 60
            self.logger.info(f"\n{separator}")
            self.logger.info(title)
            self.logger.info(separator)
        elif level == 1:
            separator = "-" * 60
            self.logger.info(f"\n{separator}")
            self.logger.info(title)
            self.logger.info(separator)
        else:
            self.logger.info(f"\n{title}")
    
    def table(self, title: str, headers: List[str], rows: List[List[str]]):
        """Display data as a table (if rich is available)"""
        if RICH_AVAILABLE:
            table = Table(title=title, box=box.ROUNDED)
            for header in headers:
                table.add_column(header)
            for row in rows:
                table.add_row(*row)
            console = Console()
            console.print(table)
        else:
            # Fallback to simple text table
            self.logger.info(f"\n{title}:")
            self.logger.info(" | ".join(headers))
            self.logger.info("-" * (sum(len(h) for h in headers) + 3 * len(headers)))
            for row in rows:
                self.logger.info(" | ".join(row))

HARTREE_TO_KCAL = 627.5094740631

# Try to import RDKit - if not available, will fall back to OpenBabel
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# Template for default YAML configuration
DEFAULT_YAML_CONFIG = """# ============================================================================
# VERDE DB MATERIALS - Gaussian Workflow Configuration
# ============================================================================
# Default: closed-shell cation ground state (charge +1, multiplicity 1).
# Conformer search defaults to CREST (see conformer_search.backend).
# Oxidation (cat_*): radical cation +1, multiplicity 2 | Reduction (an_*): radical anion -1, mult 2
# ============================================================================

project:
  name: "photophysics_workflow"
  description: "Complete photophysical characterization workflow - VERDE DB Materials"
  output_dir: "workflow_run"
  xyz_dir: "xyz"

# OpenBabel settings for InChIKey generation and conformer search
openbabel:
  module: "OpenBabel/2.4.1"
  use_inchikey: true

# Conformational search settings
conformer_search:
  enabled: true
  num_conformers: 5
  backend: "crest"            # rdkit | openbabel | crest
  allow_fallback: true        # try other providers if the preferred one fails
  method: "PM7"
  rmsd_cutoff: 0.5            # Used by RDKit pruning
  use_rdkit: false            # Backwards compatibility toggle (ignored when backend is crest)
  rdkit_optimize: true
  rdkit_max_iters: 500
  
  crest:
    # If xTB/CREST uses OpenBLAS: set OPENBLAS_NUM_THREADS=1 to avoid bad warnings in nested
    # parallel regions (exported in run_crest.slurm and local CREST subprocess).
    executable: "crest"       # Path to CREST binary
    xtb_executable: "xtb"     # Path to xTB binary (CREST dependency)
    gfn: "gfn2"               # gfn2 | gfn1 | gfn0 | gfnff
    solvent: null             # Example: "water" -> adds --alpb water
    threads: 16               # Passed via --T and OMP_NUM_THREADS
    ewin: 6.0                 # kcal/mol energy window
    rthr: 0.125               # Å RMSD threshold
    ethr: 0.05                # kcal/mol duplicate energy threshold
    mdlen: 20                 # ps, MD length
    mddump: 100               # MD dump frequency
    mdtemp: 400               # K, MD temperature
    nci: false                # Adds --nci for non-covalent complexes
    entropy: false            # Adds --entropy to include conformational entropy
    cbonds: []                # Explicit constrained bonds, e.g., ["1-2","5-7"]
    reoptimize_pm7: true      # Keep Gaussian PM7 refinement stage
    gaussian_route: "#p PM7 opt"
  
  slurm:
    cores: 4
    memory: "8GB"
    time: "00:30:00"
    partition: "short"
  
  gaussian:
    route: "#p PM7 opt"
    charge: 1                  # Match s0 closed-shell cation (CREST --chrg)
    multiplicity: 1

# Main calculation phases
calculations:
  
  # Ground state vacuum optimization (starting point for all other calcs)
  s0_opt_freq_vac:
    enabled: true
    depends_on: null
    gaussian:
      route: "#p M06/6-31+G(d,p) opt=(calcfc,maxcycles=250) freq=noraman int=ultrafine"
      charge: 1
      multiplicity: 1
      oldchk: null
    slurm:
      cores: 4
      memory: "8GB"
      time: "00:30:00"
      partition: "short"
  
  # Ground state solvated optimization
  s0_opt_freq_solv:
    enabled: true
    depends_on: "s0_opt_freq_vac"
    gaussian:
      route: "#p M06/6-31+G(d,p) opt=(recalcfc=30,maxcycles=250) freq=noraman scrf=(iefpcm,solvent=acetonitrile) int=ultrafine"
      charge: 1
      multiplicity: 1
      oldchk: "s0_opt_freq_vac"
      geom: "AllCheck"
      guess: "Read"
    slurm:
      cores: 4
      memory: "8GB"
      time: "00:30:00"
      partition: "short"
  
  # TDDFT single point for absorption spectrum
  s0_sp_tddft_solv:
    enabled: true
    depends_on: "s0_opt_freq_solv"
    gaussian:
      route: "#p M06/6-31+G(d,p) td=(50-50,nstates=20) scrf=(iefpcm,solvent=acetonitrile) int=ultrafine"
      charge: 1
      multiplicity: 1
      oldchk: "s0_opt_freq_solv"
      geom: "AllCheck"
      guess: "Read"
    slurm:
      cores: 4
      memory: "8GB"
      time: "00:30:00"
      partition: "short"
  
  # First excited singlet state
  s1_opt_freq_solv:
    enabled: true
    depends_on: "s0_opt_freq_vac"
    gaussian:
      route: "#p M06/6-31+G(d,p) opt=(maxcycles=250) td=(nstates=10,root=1) freq=noraman scrf=(iefpcm,solvent=acetonitrile) int=ultrafine"
      charge: 1
      multiplicity: 1
      oldchk: "s0_opt_freq_vac"
      geom: "AllCheck"
      guess: "Read"
    slurm:
      cores: 4
      memory: "8GB"
      time: "00:30:00"
      partition: "short"
  
  # Triplet state
  t1_opt_freq_solv:
    enabled: true
    depends_on: "s0_opt_freq_vac"
    gaussian:
      route: "#p UM06/6-31+G(d,p) opt=(calcfc,maxcycles=250) freq=noraman scrf=(iefpcm,solvent=acetonitrile) scf=(xqc,maxcycle=512) int=ultrafine"
      charge: 1
      multiplicity: 3
      oldchk: "s0_opt_freq_vac"
      geom: "AllCheck"
      guess: "Read"
    slurm:
      cores: 4
      memory: "8GB"
      time: "00:30:00"
      partition: "short"
  
  # Oxidation branch: radical cation (charge +1, doublet)
  cat_opt_freq_vac:
    enabled: true
    depends_on: "s0_opt_freq_vac"
    gaussian:
      route: "#p UM06/6-31+G(d,p) opt=(calcfc,maxcycles=250) freq=noraman scf=(xqc,maxcycle=512) int=ultrafine"
      charge: 1
      multiplicity: 2
      oldchk: "s0_opt_freq_vac"
      geom: "AllCheck"
      guess: "Read"
    slurm:
      cores: 4
      memory: "8GB"
      time: "00:30:00"
      partition: "short"
  
  # Radical cation solvated
  cat_opt_freq_solv:
    enabled: true
    depends_on: "cat_opt_freq_vac"
    gaussian:
      route: "#p UM06/6-31+G(d,p) opt=(recalcfc=30,maxcycles=250) freq=noraman scrf=(iefpcm,solvent=acetonitrile) scf=(xqc,maxcycle=512) int=ultrafine"
      charge: 1
      multiplicity: 2
      oldchk: "cat_opt_freq_vac"
      geom: "AllCheck"
      guess: "Read"
    slurm:
      cores: 4
      memory: "8GB"
      time: "00:30:00"
      partition: "short"
  
  # Reduction branch: radical anion (charge -1, doublet); adjust if closed-shell anion
  an_opt_freq_solv:
    enabled: true
    depends_on: "s0_opt_freq_vac"
    gaussian:
      route: "#p M06/6-31+G(d,p) opt=(calcfc,maxcycles=250) freq=noraman scrf=(iefpcm,solvent=acetonitrile) scf=(xqc,maxcycle=512,conver=6) int=ultrafine"
      charge: -1
      multiplicity: 2
      oldchk: "s0_opt_freq_vac"
      geom: "AllCheck"
      guess: "Read"
    slurm:
      cores: 4
      memory: "8GB"
      time: "00:30:00"
      partition: "short"

# Global SLURM settings
slurm:
  # Gaussian environment setup (choose ONE):
  #   Option A (recommended): load a module — portable across clusters
  gaussian_module: null       # e.g. "gaussian/g16"
  #   Option B: manual root path — set g16root and source profile directly
  gaussian_root: "/path/to/gaussian"
  scratch: "/scratch/$USER"
  email: "user@university.edu"
  mail_type: "END,FAIL"
  account: null
  constraint: null
  exclude_nodes: null        # Comma-separated list of nodes to exclude (e.g., "d3024" or "d3024,d3025")
  max_concurrent_jobs: 50
  sbatch_cmd: "sbatch"

# Analysis and reporting options
analysis:
  extract_energies: true
  extract_frequencies: true
  extract_orbitals: true
  generate_report: true
  compress_logs: false
  generate_fchk: true  # Enable .fchk generation for VERDE DB

# ============================================================================
# CHARGE / MULTIPLICITY NOTES:
# ============================================================================
# - Main pipeline (s0, s1, TDDFT): closed-shell cation, charge +1, multiplicity 1
# - t1: same charge +1, multiplicity 3 (triplet)
# - cat_* : radical cation, charge +1, multiplicity 2
# - an_*  : radical anion, charge -1, multiplicity 2
# - conformer_search.gaussian charge should match s0 (here +1)
# - Quick test parameters: 4 cores, 8GB RAM, 30 min time limits
# ============================================================================
"""

# VERDE-specific YAML (--VERDE): batch preset. Main S0 pipeline is neutral (charge 0, mult 1).
# For closed-shell cation S0 (+1/1) use the default --template file instead, or edit charges here.
VERDE_YAML_CONFIG = """project:
  name: "verde_crest_workflow"
  description: "VERDE DB workflow with CREST conformer search"
  output_dir: "verde_crest_run"
  xyz_dir: "xyz"

openbabel:
  module: "OpenBabel/2.4.1"
  use_inchikey: true

conformer_search:
  enabled: true
  num_conformers: 5
  backend: "crest"
  allow_fallback: true
  method: "PM7"
  rmsd_cutoff: 0.5
  use_rdkit: false
  rdkit_optimize: false
  crest:
    executable: "crest"
    xtb_executable: "xtb"
    gfn: "gfn2"
    solvent: null
    threads: 16
    ewin: 6.0
    rthr: 0.125
    ethr: 0.05
    mdlen: 20
    mddump: 100
    mdtemp: 300
    nci: false
    entropy: false
    cbonds: []
    reoptimize_pm7: true
    gaussian_route: "#p PM7 opt"
  slurm:
    cores: 8
    memory: "64GB"
    time: "4-00:00:00"
    partition: "lopez"
  gaussian:
    route: "#p PM7 opt"
    charge: 0
    multiplicity: 1

calculations:
  s0_opt_freq_vac:
    enabled: true
    depends_on: null
    gaussian:
      route: "#p M06/6-31+G(d,p) opt"
      charge: 0
      multiplicity: 1
      oldchk: null
    slurm:
      cores: 8
      memory: "64GB"
      time: "4-00:00:00"
      partition: "lopez"

  s0_opt_freq_solv:
    enabled: true
    depends_on: "s0_opt_freq_vac"
    gaussian:
      route: "#p M06/6-31+G(d,p) opt freq=noraman scrf=(iefpcm,solvent=acetonitrile)"
      charge: 0
      multiplicity: 1
      oldchk: "s0_opt_freq_vac"
      geom: "Checkpoint"
      guess: "Read"
    slurm:
      cores: 8
      memory: "64GB"
      time: "4-00:00:00"
      partition: "lopez"

  s0_sp_tddft_solv:
    enabled: true
    depends_on: "s0_opt_freq_solv"
    gaussian:
      route: "#p M06/6-31+G(d,p) td=(50-50,nstates=20) scrf=(iefpcm,solvent=acetonitrile) int=ultrafine"
      charge: 0
      multiplicity: 1
      oldchk: "s0_opt_freq_solv"
      geom: "Checkpoint"
      guess: "Read"
    slurm:
      cores: 8
      memory: "64GB"
      time: "4-00:00:00"
      partition: "lopez"

  s1_opt_freq_solv:
    enabled: true
    depends_on: "s0_opt_freq_vac"
    gaussian:
      route: "#p M06/6-31+G(d,p) td=(nstates=5,root=1) opt=(maxcycles=250) freq=noraman scrf=(solvent=acetonitrile)"
      charge: 0
      multiplicity: 1
      oldchk: "s0_opt_freq_vac"
      geom: "Checkpoint"
      guess: null
    slurm:
      cores: 8
      memory: "64GB"
      time: "4-00:00:00"
      partition: "lopez"

  t1_opt_freq_solv:
    enabled: true
    depends_on: "s0_opt_freq_vac"
    gaussian:
      route: "#p M06/6-31+G(d,p) opt freq=noraman scrf=(solvent=acetonitrile)"
      charge: 0
      multiplicity: 3
      oldchk: "s0_opt_freq_vac"
      geom: "Checkpoint"
      guess: "Mix"
    slurm:
      cores: 8
      memory: "64GB"
      time: "4-00:00:00"
      partition: "lopez"

  cat_opt_freq_solv:
    enabled: true
    depends_on: "s0_opt_freq_vac"
    gaussian:
      route: "#p M06/6-31+G(d,p) opt freq=noraman scrf=(solvent=acetonitrile)"
      charge: 1
      multiplicity: 2
      oldchk: "s0_opt_freq_vac"
      geom: "Checkpoint"
      guess: "Mix"
    slurm:
      cores: 8
      memory: "64GB"
      time: "4-00:00:00"
      partition: "lopez"

  an_opt_freq_solv:
    enabled: true
    depends_on: "s0_opt_freq_vac"
    gaussian:
      route: "#p M06/6-31+G(d,p) opt freq=noraman scrf=(solvent=acetonitrile)"
      charge: -1
      multiplicity: 2
      oldchk: "s0_opt_freq_vac"
      geom: "Checkpoint"
      guess: "Mix"
    slurm:
      cores: 8
      memory: "64GB"
      time: "4-00:00:00"
      partition: "lopez"

slurm:
  gaussian_module: null       # e.g. "gaussian/g16" — set this OR gaussian_root
  gaussian_root: "/path/to/gaussian"
  scratch: "/scratch/$USER"
  email: "user@university.edu"
  mail_type: "END,FAIL"
  account: null
  constraint: null
  max_concurrent_jobs: 100
  max_concurrent_molecules: 100
  exclude_nodes: null
  sbatch_cmd: "sbatch"

analysis:
  extract_energies: true
  extract_frequencies: true
  extract_orbitals: true
  generate_report: true
  compress_logs: false
  generate_fchk: false
"""


# ==============================================================================
# PARALLEL CONFORMER OPTIMIZATION
# ==============================================================================

def _optimize_conformer_worker(args):
    """
    Worker function for parallel UFF optimization of conformers
    
    This function is defined at module level (not inside class) to be
    pickle-able for multiprocessing.
    
    Args:
        args: Tuple of (smiles, conf_id, max_iters, random_seed)
    
    Returns:
        Tuple of (conf_id, energy, success, coords)
        coords is a list of [x, y, z] coordinates if successful
    """
    smiles, conf_id, max_iters, random_seed = args
    
    try:
        # Re-create molecule from SMILES (each worker needs its own copy)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return (conf_id, None, False, None)
        
        mol = Chem.AddHs(mol)
        
        # Re-embed just this conformer
        params = AllChem.ETKDGv3()
        params.randomSeed = random_seed + conf_id  # Unique seed per conformer
        params.numThreads = 1  # Each worker uses 1 thread
        params.useRandomCoords = True
        
        # Embed single conformer
        result = AllChem.EmbedMolecule(mol, params=params)
        if result == -1:
            return (conf_id, None, False, None)
        
        # Optimize with UFF
        AllChem.UFFOptimizeMolecule(mol, confId=0, maxIters=max_iters)
        
        # Calculate energy
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=0)
        if ff is None:
            return (conf_id, None, False, None)
        
        energy = ff.CalcEnergy()
        
        # Extract coordinates
        conf = mol.GetConformer(0)
        coords = []
        for atom_idx in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(atom_idx)
            coords.append([pos.x, pos.y, pos.z])
        
        return (conf_id, energy, True, coords)
        
    except Exception as e:
        return (conf_id, None, False, None)


class GaussianWorkflowManager:
    """Orchestrate workflow generation: reads YAML (charges, backends, routes) and writes scripts."""
    
    def __init__(self, config_file: str = "workflow_config.yaml", verbose: bool = True, force: bool = False):
        """Initialize workflow manager with configuration"""
        self.config_file = config_file
        self.force = force
        self.config = None
        self.molecules = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.last_conformer_backend = None
        self.latest_best_conformer = None
        self.conformer_metadata = {}
        self.log_file_handle = None
        self.log_file_path = Path(f"workflow_{self.timestamp}.log")
        self._stdout_original = None
        self.logger = None  # Will be initialized in start_logging()
        self.verbose = verbose
        self.xyz_files = []  # Will be populated in check_dependencies()
    
    def get_conformer_backend(self) -> str:
        """Return sanitized conformer generation backend (crest|rdkit|openbabel)."""
        if self.config is None:
            # Before load_config(); real default comes from YAML (template uses crest)
            return 'rdkit'
        backend = self.config.get('conformer_search', {}).get('backend')
        if backend is None:
            # Backwards compatibility with legacy flag
            if self.config.get('conformer_search', {}).get('use_rdkit', False):
                backend = 'rdkit'
            else:
                backend = 'openbabel'
        backend = str(backend).strip().lower()
        if backend not in {'rdkit', 'openbabel', 'crest'}:
            backend = 'rdkit'
        return backend
    
    def _get_crest_config(self) -> Tuple[str, Dict, bool]:
        """Helper to get backend, crest_cfg, and crest_reoptimize consistently
        
        Returns:
            Tuple of (backend, crest_cfg, crest_reoptimize)
        """
        if self.config is None:
            # Return defaults if config not loaded
            return 'rdkit', {}, False
        backend = (self.last_conformer_backend or self.get_conformer_backend())
        crest_cfg = self.config.get('conformer_search', {}).get('crest', {})
        crest_reoptimize = crest_cfg.get('reoptimize_pm7', True) if backend == 'crest' else False
        return backend, crest_cfg, crest_reoptimize

    def start_logging(self):
        """Initialize robust logging system with file and console handlers"""
        if self.logger is not None:
            return
        
        # Initialize WorkflowLogger
        self.logger = WorkflowLogger(self.log_file_path, verbose=self.verbose)
        
        # Log startup information
        self.logger.section("GAUSSIAN WORKFLOW MANAGER - EXECUTION LOG", level=0)
        self.logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Timestamp: {self.timestamp}")
        self.logger.info(f"Python Version: {sys.version.split()[0]}")
        self.logger.info(f"Working Directory: {os.getcwd()}")
        if RICH_AVAILABLE:
            self.logger.info("Rich console: Available")
        elif TQDM_AVAILABLE:
            self.logger.info("Progress bars: tqdm available")
        else:
            self.logger.warning("No progress bar library available (install 'rich' or 'tqdm')")
        self.logger.section("", level=0)

    def stop_logging(self, final_dir: Optional[Path] = None):
        """Finalize logging and move log file to output directory"""
        if self.logger:
            self.logger.section("", level=0)
            self.logger.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            if self.log_file_path:
                self.logger.info(f"Log File: {self.log_file_path.name}")
            self.logger.section("", level=0)
            
            # Move log file if specified
            if final_dir and final_dir.exists() and self.log_file_path and self.log_file_path.exists():
                destination = final_dir / self.log_file_path.name
                try:
                    shutil.move(str(self.log_file_path), destination)
                    self.log_file_path = destination
                    self.logger.success(f"Log file saved: {destination}")
                except Exception as e:
                    self.logger.warning(f"Could not move log file to output directory: {e}")
            
            # Close all handlers
            for handler in self.logger.logger.handlers[:]:
                handler.close()
                self.logger.logger.removeHandler(handler)
            
            self.logger = None
    
    def fix_bash_variables(self, script_content: str) -> str:
        """Fix bash variable syntax: replace ${{VAR}} with ${VAR}
        
        Python f-strings use {{ to escape braces, but bash needs single braces.
        This function fixes all bash variable references in generated scripts.
        
        Examples:
            - ${{BLUE}} -> ${BLUE}
            - ${{COMPLETED_LIST[@]}} -> ${COMPLETED_LIST[@]}
            - ${{VAR:0:8}} -> ${VAR:0:8}
            """
        # Replace ${{...}} with ${...} for all bash variable patterns
        # This handles simple variables, arrays, and parameter expansion
        # Pattern matches: ${{ followed by any content (non-greedy) until }}
        script_content = re.sub(r'\$\{\{([^}]+)\}\}', r'${\1}', script_content)
        return script_content
    
    def get_bash_header(self, script_name: str = "", description: str = "") -> str:
        """
        Generate a standard header for all bash scripts
        
        Args:
            script_name: Name of the script
            description: Brief description of what the script does
        
        Returns:
            Formatted bash header string
        """
        header = f"""#!/bin/bash
################################################################################
#
#  GAUSSIAN WORKFLOW MANAGER - {script_name}
#
################################################################################
#
#  Generated by: Gaussian Workflow Manager v2.1
#  Date:         {datetime.now().strftime("%B %Y")}
#
#  Description:  {description}
#
#  https://github.com/lopezgroup-northeastern/gaussian-workflow-manager
#
################################################################################

"""
        return header
        
    def load_config(self) -> bool:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_file):
            if self.logger:
                self.logger.warning(f"Configuration file '{self.config_file}' not found!")
                self.logger.info("Creating default template (CREST backend; see charge/mult notes in file)...")
            else:
                print(f"Configuration file '{self.config_file}' not found!")
                print("Creating default template (CREST backend; edit charge/mult to match your system)...")
            self.create_template_config()
            if self.logger:
                self.logger.info(f"Edit '{self.config_file}' (xyz_dir, gaussian_root, charge/mult) and run again.")
            else:
                print(f"Edit '{self.config_file}' (xyz_dir, gaussian_root, charge/mult) and run again.")
            return False
        
        try:
            with open(self.config_file, 'r') as f:
                self.config = yaml.safe_load(f)
            if self.logger:
                self.logger.success(f"Configuration loaded from {self.config_file}")
            else:
                print(f"[OK] Configuration loaded from {self.config_file}")
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading configuration: {e}")
            else:
                print(f"Error loading configuration: {e}")
            return False
    
    def create_template_config(self):
        """Write DEFAULT_YAML_CONFIG: CREST default, closed-shell cation S0 +1/1, cat +1/2, an -1/2."""
        with open(self.config_file, 'w') as f:
            f.write(DEFAULT_YAML_CONFIG)
        if self.logger:
            self.logger.success(f"Template configuration created: {self.config_file}")
        else:
            print(f"[OK] Template configuration created: {self.config_file}")
    
    def create_verde_config(self):
        """Write VERDE_YAML_CONFIG: batch preset; neutral S0 0/1, cat +1/2, an -1/2 (see file header)."""
        with open(self.config_file, 'w') as f:
            f.write(VERDE_YAML_CONFIG)
        if self.logger:
            self.logger.success(f"VERDE configuration created: {self.config_file}")
        else:
            print(f"[OK] VERDE configuration created: {self.config_file}")
    
    def check_dependencies(self) -> bool:
        """Check for required dependencies"""
        if self.config is None:
            if self.logger:
                self.logger.error("Configuration not loaded. Call load_config() first.")
            else:
                print("[ERROR] Configuration not loaded. Call load_config() first.")
            return False
        
        if self.logger:
            self.logger.section("Checking dependencies...", level=1)
        else:
            print("\nChecking dependencies...")
            print("=" * 50)
        
        all_good = True
        
        # Check XYZ directory
        xyz_dir = self.config['project']['xyz_dir']
        if not os.path.exists(xyz_dir):
            if self.logger:
                self.logger.error(f"XYZ directory '{xyz_dir}' not found")
                self.logger.info(f"  Create it with: mkdir {xyz_dir}")
            else:
                print(f"[ERROR] XYZ directory '{xyz_dir}' not found")
                print(f"  Create it with: mkdir {xyz_dir}")
            all_good = False
        else:
            xyz_files = list(Path(xyz_dir).glob("*.xyz"))
            if not xyz_files:
                if self.logger:
                    self.logger.error(f"No XYZ files found in {xyz_dir}")
                else:
                    print(f"[ERROR] No XYZ files found in {xyz_dir}")
                all_good = False
            else:
                if self.logger:
                    self.logger.success(f"Found {len(xyz_files)} XYZ files")
                else:
                    print(f"[OK] Found {len(xyz_files)} XYZ files")
                self.xyz_files = xyz_files
        
        backend = self.get_conformer_backend()
        conformers_enabled = self.config['conformer_search'].get('enabled', True)
        
        # Check RDKit requirements
        if conformers_enabled and backend == 'rdkit':
            if RDKIT_AVAILABLE:
                if self.logger:
                    self.logger.success("RDKit found")
                else:
                    print("[OK] RDKit found")
            else:
                if self.logger:
                    self.logger.error("RDKit not found but requested as conformer backend")
                    self.logger.info("  Install with: conda install -c conda-forge rdkit")
                else:
                    print("[ERROR] RDKit not found but requested as conformer backend")
                print("  Install with: conda install -c conda-forge rdkit")
                all_good = False
        
        # Check CREST/xTB requirements
        if conformers_enabled and backend == 'crest':
            crest_cfg = self.config['conformer_search'].get('crest', {})
            crest_exec = crest_cfg.get('executable', 'crest')
            xtb_exec = crest_cfg.get('xtb_executable', 'xtb')
            crest_path = shutil.which(crest_exec)
            xtb_path = shutil.which(xtb_exec)
            if crest_path:
                if self.logger:
                    self.logger.success(f"CREST found at {crest_path}")
                else:
                    print(f"[OK] CREST found at {crest_path}")
            else:
                if self.logger:
                    self.logger.error(f"CREST executable '{crest_exec}' not found in PATH")
                    self.logger.info("  Update 'conformer_search.crest.executable' or install CREST")
                else:
                    print(f"[ERROR] CREST executable '{crest_exec}' not found in PATH")
                    print("  Update 'conformer_search.crest.executable' or install CREST")
                all_good = False
            if xtb_path:
                if self.logger:
                    self.logger.success(f"xTB found at {xtb_path}")
                else:
                    print(f"[OK] xTB found at {xtb_path}")
            else:
                if self.logger:
                    self.logger.error(f"xTB executable '{xtb_exec}' not found in PATH (required by CREST)")
                else:
                    print(f"[ERROR] xTB executable '{xtb_exec}' not found in PATH (required by CREST)")
                all_good = False
        
        # Check OpenBabel if InChIKey is requested or if backend requires it
        needs_openbabel = self.config['openbabel']['use_inchikey'] or (
            conformers_enabled and backend in {'openbabel', 'rdkit'}
        )
        if needs_openbabel:
            try:
                result = subprocess.run(['obabel', '--version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    if self.logger:
                        self.logger.success("OpenBabel found")
                    else:
                        print("[OK] OpenBabel found")
                else:
                    if self.logger:
                        self.logger.error("OpenBabel not found")
                    else:
                        print("[ERROR] OpenBabel not found")
                    if self.config['conformer_search'].get('enabled') and not RDKIT_AVAILABLE:
                        if self.logger:
                            self.logger.error("  Neither RDKit nor OpenBabel available for conformer search!")
                        else:
                            print("  Neither RDKit nor OpenBabel available for conformer search!")
                        all_good = False
                    else:
                        if self.logger:
                            self.logger.warning("  (will use filenames instead of InChIKeys)")
                        else:
                            print("  (will use filenames instead of InChIKeys)")
                        self.config['openbabel']['use_inchikey'] = False
            except (FileNotFoundError, subprocess.SubprocessError) as e:
                if self.logger:
                    self.logger.error(f"OpenBabel not found: {e}")
                else:
                    print(f"[ERROR] OpenBabel not found: {e}")
                if self.config['conformer_search'].get('enabled') and not RDKIT_AVAILABLE:
                    if self.logger:
                        self.logger.error("  Neither RDKit nor OpenBabel available for conformer search!")
                    else:
                        print("  Neither RDKit nor OpenBabel available for conformer search!")
                    all_good = False
                else:
                    if self.logger:
                        self.logger.warning("  (will use filenames instead of InChIKeys)")
                    else:
                        print("  (will use filenames instead of InChIKeys)")
                    self.config['openbabel']['use_inchikey'] = False
        
        # Check Gaussian directory
        gauss_root = self.config['slurm']['gaussian_root']
        if os.path.exists(gauss_root):
            if self.logger:
                self.logger.success(f"Gaussian root found: {gauss_root}")
            else:
                print(f"[OK] Gaussian root found: {gauss_root}")
        else:
            if self.logger:
                self.logger.warning(f"Gaussian root '{gauss_root}' not found")
                self.logger.info("  Update 'slurm.gaussian_root' in config file")
            else:
                print(f"[WARNING] Gaussian root '{gauss_root}' not found")
                print("  Update 'slurm.gaussian_root' in config file")
        
        if all_good:
            if self.logger:
                self.logger.success("All dependencies satisfied")
            else:
                print("\n[OK] All dependencies satisfied")
        else:
            if self.logger:
                self.logger.error("Some dependencies are missing. Please fix the issues above.")
            else:
                print("\n[ERROR] Some dependencies are missing. Please fix the issues above.")
        
        return all_good
    
    def generate_inchikey(self, xyz_file: Path) -> str:
        """Generate InChIKey for molecule or return sanitized filename
        
        Always attempts to generate InChIKey first for consistent naming.
        Falls back to sanitized filename if InChIKey generation fails.
        """
        base_name = xyz_file.stem
        
        # Always try to generate InChIKey first (preferred method)
        try:
            result = subprocess.run(
                ['obabel', '-ixyz', str(xyz_file), '-oinchikey', '--append', ''],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0 and result.stdout:
                # Extract InChIKey
                inchikey = result.stdout.strip()
                if inchikey.startswith('InChIKey='):
                    inchikey = inchikey[9:]
                
                # Take only first two parts (remove charge indicator)
                parts = inchikey.split('-')
                if len(parts) >= 2:
                    inchikey = f"{parts[0]}-{parts[1]}"
                    
                    # Validate format (14 chars + 10 chars)
                    if len(parts[0]) == 14 and len(parts[1]) == 10:
                        if self.logger:
                            self.logger.debug(f"Generated InChIKey: {inchikey}")
                        else:
                            print(f"  [OK] Generated InChIKey: {inchikey}")
                        return inchikey
                    else:
                        if self.logger:
                            self.logger.warning(f"Invalid InChIKey format: {inchikey}, using filename")
                        else:
                            print(f"  [WARN] Invalid InChIKey format: {inchikey}, using filename")
                else:
                    if self.logger:
                        self.logger.warning(f"Incomplete InChIKey: {inchikey}, using filename")
                    else:
                        print(f"  [WARN] Incomplete InChIKey: {inchikey}, using filename")
            else:
                if not self.config['openbabel']['use_inchikey']:
                    # Silent fallback if InChIKey is disabled in config
                    pass
                else:
                    if self.logger:
                        self.logger.warning("Failed to generate InChIKey, using filename")
                    else:
                        print(f"  [WARN] Failed to generate InChIKey, using filename")
        except FileNotFoundError:
            if not self.config['openbabel']['use_inchikey']:
                # Silent fallback if OpenBabel not available and InChIKey disabled
                pass
            else:
                if self.logger:
                    self.logger.warning("OpenBabel not found, cannot generate InChIKey, using filename")
                else:
                    print(f"  [WARN] OpenBabel not found, cannot generate InChIKey, using filename")
        except subprocess.TimeoutExpired:
            if self.logger:
                self.logger.warning("InChIKey generation timed out, using filename")
            else:
                print(f"  [WARN] InChIKey generation timed out, using filename")
        except Exception as e:
            if not self.config['openbabel']['use_inchikey']:
                # Silent fallback if InChIKey is disabled
                pass
            else:
                if self.logger:
                    self.logger.warning(f"Error generating InChIKey: {e}, using filename")
                else:
                    print(f"  [WARN] Error generating InChIKey: {e}, using filename")
        
        # Fallback to sanitized filename
        sanitized = ''.join(c if c.isalnum() or c in '-_' else '_' for c in base_name)
        if self.logger:
            self.logger.info(f"Using sanitized filename: {sanitized}")
        else:
            print(f"  [INFO] Using sanitized filename: {sanitized}")
        return sanitized
    
    def read_xyz_file(self, xyz_file: Path) -> Tuple[List[str], List[List[float]]]:
        """Read XYZ file and return atoms and coordinates with validation"""
        atoms = []
        coords = []
        
        if not xyz_file.exists():
            raise FileNotFoundError(f"XYZ file not found: {xyz_file}")
        
        try:
            with open(xyz_file, 'r') as f:
                lines = f.readlines()
        except IOError as e:
            raise IOError(f"Cannot read XYZ file {xyz_file}: {e}")
        
        if len(lines) < 2:
            raise ValueError(f"Invalid XYZ file format: {xyz_file} (needs at least 2 lines)")
        
        # Validate atom count
        try:
            expected_atoms = int(lines[0].strip())
        except ValueError:
            raise ValueError(f"Invalid XYZ file format: {xyz_file} (first line must be atom count)")
        
        # Parse atoms and coordinates
        for i, line in enumerate(lines[2:], start=2):
            parts = line.split()
            if len(parts) < 4:
                continue  # Skip empty or malformed lines
            try:
                atoms.append(parts[0])
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except (ValueError, IndexError) as e:
                if self.logger:
                    self.logger.warning(f"Skipping malformed line {i+1} in {xyz_file.name}: {e}")
                else:
                    print(f"  [WARN] Skipping malformed line {i+1} in {xyz_file.name}: {e}")
                continue
        
        # Validate atom count matches
        if len(atoms) != expected_atoms:
            if self.logger:
                self.logger.warning(f"Atom count mismatch in {xyz_file.name}: expected {expected_atoms}, found {len(atoms)}")
            else:
                print(f"  [WARN] Atom count mismatch in {xyz_file.name}: expected {expected_atoms}, found {len(atoms)}")
        
        if not atoms:
            raise ValueError(f"No valid atoms found in XYZ file: {xyz_file}")
        
        return atoms, coords
    
    def write_xyz_file(self, filepath: Path, atoms: List[str], coords: List[List[float]], comment: str = ""):
        """Write XYZ file with validation"""
        
        if len(atoms) != len(coords):
            raise ValueError(f"Atoms and coordinates length mismatch: {len(atoms)} vs {len(coords)}")
        
        if not atoms:
            raise ValueError("Cannot write empty XYZ file")
        
        # Validate coordinates are not NaN or infinite
        for i, coord in enumerate(coords):
            if len(coord) != 3:
                raise ValueError(f"Coordinate {i} must have 3 values, got {len(coord)}")
            if any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in coord):
                raise ValueError(f"Invalid coordinate at index {i}: {coord} (must be finite numbers)")
        
        try:
            with open(filepath, 'w') as f:
                f.write(f"{len(atoms)}\n")
                f.write(f"{comment}\n")
                for atom, coord in zip(atoms, coords):
                    f.write(f"{atom:2s}  {coord[0]:12.6f}  {coord[1]:12.6f}  {coord[2]:12.6f}\n")
        except IOError as e:
            raise IOError(f"Cannot write XYZ file {filepath}: {e}")
    
    def generate_conformers_rdkit(self, xyz_file: Path, output_dir: Path, num_conf: int) -> int:
        """Generate conformers using RDKit"""
        if num_conf <= 0:
            raise ValueError(f"num_conf must be > 0, got {num_conf}")
        try:
            if self.logger:
                self.logger.info("Using RDKit for conformer generation...")
            else:
                print(f"  Using RDKit for conformer generation...")
            
            # Step 1: Convert XYZ to SMILES using OpenBabel
            temp_smi = output_dir / 'temp.smi'
            result = subprocess.run(
                ['obabel', str(xyz_file), '-O', str(temp_smi)],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode != 0:
                if self.logger:
                    self.logger.warning("Failed to convert XYZ to SMILES")
                else:
                    print(f"    Warning: Failed to convert XYZ to SMILES")
                return 0
            
            # Read SMILES
            with open(temp_smi, 'r') as f:
                smiles_line = f.read().strip()
                # Extract just the SMILES (not the name if present)
                smiles = smiles_line.split()[0] if smiles_line else None
            
            if not smiles:
                if self.logger:
                    self.logger.warning("Empty SMILES generated")
                else:
                    print(f"    Warning: Empty SMILES generated")
                temp_smi.unlink(missing_ok=True)
                return 0
            
            if self.logger:
                self.logger.debug(f"SMILES: {smiles}")
            else:
                print(f"    SMILES: {smiles}")
            
            # Step 2: Generate conformers with RDKit
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                if self.logger:
                    self.logger.warning("RDKit could not parse SMILES")
                else:
                    print(f"    Warning: RDKit could not parse SMILES")
                temp_smi.unlink(missing_ok=True)
                return 0
            
            # Add hydrogens
            mol = Chem.AddHs(mol)
            
            # Generate conformers using ETKDGv3
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            params.numThreads = 0  # Use all available cores
            params.pruneRmsThresh = self.config['conformer_search'].get('rmsd_cutoff', 1.5)
            params.useRandomCoords = True  # Force different starting geometries
            
            if self.logger:
                self.logger.info(f"Generating {num_conf} conformers...")
            else:
                print(f"    Generating {num_conf} conformers...")
            # Generate more conformers than needed to account for pruning
            conf_ids = AllChem.EmbedMultipleConfs(
                mol,
                numConfs=num_conf * 3,  # Generate 3x more to ensure diversity
                params=params
            )
            
            if self.logger:
                self.logger.debug(f"RDKit generated {len(conf_ids)} raw conformers")
            else:
                print(f"    DEBUG: RDKit generated {len(conf_ids)} raw conformers")
            # Calculate RMSD between conformers for debugging
            if len(conf_ids) > 1:
                rmsd_matrix = []
                for i in range(len(conf_ids)):
                    for j in range(i+1, len(conf_ids)):
                        rmsd = AllChem.GetBestRMS(mol, mol, i, j)
                        rmsd_matrix.append(rmsd)
                if rmsd_matrix:
                    if self.logger:
                        self.logger.debug(f"RMSD range: {min(rmsd_matrix):.2f} - {max(rmsd_matrix):.2f} Å")
                        self.logger.debug(f"Average RMSD: {sum(rmsd_matrix)/len(rmsd_matrix):.2f} Å")
                    else:
                        print(f"    DEBUG: RMSD range: {min(rmsd_matrix):.2f} - {max(rmsd_matrix):.2f} Å")
                        print(f"    DEBUG: Average RMSD: {sum(rmsd_matrix)/len(rmsd_matrix):.2f} Å")
            
            if len(conf_ids) == 0:
                if self.logger:
                    self.logger.warning("No conformers generated")
                else:
                    print(f"    Warning: No conformers generated")
                temp_smi.unlink(missing_ok=True)
                return 0
            
            if self.logger:
                self.logger.debug(f"Generated {len(conf_ids)} initial conformers")
            else:
                print(f"    Generated {len(conf_ids)} initial conformers")
            
            # Step 3: Optimize conformers if requested
            if self.config['conformer_search'].get('rdkit_optimize', True):
                max_iters = self.config['conformer_search'].get('rdkit_max_iters', 200)
                
                # Determine number of parallel workers
                n_workers = min(cpu_count(), len(conf_ids), 8)  # Cap at 8 workers
                
                if self.logger:
                    self.logger.info(f"Optimizing {len(conf_ids)} conformers with UFF (max {max_iters} iterations)...")
                    self.logger.debug(f"Using {n_workers} parallel workers for speedup...")
                else:
                    print(f"    Optimizing {len(conf_ids)} conformers with UFF (max {max_iters} iterations)...")
                    print(f"    Using {n_workers} parallel workers for speedup...")
                
                try:
                    # Prepare arguments for parallel optimization
                    # Each worker needs: (smiles, conf_id, max_iters, random_seed)
                    args_list = [(smiles, i, max_iters, 42) for i in range(len(conf_ids))]
                    
                    # Parallel optimization
                    with Pool(n_workers) as pool:
                        results = pool.map(_optimize_conformer_worker, args_list)
                    
                    # Process results
                    energies = []
                    coords_dict = {}  # Store coordinates for successful conformers
                    
                    for conf_id, energy, success, coords in results:
                        if success and energy is not None:
                            energies.append((conf_id, energy))
                            coords_dict[conf_id] = coords
                        else:
                            energies.append((conf_id, float('inf')))
                    
                    # Sort by energy
                    energies.sort(key=lambda x: x[1])
                    
                    # Filter out failed conformers
                    valid_energies = [(cid, e) for cid, e in energies if e != float('inf')]
                    
                    if len(valid_energies) == 0:
                        if self.logger:
                            self.logger.warning("All conformers failed parallel optimization")
                            self.logger.info("Falling back to sequential optimization...")
                        else:
                            print(f"    Warning: All conformers failed parallel optimization")
                            print(f"    Falling back to sequential optimization...")
                        
                        # Fallback: Sequential optimization
                        energies = []
                        coords_dict = {}
                        for conf_id in conf_ids:
                            try:
                                AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=max_iters)
                                ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
                                if ff:
                                    energy = ff.CalcEnergy()
                                    energies.append((conf_id, energy))
                                    
                                    # Store coordinates
                                    conf = mol.GetConformer(conf_id)
                                    coords = []
                                    for atom_idx in range(mol.GetNumAtoms()):
                                        pos = conf.GetAtomPosition(atom_idx)
                                        coords.append([pos.x, pos.y, pos.z])
                                    coords_dict[conf_id] = coords
                                else:
                                    energies.append((conf_id, float('inf')))
                            except (ValueError, RuntimeError) as e:
                                if self.logger:
                                    self.logger.warning(f"Conformer {conf_id} optimization failed: {e}")
                                else:
                                    print(f"    [WARN] Conformer {conf_id} optimization failed: {e}")
                                energies.append((conf_id, float('inf')))
                        
                        energies.sort(key=lambda x: x[1])
                        valid_energies = [(cid, e) for cid, e in energies if e != float('inf')]
                        
                        if len(valid_energies) == 0:
                            if self.logger:
                                self.logger.warning("All conformers failed UFF optimization")
                            else:
                                print(f"    Warning: All conformers failed UFF optimization")
                            temp_smi.unlink(missing_ok=True)
                            return 0
                    
                    energies = valid_energies
                    if self.logger:
                        self.logger.info(f"Successfully optimized {len(energies)} conformers")
                        self.logger.debug(f"Energy range: {energies[0][1]:.2f} to {energies[-1][1]:.2f} kcal/mol")
                    else:
                        print(f"    Successfully optimized {len(energies)} conformers")
                        print(f"    Energy range: {energies[0][1]:.2f} to {energies[-1][1]:.2f} kcal/mol")
                    
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Parallel optimization failed: {e}")
                        self.logger.info("Falling back to sequential optimization...")
                    else:
                        print(f"    Warning: Parallel optimization failed: {e}")
                        print(f"    Falling back to sequential optimization...")
                    
                    # Fallback: Sequential optimization
                    energies = []
                    coords_dict = {}
                    for conf_id in conf_ids:
                        try:
                            AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=max_iters)
                            ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
                            if ff:
                                energies.append((conf_id, ff.CalcEnergy()))
                                conf = mol.GetConformer(conf_id)
                                coords = []
                                for atom_idx in range(mol.GetNumAtoms()):
                                    pos = conf.GetAtomPosition(atom_idx)
                                    coords.append([pos.x, pos.y, pos.z])
                                coords_dict[conf_id] = coords
                            else:
                                energies.append((conf_id, float('inf')))
                        except Exception:
                            energies.append((conf_id, float('inf')))
                    
                    energies.sort(key=lambda x: x[1])
                    valid_energies = [(cid, e) for cid, e in energies if e != float('inf')]
                    
                    if len(valid_energies) == 0:
                        if self.logger:
                            self.logger.warning("All conformers failed UFF optimization")
                        else:
                            print(f"    Warning: All conformers failed UFF optimization")
                        temp_smi.unlink(missing_ok=True)
                        return 0
                    
                    energies = valid_energies
                    print(f"    Energy range: {energies[0][1]:.2f} to {energies[-1][1]:.2f} kcal/mol")
            else:
                energies = [(conf_id, 0) for conf_id in conf_ids]
            
            # Step 4: Write conformers as XYZ files (directly from optimized coords)
            num_written = 0
            rdkit_atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
            if not self.config['conformer_search'].get('rdkit_optimize', True):
                coords_dict = {}
                for conf_id in conf_ids:
                    conf = mol.GetConformer(conf_id)
                    coords = []
                    for atom_idx in range(mol.GetNumAtoms()):
                        pos = conf.GetAtomPosition(atom_idx)
                        coords.append([pos.x, pos.y, pos.z])
                    coords_dict[conf_id] = coords
            for i, (conf_id, energy) in enumerate(energies[:num_conf], 1):
                if conf_id not in coords_dict:
                    if self.logger:
                        self.logger.warning(f"Skipping conformer {conf_id} (no optimized coordinates)")
                    continue
                coords = coords_dict[conf_id]
                conf_file = output_dir / f'conf_{i}.xyz'
                comment = f"Conformer {i} - Energy: {energy:.2f} kcal/mol (RDKit/UFF)"
                self.write_xyz_file(conf_file, rdkit_atoms, coords, comment)
                num_written += 1
            
            # Clean up temporary files
            temp_smi.unlink(missing_ok=True)
            
            if self.logger:
                self.logger.success(f"Successfully generated {num_written} conformers with RDKit")
            else:
                print(f"    [OK] Successfully generated {num_written} conformers with RDKit")
            return num_written
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in RDKit conformer generation: {e}")
            else:
                print(f"    Error in RDKit conformer generation: {e}")
            return 0
    
    def generate_conformers_openbabel(self, xyz_file: Path, output_dir: Path, num_conf: int) -> int:
        """Generate conformers using OpenBabel (fallback method)"""
        try:
            if self.logger:
                self.logger.info("Using OpenBabel for conformer generation...")
            else:
                print(f"  Using OpenBabel for conformer generation...")
            
            # Generate conformers
            result = subprocess.run(
                ['obabel', str(xyz_file), '-O', str(output_dir / 'conf_.xyz'),
                 '--conformer', '--nconf', str(num_conf), '--writeconformers'],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode == 0:
                # OpenBabel generates conf_.xyz, conf_1.xyz, conf_2.xyz, etc.
                # Rename conf_.xyz to conf_1.xyz for consistency
                conf_base = output_dir / 'conf_.xyz'
                if conf_base.exists():
                    # Check if conf_1.xyz already exists
                    conf_1 = output_dir / 'conf_1.xyz'
                    if conf_1.exists():
                        # Multiple conformers: conf_.xyz is extra, rename to conf_0.xyz
                        conf_base.rename(output_dir / 'conf_0.xyz')
                    else:
                        # Single conformer: rename conf_.xyz to conf_1.xyz
                        conf_base.rename(conf_1)
                
                # Renumber if we have conf_0.xyz (shift all up by 1)
                if (output_dir / 'conf_0.xyz').exists():
                    # Get all conformer files and sort
                    conf_files = sorted(output_dir.glob('conf_*.xyz'))
                    # Rename in reverse to avoid conflicts
                    for i, conf_file in enumerate(reversed(conf_files)):
                        new_num = len(conf_files) - i
                        new_name = output_dir / f'conf_{new_num}.xyz'
                        if conf_file != new_name:
                            conf_file.rename(new_name)
                
                # Count generated files
                conf_files = list(output_dir.glob('conf_*.xyz'))
                if self.logger:
                    self.logger.success(f"Successfully generated {len(conf_files)} conformers with OpenBabel")
                else:
                    print(f"    [OK] Successfully generated {len(conf_files)} conformers with OpenBabel")
                return len(conf_files)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"OpenBabel conformer generation failed: {e}")
            else:
                print(f"    Warning: OpenBabel conformer generation failed: {e}")
        
        return 0
    
    def _read_multi_xyz(self, xyz_path: Path) -> List[Tuple[List[str], List[List[float]], str]]:
        """Parse multi-structure XYZ file produced by CREST"""
        structures = []
        if not xyz_path.exists():
            return structures
        
        with open(xyz_path, 'r') as f:
            while True:
                header = f.readline()
                if not header:
                    break
                header = header.strip()
                if not header:
                    continue
                try:
                    atom_count = int(header)
                except ValueError:
                    # Skip malformed blocks
                    continue
                
                comment = f.readline().strip() if atom_count else ""
                atoms = []
                coords = []
                for i in range(atom_count):
                    line = f.readline()
                    if not line:
                        if self.logger:
                            self.logger.warning(f"Unexpected EOF in XYZ block (expected {atom_count} atoms, got {i})")
                        else:
                            print(f"  [WARN] Unexpected EOF in XYZ block (expected {atom_count} atoms, got {i})")
                        break
                    parts = line.split()
                    if len(parts) < 4:
                        if self.logger:
                            self.logger.warning("Skipping malformed line in XYZ block")
                        else:
                            print(f"  [WARN] Skipping malformed line in XYZ block")
                        continue
                    try:
                        atoms.append(parts[0])
                        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    except ValueError as e:
                        if self.logger:
                            self.logger.warning(f"Invalid coordinate format in XYZ block: {e}")
                        else:
                            print(f"  [WARN] Invalid coordinate format in XYZ block: {e}")
                        continue
                
                # Validate atom count
                if len(atoms) != atom_count:
                    if self.logger:
                        self.logger.warning(f"Atom count mismatch in XYZ block: expected {atom_count}, got {len(atoms)}")
                    else:
                        print(f"  [WARN] Atom count mismatch in XYZ block: expected {atom_count}, got {len(atoms)}")
                
                if atoms:
                    structures.append((atoms, coords, comment))
        return structures
    
    def generate_conformers_crest(self, xyz_file: Path, output_dir: Path, num_conf: int) -> int:
        """Generate conformers using CREST + xTB"""
        if num_conf <= 0:
            raise ValueError(f"num_conf must be > 0, got {num_conf}")
        
        crest_cfg = self.config['conformer_search'].get('crest', {})
        crest_exec = crest_cfg.get('executable', 'crest')
        threads = crest_cfg.get('threads', min(cpu_count(), 16))
        gfn = crest_cfg.get('gfn', 'gfn2').lower()
        solvent = crest_cfg.get('solvent')
        ewin = crest_cfg.get('ewin', 6.0)
        rthr = crest_cfg.get('rthr', 0.125)
        ethr = crest_cfg.get('ethr', 0.05)
        mdlen = crest_cfg.get('mdlen', 20)
        mddump = crest_cfg.get('mddump', 100)
        mdtemp = crest_cfg.get('mdtemp', 400)
        enable_nci = crest_cfg.get('nci', False)
        enable_entropy = crest_cfg.get('entropy', False)
        cbonds = crest_cfg.get('cbonds', [])
        timeout = crest_cfg.get('timeout_seconds', 7200)
        
        crest_input = output_dir / "crest_input.xyz"
        shutil.copy2(xyz_file, crest_input)
        
        charge = self.config['conformer_search']['gaussian'].get('charge', 1)
        mult = self.config['conformer_search']['gaussian'].get('multiplicity', 1)
        uhf = max(mult - 1, 0)
        
        cmd = [crest_exec, str(crest_input.name)]
        if gfn in {'gfn0', 'gfn1', 'gfn2', 'gfnff'}:
            if gfn == 'gfn2':
                cmd.append('--gfn2')
            elif gfn == 'gfn1':
                cmd.append('--gfn1')
            elif gfn == 'gfn0':
                cmd.append('--gfn0')
            elif gfn == 'gfnff':
                cmd.append('--gfnff')
        if solvent:
            cmd.extend(['--alpb', solvent])
        cmd.extend(['--T', str(threads)])
        cmd.extend(['--ewin', str(ewin)])
        cmd.extend(['--rthr', str(rthr)])
        cmd.extend(['--ethr', str(ethr)])
        cmd.extend(['--mdlen', str(mdlen)])
        cmd.extend(['--mddump', str(mddump)])
        cmd.extend(['--mdtemp', str(mdtemp)])
        cmd.extend(['--chrg', str(charge)])
        if uhf > 0:
            cmd.extend(['--uhf', str(uhf)])
        if enable_nci:
            cmd.append('--nci')
        if enable_entropy:
            cmd.append('--entropy')
        if cbonds:
            for bond in cbonds:
                cmd.extend(['--cbonds', str(bond)])
        
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = str(threads)
        env['OPENBLAS_NUM_THREADS'] = '1'
        
        if self.logger:
            self.logger.info(f"Using CREST for conformer generation ({gfn.upper()})...")
            self.logger.debug(f"Command: {' '.join(cmd)}")
        else:
            print(f"  Using CREST for conformer generation ({gfn.upper()})...")
            print(f"    Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=output_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            if self.logger:
                self.logger.error("CREST run timed out")
            else:
                print("    [ERROR] CREST run timed out")
            return 0
        
        if result.returncode != 0:
            error_log = output_dir / "crest_error.log"
            try:
                with open(error_log, 'w') as f:
                    f.write("STDOUT:\n")
                    f.write(result.stdout)
                    f.write("\n\nSTDERR:\n")
                    f.write(result.stderr)
                if self.logger:
                    self.logger.error(f"CREST failed (see {error_log.name} for details)")
                else:
                    print(f"    [ERROR] CREST failed (see {error_log.name} for details)")
            except IOError:
                if self.logger:
                    self.logger.error("CREST failed (cannot write error log)")
                else:
                    print("    [ERROR] CREST failed (cannot write error log)")
            if self.logger:
                self.logger.debug(result.stderr.strip() or result.stdout.strip())
            else:
                print(result.stderr.strip() or result.stdout.strip())
            return 0
        
        energies_file_candidates = [
            output_dir / "crest.energies",
            output_dir / "crest_energies.dat",
            output_dir / "crest_conformers.energies",
        ]
        energies_file = next((path for path in energies_file_candidates if path.exists()), None)
        if energies_file is None:
            if self.logger:
                self.logger.error("CREST energies file not found")
            else:
                print("    [ERROR] CREST energies file not found")
            return 0
        
        conformer_file_candidates = [
            output_dir / "crest_conformers.xyz",
            output_dir / "crest_cluster.xyz",
            output_dir / "crest_best.xyz",
        ]
        conformer_file = next((path for path in conformer_file_candidates if path.exists()), None)
        if conformer_file is None:
            if self.logger:
                self.logger.error("CREST conformer XYZ not found")
            else:
                print("    [ERROR] CREST conformer XYZ not found")
            return 0
        
        structures = self._read_multi_xyz(conformer_file)
        if not structures:
            if self.logger:
                self.logger.error("CREST XYZ parsing failed")
            else:
                print("    [ERROR] CREST XYZ parsing failed")
            return 0
        
        energies = []
        with open(energies_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    idx = int(parts[0])
                    energy = float(parts[1])
                except ValueError:
                    continue
                energies.append((idx, energy))
        
        if not energies:
            if self.logger:
                self.logger.error("CREST energies file is empty")
            else:
                print("    [ERROR] CREST energies file is empty")
            return 0
        
        energies.sort(key=lambda x: x[1])
        base_energy = energies[0][1]
        
        structures_dict = {i + 1: data for i, data in enumerate(structures)}
        
        num_selected = min(num_conf, len(energies))
        if self.logger:
            self.logger.info(f"CREST generated {len(structures)} conformers, selecting top {num_selected}")
        else:
            print(f"    CREST generated {len(structures)} conformers, selecting top {num_selected}")
        
        written = 0
        best_atoms = None
        best_coords = None
        best_comment = ""
        for rank, (conf_idx, energy) in enumerate(energies[:num_selected], start=1):
            if conf_idx not in structures_dict:
                continue
            atoms, coords, comment = structures_dict[conf_idx]
            rel_kcal = (energy - base_energy) * HARTREE_TO_KCAL
            conf_file = output_dir / f"conf_{rank}.xyz"
            comment_line = (
                f"CREST conformer #{conf_idx} | E={energy:.6f} Eh | dE={rel_kcal:.2f} kcal/mol"
            )
            self.write_xyz_file(conf_file, atoms, coords, comment_line)
            written += 1
            if rank == 1:
                best_atoms = [a for a in atoms]
                best_coords = [list(c) for c in coords]
                best_comment = comment_line
        
        if written == 0:
            if self.logger:
                self.logger.error("CREST did not produce usable conformers")
            else:
                print("    [ERROR] CREST did not produce usable conformers")
            return 0
        
        self.latest_best_conformer = {
            'atoms': best_atoms,
            'coords': best_coords,
            'comment': best_comment,
            'energy': energies[0][1],
            'relative_kcal': 0.0,
            'label': f"conf_{energies[0][0]}"
        }
        
        if self.logger:
            self.logger.success(f"Successfully wrote {written} CREST conformers")
        else:
            print(f"    [OK] Successfully wrote {written} CREST conformers")
        return written
    
    def generate_conformers(self, xyz_file: Path, output_dir: Path, num_conf: int, 
                          execute_crest: bool = True) -> int:
        """Generate conformers using RDKit, OpenBabel, or CREST
        
        Args:
            xyz_file: Input XYZ file
            output_dir: Output directory for conformers
            num_conf: Number of conformers to generate
            execute_crest: If False, skip CREST execution (for SLURM mode)
        """
        if not self.config['conformer_search']['enabled']:
            return 0
        
        backend = self.get_conformer_backend()
        allow_fallback = self.config['conformer_search'].get('allow_fallback', True)
        num_generated = 0
        self.latest_best_conformer = None
        
        def run_backend(name: str) -> int:
            if name == 'rdkit':
                if not RDKIT_AVAILABLE:
                    if self.logger:
                        self.logger.warning("RDKit not available on this system")
                    else:
                        print("  [WARN] RDKit not available on this system")
                    return 0
                return self.generate_conformers_rdkit(xyz_file, output_dir, num_conf)
            if name == 'openbabel':
                return self.generate_conformers_openbabel(xyz_file, output_dir, num_conf)
            if name == 'crest':
                if not execute_crest:
                    if self.logger:
                        self.logger.info("CREST execution skipped (will run in SLURM)")
                    else:
                        print("  [INFO] CREST execution skipped (will run in SLURM)")
                    return 0
                return self.generate_conformers_crest(xyz_file, output_dir, num_conf)
            return 0
        
        num_generated = run_backend(backend)
        
        if num_generated == 0 and allow_fallback:
            fallback_chain = []
            if backend == 'crest':
                fallback_chain = ['rdkit', 'openbabel']
            elif backend == 'rdkit':
                fallback_chain = ['openbabel']
            elif backend == 'openbabel':
                fallback_chain = ['rdkit']
            for candidate in fallback_chain:
                if candidate == backend:
                    continue
                if self.logger:
                    self.logger.info(f"Falling back to {candidate.upper()} conformer generator...")
                else:
                    print(f"  [INFO] Falling back to {candidate.upper()} conformer generator...")
                num_generated = run_backend(candidate)
                if num_generated > 0:
                    backend = candidate
                    break
        
        self.last_conformer_backend = backend if num_generated > 0 else None
        return num_generated
    
    def create_gaussian_input(self, calc_name: str, calc_config: Dict, 
                            mol_name: str, atoms: List[str], 
                            coords: List[List[float]], 
                            use_coords: bool = True) -> str:
        """Create Gaussian input file content"""
        lines = []
        
        # Checkpoint file
        lines.append(f"%chk=mol_{calc_name}.chk")
        
        # Old checkpoint if specified
        if calc_config['gaussian'].get('oldchk'):
            if calc_config['gaussian']['oldchk'] == 'conformer':
                lines.append("%oldchk=../s0_conf_search/best_conformer.chk")
            else:
                parent = calc_config['gaussian']['oldchk']
                lines.append(f"%oldchk=../{parent}/mol_{parent}.chk")
        
        # Resources
        lines.append(f"%nprocshared={calc_config['slurm']['cores']}")
        lines.append(f"%mem={calc_config['slurm']['memory']}")
        
        # Route section
        route = calc_config['gaussian']['route']
        if calc_config['gaussian'].get('geom') == 'AllCheck':
            route += " Geom=AllCheck"
        elif calc_config['gaussian'].get('geom') == 'Checkpoint':
            route += " Geom=Checkpoint"
        if calc_config['gaussian'].get('guess') == 'Read':
            route += " Guess=Read"
        elif calc_config['gaussian'].get('guess') == 'Mix':
            route += " Guess=Mix"
        lines.append(route)
        
        # Title
        lines.append("")
        lines.append(f"{mol_name} - {calc_name}")
        lines.append("")
        
        # Charge and multiplicity
        charge = calc_config['gaussian']['charge']
        mult = calc_config['gaussian']['multiplicity']
        lines.append(f"{charge} {mult}")
        
        # Coordinates (only if not using AllCheck or Checkpoint)
        geom_type = calc_config['gaussian'].get('geom')
        if use_coords and geom_type != 'AllCheck' and geom_type != 'Checkpoint':
            for atom, coord in zip(atoms, coords):
                lines.append(f"{atom:2s}  {coord[0]:12.6f}  {coord[1]:12.6f}  {coord[2]:12.6f}")
        
        lines.append("")
        lines.append("")
        
        return '\n'.join(lines)
    
    def create_slurm_script(self, calc_name: str, calc_config: Dict, 
                          mol_name: str, depends_on: Optional[str] = None) -> str:
        """Create SLURM submission script"""
        slurm = calc_config['slurm']
        global_slurm = self.config['slurm']
        
        # Generate header
        header = self.get_bash_header(
            script_name=f"SLURM Job Script - {calc_name}",
            description=f"SLURM submission script for {calc_name} calculation"
        )
        
        lines = []
        lines.append(header.rstrip())  # Add header without extra newline
        lines.append(f"#SBATCH --job-name={mol_name[:8]}_{calc_name[:8]}")
        lines.append(f"#SBATCH --partition={slurm['partition']}")
        lines.append(f"#SBATCH --time={slurm['time']}")
        lines.append(f"#SBATCH --nodes=1")
        lines.append(f"#SBATCH --ntasks={slurm['cores']}")
        lines.append(f"#SBATCH --mem={slurm['memory']}")
        lines.append(f"#SBATCH --output=%x_%j.out")
        lines.append(f"#SBATCH --error=%x_%j.err")
        lines.append(f"#SBATCH --mail-type={global_slurm['mail_type']}")
        lines.append(f"#SBATCH --mail-user={global_slurm['email']}")
        
        if global_slurm.get('account'):
            lines.append(f"#SBATCH --account={global_slurm['account']}")
        
        if global_slurm.get('constraint'):
            lines.append(f"#SBATCH --constraint={global_slurm['constraint']}")
        
        if global_slurm.get('exclude_nodes'):
            lines.append(f"#SBATCH --exclude={global_slurm['exclude_nodes']}")
        
        lines.append("")
        lines.append("# Setup environment")
        if global_slurm.get('gaussian_module'):
            lines.append(f"module load {global_slurm['gaussian_module']}")
        else:
            lines.append("export PATH=/usr/bin:$PATH")
            lines.append(f"export g16root={global_slurm['gaussian_root']}")
            lines.append('source "$g16root/g16/bsd/g16.profile"')
        lines.append("export GAUSS_SCRDIR=/scratch/$USER/$SLURM_JOB_ID")
        lines.append("mkdir -p $GAUSS_SCRDIR")
        lines.append("")
        lines.append(f"INPUT_FILE=mol_{calc_name}.com")
        lines.append(f"OUTPUT_FILE=mol_{calc_name}.log")
        lines.append("")
        lines.append('echo "Starting Gaussian calculation: $INPUT_FILE"')
        lines.append('echo "Job ID: $SLURM_JOB_ID"')
        lines.append('echo "Start time: $(date)"')
        lines.append("")
        lines.append("# Run Gaussian")
        if global_slurm.get('gaussian_module'):
            lines.append("g16 < $INPUT_FILE > $OUTPUT_FILE 2>&1")
        else:
            lines.append(f"{global_slurm['gaussian_root']}/g16/g16 < $INPUT_FILE > $OUTPUT_FILE 2>&1")
        lines.append("status=$?")
        lines.append("")
        lines.append("# Check completion")
        lines.append('if [[ $status -eq 0 ]] && grep -q "Normal termination" $OUTPUT_FILE; then')
        lines.append('    echo "SUCCESS" > status.txt')
        lines.append('    echo "Calculation completed successfully at $(date)"')
        lines.append('    ')
        lines.append('    # Copy checkpoint from scratch if exists')
        lines.append(f'    if [[ -f "$GAUSS_SCRDIR/mol_{calc_name}.chk" ]]; then')
        lines.append(f'        cp "$GAUSS_SCRDIR/mol_{calc_name}.chk" mol_{calc_name}.chk')
        lines.append('    fi')
        lines.append('    ')
        # Generate formatted checkpoint only if enabled
        if self.config['analysis'].get('generate_fchk', False):
            lines.append('    # Generate formatted checkpoint')
            lines.append(f'    if [[ -f mol_{calc_name}.chk ]]; then')
            lines.append(f'        formchk mol_{calc_name}.chk mol_{calc_name}.fchk 2>/dev/null || true')
            lines.append('    fi')
        lines.append('else')
        lines.append('    echo "FAILED" > status.txt')
        lines.append('    echo "Calculation failed at $(date)"')
        lines.append('    exit 1')
        lines.append('fi')
        lines.append("")
        lines.append("# Cleanup")
        lines.append("rm -rf $GAUSS_SCRDIR")
        lines.append("")
        lines.append("exit 0")
        
        return '\n'.join(lines)
    
    def _write_best_conformer_files(self, conf_dir: Path, comment: str,
                                    best_atoms: List[str], best_coords: List[List[float]],
                                    label: str, energy: Optional[float]):
        """Create helper artifacts for best conformer when no Gaussian refinement is used"""
        if best_atoms is None or best_coords is None:
            return
        best_xyz = conf_dir / "best_conformer.xyz"
        self.write_xyz_file(best_xyz, best_atoms, best_coords, comment or "Best conformer")
        with open(conf_dir / "best_conformer.txt", 'w') as f:
            f.write(f"{label or 'best_conformer'}\n")
            if energy is not None:
                f.write(f"{energy}\n")
            else:
                f.write("0.0\n")
    
    def create_conformer_scripts(self, mol_dir: Path, mol_name: str, 
                               atoms: List[str], num_conf: int) -> Dict[str, Optional[object]]:
        """Create conformer search scripts"""
        conf_dir = mol_dir / "s0_conf_search"
        conf_dir.mkdir(exist_ok=True)
        conf_config = self.config['conformer_search']
        gaussian_settings = conf_config['gaussian']
        backend, crest_cfg, crest_reoptimize = self._get_crest_config()
        
        metadata = {
            'has_jobs': True,
            'needs_checkpoint': True,
            'best_atoms': None,
            'best_coords': None
        }
        
        # Handle CREST with PM7 reoptimization (CREST runs in SLURM)
        if backend == 'crest' and crest_reoptimize:
            # Use custom route if specified
            custom_route = crest_cfg.get('gaussian_route')
            if custom_route:
                gaussian_settings = copy.deepcopy(gaussian_settings)
                gaussian_settings['route'] = custom_route
            
            # CREST will run in SLURM, just prepare input file
            if self.logger:
                self.logger.info("CREST will run in SLURM job (reoptimize_pm7: true)")
                self.logger.info("Preparing CREST input file...")
            else:
                print(f"  [INFO] CREST will run in SLURM job (reoptimize_pm7: true)")
                print(f"  [INFO] Preparing CREST input file...")
            crest_input = conf_dir / "crest_input.xyz"
            shutil.copy2(mol_dir / "original.xyz", crest_input)
            num_generated = 0  # Will be determined after CREST runs in SLURM
            # Don't create manual conformer - CREST will generate them
        else:
            # Try to generate conformers now (for CREST without PM7 or other backends)
            # Don't execute CREST if it will run in SLURM
            execute_crest = not (backend == 'crest' and crest_reoptimize)
            num_generated = self.generate_conformers(
                mol_dir / "original.xyz", conf_dir, num_conf, execute_crest=execute_crest
            )
        
        # If conformer generation fails (and it's not CREST in SLURM), create manual conformer
        if num_generated == 0 and not (backend == 'crest' and crest_reoptimize):
            if self.logger:
                self.logger.warning("Conformer generation failed, creating manual conformer from original geometry")
            else:
                print(f"  [WARNING] Conformer generation failed, creating manual conformer from original geometry")
            # Create single conformer directory with original geometry
            conf_subdir = conf_dir / "conf_1"
            conf_subdir.mkdir(exist_ok=True)
            
            # Copy original XYZ as conformer
            shutil.copy2(mol_dir / "original.xyz", conf_subdir / "conf_1.xyz")
            
            # Read original coordinates properly and create Gaussian input
            original_atoms, original_coords = self.read_xyz_file(mol_dir / "original.xyz")
            input_content = self.create_gaussian_input(
                "conf_1", 
                {'gaussian': gaussian_settings, 'slurm': conf_config['slurm']},
                mol_name, original_atoms, original_coords
            )
            
            with open(conf_subdir / "mol_conf_1.com", 'w') as f:
                f.write(input_content)
            
            num_generated = 1
            print(f"  [OK] Created 1 manual conformer (will be optimized with {gaussian_settings['route']})")
        else:
            if backend == 'crest' and not crest_reoptimize:
                best_atoms = None
                best_coords = None
                best_comment = ""
                best_label = "best_conformer"
                best_energy = None
                if self.latest_best_conformer:
                    best_atoms = self.latest_best_conformer.get('atoms')
                    best_coords = self.latest_best_conformer.get('coords')
                    best_comment = self.latest_best_conformer.get('comment', "")
                    best_energy = self.latest_best_conformer.get('energy')
                    best_label = self.latest_best_conformer.get('label', best_label)
                if best_atoms is None or best_coords is None:
                    first_conf = conf_dir / "conf_1.xyz"
                    if first_conf.exists():
                        best_atoms, best_coords = self.read_xyz_file(first_conf)
                        best_comment = "Best conformer (fallback conf_1.xyz)"
                        best_label = 'conf_1'
                metadata.update({
                    'has_jobs': False,
                    'needs_checkpoint': False,
                    'best_atoms': best_atoms,
                    'best_coords': best_coords
                })
                self._write_best_conformer_files(conf_dir, best_comment, best_atoms, best_coords, best_label, best_energy)
                return metadata
            # Create input for each conformer
            for conf_file in conf_dir.glob('conf_*.xyz'):
                conf_num = conf_file.stem.replace('conf_', '')
                conf_subdir = conf_dir / f"conf_{conf_num}"
                conf_subdir.mkdir(exist_ok=True)
                
                # Read conformer coordinates
                conf_atoms, conf_coords = self.read_xyz_file(conf_file)
                
                # Create Gaussian input
                input_content = self.create_gaussian_input(
                    f"conf_{conf_num}", 
                    {'gaussian': gaussian_settings, 'slurm': conf_config['slurm']},
                    mol_name, conf_atoms, conf_coords
                )
                
                with open(conf_subdir / f"mol_conf_{conf_num}.com", 'w') as f:
                    f.write(input_content)
                
                # Move XYZ file
                shutil.move(str(conf_file), str(conf_subdir))
        
        # Create SLURM scripts
        if backend == 'crest' and crest_reoptimize:
            # Create CREST SLURM script first, then array for PM7
            self.create_crest_slurm_script(conf_dir, mol_name, num_conf)
            # Create script to prepare conformer inputs after CREST completes
            self.create_prepare_conformers_script(conf_dir, mol_name, num_conf)
            # Array size will be determined dynamically after CREST runs
            # Use num_conf as maximum estimate for array size
            self.create_conformer_slurm_array(conf_dir, mol_name, num_conf)
        else:
            # Create SLURM array script for PM7 optimization
            if num_generated > 0:
                self.create_conformer_slurm_array(conf_dir, mol_name, num_generated)
            else:
                # Fallback: create array for at least 1 conformer
                self.create_conformer_slurm_array(conf_dir, mol_name, 1)
        
        # Create analysis script
        self.create_conformer_analysis_script(conf_dir)
        
        metadata['has_jobs'] = True
        metadata['needs_checkpoint'] = True
        return metadata
    
    def create_conformer_slurm_array(self, conf_dir: Path, mol_name: str, num_conf: int):
        """Create SLURM array script for conformer search"""
        conf_config = self.config['conformer_search']
        global_slurm = self.config['slurm']
        
        # Generate header
        header = self.get_bash_header(
            script_name="Conformer Search Array Job",
            description=f"SLURM array job for optimizing {num_conf} conformers"
        )
        
        # Build exclude_nodes directive if configured
        exclude_directive = ""
        if global_slurm.get('exclude_nodes'):
            exclude_directive = f"#SBATCH --exclude={global_slurm['exclude_nodes']}\n"
        
        script = f"""{header}#SBATCH --job-name={mol_name[:8]}_conf
#SBATCH --partition={conf_config['slurm']['partition']}
#SBATCH --time={conf_config['slurm']['time']}
#SBATCH --nodes=1
#SBATCH --ntasks={conf_config['slurm']['cores']}
#SBATCH --mem={conf_config['slurm']['memory']}
#SBATCH --array=1-{num_conf}
#SBATCH --output=conf_%a/conf_%a_%j.out
#SBATCH --error=conf_%a/conf_%a_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user={global_slurm['email']}
{exclude_directive}

# Setup environment
"""
        if global_slurm.get('gaussian_module'):
            script += f"""module load {global_slurm['gaussian_module']}
"""
        else:
            script += f"""export PATH=/usr/bin:$PATH
export g16root={global_slurm['gaussian_root']}
source "$g16root/g16/bsd/g16.profile"
"""
        script += f"""export GAUSS_SCRDIR=/scratch/$USER/$SLURM_JOB_ID
mkdir -p $GAUSS_SCRDIR

CONF_ID=$SLURM_ARRAY_TASK_ID
CONF_DIR="conf_${{CONF_ID}}"
INPUT_FILE="mol_conf_${{CONF_ID}}.com"

# Check conformer count file (in current directory, not parent)
COUNT_FILE="conformer_count.txt"
AVAILABLE_CONFS=0

# Validate conformer_count.txt exists and is valid
if [[ ! -f "$COUNT_FILE" ]]; then
    echo "ERROR: conformer_count.txt not found in current directory"
    echo "  This file should be created by prepare_conformers.sh after CREST completes"
    echo "  Current directory: $(pwd)"
    echo "  Expected file: $COUNT_FILE"
    exit 1
fi

AVAILABLE_CONFS=$(cat "$COUNT_FILE" 2>/dev/null | tr -d '[:space:]')

# Validate it's a number and greater than 0
if [[ -z "$AVAILABLE_CONFS" ]] || [[ ! "$AVAILABLE_CONFS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Invalid conformer count in conformer_count.txt: '$AVAILABLE_CONFS'"
    echo "  Expected a positive integer, got: '$AVAILABLE_CONFS'"
    echo "  File contents: $(cat "$COUNT_FILE" 2>/dev/null | head -1)"
    exit 1
fi

if [[ "$AVAILABLE_CONFS" -eq 0 ]]; then
    echo "ERROR: conformer_count.txt contains zero (0)"
    echo "  No conformers available for optimization"
    echo "  Check CREST output and prepare_conformers.sh execution"
    exit 1
fi

# Skip if this conformer ID exceeds available count
if [[ "$CONF_ID" -gt "$AVAILABLE_CONFS" ]]; then
    echo "SKIP: Conformer $CONF_ID exceeds available ($AVAILABLE_CONFS)."
    exit 0
fi

# Check if directory and input file exist
if [[ ! -d "$CONF_DIR" ]]; then
    echo "SKIP: Directory $CONF_DIR does not exist yet (conformer pruned)."
    exit 0
fi

cd "$CONF_DIR" || exit 1

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "SKIP: Input file $INPUT_FILE not found (conformer pruned)."
    exit 0
fi

echo "Starting conformer optimization: $INPUT_FILE"
echo "Conformer: $CONF_ID of {num_conf}"
echo "Start time: $(date)"

# Run Gaussian
"""
        if global_slurm.get('gaussian_module'):
            script += """g16 < "$INPUT_FILE" > "${INPUT_FILE%.com}.log" 2>&1"""
        else:
            script += f"""{global_slurm['gaussian_root']}/g16/g16 < "$INPUT_FILE" > "${{INPUT_FILE%.com}}.log" 2>&1"""
        script += """
status=$?

if [[ $status -eq 0 ]] && grep -q "Normal termination" "${{INPUT_FILE%.com}}.log"; then
    echo "SUCCESS" > status.txt
    if [[ -f "$GAUSS_SCRDIR/mol_conf_${{CONF_ID}}.chk" ]]; then
        cp "$GAUSS_SCRDIR/mol_conf_${{CONF_ID}}.chk" mol_conf_${{CONF_ID}}.chk
    fi"""
        
        # Add formchk generation if enabled
        if self.config['analysis'].get('generate_fchk', False):
            script += """
    # Generate formatted checkpoint
    if [[ -f mol_conf_${{CONF_ID}}.chk ]]; then
        formchk mol_conf_${{CONF_ID}}.chk mol_conf_${{CONF_ID}}.fchk 2>/dev/null || true
    fi"""
        
        script += """
else
    echo "FAILED" > status.txt
    exit 1
fi

rm -rf $GAUSS_SCRDIR
"""
        
        # Fix bash variable syntax
        script = self.fix_bash_variables(script)
        
        with open(conf_dir / "run_conformers.slurm", 'w') as f:
            f.write(script)
        os.chmod(conf_dir / "run_conformers.slurm", 0o755)
    
    def create_crest_slurm_script(self, conf_dir: Path, mol_name: str, num_conf: int):
        """Create SLURM script to run CREST conformer search"""
        conf_config = self.config['conformer_search']
        crest_cfg = conf_config.get('crest', {})
        global_slurm = self.config['slurm']
        
        crest_exec = crest_cfg.get('executable', 'crest')
        xtb_exec = crest_cfg.get('xtb_executable', 'xtb')
        threads = crest_cfg.get('threads', min(cpu_count(), 16))
        gfn = crest_cfg.get('gfn', 'gfn2').lower()
        solvent = crest_cfg.get('solvent')
        ewin = crest_cfg.get('ewin', 6.0)
        rthr = crest_cfg.get('rthr', 0.125)
        ethr = crest_cfg.get('ethr', 0.05)
        mdlen = crest_cfg.get('mdlen', 20)
        mddump = crest_cfg.get('mddump', 100)
        mdtemp = crest_cfg.get('mdtemp', 400)
        enable_nci = crest_cfg.get('nci', False)
        enable_entropy = crest_cfg.get('entropy', False)
        cbonds = crest_cfg.get('cbonds', [])
        
        charge = conf_config['gaussian'].get('charge', 1)
        mult = conf_config['gaussian'].get('multiplicity', 1)
        uhf = max(mult - 1, 0)
        
        # Build CREST command
        cmd_parts = [crest_exec, 'crest_input.xyz']
        if gfn in {'gfn0', 'gfn1', 'gfn2', 'gfnff'}:
            if gfn == 'gfn2':
                cmd_parts.append('--gfn2')
            elif gfn == 'gfn1':
                cmd_parts.append('--gfn1')
            elif gfn == 'gfn0':
                cmd_parts.append('--gfn0')
            elif gfn == 'gfnff':
                cmd_parts.append('--gfnff')
        if solvent:
            cmd_parts.extend(['--alpb', solvent])
        cmd_parts.extend(['--T', str(threads)])
        cmd_parts.extend(['--ewin', str(ewin)])
        cmd_parts.extend(['--rthr', str(rthr)])
        cmd_parts.extend(['--ethr', str(ethr)])
        cmd_parts.extend(['--mdlen', str(mdlen)])
        cmd_parts.extend(['--mddump', str(mddump)])
        cmd_parts.extend(['--mdtemp', str(mdtemp)])
        cmd_parts.extend(['--chrg', str(charge)])
        if uhf > 0:
            cmd_parts.extend(['--uhf', str(uhf)])
        if enable_nci:
            cmd_parts.append('--nci')
        if enable_entropy:
            cmd_parts.append('--entropy')
        if cbonds:
            for bond in cbonds:
                cmd_parts.extend(['--cbonds', str(bond)])
        
        # Add xTB path if specified
        if xtb_exec and xtb_exec != 'xtb':
            cmd_parts.extend(['-xnam', xtb_exec])
        
        crest_cmd = ' '.join(cmd_parts)
        
        # Generate header
        header = self.get_bash_header(
            script_name="CREST Conformer Search",
            description=f"Run CREST to generate conformers for {mol_name}"
        )
        
        # Build exclude_nodes directive if configured
        exclude_directive = ""
        if global_slurm.get('exclude_nodes'):
            exclude_directive = f"#SBATCH --exclude={global_slurm['exclude_nodes']}\n"
        
        script = f"""{header}#SBATCH --job-name={mol_name[:8]}_crest
#SBATCH --partition={conf_config['slurm']['partition']}
#SBATCH --time={conf_config['slurm']['time']}
#SBATCH --nodes=1
#SBATCH --ntasks={threads}
#SBATCH --mem={conf_config['slurm']['memory']}
#SBATCH --output=crest_%j.out
#SBATCH --error=crest_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user={global_slurm['email']}
{exclude_directive}
# Trap signals to handle job cancellation/timeout gracefully
cleanup_on_signal() {{
    echo ""
    echo "=========================================="
    echo "CREST job interrupted or killed"
    echo "Signal received at: $(date)"
    echo "=========================================="
    echo ""
    echo "KILLED" > crest_status.txt
    exit 130
}}

trap cleanup_on_signal SIGTERM SIGINT

# Setup environment
export OMP_NUM_THREADS={threads}
export OMP_STACKSIZE=1G
export OPENBLAS_NUM_THREADS=1

echo "=========================================="
echo "CREST Conformer Search"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# Run CREST
{crest_cmd} > crest.out 2>&1
CREST_STATUS=$?

if [[ $CREST_STATUS -eq 0 ]]; then
    echo ""
    echo "CREST completed successfully"
    echo "End time: $(date)"
    
    # Check for output files
    if [[ -f "crest_conformers.xyz" ]] || [[ -f "crest.energies" ]]; then
        echo "CREST output files found"
        echo "SUCCESS" > crest_status.txt

        if [[ -x "./prepare_conformers.sh" ]]; then
            echo ""
            echo "Preparing Gaussian inputs from CREST structures..."
            ./prepare_conformers.sh
            PREP_STATUS=$?
            if [[ $PREP_STATUS -ne 0 ]]; then
                echo "ERROR: prepare_conformers.sh failed with status $PREP_STATUS"
                echo "FAILED" > crest_status.txt
                exit 1
            fi
            
            # Verify preparation was successful
            if [[ ! -f "conformer_count.txt" ]]; then
                echo "ERROR: conformer_count.txt was not created by prepare_conformers.sh"
                echo "FAILED" > crest_status.txt
                exit 1
            fi
            
            ACTUAL_COUNT=$(cat conformer_count.txt | tr -d '[:space:]')
            if [[ -z "$ACTUAL_COUNT" ]] || [[ ! "$ACTUAL_COUNT" =~ ^[0-9]+$ ]] || [[ "$ACTUAL_COUNT" -eq 0 ]]; then
                echo "ERROR: Invalid or zero conformer count: '$ACTUAL_COUNT'"
                echo "FAILED" > crest_status.txt
                exit 1
            fi
            
            echo "[OK] Successfully prepared $ACTUAL_COUNT conformer input files"
        else
            echo "ERROR: prepare_conformers.sh not found or not executable"
            echo "FAILED" > crest_status.txt
            exit 1
        fi
    else
        echo "WARNING: CREST output files not found"
        echo "FAILED" > crest_status.txt
        exit 1
    fi
else
    echo ""
    echo "CREST failed with exit code: $CREST_STATUS"
    echo "End time: $(date)"
    echo "FAILED" > crest_status.txt
    exit 1
fi

exit 0
"""
        
        # Fix bash variable syntax
        script = self.fix_bash_variables(script)
        
        with open(conf_dir / "run_crest.slurm", 'w') as f:
            f.write(script)
        os.chmod(conf_dir / "run_crest.slurm", 0o755)
    
    def create_prepare_conformers_script(self, conf_dir: Path, mol_name: str, num_conf: int):
        """Create script to prepare Gaussian input files from CREST output"""
        conf_config = self.config['conformer_search']
        gaussian_settings = conf_config['gaussian']
        crest_cfg = conf_config.get('crest', {})
        
        # Use custom route if specified
        custom_route = crest_cfg.get('gaussian_route')
        if custom_route:
            gaussian_settings = copy.deepcopy(gaussian_settings)
            gaussian_settings['route'] = custom_route
        
        # Generate header
        header = self.get_bash_header(
            script_name="Prepare Conformers",
            description="Prepares Gaussian input files from CREST conformer output"
        )
        
        # Create Python script to generate inputs (more reliable than bash parsing)
        python_script_content = f'''#!/usr/bin/env python3
import sys
import shutil
from pathlib import Path

# Configuration
mol_name = "{mol_name}"
num_conf = {num_conf}
gaussian_route = "{gaussian_settings['route']}"
charge = {gaussian_settings['charge']}
multiplicity = {gaussian_settings['multiplicity']}
cores = {conf_config['slurm']['cores']}
memory = "{conf_config['slurm']['memory']}"

# Single route line: YAML route usually includes "#p ..."; do not prepend "#p" again
_route_raw = gaussian_route.strip()
if _route_raw.startswith("#p"):
    route_line = _route_raw
elif _route_raw.startswith("#"):
    route_line = _route_raw
else:
    route_line = "#p " + _route_raw

# Find CREST output file (prioritize files with more structures)
crest_files = ["crest_conformers.xyz", "crest_rotamers.xyz", "crest_cluster.xyz", "crest_best.xyz"]

def count_structures(xyz_path):
    """Count number of structures in multi-XYZ file"""
    if not xyz_path.exists():
        return 0
    count = 0
    with open(xyz_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Check if line starts with a number (atom count)
            # Atom count lines are typically just a number, no decimals
            parts = line.split()
            if parts and parts[0].isdigit():
                # This looks like an atom count line (first line of a structure)
                try:
                    atom_count = int(parts[0])
                    # Valid atom count should be > 0 and reasonable (< 10000)
                    if atom_count > 0 and atom_count < 10000:
                        count += 1
                except (ValueError, IndexError):
                    pass
    return count

# Find file with most structures
crest_file = None
max_structures = 0
for f in crest_files:
    path = Path(f)
    if path.exists():
        struct_count = count_structures(path)
        print(f"  Found {{f}} with {{struct_count}} structures")
        if struct_count > max_structures:
            max_structures = struct_count
            crest_file = f

if not crest_file:
    print("ERROR: CREST output file not found!")
    print("  Expected one of: " + ", ".join(crest_files))
    sys.exit(1)

print(f"Using {{crest_file}} with {{max_structures}} structures")

print(f"Reading conformers from: {{crest_file}}")

# Clean previous conformer directories to avoid stale data
for existing in sorted(Path('.').glob('conf_*')):
    if existing.is_dir():
        shutil.rmtree(existing)

# Read XYZ file - improved parsing
def read_multi_xyz(xyz_path):
    """Parse multi-structure XYZ file produced by CREST"""
    structures = []
    if not xyz_path.exists():
        return structures
    
    with open(xyz_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        # Skip empty lines
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        # Try to read atom count
        try:
            atom_count = int(line)
        except ValueError:
            i += 1
            continue
        
        # Read comment line
        if i + 1 >= len(lines):
            break
        comment = lines[i+1].strip()
        
        # Read atoms
        atoms = []
        coords = []
        start_idx = i + 2
        end_idx = start_idx + atom_count
        
        if end_idx > len(lines):
            print(f"WARNING: Incomplete structure at line {{i+1}}, expected {{atom_count}} atoms but file ends")
            break
        
        for j in range(start_idx, end_idx):
            if j >= len(lines):
                break
            parts = lines[j].split()
            if len(parts) < 4:
                continue
            try:
                atoms.append(parts[0])
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except (ValueError, IndexError) as e:
                print(f"WARNING: Skipping malformed line {{j+1}}: {{e}}")
                continue
        
        # Validate we got the right number of atoms
        if len(atoms) == atom_count and len(coords) == atom_count:
            structures.append((atoms, coords, comment))
        else:
            print(f"WARNING: Structure at line {{i+1}} has {{len(atoms)}} atoms, expected {{atom_count}}")
        
        # Move to next structure
        i = end_idx
    
    return structures

conformers = read_multi_xyz(Path(crest_file))

if not conformers:
    print("ERROR: No conformers found in CREST output file!")
    print("  Check that the file is a valid multi-XYZ format")
    sys.exit(1)

print(f"Found {{len(conformers)}} conformers")
selected = conformers[:num_conf]
actual_count = len(selected)

if actual_count == 0:
    print("ERROR: No conformers selected for PM7 optimization!")
    sys.exit(1)

if actual_count < num_conf:
    print(f"WARNING: Only {{actual_count}} conformers available, requested {{num_conf}}")

count_file = Path("conformer_count.txt")
count_file.write_text(str(actual_count))
print(f"Selected {{actual_count}} conformers for PM7 optimization")

# Generate Gaussian input files
created_count = 0
for conf_num, (atoms, coords, comment) in enumerate(selected, 1):
    conf_dir = Path(f"conf_{{conf_num}}")
    conf_dir.mkdir(exist_ok=True)
    
    input_file = conf_dir / f"mol_conf_{{conf_num}}.com"
    try:
        with open(input_file, 'w') as f:
            f.write(f"%chk=mol_conf_{{conf_num}}.chk\\n")
            f.write(f"%mem={{memory}}\\n")
            f.write(f"%nprocshared={{cores}}\\n")
            f.write(f"{{route_line}}\\n")
            f.write(f"\\n")
            f.write(f"Conformer {{conf_num}} from CREST\\n")
            f.write(f"\\n")
            f.write(f"{{charge}} {{multiplicity}}\\n")
            for atom, coord in zip(atoms, coords):
                f.write(f"{{atom:>3}} {{coord[0]:>15.10f}} {{coord[1]:>15.10f}} {{coord[2]:>15.10f}}\\n")
            f.write(f"\\n")
        
        # Also create XYZ file
        xyz_file = conf_dir / f"conf_{{conf_num}}.xyz"
        with open(xyz_file, 'w') as f:
            f.write(f"{{len(atoms)}}\\n")
            f.write(f"{{comment}}\\n")
            for atom, coord in zip(atoms, coords):
                f.write(f"{{atom}} {{coord[0]:>15.10f}} {{coord[1]:>15.10f}} {{coord[2]:>15.10f}}\\n")
        
        # Verify file was created and has content
        if input_file.exists() and input_file.stat().st_size > 0:
            print(f"  [OK] Created: {{input_file}}")
            created_count += 1
        else:
            print(f"  [ERROR] Failed to create {{input_file}}")
    except Exception as e:
        print(f"  [ERROR] creating {{input_file}}: {{e}}")
        sys.exit(1)

if created_count != actual_count:
    print(f"\\nERROR: Created {{created_count}} files but expected {{actual_count}}!")
    sys.exit(1)

print(f"\\n[OK] Preparation complete: {{created_count}} conformer input files created")
print(f"  conformer_count.txt updated with: {{actual_count}}")
'''
        
        # Write Python script
        python_script_path = conf_dir / "prepare_conformers.py"
        with open(python_script_path, 'w') as f:
            f.write(python_script_content)
        os.chmod(python_script_path, 0o755)
        
        # Create bash wrapper
        script = header + f'''
echo "======================================"
echo "  PREPARING CONFORMER INPUTS"
echo "  $(date)"
echo "======================================"
echo ""

# Run Python script to generate inputs
python3 prepare_conformers.py
STATUS=$?

if [[ $STATUS -ne 0 ]]; then
    echo ""
    echo "ERROR: prepare_conformers.py failed with status $STATUS"
    exit 1
fi

# Verify that conformer_count.txt was created
if [[ ! -f "conformer_count.txt" ]]; then
    echo "ERROR: conformer_count.txt was not created!"
    exit 1
fi

ACTUAL_COUNT=$(cat conformer_count.txt | tr -d '[:space:]')
if [[ -z "$ACTUAL_COUNT" ]] || [[ ! "$ACTUAL_COUNT" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Invalid conformer count in conformer_count.txt: '$ACTUAL_COUNT'"
    exit 1
fi

echo ""
echo "Verifying created input files..."
MISSING=0
for i in $(seq 1 $ACTUAL_COUNT); do
    if [[ ! -f "conf_$i/mol_conf_$i.com" ]]; then
        echo "  [ERROR] Missing: conf_$i/mol_conf_$i.com"
        ((MISSING++))
    fi
done

if [[ $MISSING -gt 0 ]]; then
    echo ""
    echo "ERROR: $MISSING input file(s) are missing!"
    exit 1
fi

echo "  [OK] All $ACTUAL_COUNT input files verified"
echo ""
echo "======================================"
echo "  PREPARATION COMPLETE"
echo "  Created $ACTUAL_COUNT conformer inputs"
echo "======================================"
exit 0
'''
        
        # Fix bash variable syntax
        script = self.fix_bash_variables(script)
        
        with open(conf_dir / "prepare_conformers.sh", 'w') as f:
            f.write(script)
        os.chmod(conf_dir / "prepare_conformers.sh", 0o755)
    
    def create_conformer_analysis_script(self, conf_dir: Path):
        """Create script to analyze conformer results"""
        # Generate header
        header = self.get_bash_header(
            script_name="Conformer Analysis",
            description="Analyzes conformer results and selects the lowest energy conformer"
        )
        
        script = header + '''
echo "======================================"
echo "  CONFORMER ANALYSIS"
echo "  $(date)"
echo "======================================"

# Find lowest energy conformer
MIN_ENERGY=999999.0
BEST_CONF=""
COMPLETED=0
FAILED=0

# Get expected conformer count
EXPECTED_COUNT=0
if [[ -f "conformer_count.txt" ]]; then
    EXPECTED_COUNT=$(cat conformer_count.txt | tr -d '[:space:]')
    if [[ ! "$EXPECTED_COUNT" =~ ^[0-9]+$ ]]; then
        EXPECTED_COUNT=0
    fi
fi

echo ""
echo "Conformer Results:"
echo "-------------------"
printf "%-15s %-20s %-10s\\n" "Conformer" "Energy (Hartree)" "Status"
printf "%-15s %-20s %-10s\\n" "---------" "-----------------" "------"

# Check conformers in order
# Use a reasonable default if EXPECTED_COUNT is 0 or invalid
MAX_CHECK=${EXPECTED_COUNT:-10}
if [[ $MAX_CHECK -eq 0 ]]; then
    MAX_CHECK=10
fi

for i in $(seq 1 $MAX_CHECK); do
    CONF_ID="conf_$i"
    log="$CONF_ID/mol_conf_$i.log"
    
    if [[ ! -f "$log" ]]; then
        # Check if directory exists (might have been pruned)
        if [[ -d "$CONF_ID" ]]; then
            printf "%-15s %-20s %-10s\\n" "$CONF_ID" "N/A" "NO LOG"
            ((FAILED++))
        fi
        continue
    fi
    
    if grep -q "Normal termination" "$log" 2>/dev/null; then
        ENERGY_RAW=$(grep "SCF Done" "$log" | tail -1 | awk '{print $5}')
        # Convert energy to decimal format (handles both normal and scientific notation)
        # awk automatically handles scientific notation like 0.8235E-01 or -0.2895
        ENERGY=$(awk -v e="$ENERGY_RAW" 'BEGIN {printf "%.10f", e}' 2>/dev/null)
        
        # Validate that ENERGY is a valid number (not empty and not "nan" or "inf")
        if [[ -n "$ENERGY_RAW" ]] && [[ -n "$ENERGY" ]] && \
           [[ "$ENERGY" != "nan" ]] && [[ "$ENERGY" != "inf" ]] && [[ "$ENERGY" != "-inf" ]] && \
           awk -v e="$ENERGY" 'BEGIN {exit !(e+0 == e)}' 2>/dev/null; then
            printf "%-15s %-20s %-10s\\n" "$CONF_ID" "$ENERGY" "COMPLETE"
            
            # Check if lowest energy (using awk for comparison, more portable than bc)
            IS_LOWER=$(awk -v e1="$ENERGY" -v e2="$MIN_ENERGY" 'BEGIN {print (e1 < e2)}')
            if [[ "$IS_LOWER" == "1" ]]; then
                MIN_ENERGY=$ENERGY
                BEST_CONF=$CONF_ID
            fi
            ((COMPLETED++))
        else
            printf "%-15s %-20s %-10s\\n" "$CONF_ID" "N/A" "NO ENERGY"
            ((FAILED++))
        fi
    else
        printf "%-15s %-20s %-10s\\n" "$CONF_ID" "N/A" "FAILED"
        ((FAILED++))
    fi
done

echo ""
echo "Summary:"
if [[ $EXPECTED_COUNT -gt 0 ]]; then
    echo "  Expected: $EXPECTED_COUNT"
fi
echo "  Completed: $COMPLETED"
echo "  Failed: $FAILED"

if [[ $COMPLETED -eq 0 ]]; then
    echo ""
    echo "ERROR: No conformers completed successfully!"
    echo "  Check individual log files for errors"
    exit 1
fi

if [[ -n "$BEST_CONF" ]]; then
    echo ""
    echo "LOWEST ENERGY CONFORMER: $BEST_CONF"
    echo "Energy: $MIN_ENERGY Hartree"
    
    # Copy best conformer checkpoint
    if [[ -f "$BEST_CONF/mol_conf_${BEST_CONF#conf_}.chk" ]]; then
        cp "$BEST_CONF/mol_conf_${BEST_CONF#conf_}.chk" best_conformer.chk
        echo "[OK] Checkpoint copied to best_conformer.chk"'''
        
        # Add formchk generation if enabled
        if self.config['analysis'].get('generate_fchk', False):
            script += '''
        
        # Generate formatted checkpoint for best conformer
        formchk best_conformer.chk best_conformer.fchk 2>/dev/null || true
        if [[ -f best_conformer.fchk ]]; then
            echo "[OK] Formatted checkpoint created: best_conformer.fchk"
        fi'''
        
        script += '''
    fi
    
    # Save info
    echo "$BEST_CONF" > best_conformer.txt
    echo "$MIN_ENERGY" >> best_conformer.txt
else
    echo ""
    echo "ERROR: No conformers completed successfully!"
    exit 1
fi
'''
        
        # Fix bash variable syntax
        script = self.fix_bash_variables(script)
        
        with open(conf_dir / "analyze_conformers.sh", 'w') as f:
            f.write(script)
        os.chmod(conf_dir / "analyze_conformers.sh", 0o755)
    
    def create_molecule_workflow(self, xyz_file: Path, mol_name: str, base_dir: Path):
        """Create complete workflow for one molecule"""
        # Note: Processing message is now printed in run() method
        # This avoids duplicate messages in the log
        
        # Create molecule directory
        mol_dir = base_dir / mol_name
        mol_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy original XYZ
        shutil.copy2(xyz_file, mol_dir / "original.xyz")
        
        # Read coordinates
        atoms, coords = self.read_xyz_file(xyz_file)
        
        # Save molecule info
        info = {
            'original_file': xyz_file.name,
            'molecule_id': mol_name,
            'num_atoms': len(atoms),
            'timestamp': self.timestamp,
            'config_file': self.config_file,
            'conformer_method': (self.last_conformer_backend or self.get_conformer_backend()).upper()
        }
        with open(mol_dir / "molecule_info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        # Create conformer search if enabled
        conf_metadata = {'has_jobs': False, 'needs_checkpoint': False, 'best_atoms': None, 'best_coords': None}
        if self.config['conformer_search']['enabled']:
            conf_metadata = self.create_conformer_scripts(
                mol_dir, mol_name, atoms, 
                self.config['conformer_search']['num_conformers']
            )
        self.conformer_metadata[mol_name] = conf_metadata
        
        # Create calculation directories and inputs
        enabled_calcs = []
        for calc_name, calc_config in self.config['calculations'].items():
            if not calc_config.get('enabled', True):
                continue
            
            calc_dir = mol_dir / calc_name
            calc_dir.mkdir(exist_ok=True)
            enabled_calcs.append(calc_name)
            
            # Make a copy of calc_config to avoid modifying the original
            calc_config_copy = copy.deepcopy(calc_config)
            
            # Determine if we should include coordinates
            geom_type = calc_config_copy['gaussian'].get('geom')
            use_coords = not (geom_type == 'AllCheck' or geom_type == 'Checkpoint')
            atoms_for_input = atoms
            coords_for_input = coords
            
            # Special handling for s0_opt_freq_vac with conformers
            if calc_name == 's0_opt_freq_vac' and self.config['conformer_search']['enabled']:
                if conf_metadata.get('needs_checkpoint', False):
                    calc_config_copy['gaussian']['oldchk'] = 'conformer'
                    calc_config_copy['gaussian']['geom'] = 'AllCheck'
                    calc_config_copy['gaussian']['guess'] = 'Read'
                    use_coords = False
                else:
                    calc_config_copy['gaussian'].pop('oldchk', None)
                    calc_config_copy['gaussian'].pop('geom', None)
                    calc_config_copy['gaussian'].pop('guess', None)
                    best_atoms = conf_metadata.get('best_atoms')
                    best_coords = conf_metadata.get('best_coords')
                    if best_atoms and best_coords:
                        atoms_for_input = best_atoms
                        coords_for_input = best_coords
                    use_coords = True
            
            # Create Gaussian input
            input_content = self.create_gaussian_input(
                calc_name, calc_config_copy, mol_name, atoms_for_input, coords_for_input, use_coords
            )
            with open(calc_dir / f"mol_{calc_name}.com", 'w') as f:
                f.write(input_content)
            
            # Create SLURM script
            slurm_content = self.create_slurm_script(
                calc_name, calc_config_copy, mol_name, 
                calc_config_copy.get('depends_on')
            )
            with open(calc_dir / "run.slurm", 'w') as f:
                f.write(slurm_content)
            os.chmod(calc_dir / "run.slurm", 0o755)
        
        # Create individual submission script
        self.create_molecule_submit_script(mol_dir, mol_name, conf_metadata, enabled_calcs)
        
        # Create status script
        self.create_molecule_status_script(mol_dir, mol_name, conf_metadata, enabled_calcs)
        
        # Create cancel script
        self.create_molecule_cancel_script(mol_dir, mol_name, conf_metadata, enabled_calcs)
        
        if self.logger:
            self.logger.success(f"Workflow created with {len(enabled_calcs)} calculations")
        else:
            print(f"  [OK] Workflow created with {len(enabled_calcs)} calculations")
        
        return mol_name
    
    def create_molecule_submit_script(self, mol_dir: Path, mol_name: str, 
                                     conf_metadata: Dict[str, Optional[object]], enabled_calcs: List[str]):
        """Create submission script for molecule"""
        sbatch = self.config['slurm']['sbatch_cmd']
        has_conformer_jobs = conf_metadata.get('has_jobs', False)
        
        # Generate header
        header = self.get_bash_header(
            script_name=f"Submission Script - {mol_name}",
            description=f"Submits all workflow calculations for molecule {mol_name}"
        )
        
        lines = [header.rstrip()]
        lines.append('echo "============================================"')
        lines.append(f'echo "  Submitting workflow for: {mol_name}"')
        lines.append('echo "  $(date)"')
        lines.append('echo "============================================"')
        lines.append('')
        lines.append('# Track job IDs')
        lines.append('declare -A JOB_IDS')
        lines.append('')
        
        if has_conformer_jobs:
            lines.append('# Submit conformer search')
            lines.append('SUBMIT_BASE_DIR=$(pwd)')
            lines.append('cd s0_conf_search || exit 1')
            # Get CREST configuration (calculated once)
            backend, crest_cfg, crest_reoptimize = self._get_crest_config()
            
            if backend == 'crest' and crest_reoptimize:
                # First submit CREST job
                lines.append('# Submit CREST conformer search')
                lines.append(f'CREST_JOB=$({sbatch} --parsable run_crest.slurm 2>&1)')
                lines.append('if [[ -z "$CREST_JOB" ]] || [[ ! "$CREST_JOB" =~ ^[0-9]+$ ]]; then')
                lines.append('    echo "ERROR: Failed to submit CREST - invalid job ID: $CREST_JOB"')
                lines.append('    exit 1')
                lines.append('else')
                lines.append('    echo "[OK] CREST conformer search submitted: $CREST_JOB"')
                lines.append('    echo "$CREST_JOB" > .crest_jobid')
                lines.append('    sync')
                lines.append('fi')
                lines.append('')
                lines.append('# Submit conformer PM7 optimization array (depends on CREST)')
                lines.append(f'CONF_JOB=$({sbatch} --parsable --dependency=afterok:$CREST_JOB run_conformers.slurm 2>&1)')
            else:
                # Direct conformer array submission
                lines.append(f'CONF_JOB=$({sbatch} --parsable run_conformers.slurm 2>&1)')
            lines.append('if [[ -z "$CONF_JOB" ]] || [[ ! "$CONF_JOB" =~ ^[0-9]+$ ]]; then')
            lines.append('    echo "ERROR: Failed to submit conformer search - invalid job ID: $CONF_JOB"')
            lines.append('    exit 1')
            lines.append('else')
            lines.append('    echo "[OK] Conformer search submitted: $CONF_JOB"')
            lines.append('    echo "$CONF_JOB" > .jobid')
            lines.append('    sync')
            lines.append('fi')
            lines.append('cd "$SUBMIT_BASE_DIR"')
            lines.append('')
            lines.append('# Submit conformer analysis as dependency')
            lines.append(f'ANALYSIS_JOB=$({sbatch} --parsable --dependency=afterok:$CONF_JOB --chdir=s0_conf_search --wrap "./analyze_conformers.sh" 2>&1)')
            lines.append('if [[ -z "$ANALYSIS_JOB" ]] || [[ ! "$ANALYSIS_JOB" =~ ^[0-9]+$ ]]; then')
            lines.append('    echo "ERROR: Failed to submit conformer analysis - invalid job ID: $ANALYSIS_JOB"')
            lines.append('    exit 1')
            lines.append('else')
            lines.append('    echo "[OK] Conformer analysis queued: $ANALYSIS_JOB"')
            lines.append('    echo "$ANALYSIS_JOB" > s0_conf_search/.analysis_jobid')
            lines.append('    sync')
            lines.append('fi')
            lines.append('')
            lines.append('# Submit s0_opt_freq_vac depending on analysis')
            lines.append('cd s0_opt_freq_vac || exit 1')
            lines.append(f'JOB_IDS[s0_opt_freq_vac]=$({sbatch} --parsable --dependency=afterok:$ANALYSIS_JOB run.slurm 2>&1)')
            lines.append('if [[ -z "${JOB_IDS[s0_opt_freq_vac]}" ]] || [[ ! "${JOB_IDS[s0_opt_freq_vac]}" =~ ^[0-9]+$ ]]; then')
            lines.append('    echo "ERROR: Failed to submit s0_opt_freq_vac - invalid job ID"')
            lines.append('    exit 1')
            lines.append('else')
            lines.append('    echo "[OK] s0_opt_freq_vac queued: ${JOB_IDS[s0_opt_freq_vac]}"')
            lines.append('    echo "${JOB_IDS[s0_opt_freq_vac]}" > .jobid')
            lines.append('    sync')
            lines.append('fi')
            lines.append('cd "$SUBMIT_BASE_DIR"')
        else:
            # No conformer jobs (CREST without PM7 reoptimization or conformers disabled)
            # Check if best_conformer.txt exists (CREST completed)
            lines.append('# Check if conformer search completed (CREST without PM7)')
            lines.append('if [[ -f "s0_conf_search/best_conformer.txt" ]]; then')
            lines.append('    echo "[OK] Conformer search completed (CREST, no PM7 refinement)"')
            lines.append('else')
            lines.append('    echo "[WARN] best_conformer.txt not found, but continuing..."')
            lines.append('fi')
            lines.append('')
            lines.append('# Submit first calculation (s0_opt_freq_vac)')
            lines.append('cd s0_opt_freq_vac')
            lines.append('if [[ ! -f "run.slurm" ]]; then')
            lines.append('    echo "ERROR: run.slurm not found in s0_opt_freq_vac directory"')
            lines.append('    exit 1')
            lines.append('fi')
            lines.append(f'JOB_IDS[s0_opt_freq_vac]=$({sbatch} --parsable run.slurm 2>&1)')
            lines.append('if [[ -z "${JOB_IDS[s0_opt_freq_vac]}" ]] || [[ ! "${JOB_IDS[s0_opt_freq_vac]}" =~ ^[0-9]+$ ]]; then')
            lines.append('    echo "ERROR: Failed to submit s0_opt_freq_vac - invalid job ID: ${JOB_IDS[s0_opt_freq_vac]}"')
            lines.append('    echo "  Check that sbatch is available and run.slurm is valid"')
            lines.append('    exit 1')
            lines.append('else')
            lines.append('    echo "[OK] s0_opt_freq_vac submitted: ${JOB_IDS[s0_opt_freq_vac]}"')
            lines.append('fi')
            lines.append('cd ..')
        
        lines.append('')
        lines.append('# Submit dependent calculations')
        
        # Get topologically sorted calculation order
        workflow_order = self._get_calculation_order(enabled_calcs, conf_metadata)
        
        # Submit each calculation in order
        for calc in workflow_order:
            if calc == 's0_opt_freq_vac':
                continue  # Already submitted above
            depends = self.config['calculations'][calc].get('depends_on')
            lines.append(f'cd {calc}')
            if depends and depends in enabled_calcs:
                lines.append(f'JOB_IDS[{calc}]=$({sbatch} --parsable --dependency=afterok:${{JOB_IDS[{depends}]}} run.slurm 2>&1)')
            else:
                lines.append(f'JOB_IDS[{calc}]=$({sbatch} --parsable run.slurm 2>&1)')
            lines.append(f'if [[ -z "${{JOB_IDS[{calc}]}}" ]] || [[ ! "${{JOB_IDS[{calc}]}}" =~ ^[0-9]+$ ]]; then')
            lines.append(f'    echo "  [ERROR] Failed to submit {calc} - invalid job ID"')
            lines.append('    exit 1')
            lines.append('else')
            lines.append(f'    echo "  [OK] {calc} queued: ${{JOB_IDS[{calc}]}}"')
            lines.append('fi')
            lines.append('cd ..')
        
        lines.append('')
        lines.append('# Save remaining calculation job IDs to files for status tracking')
        lines.append('for calc in "${!JOB_IDS[@]}"; do')
        lines.append('    if [[ -n "${JOB_IDS[$calc]}" ]] && [[ "${JOB_IDS[$calc]}" =~ ^[0-9]+$ ]]; then')
        lines.append('        cd "$calc" || continue')
        lines.append('        echo "${JOB_IDS[$calc]}" > .jobid')
        lines.append('        sync')
        lines.append('        cd "$SUBMIT_BASE_DIR"')
        lines.append('        echo "  [OK] Saved Job ID for $calc: ${JOB_IDS[$calc]}"')
        lines.append('    else')
        lines.append('        echo "  [WARN] Invalid Job ID for $calc: ${JOB_IDS[$calc]}"')
        lines.append('    fi')
        lines.append('done')
        lines.append('')
        lines.append('echo ""')
        lines.append('echo "Workflow submitted successfully!"')
        lines.append('echo "Monitor with: ./status.sh"')
        lines.append('echo "Cancel with: ./cancel.sh [all|calc_name]"')
        
        script_content = '\n'.join(lines)
        script_content = self.fix_bash_variables(script_content)
        
        with open(mol_dir / "submit.sh", 'w') as f:
            f.write(script_content)
        os.chmod(mol_dir / "submit.sh", 0o755)
    
    def _get_enabled_calculations(self) -> List[str]:
        """Get list of enabled calculations from config (DRY helper)"""
        enabled_calcs = []
        for calc_name, calc_config in self.config['calculations'].items():
            if calc_config.get('enabled', True):
                enabled_calcs.append(calc_name)
        return enabled_calcs
    
    def _get_calculation_order(self, enabled_calcs: List[str], conf_metadata: Dict[str, Optional[object]]) -> List[str]:
        """Dynamically order calculations based on dependencies from YAML config
        
        Returns:
            List of calculation names in execution order
        """
        deps_map = {}
        for calc in enabled_calcs:
            dep = self.config['calculations'][calc].get('depends_on')
            deps_map[calc] = dep
        
        # Build dependency graph and resolve order using topological sort
        def get_dependency_level(calc_name, visited=None):
            """Recursively determine dependency level (0 = no deps, higher = more deps)"""
            if visited is None:
                visited = set()
            if calc_name in visited:
                # Circular dependency detected, return 0 to break cycle
                return 0
            visited.add(calc_name)
            
            dep = deps_map.get(calc_name)
            if dep is None or dep not in enabled_calcs:
                # No dependency or dependency not enabled
                visited.remove(calc_name)
                return 0
            
            # Recursively get level of dependency
            parent_level = get_dependency_level(dep, visited)
            visited.remove(calc_name)
            return parent_level + 1
        
        # Sort by dependency level, then alphabetically for same level
        workflow_order = sorted(enabled_calcs, key=lambda x: (get_dependency_level(x), x))
        
        # If conformer search is enabled and has jobs, it should come first
        include_conformer_stage = conf_metadata.get('has_jobs', False)
        if include_conformer_stage:
            # Ensure s0_opt_freq_vac comes after conformer search if it exists
            if 's0_opt_freq_vac' in workflow_order:
                workflow_order.remove('s0_opt_freq_vac')
                # Insert after conformer stage (at position 0 or 1)
                workflow_order.insert(0, 's0_opt_freq_vac')
        
        return workflow_order
    
    def _get_calculation_display_name(self, calc_name: str) -> str:
        """Get human-readable display name for a calculation"""
        calc_config = self.config['calculations'].get(calc_name, {})
        display_name = calc_config.get('display_name')
        if display_name:
            return display_name
        
        # Generate from name if no display_name specified
        if calc_name == 's0_conf_search':
            return 'Conformer Search'
        # Try to extract meaningful parts from name
        parts = calc_name.split('_')
        if len(parts) >= 2:
            state = parts[0].upper()
            calc_type = parts[1]
            if 'opt' in calc_type:
                calc_type = 'Optimization'
            elif 'freq' in calc_type:
                calc_type = 'Freq'
            elif 'sp' in calc_type:
                calc_type = 'Single Point'
            env = ''
            if 'vac' in calc_name:
                env = ' (Vacuum)'
            elif 'solv' in calc_name:
                env = ' (Solvated)'
            return f"{state} {calc_type}{env}"
        return calc_name
    
    def _get_calculation_abbreviation(self, calc_name: str) -> str:
        """Get short abbreviation for a calculation (for compact status display)"""
        calc_config = self.config['calculations'].get(calc_name, {})
        abbrev = calc_config.get('abbreviation')
        if abbrev:
            return abbrev
        
        # Generate abbreviation from name
        if calc_name == 's0_conf_search':
            return 'CONF'
        
        # Extract key parts
        parts = calc_name.split('_')
        abbrev_parts = []
        for part in parts:
            if part in ['s0', 's1', 't1', 'cat', 'an']:
                abbrev_parts.append(part.upper())
            elif 'opt' in part:
                abbrev_parts.append('OPT')
            elif 'freq' in part:
                abbrev_parts.append('FRQ')
            elif 'sp' in part:
                abbrev_parts.append('SP')
            elif 'tddft' in part:
                abbrev_parts.append('TD')
            elif 'vac' in part:
                abbrev_parts.append('V')
            elif 'solv' in part:
                abbrev_parts.append('S')
        
        return '_'.join(abbrev_parts) if abbrev_parts else calc_name[:8].upper()
    
    def create_molecule_status_script(self, mol_dir: Path, mol_name: str,
                                     conf_metadata: Dict[str, Optional[object]], enabled_calcs: List[str]):
        """Create status checking script for molecule (dynamically generated from YAML)"""
        # Generate header
        header = self.get_bash_header(
            script_name=f"Status Script - {mol_name}",
            description=f"Checks the status of all calculations for molecule {mol_name}"
        )
        
        script = header + f'''
# Colors
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
BLUE='\\033[0;34m'
NC='\\033[0m'

# Cache SLURM queue state (single call for entire status check)
SQUEUE_CACHE=$(squeue -u $USER -o "%.18i %.30j %.8T %.10M" 2>/dev/null || echo "")

extract_energy() {{
    local log="$1"
    local energy=""
    energy=$(grep "SCF Done" "$log" 2>/dev/null | tail -1 | awk '{{print $5}}')
    if [[ -z "$energy" ]]; then
        local free_line=$(grep "Sum of electronic and thermal Free Energies" "$log" 2>/dev/null | tail -1)
        [[ -n "$free_line" ]] && energy=$(echo "$free_line" | awk '{{print $NF}}')
    fi
    if [[ -z "$energy" ]]; then
        local tddft=$(grep "Total Energy, E" "$log" 2>/dev/null | tail -1 | awk '{{print $NF}}')
        [[ -n "$tddft" ]] && energy="$tddft"
    fi
    echo "${{energy:--}}"
}}

# Initialize all counters and arrays at the start
    TOTAL_CALCS=0
    COMPLETED_CALCS=0
    FAILED_CALCS=0
    RUNNING_CALCS=0
    PENDING_CALCS=0
    declare -a COMPLETED_LIST=()
    declare -a FAILED_LIST=()
    declare -a RUNNING_LIST=()
    declare -a PENDING_LIST=()

echo "================================================================================"
echo "  DETAILED STATUS: {mol_name}"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"
echo ""
'''
        
        if self.config['conformer_search']['enabled']:
            script += '''# CONFORMER SEARCH STATUS
echo -e "${BLUE}=== CONFORMER SEARCH ===${NC}"
if [[ -f "s0_conf_search/best_conformer.txt" ]]; then
    BEST=$(head -1 s0_conf_search/best_conformer.txt)
    ENERGY=$(tail -1 s0_conf_search/best_conformer.txt)
    echo -e "  Status: ${GREEN}[OK] COMPLETED${NC}"
    echo "  Best conformer: $BEST"
    echo "  Energy: $ENERGY Hartree"
    
    # Show all conformer energies if available
    if [[ -d "s0_conf_search" ]]; then
        echo ""
        echo "  All conformers:"
        for log in s0_conf_search/conf_*/mol_conf_*.log; do
            if [[ -f "$log" ]]; then
                CONF_ID=$(basename "$(dirname "$log")")
                if grep -q "Normal termination" "$log" 2>/dev/null; then
                    ENERGY=$(extract_energy "$log")
                    if [[ "$CONF_ID" == "$BEST" ]]; then
                        echo -e "    ${GREEN}> $CONF_ID: $ENERGY Hartree (BEST)${NC}"
                    else
                        echo "    - $CONF_ID: $ENERGY Hartree"
                    fi
                fi
            fi
        done
    fi
else
    CONF_DONE=0
    CONF_TOTAL=0
    CONF_DONE=$(ls -1 s0_conf_search/conf_*/status.txt 2>/dev/null | wc -l)
    CONF_TOTAL=$(ls -1d s0_conf_search/conf_*/ 2>/dev/null | wc -l)
    # Ensure they are numbers
    CONF_DONE=${{CONF_DONE:-0}}
    CONF_TOTAL=${{CONF_TOTAL:-0}}
    if [[ $CONF_DONE -eq 0 ]]; then
        echo -e "  Status: ${YELLOW}[.] PENDING${NC} ($CONF_DONE/$CONF_TOTAL completed)"
    else
        echo -e "  Status: ${YELLOW}[>] RUNNING${NC} ($CONF_DONE/$CONF_TOTAL completed)"
        if [[ "$CONF_TOTAL" -gt 0 ]]; then
            FILLED=$(( CONF_DONE * 20 / CONF_TOTAL ))
            EMPTY=$(( 20 - FILLED ))
            echo "  Progress: [$(printf '%*s' "$FILLED" "" | tr ' ' '#')$(printf '%*s' "$EMPTY" "" | tr ' ' '-')]"
        fi
    fi
fi
echo ""
'''
        
        script += '''# MAIN CALCULATIONS STATUS
echo -e "${BLUE}=== WORKFLOW CALCULATIONS ===${NC}"
echo ""
printf "%-30s %-12s %-20s %-15s\\n" "Calculation" "Status" "Job ID" "Runtime"
printf "%-30s %-12s %-20s %-15s\\n" "$(printf -- '-%.0s' {1..30})" "$(printf -- '-%.0s' {1..12})" "$(printf -- '-%.0s' {1..20})" "$(printf -- '-%.0s' {1..15})"

'''
        
        # Get workflow order dynamically from YAML dependencies
        workflow_order = self._get_calculation_order(enabled_calcs, conf_metadata)
        
        # Generate display names in Python to avoid duplication in bash
        calc_display_names = {}
        for calc in workflow_order:
            calc_display_names[calc] = self._get_calculation_display_name(calc)
        
        # Add helper function to bash script for getting display names
        script += '''
# Helper function to get calculation display name (avoids duplication)
get_calc_display_name() {
    local calc_name="$1"
    case "$calc_name" in
        "s0_conf_search") echo "Conformer Search" ;;
        *)
            # Generate from name pattern
            local clean_name=$(echo "$calc_name" | tr '_' ' ')
            if [[ "$calc_name" == *"opt"* ]] && [[ "$calc_name" == *"freq"* ]]; then
                if [[ "$calc_name" == *"vac"* ]]; then
                    echo "$clean_name (Vacuum)"
                elif [[ "$calc_name" == *"solv"* ]]; then
                    echo "$clean_name (Solvated)"
                else
                    echo "$clean_name"
                fi
            elif [[ "$calc_name" == *"sp"* ]] && [[ "$calc_name" == *"tddft"* ]]; then
                echo "TD-DFT Excitations"
            elif [[ "$calc_name" == *"sp"* ]]; then
                echo "$clean_name - Single Point"
            else
                echo "$clean_name"
            fi
            ;;
    esac
}

'''
        
        for calc in workflow_order:
            # Construct job name pattern for grep
            job_pattern = f"{mol_name[:8]}_{calc[:8]}"
            display_name = calc_display_names[calc]
            script += f'''
# Check {calc}
CALC_NAME="{calc}"
CALC_DISPLAY="{display_name}"

# Read saved Job ID if available (try multiple locations)
SAVED_JOB_ID=""
if [[ -f "$CALC_NAME/.jobid" ]]; then
    SAVED_JOB_ID=$(cat "$CALC_NAME/.jobid" 2>/dev/null | tr -d '[:space:]')
elif [[ "$CALC_NAME" == "s0_conf_search" ]] && [[ -f "s0_conf_search/.crest_jobid" ]]; then
    # For CREST, try .crest_jobid first
    SAVED_JOB_ID=$(cat "s0_conf_search/.crest_jobid" 2>/dev/null | tr -d '[:space:]')
fi

# Validate Job ID is a number
if [[ -n "$SAVED_JOB_ID" ]] && [[ ! "$SAVED_JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "  [WARN] Invalid Job ID format for $CALC_NAME: '$SAVED_JOB_ID', ignoring"
    SAVED_JOB_ID=""
fi

# FIRST: Check if job is in queue (running or pending)
# Try using saved Job ID first, then fall back to name-based search
if [[ -n "$SAVED_JOB_ID" ]]; then
    JOB_INFO=$(echo "$SQUEUE_CACHE" | awk -v id="$SAVED_JOB_ID" '$1 == id {{print; exit}}')
fi

# If not found by Job ID, try by name pattern
if [[ -z "$JOB_INFO" ]]; then
    JOB_INFO=$(echo "$SQUEUE_CACHE" | grep "{job_pattern}" | head -1)
fi

if [[ -n "$JOB_INFO" ]]; then
    # Job is in SLURM queue (SQUEUE_CACHE: id name state time)
    JOB_ID=$(echo "$JOB_INFO" | awk '{{print $1}}')
    JOB_STATE=$(echo "$JOB_INFO" | awk '{{print $3}}')
    RUNTIME=$(echo "$JOB_INFO" | awk '{{print $4}}')
    
    if [[ "$JOB_STATE" == "RUNNING" ]] || [[ "$JOB_STATE" == "R" ]]; then
        STATUS_SYMBOL="${{BLUE}}[>] RUNNING${{NC}}"
    elif [[ "$JOB_STATE" == "PENDING" ]] || [[ "$JOB_STATE" == "PD" ]]; then
        STATUS_SYMBOL="${{YELLOW}}[.] QUEUED${{NC}}"
        RUNTIME="-"
    else
        STATUS_SYMBOL="${{YELLOW}}[.] $JOB_STATE${{NC}}"
    fi
    EXTRA_INFO=""
    
elif [[ -f "$CALC_NAME/mol_$CALC_NAME.log" ]]; then
    # Job not in queue, check log file
    # Use saved job ID if available, even if job is completed
    if [[ -z "$JOB_ID" ]] && [[ -n "$SAVED_JOB_ID" ]]; then
        JOB_ID="$SAVED_JOB_ID"
    fi
    
    if grep -q "Normal termination" "$CALC_NAME/mol_$CALC_NAME.log" 2>/dev/null; then
        # Completed successfully
        STATUS_SYMBOL="${{GREEN}}[OK] COMPLETE${{NC}}"
        ENERGY=$(extract_energy "$CALC_NAME/mol_$CALC_NAME.log")
        if [[ -n "$ENERGY" ]] && [[ "$ENERGY" != "" ]] && [[ "$ENERGY" != "-" ]]; then
            EXTRA_INFO="E=$ENERGY Ha"
        else
            EXTRA_INFO=""
        fi
        # Keep job ID if we have it, otherwise show "-"
        [[ -z "$JOB_ID" ]] && JOB_ID="-"
        RUNTIME="-"
    elif grep -q "Error termination" "$CALC_NAME/mol_$CALC_NAME.log" 2>/dev/null; then
        # Failed with error
        STATUS_SYMBOL="${{RED}}[X] FAILED${{NC}}"
        # Get error message (first 20 chars)
        ERROR_MSG=$(grep -i "error" "$CALC_NAME/mol_$CALC_NAME.log" 2>/dev/null | tail -1 | cut -c1-20)
        if [[ -n "$ERROR_MSG" ]]; then
            EXTRA_INFO="$ERROR_MSG"
        else
            EXTRA_INFO=""
        fi
        # Keep job ID if we have it, otherwise show "-"
        [[ -z "$JOB_ID" ]] && JOB_ID="-"
        RUNTIME="-"
    else
        # Log exists but job not in queue and no termination message
        # This happens when job just finished but files not yet updated
        STATUS_SYMBOL="${{YELLOW}}[.] FINISHING${{NC}}"
        EXTRA_INFO=""
        # Keep job ID if we have it, otherwise show "-"
        [[ -z "$JOB_ID" ]] && JOB_ID="-"
        RUNTIME="-"
    fi
    
elif [[ -f "$CALC_NAME/status.txt" ]]; then
    # Check status file
    # Use saved job ID if available
    if [[ -z "$JOB_ID" ]] && [[ -n "$SAVED_JOB_ID" ]]; then
        JOB_ID="$SAVED_JOB_ID"
    fi
    
    STATUS=$(cat "$CALC_NAME/status.txt" 2>/dev/null)
    if [[ "$STATUS" == "SUCCESS" ]]; then
        STATUS_SYMBOL="${{GREEN}}[OK] COMPLETE${{NC}}"
        # Keep job ID if we have it, otherwise show "-"
        [[ -z "$JOB_ID" ]] && JOB_ID="-"
        RUNTIME="-"
        EXTRA_INFO=""
    else
        STATUS_SYMBOL="${{RED}}[X] FAILED${{NC}}"
        # Keep job ID if we have it, otherwise show "-"
        [[ -z "$JOB_ID" ]] && JOB_ID="-"
        RUNTIME="-"
        EXTRA_INFO=""
    fi
    
else
    # No log, no status, not in queue = not started yet
    # Use saved job ID if available (might be queued but not yet started)
    if [[ -z "$JOB_ID" ]] && [[ -n "$SAVED_JOB_ID" ]]; then
        JOB_ID="$SAVED_JOB_ID"
    fi
    
    STATUS_SYMBOL="${{YELLOW}}[.] PENDING${{NC}}"
    [[ -z "$JOB_ID" ]] && JOB_ID="-"
    RUNTIME="-"
    EXTRA_INFO=""
fi

# Accumulate summary (single pass; matches status logic above)
if [[ -n "$JOB_INFO" ]]; then
    ((TOTAL_CALCS++))
    ((RUNNING_CALCS++))
    RUNNING_LIST+=("$CALC_NAME")
elif [[ -f "$CALC_NAME/mol_$CALC_NAME.log" ]] && grep -q "Normal termination" "$CALC_NAME/mol_$CALC_NAME.log" 2>/dev/null; then
    ((TOTAL_CALCS++))
    ((COMPLETED_CALCS++))
    COMPLETED_LIST+=("$CALC_NAME")
elif [[ -f "$CALC_NAME/mol_$CALC_NAME.log" ]] && grep -q "Error termination" "$CALC_NAME/mol_$CALC_NAME.log" 2>/dev/null; then
    ((TOTAL_CALCS++))
    ((FAILED_CALCS++))
    FAILED_LIST+=("$CALC_NAME")
elif [[ -f "$CALC_NAME/mol_$CALC_NAME.log" ]]; then
    ((TOTAL_CALCS++))
    PENDING_LIST+=("$CALC_NAME")
elif [[ -f "$CALC_NAME/status.txt" ]]; then
    ((TOTAL_CALCS++))
    STATUS_SUM=$(cat "$CALC_NAME/status.txt" 2>/dev/null)
    if [[ "$STATUS_SUM" == "SUCCESS" ]]; then
        ((COMPLETED_CALCS++))
        COMPLETED_LIST+=("$CALC_NAME")
    else
        ((FAILED_CALCS++))
        FAILED_LIST+=("$CALC_NAME")
    fi
else
    ((TOTAL_CALCS++))
    PENDING_LIST+=("$CALC_NAME")
fi

printf "%-30s " "$CALC_NAME"
printf "%-20b " "$STATUS_SYMBOL"
printf "%-20s %-15s\\n" "$JOB_ID" "$RUNTIME"
if [[ -n "$EXTRA_INFO" ]]; then
    echo "  -> $EXTRA_INFO"
fi
'''
        
        script += '''
echo ""
echo "================================================================================"
echo ""

'''
        
        # Create a bash variable with the molecule name prefix
        mol_prefix = mol_name[:8]
        script += f'''
# Set molecule name prefix for job name matching
MOL_PREFIX="{mol_prefix}"

'''
        
        if self.config['conformer_search']['enabled']:
            script += '''# Add conformer search to totals (not in workflow_order)
if [[ -d "s0_conf_search" ]]; then
    ((TOTAL_CALCS++))
    if [[ -f "s0_conf_search/best_conformer.txt" ]]; then
        ((COMPLETED_CALCS++))
        COMPLETED_LIST+=("s0_conf_search")
    else
        CONF_RUNNING=$(echo "$SQUEUE_CACHE" | grep -c "${MOL_PREFIX}_conf" || echo "0")
        if [[ $CONF_RUNNING -gt 0 ]]; then
            ((RUNNING_CALCS++))
            RUNNING_LIST+=("s0_conf_search")
        else
            PENDING_LIST+=("s0_conf_search")
        fi
    fi
fi

'''
        
        script += '''
# Summary statistics
if [[ -z "$TOTAL_CALCS" ]] || [[ "$TOTAL_CALCS" -eq 0 ]]; then
    TOTAL_CALCS=0
fi
if [[ -z "$COMPLETED_CALCS" ]]; then
    COMPLETED_CALCS=0
fi
if [[ -z "$FAILED_CALCS" ]]; then
    FAILED_CALCS=0
fi
if [[ -z "$RUNNING_CALCS" ]]; then
    RUNNING_CALCS=0
fi

if [[ $TOTAL_CALCS -gt 0 ]]; then
    PERCENT_COMPLETE=$(( COMPLETED_CALCS * 100 / TOTAL_CALCS ))
    
    echo "SUMMARY:"
    printf "  %-15s %b%d/%d (%d%%)%b\\n" "Completed:" "${{GREEN}}" $COMPLETED_CALCS $TOTAL_CALCS $PERCENT_COMPLETE "${{NC}}"
    [[ $FAILED_CALCS -gt 0 ]] && printf "  %-15s %b%d%b\\n" "Failed:" "${{RED}}" $FAILED_CALCS "${{NC}}"
    [[ $RUNNING_CALCS -gt 0 ]] && printf "  %-15s %b%d%b\\n" "Running:" "${{BLUE}}" $RUNNING_CALCS "${{NC}}"
    PENDING_CALCS=$(( TOTAL_CALCS - COMPLETED_CALCS - FAILED_CALCS - RUNNING_CALCS ))
    [[ $PENDING_CALCS -gt 0 ]] && printf "  %-15s %b%d%b\\n" "Pending:" "${{YELLOW}}" $PENDING_CALCS "${{NC}}"
    
    echo ""
    # Progress bar
    BAR_LENGTH=40
    FILLED=$(( PERCENT_COMPLETE * BAR_LENGTH / 100 ))
    EMPTY=$(( BAR_LENGTH - FILLED ))
    printf "  Progress: ["
    printf "%b" "${{GREEN}}"
    printf '%*s' "$FILLED" "" | tr ' ' '#'
    printf "%b" "${{NC}}"
    printf "%b" "${{YELLOW}}"
    printf '%*s' "$EMPTY" "" | tr ' ' '-'
    printf "%b" "${{NC}}"
    printf "] %d%%\\n" "$PERCENT_COMPLETE"
    
    echo ""
    if [[ $COMPLETED_CALCS -eq $TOTAL_CALCS ]]; then
        echo -e "  ${{GREEN}}[OK] ALL CALCULATIONS COMPLETE!${{NC}}"
    elif [[ $FAILED_CALCS -gt 0 ]]; then
        echo -e "  ${{RED}}[!] $FAILED_CALCS CALCULATION(S) FAILED${{NC}}"
    elif [[ $RUNNING_CALCS -gt 0 ]]; then
        echo -e "  ${{BLUE}}[>] $RUNNING_CALCS JOB(S) RUNNING${{NC}}"
    else
        echo -e "  ${{YELLOW}}[.] WORKFLOW QUEUED${{NC}}"
    fi
fi

echo ""
echo "================================================================================"
'''
        
        # Fix bash variable syntax
        script = self.fix_bash_variables(script)
        
        with open(mol_dir / "status.sh", 'w') as f:
            f.write(script)
        os.chmod(mol_dir / "status.sh", 0o755)
    
    def create_molecule_cancel_script(self, mol_dir: Path, mol_name: str,
                                     conf_metadata: Dict[str, Optional[object]], enabled_calcs: List[str]):
        """Create job cancellation script for molecule"""
        # Generate header
        header = self.get_bash_header(
            script_name=f"Cancel Script - {mol_name}",
            description=f"Cancels running/pending jobs for molecule {mol_name}"
        )
        
        has_conformer_jobs = conf_metadata.get('has_jobs', False)
        script = header + '''
echo "============================================"
echo "  CANCEL JOBS FOR: ''' + mol_name + '''"
echo "  $(date)"
echo "============================================"
echo ""

# Check if argument provided
if [[ $# -eq 0 ]]; then
    echo "Usage:"
    echo "  ./cancel.sh all           # Cancel all jobs"
    echo "  ./cancel.sh ALL           # Same as above (case-insensitive)"
    echo "  ./cancel.sh <calc_name>   # Cancel specific calculation"
    echo ""
    echo "Available calculations:"
'''
        
        if self.config['conformer_search']['enabled'] and has_conformer_jobs:
            script += '''    echo "  - s0_conf_search (conformer search)"
'''
        
        for calc in enabled_calcs:
            script += f'''    echo "  - {calc}"
'''
        
        script += '''    exit 0
fi

TARGET="$1"
CANCELLED=0

# Convert target to lowercase for case-insensitive comparison
TARGET_LOWER=$(echo "$TARGET" | tr '[:upper:]' '[:lower:]')

# Function to cancel a job by reading .jobid file
cancel_job() {
    local calc_dir="$1"
    local calc_name="$2"
    
    if [[ -f "$calc_dir/.jobid" ]]; then
        JOB_ID=$(cat "$calc_dir/.jobid" 2>/dev/null | tr -d '[:space:]')
        if [[ -n "$JOB_ID" ]]; then
            # Check if job is still in queue
            if squeue -j "$JOB_ID" &>/dev/null; then
                echo "Cancelling $calc_name (Job ID: $JOB_ID)..."
                scancel "$JOB_ID"
                if [[ $? -eq 0 ]]; then
                    echo "  [OK] Job $JOB_ID cancelled"
                    ((CANCELLED++))
                    # Remove jobid file
                    rm -f "$calc_dir/.jobid"
                else
                    echo "  [X] Failed to cancel job $JOB_ID"
                fi
            else
                echo "  [-] Job $JOB_ID not in queue (already finished or cancelled)"
                rm -f "$calc_dir/.jobid"
            fi
        else
            echo "  [-] No job ID found for $calc_name"
        fi
    else
        echo "  [-] No .jobid file for $calc_name (not submitted or already finished)"
    fi
}

if [[ "$TARGET_LOWER" == "all" ]]; then
    echo "Cancelling ALL jobs for ''' + mol_name + '''..."
    echo ""
    
'''
        
        if has_conformer_jobs:
            script += '''    # Cancel conformer search
    if [[ -d "s0_conf_search" ]]; then
        cancel_job "s0_conf_search" "Conformer Search"
        # Also cancel analysis job
        if [[ -f "s0_conf_search/.analysis_jobid" ]]; then
            ANALYSIS_JOB=$(cat "s0_conf_search/.analysis_jobid" 2>/dev/null | tr -d '[:space:]')
            if [[ -n "$ANALYSIS_JOB" ]] && squeue -j "$ANALYSIS_JOB" &>/dev/null; then
                echo "Cancelling Conformer Analysis (Job ID: $ANALYSIS_JOB)..."
                scancel "$ANALYSIS_JOB"
                echo "  [OK] Job $ANALYSIS_JOB cancelled"
                ((CANCELLED++))
                rm -f "s0_conf_search/.analysis_jobid"
            fi
        fi
    fi
    
'''
        
        for calc in enabled_calcs:
            script += f'''    cancel_job "{calc}" "{calc}"
'''
        
        script += '''
    echo ""
    if [[ $CANCELLED -gt 0 ]]; then
        echo "[OK] Cancelled $CANCELLED job(s)"
    else
        echo "[-] No active jobs to cancel"
    fi
    
else
    # Cancel specific calculation
    echo "Cancelling job: $TARGET"
    echo ""
    
    if [[ "$TARGET" == "s0_conf_search" ]]; then
        cancel_job "s0_conf_search" "Conformer Search"
        # Also ask about analysis
        if [[ -f "s0_conf_search/.analysis_jobid" ]]; then
            ANALYSIS_JOB=$(cat "s0_conf_search/.analysis_jobid" 2>/dev/null | tr -d '[:space:]')
            if [[ -n "$ANALYSIS_JOB" ]] && squeue -j "$ANALYSIS_JOB" &>/dev/null; then
                echo ""
                read -p "Also cancel conformer analysis job? (y/n): " -n 1 -r
                echo ""
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    scancel "$ANALYSIS_JOB"
                    echo "  [OK] Analysis job cancelled"
                    rm -f "s0_conf_search/.analysis_jobid"
                fi
            fi
        fi
    elif [[ -d "$TARGET" ]]; then
        cancel_job "$TARGET" "$TARGET"
    else
        echo "[X] Error: Calculation '$TARGET' not found"
        echo "    Run './cancel.sh' without arguments to see available calculations"
        exit 1
    fi
    
    if [[ $CANCELLED -gt 0 ]]; then
        echo ""
        echo "[OK] Job cancelled successfully"
    fi
fi

echo ""
echo "Done. Check status with: ./status.sh"
'''
        
        # Fix bash variable syntax
        script = self.fix_bash_variables(script)
        
        with open(mol_dir / "cancel.sh", 'w') as f:
            f.write(script)
        os.chmod(mol_dir / "cancel.sh", 0o755)
    
    def create_master_workflow_script(self, base_dir: Path, molecules: List[str]):
        """Create master workflow.sh script (dynamically generated from YAML)"""
        # Get enabled calculations from config (using helper to avoid duplication)
        enabled_calcs = self._get_enabled_calculations()
        
        # Add conformer search if enabled
        calc_names_list = []
        calc_order_list = []
        if self.config['conformer_search'].get('enabled', True):
            calc_names_list.append('s0_conf_search')
            calc_order_list.append(self._get_calculation_abbreviation('s0_conf_search'))
        
        # Add all enabled calculations
        for calc_name in enabled_calcs:
            calc_names_list.append(calc_name)
            calc_order_list.append(self._get_calculation_abbreviation(calc_name))
        
        # Generate header
        header = self.get_bash_header(
            script_name="Master Workflow Manager",
            description="Main control script for managing all molecular workflows"
        )
        
        script = header + f'''
# Generated: {self.timestamp}
# Config: {self.config_file}

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
BASE_DIR="$SCRIPT_DIR"

# Colors
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
BLUE='\\033[0;34m'
NC='\\033[0m'

# Molecule list
MOLECULES=({' '.join([f'"{m}"' for m in molecules])})

# Calculation order (dynamically generated from YAML config)
CALC_ORDER=({' '.join([f'"{abbr}"' for abbr in calc_order_list])})
CALC_NAMES=({' '.join([f'"{name}"' for name in calc_names_list])})

extract_energy() {{
    local log="$1"
    local energy=""
    energy=$(grep "SCF Done" "$log" 2>/dev/null | tail -1 | awk '{{print $5}}')
    if [[ -z "$energy" ]]; then
        local free_line=$(grep "Sum of electronic and thermal Free Energies" "$log" 2>/dev/null | tail -1)
        [[ -n "$free_line" ]] && energy=$(echo "$free_line" | awk '{{print $NF}}')
    fi
    if [[ -z "$energy" ]]; then
        local tddft=$(grep "Total Energy, E" "$log" 2>/dev/null | tail -1 | awk '{{print $NF}}')
        [[ -n "$tddft" ]] && energy="$tddft"
    fi
    echo "${{energy:--}}"
}}

# Helper function to get calculation display name (avoids duplication)
get_calc_display_name() {{
    local calc_name="$1"
    case "$calc_name" in
        "s0_conf_search") echo "Conformer Search" ;;
        *)
            # Generate from name pattern
            local clean_name=$(echo "$calc_name" | tr '_' ' ')
            if [[ "$calc_name" == *"opt"* ]] && [[ "$calc_name" == *"freq"* ]]; then
                if [[ "$calc_name" == *"vac"* ]]; then
                    echo "$clean_name (Vacuum)"
                elif [[ "$calc_name" == *"solv"* ]]; then
                    echo "$clean_name (Solvated)"
                else
                    echo "$clean_name"
                fi
            elif [[ "$calc_name" == *"sp"* ]] && [[ "$calc_name" == *"tddft"* ]]; then
                echo "TD-DFT Excitations"
            elif [[ "$calc_name" == *"sp"* ]]; then
                echo "$clean_name - Single Point"
            else
                echo "$clean_name"
            fi
            ;;
    esac
}}

usage() {{
    echo "Usage: $0 {{submit|status|cancel|extract|clean}} [molecule] [target]"
    echo ""
    echo "Commands:"
    echo "  submit [mol]         - Submit workflows (all or specific molecule)"
    echo "  status [mol] [--verbose|-v] - Check status (compact; add --verbose for per-molecule detail)"
    echo "  cancel [mol] [calc]  - Cancel jobs (all molecules, specific molecule, or specific calc)"
    echo "  extract              - Extract results from all molecules"
    echo "  clean                - Clean temporary files"
    echo ""
    echo "Cancel examples:"
    echo "  ./workflow.sh cancel                        # Show help"
    echo "  ./workflow.sh cancel all                    # Cancel ALL jobs from ALL molecules"
    echo "  ./workflow.sh cancel ALL                    # Same as above (case-insensitive)"
    echo "  ./workflow.sh cancel MOLECULE_ID            # Cancel all jobs for specific molecule"
    echo "  ./workflow.sh cancel MOLECULE_ID all        # Cancel all jobs for specific molecule"
    echo "  ./workflow.sh cancel MOLECULE_ID ALL        # Same as above (case-insensitive)"
    echo "  ./workflow.sh cancel MOLECULE_ID calc_name  # Cancel specific calculation"
    echo ""
    echo "Available molecules:"
    for mol in "${{MOLECULES[@]}}"; do
        echo "    $mol"
    done
}}

submit_workflows() {{
    local target="${{1:-all}}"
    
    if [[ "$target" == "all" ]]; then
        echo "Submitting all molecules..."
        echo "=========================="
        for mol in "${{MOLECULES[@]}}"; do
            echo -e "\\n${{BLUE}}Submitting $mol...${{NC}}"
            cd "$BASE_DIR/$mol" && ./submit.sh
        done
    else
        if [[ -d "$BASE_DIR/$target" ]]; then
            echo -e "${{BLUE}}Submitting $target...${{NC}}"
            cd "$BASE_DIR/$target" && ./submit.sh
        else
            echo -e "${{RED}}Error: Molecule $target not found${{NC}}"
            exit 1
        fi
    fi
}}

check_status() {{
    local target="${{1:-all}}"
    VERBOSE=false
    [[ "${{2:-}}" == "--verbose" || "${{2:-}}" == "-v" ]] && VERBOSE=true
    
    if [[ "$target" == "all" ]]; then
        # Cache SLURM queue state (single call for entire status check)
        SQUEUE_CACHE=$(squeue -u $USER -o "%.18i %.30j %.8T %.10M" 2>/dev/null || echo "")
        
        echo "======================================================================================================="
        echo "  BATCH WORKFLOW STATUS - $(date '+%Y-%m-%d %H:%M:%S')"
        echo "======================================================================================================="
        echo ""
        
        # Calculation order (dynamically generated from YAML config)
        # These arrays are defined at the top of the script
        
        # Show legend before table
        echo -e "Status Legend: ${{GREEN}}[OK]${{NC}}=Complete  ${{RED}}[FAIL]${{NC}}=Failed  ${{BLUE}}[RUN]${{NC}}=Running  ${{YELLOW}}[WAIT]${{NC}}=Pending  -=Disabled"
        echo ""
        
        # Table header - compact view with calculation columns
        printf "${{BLUE}}%-30s" "Molecule"
        for abbr in "${{CALC_ORDER[@]}}"; do
            printf " %-6s" "$abbr"
        done
        printf " %-12s %-12s${{NC}}\\n" "Progress" "Status"
        
        # Separator line
        printf "%-30s" "$(printf '=%.0s' {{1..30}})"
        for abbr in "${{CALC_ORDER[@]}}"; do
            printf " %-6s" "======"
        done
        printf " %-12s %-12s\\n" "============" "============"
        
        GRAND_TOTAL_COMPLETE=0
        GRAND_TOTAL_FAILED=0
        GRAND_TOTAL_PENDING=0
        GRAND_TOTAL_CALCS=0
        
        for mol in "${{MOLECULES[@]}}"; do
            if [[ -d "$BASE_DIR/$mol" ]]; then
                cd "$BASE_DIR/$mol"
                
                # Count statuses
                COMPLETE=0
                FAILED=0
                PENDING=0
                TOTAL=0
                
                # Store status symbols for each calculation
                declare -A CALC_STATUS
                
                # Check conformer search
                if [[ -d "s0_conf_search" ]]; then
                    ((TOTAL++))
                    if [[ -f "s0_conf_search/best_conformer.txt" ]]; then
                        CALC_STATUS["s0_conf_search"]="${{GREEN}}+${{NC}}"
                        ((COMPLETE++))
                    else
                        # Check if conformer search job is in queue and get its state
                        CONF_STATE=""
                        if [[ -f "s0_conf_search/.jobid" ]]; then
                            CONF_JOB_ID=$(cat "s0_conf_search/.jobid" 2>/dev/null | tr -d '[:space:]')
                            if [[ -n "$CONF_JOB_ID" ]]; then
                                CONF_STATE=$(echo "$SQUEUE_CACHE" | awk -v id="$CONF_JOB_ID" '$1 == id {{print $3}}')
                            fi
                        fi
                        
                        # If not found by jobid, try by name
                        if [[ -z "$CONF_STATE" ]]; then
                            CONF_INFO=$(echo "$SQUEUE_CACHE" | grep "${{mol:0:8}}_conf" | head -1)
                            if [[ -n "$CONF_INFO" ]]; then
                                CONF_STATE=$(echo "$CONF_INFO" | awk '{{print $3}}')
                            fi
                        fi
                        
                        if [[ "$CONF_STATE" == "RUNNING" ]] || [[ "$CONF_STATE" == "R" ]]; then
                            # Actively running
                            CALC_STATUS["s0_conf_search"]="${{BLUE}}>${{NC}}"
                        elif [[ "$CONF_STATE" == "PENDING" ]] || [[ "$CONF_STATE" == "PD" ]]; then
                            # Queued/pending
                            CALC_STATUS["s0_conf_search"]="${{YELLOW}}.${{NC}}"
                        else
                            # Not in queue, check progress
                            CONF_DONE=$(ls -1 s0_conf_search/conf_*/status.txt 2>/dev/null | wc -l)
                            CONF_TOTAL=$(ls -1d s0_conf_search/conf_*/ 2>/dev/null | wc -l)
                            if [[ $CONF_DONE -gt 0 ]] && [[ $CONF_TOTAL -gt 0 ]]; then
                                CALC_STATUS["s0_conf_search"]="${{YELLOW}}${{CONF_DONE}}/${{CONF_TOTAL}}${{NC}}"
                            else
                                CALC_STATUS["s0_conf_search"]="${{YELLOW}}.${{NC}}"
                            fi
                        fi
                        ((PENDING++))
                    fi
                else
                    CALC_STATUS["s0_conf_search"]="-"
                fi
                
                # Check main calculations
                for calc_name in "${{CALC_NAMES[@]}}"; do
                    if [[ "$calc_name" == "s0_conf_search" ]]; then
                        continue
                    fi
                    
                    if [[ -d "$calc_name" ]]; then
                        ((TOTAL++))
                        
                        # Read saved Job ID if available
                        SAVED_JOB_ID=""
                        if [[ -f "$calc_name/.jobid" ]]; then
                            SAVED_JOB_ID=$(cat "$calc_name/.jobid" 2>/dev/null | tr -d '[:space:]')
                        fi
                        
                        # Check if job is in queue and get its state
                        JOB_STATE=""
                        if [[ -n "$SAVED_JOB_ID" ]]; then
                            JOB_STATE=$(echo "$SQUEUE_CACHE" | awk -v id="$SAVED_JOB_ID" '$1 == id {{print $3}}')
                        fi
                        
                        # If not found by Job ID, try by name pattern
                        if [[ -z "$JOB_STATE" ]]; then
                            JOB_INFO=$(echo "$SQUEUE_CACHE" | grep "${{mol:0:8}}_${{calc_name:0:8}}" | head -1)
                            if [[ -n "$JOB_INFO" ]]; then
                                JOB_STATE=$(echo "$JOB_INFO" | awk '{{print $3}}')
                            fi
                        fi
                        
                        if [[ "$JOB_STATE" == "RUNNING" ]] || [[ "$JOB_STATE" == "R" ]]; then
                            # Job is actively running
                            CALC_STATUS["$calc_name"]="${{BLUE}}>${{NC}}"
                            ((PENDING++))
                        elif [[ "$JOB_STATE" == "PENDING" ]] || [[ "$JOB_STATE" == "PD" ]]; then
                            # Job is queued/pending
                            CALC_STATUS["$calc_name"]="${{YELLOW}}.${{NC}}"
                            ((PENDING++))
                        elif [[ -f "$calc_name/mol_${{calc_name}}.log" ]] && grep -q "Normal termination" "$calc_name/mol_${{calc_name}}.log" 2>/dev/null; then
                            # Completed successfully
                            CALC_STATUS["$calc_name"]="${{GREEN}}+${{NC}}"
                            ((COMPLETE++))
                        elif [[ -f "$calc_name/status.txt" ]]; then
                            # Check status file
                            STATUS=$(cat "$calc_name/status.txt" 2>/dev/null)
                            if [[ "$STATUS" == "SUCCESS" ]]; then
                                CALC_STATUS["$calc_name"]="${{GREEN}}+${{NC}}"
                                ((COMPLETE++))
                            else
                                CALC_STATUS["$calc_name"]="${{RED}}X${{NC}}"
                                ((FAILED++))
                            fi
                        elif [[ -f "$calc_name/mol_${{calc_name}}.log" ]] && grep -q "Error termination" "$calc_name/mol_${{calc_name}}.log" 2>/dev/null; then
                            # Failed
                            CALC_STATUS["$calc_name"]="${{RED}}X${{NC}}"
                            ((FAILED++))
                        else
                            # Pending (not started or waiting for dependencies)
                            CALC_STATUS["$calc_name"]="${{YELLOW}}.${{NC}}"
                            ((PENDING++))
                        fi
                    else
                        CALC_STATUS["$calc_name"]="-"
                    fi
                done
                
                # Determine overall molecule status
                if [[ $TOTAL -eq 0 ]]; then
                    MOL_COLOR=$NC
                    STATUS_TEXT="N/A"
                elif [[ $FAILED -gt 0 ]]; then
                    MOL_COLOR=$RED
                    STATUS_TEXT="FAILED"
                elif [[ $PENDING -eq 0 ]]; then
                    MOL_COLOR=$GREEN
                    STATUS_TEXT="COMPLETE"
                else
                    MOL_COLOR=$YELLOW
                    STATUS_TEXT="RUNNING"
                fi
                
                # Print molecule row
                printf "%b" "${{MOL_COLOR}}"
                printf "%-30s" "${{mol:0:29}}"
                printf "%b" "${{NC}}"
                
                # Print status for each calculation (compact symbols)
                for calc_name in "${{CALC_NAMES[@]}}"; do
                    status_symbol="${{CALC_STATUS[$calc_name]:-"-"}}"
                    # Extract just the symbol without color codes for display
                    if [[ "$status_symbol" == *"+"* ]]; then
                        printf " %b%-6s%b" "${{GREEN}}" "[OK]" "${{NC}}"
                    elif [[ "$status_symbol" == *"X"* ]]; then
                        printf " %b%-6s%b" "${{RED}}" "[X]" "${{NC}}"
                    elif [[ "$status_symbol" == *">"* ]]; then
                        printf " %b%-6s%b" "${{BLUE}}" "[>]" "${{NC}}"
                    elif [[ "$status_symbol" == *"."* ]]; then
                        printf " %b%-6s%b" "${{YELLOW}}" "[.]" "${{NC}}"
                    else
                        printf " %-6s" "-"
                    fi
                done
                
                # Print progress and status
                if [[ $TOTAL -gt 0 ]]; then
                    PERCENT=$(( COMPLETE * 100 / TOTAL ))
                    printf " %-12s" "$COMPLETE/$TOTAL"
                    printf " %-12s\\n" "$STATUS_TEXT"
                else
                    printf " %-12s" "0/0"
                    printf " %-12s\\n" "N/A"
                fi
                
                # Optional per-molecule detailed table (verbose only)
                if $VERBOSE; then
                echo ""
                echo "  Detailed Status:"
                printf "  %-25s %-10s %-15s %-12s %-20s\\n" "Calculation" "Status" "Job ID" "Energy (Ha)" "Info"
                printf "  %-25s %-10s %-15s %-12s %-20s\\n" "-------------------------" "----------" "---------------" "------------" "--------------------"
                
                for calc_name in "${{CALC_NAMES[@]}}"; do
                    status_symbol="${{CALC_STATUS[$calc_name]:-"-"}}"
                    
                    # Skip if not a directory (disabled calculation)
                    if [[ ! -d "$calc_name" && "$calc_name" != "s0_conf_search" ]]; then
                        continue
                    fi
                    
                    # Get calculation display name using helper function (avoids duplication)
                    CALC_DISPLAY=$(get_calc_display_name "$calc_name")
                    
                    # Special handling for conformer search job names
                    if [[ "$calc_name" == "s0_conf_search" ]]; then
                        JOB_PATTERN="conf"
                    else
                        JOB_PATTERN="${{mol:0:8}}_${{calc_name:0:8}}"
                    fi
                    
                    # Initialize variables
                    JOB_ID="-"
                    RUNTIME="-"
                    JOB_STATE="-"
                    ENERGY="-"
                    INFO_TEXT="Not started"
                    
                    # Read saved Job ID if available
                    SAVED_JOB_ID=""
                    if [[ "$calc_name" == "s0_conf_search" ]]; then
                        if [[ -f "s0_conf_search/.jobid" ]]; then
                            SAVED_JOB_ID=$(cat "s0_conf_search/.jobid" 2>/dev/null | tr -d '[:space:]')
                        fi
                    elif [[ -f "$calc_name/.jobid" ]]; then
                        SAVED_JOB_ID=$(cat "$calc_name/.jobid" 2>/dev/null | tr -d '[:space:]')
                    fi
                    
                    # Check if job is in SLURM queue (running or pending)
                    # Try by Job ID first (more reliable), then by name pattern
                    if [[ -n "$SAVED_JOB_ID" ]]; then
                        JOB_STATE=$(echo "$SQUEUE_CACHE" | awk -v id="$SAVED_JOB_ID" '$1 == id {{print $3}}')
                        RUNTIME=$(echo "$SQUEUE_CACHE" | awk -v id="$SAVED_JOB_ID" '$1 == id {{print $4}}')
                        if [[ -n "$JOB_STATE" ]]; then
                            JOB_ID="$SAVED_JOB_ID"
                            # Set info based on job state
                            if [[ "$JOB_STATE" == "RUNNING" ]] || [[ "$JOB_STATE" == "R" ]]; then
                                INFO_TEXT="Running ($RUNTIME)"
                            elif [[ "$JOB_STATE" == "PENDING" ]] || [[ "$JOB_STATE" == "PD" ]]; then
                                INFO_TEXT="Queued"
                                RUNTIME="-"
                            else
                                INFO_TEXT="$JOB_STATE"
                            fi
                        fi
                    fi
                    
                    # If not found by Job ID, try by name pattern
                    if [[ -z "$JOB_ID" ]] || [[ "$JOB_ID" == "-" ]]; then
                        JOB_INFO=$(echo "$SQUEUE_CACHE" | grep "$JOB_PATTERN" | head -1)
                        if [[ -n "$JOB_INFO" ]]; then
                            JOB_ID=$(echo "$JOB_INFO" | awk '{{print $1}}')
                            RUNTIME=$(echo "$JOB_INFO" | awk '{{print $4}}')
                            JOB_STATE=$(echo "$JOB_INFO" | awk '{{print $3}}')
                            
                            # Set info based on job state
                            if [[ "$JOB_STATE" == "RUNNING" ]] || [[ "$JOB_STATE" == "R" ]]; then
                                INFO_TEXT="Running ($RUNTIME)"
                            elif [[ "$JOB_STATE" == "PENDING" ]] || [[ "$JOB_STATE" == "PD" ]]; then
                                INFO_TEXT="Queued"
                                RUNTIME="-"
                            else
                                INFO_TEXT="$JOB_STATE"
                            fi
                        fi
                    fi
                    
                    # Special check for conformer search
                    if [[ "$calc_name" == "s0_conf_search" ]] && [[ -f "s0_conf_search/best_conformer.txt" ]]; then
                        ENERGY="-" 
                        INFO_TEXT="Completed"
                        # Keep job ID if we have it, otherwise show "-"
                        [[ -z "$JOB_ID" ]] && JOB_ID="-"
                    # Check if completed (overwrites job info if finished)
                    elif [[ -f "$calc_name/mol_${{calc_name}}.log" ]]; then
                        # Use saved job ID if available and not already set
                        if [[ -z "$JOB_ID" ]] || [[ "$JOB_ID" == "-" ]]; then
                            if [[ -n "$SAVED_JOB_ID" ]]; then
                                JOB_ID="$SAVED_JOB_ID"
                            fi
                        fi
                        
                        if grep -q "Normal termination" "$calc_name/mol_${{calc_name}}.log" 2>/dev/null; then
                            ENERGY=$(extract_energy "$calc_name/mol_${{calc_name}}.log")
                            [[ -z "$ENERGY" || "$ENERGY" == "-" ]] && ENERGY="-"
                            INFO_TEXT="Completed"
                            # Keep job ID if we have it, otherwise show "-"
                            [[ -z "$JOB_ID" ]] && JOB_ID="-"
                        else
                            ERROR_MSG=$(grep -i "error\\|Error termination" "$calc_name/mol_${{calc_name}}.log" 2>/dev/null | tail -1 | cut -c1-20)
                            [[ -n "$ERROR_MSG" ]] && INFO_TEXT="$ERROR_MSG" || INFO_TEXT="Failed"
                            # Keep job ID if we have it, otherwise show "-"
                            [[ -z "$JOB_ID" ]] && JOB_ID="-"
                        fi
                    elif [[ -f "$calc_name/status.txt" ]]; then
                        # Use saved job ID if available and not already set
                        if [[ -z "$JOB_ID" ]] || [[ "$JOB_ID" == "-" ]]; then
                            if [[ -n "$SAVED_JOB_ID" ]]; then
                                JOB_ID="$SAVED_JOB_ID"
                            fi
                        fi
                        
                        # Check status file
                        STATUS=$(cat "$calc_name/status.txt" 2>/dev/null)
                        if [[ "$STATUS" == "SUCCESS" ]]; then
                            INFO_TEXT="Completed"
                            if [[ -f "$calc_name/mol_${{calc_name}}.log" ]]; then
                                ENERGY=$(extract_energy "$calc_name/mol_${{calc_name}}.log")
                                [[ -z "$ENERGY" || "$ENERGY" == "-" ]] && ENERGY="-"
                            fi
                            # Keep job ID if we have it, otherwise show "-"
                            [[ -z "$JOB_ID" ]] && JOB_ID="-"
                        elif [[ "$STATUS" == "FAILED" ]]; then
                            INFO_TEXT="Failed"
                            # Keep job ID if we have it, otherwise show "-"
                            [[ -z "$JOB_ID" ]] && JOB_ID="-"
                        fi
                    fi
                    
                    # Print row with proper formatting
                    printf "  %-25s " "${{CALC_DISPLAY:0:25}}"
                    
                    # Print status with color, but maintain alignment
                    if [[ "$status_symbol" == *"+"* ]]; then
                        printf "%b%-10s%b " "${{GREEN}}" "[OK]" "${{NC}}"
                    elif [[ "$status_symbol" == *"X"* ]]; then
                        printf "%b%-10s%b " "${{RED}}" "[FAIL]" "${{NC}}"
                    elif [[ "$status_symbol" == *">"* ]]; then
                        printf "%b%-10s%b " "${{BLUE}}" "[RUN]" "${{NC}}"
                    elif [[ "$status_symbol" == *"."* ]]; then
                        printf "%b%-10s%b " "${{YELLOW}}" "[WAIT]" "${{NC}}"
                    else
                        printf "%-10s " "-"
                    fi
                    
                    printf "%-15s " "${{JOB_ID:0:15}}"
                    printf "%-12s " "${{ENERGY:0:12}}"
                    printf "%-20s\\n" "${{INFO_TEXT:0:20}}"
                done
                
                echo ""
                fi
                
                # Update grand totals
                ((GRAND_TOTAL_COMPLETE += COMPLETE))
                ((GRAND_TOTAL_FAILED += FAILED))
                ((GRAND_TOTAL_PENDING += PENDING))
                ((GRAND_TOTAL_CALCS += TOTAL))
            fi
        done
        
        # Print summary separator
        echo ""
        printf "%-30s" "$(printf '=%.0s' {{1..30}})"
        for abbr in "${{CALC_ORDER[@]}}"; do
            printf " %-6s" "======"
        done
        printf " %-12s %-12s\\n" "============" "============"
        
        # Print totals
        printf "%b" "${{BLUE}}"
        printf "%-30s" "TOTAL"
        printf "%b" "${{NC}}"
        # Print empty spaces for each calculation column (dynamic)
        for abbr in "${{CALC_ORDER[@]}}"; do
            printf " %-6s" ""
        done
        printf " %b" "${{GREEN}}"
        printf "%-12s" "$GRAND_TOTAL_COMPLETE/$GRAND_TOTAL_CALCS"
        printf "%b" "${{NC}}"
        if [[ $GRAND_TOTAL_FAILED -gt 0 ]]; then
            printf " %b" "${{RED}}"
            printf "%-12s" "$(( GRAND_TOTAL_FAILED )) FAIL"
            printf "%b\\n" "${{NC}}"
        else
            printf " %b" "${{GREEN}}"
            printf "%-12s" "OK"
            printf "%b\\n" "${{NC}}"
        fi
        
        echo ""
        echo "======================================================================================================="
        echo ""
        
        # Calculate and show percentages
        if [[ $GRAND_TOTAL_CALCS -gt 0 ]]; then
            PERCENT_COMPLETE=$(( GRAND_TOTAL_COMPLETE * 100 / GRAND_TOTAL_CALCS ))
            PERCENT_FAILED=$(( GRAND_TOTAL_FAILED * 100 / GRAND_TOTAL_CALCS ))
            PERCENT_PENDING=$(( GRAND_TOTAL_PENDING * 100 / GRAND_TOTAL_CALCS ))
            
            echo "Overall Progress Summary:"
            echo -e "  ${{GREEN}}Complete: $GRAND_TOTAL_COMPLETE/$GRAND_TOTAL_CALCS ($PERCENT_COMPLETE%)${{NC}}"
            echo -e "  ${{RED}}Failed: $GRAND_TOTAL_FAILED ($PERCENT_FAILED%)${{NC}}"
            echo -e "  ${{YELLOW}}Pending: $((GRAND_TOTAL_CALCS - GRAND_TOTAL_COMPLETE - GRAND_TOTAL_FAILED)) ($PERCENT_PENDING%)${{NC}}"
            echo ""
            
            # Progress bar
            BAR_LENGTH=50
            FILLED=$(( PERCENT_COMPLETE * BAR_LENGTH / 100 ))
            FAILED_BAR=$(( PERCENT_FAILED * BAR_LENGTH / 100 ))
            PENDING_BAR=$(( BAR_LENGTH - FILLED - FAILED_BAR ))
            
            printf "["
            printf "%b" "${{GREEN}}"
            printf '%*s' "$FILLED" "" | tr ' ' '#'
            printf "%b" "${{NC}}"
            printf "%b" "${{RED}}"
            printf '%*s' "$FAILED_BAR" "" | tr ' ' 'X'
            printf "%b" "${{NC}}"
            printf "%b" "${{YELLOW}}"
            printf '%*s' "$PENDING_BAR" "" | tr ' ' '-'
            printf "%b" "${{NC}}"
            printf "]\\n"
            echo ""
        fi
        
        echo ""
        echo "Detailed Calculation Information:"
        # Generate calculation info dynamically from CALC_ORDER and CALC_NAMES
        for i in "${{!CALC_ORDER[@]}}"; do
            abbr="${{CALC_ORDER[$i]}}"
            calc_name="${{CALC_NAMES[$i]}}"
            # Generate description from calculation name
            CALC_NAME_CLEAN=$(echo "$calc_name" | tr '_' ' ')
            if [[ "$calc_name" == "s0_conf_search" ]]; then
                desc="Conformer search (YAML backend: CREST/RDKit/OpenBabel; PM7 optional)"
            elif [[ "$calc_name" == *"opt"* ]] && [[ "$calc_name" == *"freq"* ]]; then
                if [[ "$calc_name" == *"vac"* ]]; then
                    desc="$CALC_NAME_CLEAN - Optimization + Frequency (Vacuum)"
                elif [[ "$calc_name" == *"solv"* ]]; then
                    desc="$CALC_NAME_CLEAN - Optimization + Frequency (Solvated, PCM)"
                else
                    desc="$CALC_NAME_CLEAN - Optimization + Frequency"
                fi
            elif [[ "$calc_name" == *"sp"* ]] && [[ "$calc_name" == *"tddft"* ]]; then
                desc="TD-DFT Single Point - vertical excitations"
            elif [[ "$calc_name" == *"sp"* ]]; then
                desc="$CALC_NAME_CLEAN - Single Point calculation"
            else
                desc="$CALC_NAME_CLEAN"
            fi
            printf "  %-6s = %s\\n" "$abbr" "$desc"
        done
        echo ""
        echo "For detailed status of a specific molecule: ./workflow.sh status <molecule_name>"
        echo "For per-molecule detail in batch view: ./workflow.sh status all --verbose"
        
    else
        if [[ -d "$BASE_DIR/$target" ]]; then
            cd "$BASE_DIR/$target" && ./status.sh
        else
            echo -e "${{RED}}Error: Molecule $target not found${{NC}}"
            exit 1
        fi
    fi
}}

extract_results() {{
    echo "======================================"
    echo "  EXTRACTING RESULTS"
    echo "  $(date)"
    echo "======================================"
    echo ""
    
    RESULTS_FILE="$BASE_DIR/results_{self.timestamp}.csv"
    
    # Header
    echo "Molecule,Calculation,Status,Energy_Hartree,HOMO_eV,LUMO_eV,Gap_eV" > "$RESULTS_FILE"
    
    for mol in "${{MOLECULES[@]}}"; do
        if [[ -d "$BASE_DIR/$mol" ]]; then
            echo "Processing $mol..."
            
            # Check each calculation
            for calc_dir in "$BASE_DIR/$mol"/*/; do
                if [[ -d "$calc_dir" ]] && [[ "${{calc_dir}}" != *"s0_conf_search"* ]]; then
                    calc=$(basename "$calc_dir")
                    log_file="$calc_dir/mol_${{calc}}.log"
                    
                    if [[ -f "$log_file" ]]; then
                        if grep -q "Normal termination" "$log_file" 2>/dev/null; then
                            ENERGY=$(extract_energy "$log_file")
                            
                            HOMO=$(grep "Alpha  occ. eigenvalues" "$log_file" | tail -1 | awk '{{print $NF}}')
                            LUMO=$(grep "Alpha virt. eigenvalues" "$log_file" | head -1 | awk '{{print $5}}')
                            
                            # Convert to eV and calculate gap
                            if [[ -n "$HOMO" ]] && [[ -n "$LUMO" ]]; then
                                HOMO_EV=$(echo "$HOMO * 27.2114" | bc -l 2>/dev/null || echo "N/A")
                                LUMO_EV=$(echo "$LUMO * 27.2114" | bc -l 2>/dev/null || echo "N/A")
                                GAP=$(echo "($LUMO - $HOMO) * 27.2114" | bc -l 2>/dev/null || echo "N/A")
                            else
                                HOMO_EV="N/A"
                                LUMO_EV="N/A"
                                GAP="N/A"
                            fi
                            
                            echo "$mol,$calc,SUCCESS,${{ENERGY:-N/A}},${{HOMO_EV:-N/A}},${{LUMO_EV:-N/A}},${{GAP:-N/A}}" >> "$RESULTS_FILE"
                        else
                            echo "$mol,$calc,FAILED,N/A,N/A,N/A,N/A" >> "$RESULTS_FILE"
                        fi
                    else
                        echo "$mol,$calc,PENDING,N/A,N/A,N/A,N/A" >> "$RESULTS_FILE"
                    fi
                fi
            done
        fi
    done
    
    echo ""
    echo -e "${{GREEN}}Results saved to: $RESULTS_FILE${{NC}}"
    echo ""
    echo "Quick statistics:"
    echo "  Total molecules: ${{#MOLECULES[@]}}"
    echo "  Successful calcs: $(grep -c "SUCCESS" "$RESULTS_FILE")"
    echo "  Failed calcs: $(grep -c "FAILED" "$RESULTS_FILE")"
    echo "  Pending calcs: $(grep -c "PENDING" "$RESULTS_FILE")"
}}

clean_files() {{
    echo "Cleaning temporary files..."
    
    # Remove checkpoint files if requested
    read -p "Remove checkpoint files (.chk)? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        find "$BASE_DIR" -name "*.chk" -delete
        echo "  [OK] Checkpoint files removed"
    fi
    
    # Compress log files
    read -p "Compress log files (.log)? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        find "$BASE_DIR" -name "*.log" -exec gzip -9 {{}} \\;
        echo "  [OK] Log files compressed"
    fi
    
    echo "Cleanup complete"
}}

cancel_jobs() {{
    local mol_target="${{1:-}}"
    local calc_target="${{2:-}}"
    
    if [[ -z "$mol_target" ]]; then
        echo "Cancel Jobs - Usage:"
        echo "===================="
        echo ""
        echo "  ./workflow.sh cancel all                        # Cancel ALL jobs"
        echo "  ./workflow.sh cancel ALL                        # Cancel ALL jobs (case-insensitive)"
        echo "  ./workflow.sh cancel MOLECULE_ID                # Cancel all jobs for molecule"
        echo "  ./workflow.sh cancel MOLECULE_ID all            # Cancel all jobs for molecule"
        echo "  ./workflow.sh cancel MOLECULE_ID <calc_name>    # Cancel specific calculation"
        echo ""
        echo "Available molecules:"
        for mol in "${{MOLECULES[@]}}"; do
            echo "  - $mol"
        done
        echo ""
        return 0
    fi
    
    # Convert to lowercase for comparison (case-insensitive)
    local mol_target_lower=$(echo "$mol_target" | tr '[:upper:]' '[:lower:]')
    
    if [[ "$mol_target_lower" == "all" ]]; then
        echo -e "${{RED}}WARNING: This will cancel ALL jobs for ALL molecules!${{NC}}"
        read -p "Are you sure? (yes/no): " -r
        echo ""
        if [[ ! $REPLY == "yes" ]]; then
            echo "Cancelled. No jobs were cancelled."
            return 0
        fi
        
        echo "Cancelling all jobs..."
        echo "====================="
        for mol in "${{MOLECULES[@]}}"; do
            if [[ -d "$BASE_DIR/$mol" ]]; then
                echo ""
                echo -e "${{BLUE}}Cancelling jobs for: $mol${{NC}}"
                cd "$BASE_DIR/$mol" && ./cancel.sh all
            fi
        done
        echo ""
        echo "Done cancelling all jobs"
        
    elif [[ -d "$BASE_DIR/$mol_target" ]]; then
        # Cancel jobs for specific molecule
        cd "$BASE_DIR/$mol_target"
        
        if [[ -z "$calc_target" ]]; then
            # No specific calc, cancel all for this molecule
            echo -e "${{BLUE}}Cancelling all jobs for: $mol_target${{NC}}"
            ./cancel.sh all
        else
            # Cancel specific calculation (convert to lowercase for 'all' check)
            local calc_target_lower=$(echo "$calc_target" | tr '[:upper:]' '[:lower:]')
            if [[ "$calc_target_lower" == "all" ]]; then
                echo -e "${{BLUE}}Cancelling all jobs for: $mol_target${{NC}}"
                ./cancel.sh all
            else
                echo -e "${{BLUE}}Cancelling $calc_target for: $mol_target${{NC}}"
                ./cancel.sh "$calc_target"
            fi
        fi
        
    else
        echo -e "${{RED}}Error: Molecule '$mol_target' not found${{NC}}"
        echo ""
        echo "Available molecules:"
        for mol in "${{MOLECULES[@]}}"; do
            echo "  - $mol"
        done
        exit 1
    fi
}}

# Main
case "${{1}}" in
    submit)   submit_workflows "${{2}}" ;;
    status)   check_status "${{2}}" "${{3}}" ;;
    cancel)   cancel_jobs "${{2}}" "${{3}}" ;;
    extract)  extract_results ;;
    clean)    clean_files ;;
    *)        usage ;;
esac
'''
        
        # Fix bash variable syntax
        script = self.fix_bash_variables(script)
        
        with open(base_dir / "workflow.sh", 'w') as f:
            f.write(script)
        os.chmod(base_dir / "workflow.sh", 0o755)
    
    def create_diagnostic_script(self, base_dir: Path):
        """Create diagnostic script to help troubleshoot workflow issues"""
        # Generate header
        header = self.get_bash_header(
            script_name="Diagnostic Script",
            description="Diagnostic tool to troubleshoot workflow issues"
        )
        
        script = header + '''
echo "============================================"
echo "  WORKFLOW DIAGNOSTIC TOOL"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$SCRIPT_DIR"

# Colors
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
BLUE='\\033[0;34m'
NC='\\033[0m'

echo -e "${BLUE}=== SYSTEM CHECKS ===${NC}"
echo ""

# Check SLURM availability
echo -n "SLURM commands: "
if command -v squeue &> /dev/null && command -v sbatch &> /dev/null; then
    echo -e "${GREEN}[OK]${NC}"
    echo "  squeue: $(which squeue)"
    echo "  sbatch: $(which sbatch)"
else
    echo -e "${RED}[FAIL]${NC}"
    echo "  SLURM commands not found in PATH"
fi

# Check Gaussian
echo -n "Gaussian root: "
GAUSS_ROOT=""
if [[ -f "$BASE_DIR/workflow_config.yaml" ]]; then
    GAUSS_ROOT=$(grep "gaussian_root:" "$BASE_DIR/workflow_config.yaml" | awk '{print $2}' | tr -d '"')
fi
if [[ -n "$GAUSS_ROOT" ]] && [[ -d "$GAUSS_ROOT" ]]; then
    echo -e "${GREEN}[OK]${NC} $GAUSS_ROOT"
    if [[ -f "$GAUSS_ROOT/g16/bsd/g16.profile" ]]; then
        echo "  Profile script found"
    else
        echo -e "  ${YELLOW}[WARN] Profile script not found${NC}"
    fi
else
    echo -e "${RED}[FAIL]${NC} Not found or invalid"
fi

echo ""
echo -e "${BLUE}=== CONFIGURATION CHECK ===${NC}"
echo ""

if [[ -f "$BASE_DIR/workflow_config.yaml" ]]; then
    echo -e "${GREEN}[OK]${NC} Configuration file found"
    
    # Check exclude_nodes
    EXCLUDE_NODES=$(grep "exclude_nodes:" "$BASE_DIR/workflow_config.yaml" | awk '{print $2}' | tr -d '"')
    if [[ -n "$EXCLUDE_NODES" ]] && [[ "$EXCLUDE_NODES" != "null" ]]; then
        echo "  Exclude nodes: $EXCLUDE_NODES"
    else
        echo -e "  ${YELLOW}[INFO] No nodes excluded${NC}"
    fi
else
    echo -e "${RED}[FAIL]${NC} Configuration file not found"
fi

echo ""
echo -e "${BLUE}=== MOLECULE STATUS ===${NC}"
echo ""

MOL_COUNT=0
MOL_WITH_ISSUES=0

for mol_dir in "$BASE_DIR"/*/; do
    if [[ ! -d "$mol_dir" ]]; then
        continue
    fi
    
    mol=$(basename "$mol_dir")
    if [[ "$mol" == "workflow_"* ]] || [[ "$mol" == "."* ]]; then
        continue
    fi
    
    ((MOL_COUNT++))
    ISSUES=0
    
    echo "Molecule: $mol"
    
    # Check conformer search
    if [[ -d "$mol_dir/s0_conf_search" ]]; then
        echo -n "  Conformer search: "
        
        if [[ -f "$mol_dir/s0_conf_search/best_conformer.txt" ]]; then
            echo -e "${GREEN}[OK]${NC} Completed"
        elif [[ -f "$mol_dir/s0_conf_search/crest_status.txt" ]]; then
            STATUS=$(cat "$mol_dir/s0_conf_search/crest_status.txt" 2>/dev/null | tr -d '[:space:]')
            if [[ "$STATUS" == "SUCCESS" ]]; then
                echo -e "${GREEN}[OK]${NC} CREST completed"
            elif [[ "$STATUS" == "KILLED" ]]; then
                echo -e "${RED}[FAIL]${NC} CREST was killed/interrupted"
                ((ISSUES++))
            else
                echo -e "${RED}[FAIL]${NC} CREST failed"
                ((ISSUES++))
            fi
        elif [[ -f "$mol_dir/s0_conf_search/run_crest.slurm" ]]; then
            # Check if job is running
            JOB_ID=""
            if [[ -f "$mol_dir/s0_conf_search/.crest_jobid" ]]; then
                JOB_ID=$(cat "$mol_dir/s0_conf_search/.crest_jobid" 2>/dev/null | tr -d '[:space:]')
            fi
            
            if [[ -n "$JOB_ID" ]] && squeue -j "$JOB_ID" &>/dev/null 2>&1; then
                JOB_STATE=$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null)
                echo -e "${BLUE}[RUNNING]${NC} Job $JOB_ID ($JOB_STATE)"
            else
                echo -e "${YELLOW}[PENDING]${NC} Not started or finished"
            fi
        else
            echo -e "${YELLOW}[PENDING]${NC} Not configured"
        fi
        
        # Check conformer_count.txt
        if [[ -f "$mol_dir/s0_conf_search/conformer_count.txt" ]]; then
            COUNT=$(cat "$mol_dir/s0_conf_search/conformer_count.txt" 2>/dev/null | tr -d '[:space:]')
            if [[ -n "$COUNT" ]] && [[ "$COUNT" =~ ^[0-9]+$ ]] && [[ "$COUNT" -gt 0 ]]; then
                echo "    Conformers available: $COUNT"
            else
                echo -e "    ${RED}[ERROR]${NC} Invalid conformer count: '$COUNT'"
                ((ISSUES++))
            fi
        fi
    fi
    
    # Check main calculations
    for calc_dir in "$mol_dir"*/; do
        calc=$(basename "$calc_dir")
        if [[ "$calc" == "s0_conf_search" ]] || [[ ! -d "$calc_dir" ]]; then
            continue
        fi
        
        echo -n "  $calc: "
        
        if [[ -f "$calc_dir/mol_$calc.log" ]] && grep -q "Normal termination" "$calc_dir/mol_$calc.log" 2>/dev/null; then
            echo -e "${GREEN}[OK]${NC} Completed"
        elif [[ -f "$calc_dir/mol_$calc.log" ]] && grep -q "Error termination" "$calc_dir/mol_$calc.log" 2>/dev/null; then
            echo -e "${RED}[FAIL]${NC} Error in log"
            ((ISSUES++))
        else
            # Check job status
            JOB_ID=""
            if [[ -f "$calc_dir/.jobid" ]]; then
                JOB_ID=$(cat "$calc_dir/.jobid" 2>/dev/null | tr -d '[:space:]')
            fi
            
            if [[ -n "$JOB_ID" ]] && squeue -j "$JOB_ID" &>/dev/null 2>&1; then
                JOB_STATE=$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null)
                echo -e "${BLUE}[RUNNING]${NC} Job $JOB_ID ($JOB_STATE)"
            else
                echo -e "${YELLOW}[PENDING]${NC}"
            fi
        fi
    done
    
    if [[ $ISSUES -gt 0 ]]; then
        ((MOL_WITH_ISSUES++))
        echo -e "  ${RED}-> $ISSUES issue(s) found${NC}"
    fi
    
    echo ""
done

echo "============================================"
echo "Summary:"
echo "  Total molecules: $MOL_COUNT"
if [[ $MOL_WITH_ISSUES -gt 0 ]]; then
    echo -e "  ${RED}Molecules with issues: $MOL_WITH_ISSUES${NC}"
else
    echo -e "  ${GREEN}No issues detected${NC}"
fi
echo "============================================"
echo ""
echo "For detailed status: ./workflow.sh status"
echo "For specific molecule: ./workflow.sh status <molecule_name>"
'''
        
        # Fix bash variable syntax
        script = self.fix_bash_variables(script)
        
        with open(base_dir / "diagnose.sh", 'w') as f:
            f.write(script)
        os.chmod(base_dir / "diagnose.sh", 0o755)
    
    def print_workflow_diagram(self):
        """Print visual diagram of workflow execution order"""
        if self.logger:
            self.logger.section("WORKFLOW EXECUTION PLAN", level=0)
        else:
            print("\n" + "=" * 60)
            print("WORKFLOW EXECUTION PLAN")
            print("=" * 60)
        
        # Get enabled calculations
        enabled_calcs = []
        for calc_name, calc_config in self.config['calculations'].items():
            if calc_config.get('enabled', True):
                enabled_calcs.append({
                    'name': calc_name,
                    'depends_on': calc_config.get('depends_on'),
                    'route': calc_config['gaussian']['route'][:50] + '...' if len(calc_config['gaussian']['route']) > 50 else calc_config['gaussian']['route'],
                    'charge': calc_config['gaussian']['charge'],
                    'mult': calc_config['gaussian']['multiplicity']
                })
        
        # Build dependency tree
        if self.logger:
            self.logger.info("\nExecution Order (with dependencies):")
            self.logger.info("-" * 60)
        else:
            print("\nExecution Order (with dependencies):")
            print("-" * 60)
        
        # Stage 0: Conformer search (if enabled)
        stage = 0
        if self.config['conformer_search']['enabled']:
            num_conf = self.config['conformer_search']['num_conformers']
            method = self.config['conformer_search']['method']
            backend_label = (self.last_conformer_backend or self.get_conformer_backend()).upper()
            if self.logger:
                self.logger.info(f"\nSTAGE {stage}: Conformer Search")
                self.logger.info(f"  +-- s0_conf_search")
                self.logger.info(f"      |   Method: {method}")
                self.logger.info(f"      |   Generator: {backend_label}")
                self.logger.info(f"      |   Number: {num_conf} conformers")
                self.logger.info(f"      |   RMSD cutoff: {self.config['conformer_search'].get('rmsd_cutoff', 0.5)} A")
                if backend_label == 'CREST':
                    crest_cfg = self.config['conformer_search'].get('crest', {})
                    self.logger.info(f"      |   CREST GFN: {crest_cfg.get('gfn', 'gfn2').upper()}")
                    self.logger.info(f"      |   Energy window: {crest_cfg.get('ewin', 6.0)} kcal/mol")
                    self.logger.info(f"      |   Threads: {crest_cfg.get('threads', min(cpu_count(), 16))}")
                self.logger.info(f"      +-- Analysis: Select best conformer by energy")
            else:
                print(f"\nSTAGE {stage}: Conformer Search")
                print(f"  +-- s0_conf_search")
                print(f"      |   Method: {method}")
                print(f"      |   Generator: {backend_label}")
                print(f"      |   Number: {num_conf} conformers")
                print(f"      |   RMSD cutoff: {self.config['conformer_search'].get('rmsd_cutoff', 0.5)} A")
                if backend_label == 'CREST':
                    crest_cfg = self.config['conformer_search'].get('crest', {})
                    print(f"      |   CREST GFN: {crest_cfg.get('gfn', 'gfn2').upper()}")
                    print(f"      |   Energy window: {crest_cfg.get('ewin', 6.0)} kcal/mol")
                    print(f"      |   Threads: {crest_cfg.get('threads', min(cpu_count(), 16))}")
                print(f"      +-- Analysis: Select best conformer by energy")
            stage += 1
        
        # Group calculations by dependency level
        dependency_levels = {}
        
        def get_level(calc):
            """Recursively determine the dependency level of a calculation"""
            if calc['depends_on'] is None or calc['depends_on'] not in [c['name'] for c in enabled_calcs]:
                return 0
            parent = next((c for c in enabled_calcs if c['name'] == calc['depends_on']), None)
            if parent:
                return get_level(parent) + 1
            return 0
        
        # Calculate levels for all calculations
        for calc in enabled_calcs:
            level = get_level(calc)
            if level not in dependency_levels:
                dependency_levels[level] = []
            dependency_levels[level].append(calc)
        
        # Print each stage
        for level in sorted(dependency_levels.keys()):
            calcs_at_level = dependency_levels[level]
            stage_title = ""
            if level == 0:
                stage_title = "Initial Calculations"
            else:
                parents = set(c['depends_on'] for c in calcs_at_level if c['depends_on'])
                if len(parents) == 1:
                    stage_title = f"Dependent on {list(parents)[0]}"
                else:
                    stage_title = "Mixed Dependencies"
            
            if self.logger:
                self.logger.info(f"\nSTAGE {stage}: {stage_title}")
            else:
                print(f"\nSTAGE {stage}: {stage_title}")
            
            for calc in calcs_at_level:
                symbol = "  +--" if calc == calcs_at_level[-1] else "  +--"
                if self.logger:
                    self.logger.info(f"{symbol} {calc['name']}")
                    self.logger.info(f"      |   Route: {calc['route']}")
                    self.logger.info(f"      |   Charge: {calc['charge']}, Multiplicity: {calc['mult']}")
                    if calc['depends_on']:
                        self.logger.info(f"      |   Depends on: {calc['depends_on']}")
                    if calc['depends_on']:
                        self.logger.info(f"      +-- Uses checkpoint from: {calc['depends_on']}")
                    else:
                        self.logger.info(f"      +-- Fresh calculation (no checkpoint)")
                else:
                    print(f"{symbol} {calc['name']}")
                    print(f"      |   Route: {calc['route']}")
                    print(f"      |   Charge: {calc['charge']}, Multiplicity: {calc['mult']}")
                    if calc['depends_on']:
                        print(f"      |   Depends on: {calc['depends_on']}")
                    if calc['depends_on']:
                        print(f"      +-- Uses checkpoint from: {calc['depends_on']}")
                    else:
                        print(f"      +-- Fresh calculation (no checkpoint)")
            
            stage += 1
        
        # Summary
        total_calcs = len(enabled_calcs)
        if self.config['conformer_search']['enabled']:
            total_calcs += 1
        
        if self.logger:
            self.logger.section("Summary", level=1)
            self.logger.info(f"Total stages: {stage}")
            self.logger.info(f"Total calculations per molecule: {total_calcs}")
            self.logger.info("Parallel execution: Calculations at same stage run in parallel")
            self.logger.info("Sequential execution: Each stage waits for previous to complete")
            
            self.logger.section("SLURM Job Dependencies", level=1)
            if self.config['conformer_search']['enabled']:
                self.logger.info("  [Conformer Search] --> [Analysis] --> [s0_opt_freq_vac]")
            else:
                self.logger.info("  [s0_opt_freq_vac] (starts immediately)")
            
            # Show dependency arrows
            for level in sorted(dependency_levels.keys()):
                for calc in dependency_levels[level]:
                    if calc['depends_on']:
                        self.logger.info(f"  [{calc['depends_on']}] --> [{calc['name']}]")
        else:
            print("\n" + "-" * 60)
            print("Summary:")
            print(f"  Total stages: {stage}")
            print(f"  Total calculations per molecule: {total_calcs}")
            print(f"  Parallel execution: Calculations at same stage run in parallel")
            print(f"  Sequential execution: Each stage waits for previous to complete")
            print("\n" + "-" * 60)
            print("SLURM Job Dependencies:")
            if self.config['conformer_search']['enabled']:
                print("  [Conformer Search] --> [Analysis] --> [s0_opt_freq_vac]")
            else:
                print("  [s0_opt_freq_vac] (starts immediately)")
            for level in sorted(dependency_levels.keys()):
                for calc in dependency_levels[level]:
                    if calc['depends_on']:
                        print(f"  [{calc['depends_on']}] --> [{calc['name']}]")
            print("=" * 60)
    
    def run(self):
        """Main execution method"""
        self.start_logging()
        output_dir = None
        try:
            self.logger.section("GAUSSIAN WORKFLOW MANAGER", level=0)
            
            # Load configuration
            if not self.load_config():
                return 1
            
            # Check dependencies
            if not self.check_dependencies():
                self.logger.error("Fix dependency issues and try again.")
                return 1
            
            # Create output directory
            output_dir = Path(self.config['project']['output_dir'])
            if output_dir.exists():
                if self.force:
                    self.logger.warning(f"Output directory '{output_dir}' exists, overwriting (--force)")
                    shutil.rmtree(output_dir)
                elif sys.stdin.isatty():
                    self.logger.warning(f"Output directory '{output_dir}' already exists!")
                    response = input("Overwrite? [y/N]: ").strip().lower()
                    if response != 'y':
                        self.logger.info("Aborted.")
                        return 1
                    shutil.rmtree(output_dir)
                else:
                    self.logger.error(f"Output directory '{output_dir}' exists. Use --force to overwrite.")
                    return 1
            
            output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.success(f"Created output directory: {output_dir}")
            
            # Copy configuration to output directory
            shutil.copy2(self.config_file, output_dir / "workflow_config.yaml")
            
            # Process each XYZ file with progress bar
            processed = []
            self.logger.section("PROCESSING MOLECULES", level=1)
            
            # Ensure xyz_files is initialized
            if not hasattr(self, 'xyz_files') or not self.xyz_files:
                self.logger.error("No XYZ files found. Cannot proceed.")
                return 1
            
            # Create progress bar for molecule processing
            if RICH_AVAILABLE and self.verbose:
                console = Console()
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeRemainingColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task(
                        "[cyan]Processing molecules...",
                        total=len(self.xyz_files)
                    )
                    
                    for idx, xyz_file in enumerate(self.xyz_files, 1):
                        progress.update(task, description=f"[cyan]Processing: {xyz_file.name}")
                        self.logger.debug(f"Processing molecule {idx}/{len(self.xyz_files)}: {xyz_file.name}")
                        self.logger.debug(f"File path: {xyz_file}")
                        
                        mol_name = self.generate_inchikey(xyz_file)
                        
                        # Check for duplicates
                        if mol_name in processed:
                            original_name = mol_name
                            mol_name = f"{mol_name}_{len(processed)}"
                            self.logger.info(f"Duplicate InChIKey detected, renamed: {original_name} -> {mol_name}")
                        
                        self.logger.debug(f"Molecule ID: {mol_name}")
                        mol_name = self.create_molecule_workflow(xyz_file, mol_name, output_dir)
                        processed.append(mol_name)
                        self.logger.success(f"Workflow created for {mol_name}")
                        progress.update(task, advance=1)
            elif TQDM_AVAILABLE and self.verbose:
                from tqdm import tqdm
                for idx, xyz_file in enumerate(tqdm(self.xyz_files, desc="Processing molecules", unit="mol"), 1):
                    self.logger.debug(f"Processing molecule {idx}/{len(self.xyz_files)}: {xyz_file.name}")
                    self.logger.debug(f"File path: {xyz_file}")
                    
                    mol_name = self.generate_inchikey(xyz_file)
                    
                    # Check for duplicates
                    if mol_name in processed:
                        original_name = mol_name
                        mol_name = f"{mol_name}_{len(processed)}"
                        self.logger.info(f"Duplicate InChIKey detected, renamed: {original_name} -> {mol_name}")
                    
                    self.logger.debug(f"Molecule ID: {mol_name}")
                    mol_name = self.create_molecule_workflow(xyz_file, mol_name, output_dir)
                    processed.append(mol_name)
                    self.logger.success(f"Workflow created for {mol_name}")
            else:
                # Fallback: no progress bar
                for idx, xyz_file in enumerate(self.xyz_files, 1):
                    self.logger.info(f"[{idx}/{len(self.xyz_files)}] Processing: {xyz_file.name}")
                    self.logger.debug(f"File path: {xyz_file}")
                    
                    mol_name = self.generate_inchikey(xyz_file)
                    
                    # Check for duplicates
                    if mol_name in processed:
                        original_name = mol_name
                        mol_name = f"{mol_name}_{len(processed)}"
                        self.logger.info(f"Duplicate InChIKey detected, renamed: {original_name} -> {mol_name}")
                    
                    self.logger.info(f"Molecule ID: {mol_name}")
                    mol_name = self.create_molecule_workflow(xyz_file, mol_name, output_dir)
                    processed.append(mol_name)
                    self.logger.success(f"Workflow created for {mol_name}")
            
            # Create master workflow script
            self.logger.section("CREATING MASTER SCRIPTS", level=1)
            self.create_master_workflow_script(output_dir, processed)
            self.logger.success("Master workflow scripts created")
            
            # Create diagnostic script
            self.create_diagnostic_script(output_dir)
            self.logger.success("Diagnostic script created")
            
            # Create summary log
            summary = {
                'timestamp': self.timestamp,
                'config_file': self.config_file,
                'molecules': processed,
                'total_molecules': len(processed),
                'calculations_per_molecule': len([c for c in self.config['calculations'].values() if c.get('enabled', True)]),
                'conformer_search': self.config['conformer_search']['enabled'],
                'conformer_method': (self.last_conformer_backend or self.get_conformer_backend()).upper()
            }
            
            with open(output_dir / "workflow_summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Final report
            self.logger.section("WORKFLOW GENERATION COMPLETE", level=0)
            self.logger.success(f"Processed {len(processed)} molecule(s)")
            self.logger.success(f"Output directory: {output_dir}/")
            self.logger.success(f"Configuration saved: {output_dir}/workflow_config.yaml")
            backend_label = (self.last_conformer_backend or self.get_conformer_backend()).upper()
            self.logger.success(f"Conformer method: {backend_label}")
            
            # List all processed molecules with their InChIKeys
            if processed:
                self.logger.info("\nProcessed molecules:")
                for mol in processed:
                    self.logger.info(f"  - {mol}")
            
            # Log file location
            if self.log_file_path and self.log_file_path.exists():
                self.logger.success(f"Execution log: {self.log_file_path}")
            
            # Show workflow execution order
            self.print_workflow_diagram()
            
            # Display directory structure as table if rich is available
            if RICH_AVAILABLE and self.verbose:
                dir_structure = [
                    ["workflow.sh", "Master control script"],
                    ["workflow_config.yaml", "Configuration"],
                    ["workflow_summary.json", "Summary"]
                ]
                for mol in processed[:3]:
                    dir_structure.append([f"{mol}/", "Molecule directory"])
                    dir_structure.append(["  submit.sh", "Submit workflow"])
                    dir_structure.append(["  status.sh", "Check status"])
                if self.config['conformer_search']['enabled']:
                    dir_structure.append(["  s0_conf_search/", "Conformer search"])
                for calc in list(self.config['calculations'].keys())[:2]:
                    if self.config['calculations'][calc].get('enabled', True):
                        dir_structure.append([f"  {calc}/", "Calculation directory"])
                if len(processed) > 3:
                    dir_structure.append([f"... ({len(processed) - 3} more)", ""])
                
                self.logger.table("Directory Structure", ["Path", "Description"], dir_structure)
            else:
                self.logger.info("\nDirectory structure:")
                self.logger.info(f"  {output_dir}/")
                self.logger.info(f"  +-- workflow.sh              # Master control script")
                self.logger.info(f"  +-- workflow_config.yaml     # Configuration")
                self.logger.info(f"  +-- workflow_summary.json    # Summary")
                for mol in processed[:3]:
                    self.logger.info(f"  +-- {mol}/")
                    self.logger.info(f"  |   +-- submit.sh           # Submit workflow")
                    self.logger.info(f"  |   +-- status.sh           # Check status")
                    if self.config['conformer_search']['enabled']:
                        self.logger.info(f"  |   +-- s0_conf_search/     # Conformer search")
                    for calc in list(self.config['calculations'].keys())[:2]:
                        if self.config['calculations'][calc].get('enabled', True):
                            self.logger.info(f"  |   +-- {calc}/")
                    self.logger.info(f"  |   +-- ...")
                if len(processed) > 3:
                    self.logger.info(f"  +-- ... ({len(processed) - 3} more molecules)")
            
            self.logger.section("NEXT STEPS", level=1)
            self.logger.info(f"\ncd {output_dir}")
            self.logger.info("\n# Submit all workflows:")
            self.logger.info("./workflow.sh submit")
            self.logger.info("\n# Submit specific molecule:")
            if processed:
                self.logger.info(f"./workflow.sh submit {processed[0]}")
            self.logger.info("\n# Check status:")
            self.logger.info("./workflow.sh status")
            self.logger.info("\n# Extract results:")
            self.logger.info("./workflow.sh extract")
            
            return 0
        finally:
            self.stop_logging(output_dir)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Gaussian Workflow Manager: generate multi-stage Gaussian/SLURM workflows from YAML.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s --template          # Write workflow_config.yaml (CREST default; cation S0 +1/1 in template notes)
  %(prog)s                     # Read workflow_config.yaml and build workflows
  %(prog)s my_config.yaml      # Custom YAML
  %(prog)s --VERDE             # VERDE batch preset YAML (neutral S0 0/1 in that file), then run
  %(prog)s --VERDE --force     # Overwrite output dir without prompt
        '''
    )
    
    parser.add_argument('config', nargs='?', default='workflow_config.yaml',
                       help='YAML configuration file (default: workflow_config.yaml)')
    parser.add_argument('--template', action='store_true',
                       help='Generate template configuration and exit')
    parser.add_argument('--VERDE', action='store_true',
                       help='Generate VERDE-specific configuration and run workflow')
    parser.add_argument('--force', '-y', action='store_true',
                       help='Overwrite output directory without prompting')
    
    args = parser.parse_args()
    
    # Handle --VERDE flag
    if args.VERDE:
        # Use default config name or specified one
        config_file = args.config if args.config != 'workflow_config.yaml' else 'workflow_config.yaml'
        manager = GaussianWorkflowManager(config_file, force=args.force)
        
        # Create VERDE config (before starting logging to avoid duplicate messages)
        manager.create_verde_config()
        
        # Run workflow generation (this will start logging and load config internally)
        return manager.run()
    
    # Create manager
    manager = GaussianWorkflowManager(args.config, force=args.force)
    
    # Generate template if requested
    if args.template:
        manager.create_template_config()
        if manager.logger:
            manager.logger.success(f"Template created: {args.config}")
            manager.logger.info("Adjust xyz_dir, slurm paths, and gaussian charge/mult; then run without --template")
        else:
            print(f"\n[OK] Template created: {args.config}")
            print("Edit xyz_dir, slurm, and charge/mult (template: cation +1/1, cat +1/2, an -1/2); run without --template")
        return 0
    
    # Run workflow generation
    return manager.run()


if __name__ == "__main__":
    sys.exit(main())




