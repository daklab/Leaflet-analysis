#!/usr/bin/env python
"""
LeafletFA Model Training Script - Optimized Version
Trains a factor analysis model on mouse splicing foundation data with CPU optimizations.
"""

import os
import sys
import time
import psutil
import numpy as np
import pandas as pd
import torch
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import mudata as mu
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.decomposition import TruncatedSVD
from datetime import datetime, timedelta
import json
import gc
import gzip
import pickle
from contextlib import contextmanager
import logging

# =============================================================================
# Performance Monitoring Utilities
# =============================================================================

class PerformanceMonitor:
    """Track timing and resource usage throughout training"""
    
    def __init__(self):
        self.timings = {}
        self.memory_usage = {}
        self.start_time = None
        self.process = psutil.Process()
        
    def start(self):
        """Start overall timing"""
        self.start_time = time.time()
        self.memory_usage['start'] = self.get_memory_info()
        
    def get_memory_info(self):
        """Get current memory usage"""
        mem_info = self.process.memory_info()
        return {
            'rss_gb': mem_info.rss / (1024**3),  # Resident Set Size in GB
            'vms_gb': mem_info.vms / (1024**3),  # Virtual Memory Size in GB
            'percent': self.process.memory_percent()
        }
    
    @contextmanager
    def timer(self, name):
        """Context manager for timing code blocks"""
        start = time.time()
        start_mem = self.get_memory_info()
        
        print(f"\n{'='*60}")
        print(f"Starting: {name}")
        print(f"Memory: {start_mem['rss_gb']:.2f} GB RSS ({start_mem['percent']:.1f}%)")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        try:
            yield
        finally:
            elapsed = time.time() - start
            end_mem = self.get_memory_info()
            mem_delta = end_mem['rss_gb'] - start_mem['rss_gb']
            
            self.timings[name] = elapsed
            self.memory_usage[name] = {
                'start': start_mem,
                'end': end_mem,
                'delta_gb': mem_delta
            }
            
            print(f"\n✓ Completed: {name}")
            print(f"  Time: {self.format_time(elapsed)}")
            print(f"  Memory Δ: {mem_delta:+.2f} GB")
            print(f"  Current Memory: {end_mem['rss_gb']:.2f} GB")
            print(f"{'='*60}\n")
    
    @staticmethod
    def format_time(seconds):
        """Format seconds into human-readable string"""
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 3600:
            return f"{seconds/60:.2f} minutes"
        else:
            return f"{seconds/3600:.2f} hours"
    
    def get_total_time(self):
        """Get total elapsed time"""
        if self.start_time:
            return time.time() - self.start_time
        return 0
    
    def print_summary(self):
        """Print comprehensive performance summary"""
        total_time = self.get_total_time()
        
        print("\n" + "="*80)
        print(" PERFORMANCE SUMMARY ".center(80, "="))
        print("="*80)
        
        # Overall metrics
        print(f"\n📊 OVERALL METRICS:")
        print(f"  Total Runtime: {self.format_time(total_time)}")
        print(f"  Start Time: {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Memory usage
        current_mem = self.get_memory_info()
        print(f"\n💾 MEMORY USAGE:")
        print(f"  Initial: {self.memory_usage['start']['rss_gb']:.2f} GB")
        print(f"  Current: {current_mem['rss_gb']:.2f} GB")
        print(f"  Peak Delta: {max([m.get('delta_gb', 0) for m in self.memory_usage.values() if isinstance(m, dict)]):.2f} GB")
        
        # Timing breakdown
        print(f"\n⏱️  TIMING BREAKDOWN:")
        sorted_timings = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)
        for name, duration in sorted_timings:
            percentage = (duration / total_time) * 100
            print(f"  {name:<40} {self.format_time(duration):>15} ({percentage:5.1f}%)")
        
        # Model training specific metrics
        if 'Model Training' in self.timings:
            training_time = self.timings['Model Training']
            print(f"\n🎯 TRAINING METRICS:")
            print(f"  Training Time: {self.format_time(training_time)}")
            print(f"  Training Efficiency: {(training_time/total_time)*100:.1f}% of total runtime")
        
        print("\n" + "="*80)
    
    def save_report(self, filepath):
        """Save performance report to JSON"""
        report = {
            'total_time_seconds': self.get_total_time(),
            'total_time_formatted': self.format_time(self.get_total_time()),
            'start_timestamp': self.start_time,
            'end_timestamp': time.time(),
            'timings': self.timings,
            'memory_usage': self.memory_usage,
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'cpu_count_physical': psutil.cpu_count(logical=False),
                'total_memory_gb': psutil.virtual_memory().total / (1024**3),
                'available_memory_gb': psutil.virtual_memory().available / (1024**3)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"📝 Performance report saved to: {filepath}")

# =============================================================================
# Setup Logging
# =============================================================================

def setup_logging(output_dir):
    """Configure logging with both file and console output"""
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# =============================================================================
# CPU Optimization Setup
# =============================================================================

def setup_cpu_optimizations():
    """Configure system for optimal CPU performance"""
    import multiprocessing
    
    # Get CPU information
    n_physical_cores = psutil.cpu_count(logical=False)
    n_logical_cores = psutil.cpu_count(logical=True)
    
    # Set optimal thread count (usually physical cores is best for numerical work)
    n_threads = min(n_physical_cores, 32)  # Cap at 32 for diminishing returns
    
    # Configure PyTorch
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(2)
    
    # Set environment variables for numerical libraries
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
    
    # Enable MKL optimizations if available
    if torch.backends.mkl.is_available():
        torch.backends.mkl.benchmark = True
    
    print("\n" + "="*60)
    print(" SYSTEM CONFIGURATION ".center(60, "="))
    print("="*60)
    print(f"🖥️  CPU Information:")
    print(f"  Physical Cores: {n_physical_cores}")
    print(f"  Logical Cores: {n_logical_cores}")
    print(f"  Threads for Computation: {n_threads}")
    print(f"  Total System Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"  Available Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    
    print(f"\n🔧 PyTorch Configuration:")
    print(f"  Version: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    print(f"  MKL Available: {torch.backends.mkl.is_available()}")
    print(f"  OpenMP Threads: {os.environ.get('OMP_NUM_THREADS', 'not set')}")
    
    if torch.cuda.is_available():
        print(f"  CUDA Device Count: {torch.cuda.device_count()}")
        print(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
    
    print("="*60 + "\n")
    
    return n_threads

# =============================================================================
# Main Configuration
# =============================================================================

# File paths
TRAIN_ADATA_PATH = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/072025/train_70_30_ge_splice_combined_20250730_164104.h5mu"
OUTPUT_DIR = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_OUTPUT/082025"

# Model parameters
MODEL_PARAMS = {
    "K": 20,
    "junc_specific_prior": True,
    "waypoints_use": True,
    "input_conc": None,
    "delta_fixed": None,
    "num_epochs": 300,
    "ELBO_num_particles": 3,
    "lr": 0.5,
    "gamma": 0.1,
    "min_delta": 10,
    "patience": 3,
    "num_samples": 100,
    "num_inits": 1,
    "output_dir": OUTPUT_DIR,
    "enable_cpu_optimization": True  # Enable CPU optimizations
}

# Analysis parameters
N_WAYPOINTS = 20
N_PCA_COMPONENTS = 20
N_DIM_COMPONENTS = 20
METACELL_SIZE = 200

# =============================================================================
# Main Training Function
# =============================================================================

def main():
    """Main training pipeline with performance monitoring"""
    
    # Initialize performance monitor
    monitor = PerformanceMonitor()
    monitor.start()
    
    # Setup output directory and logging
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger = setup_logging(OUTPUT_DIR)
    
    # Setup CPU optimizations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        n_threads = setup_cpu_optimizations()
        logger.info(f"CPU optimizations enabled with {n_threads} threads")
    else:
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    
    print(f"\n🚀 Starting LeafletFA Training Pipeline")
    print(f"   Device: {device}")
    print(f"   Output Directory: {OUTPUT_DIR}\n")
    
    # Configure PyTorch
    torch.set_default_tensor_type("torch.FloatTensor" if device.type == "cpu" else "torch.cuda.FloatTensor")
    torch.manual_seed(0)
    
    # Configure plotting
    sns.set_theme()
    sc.set_figure_params(figsize=(7, 7), frameon=True, dpi=80, facecolor='white')
    
    # Add custom module paths
    src_path = "/gpfs/commons/home/kisaev/Leaflet-private/src/"
    if src_path not in sys.path:
        sys.path.append(src_path)
    
    import BetaDirichletFactor.LeafletFA as LeafletFA
    import BetaDirichletFactor.utils as utils
    import BetaDirichletFactor.waypoints as wayp
    
    # =============================================================================
    # Data Loading and Preprocessing
    # =============================================================================
    
    with monitor.timer("Data Loading"):
        logger.info(f"Loading Training MuData from {TRAIN_ADATA_PATH}")
        mdata = mu.read_h5mu(TRAIN_ADATA_PATH)
        ad = mdata["splicing"]
        logger.info(f"Loaded data shape: {ad.shape}")
        logger.info(f"Found layers: {list(ad.layers.keys())}")
    
    with monitor.timer("Data Preprocessing"):
        # Reset and prepare cell indices
        ad.obs.reset_index(drop=True, inplace=True)
        ad.obs["cell_id_index"] = ad.obs.index
        
        # Reduce to junctions in ATSEs with <= 3 junctions
        original_shape = ad.shape
        ad = ad[:, ad.var["num_junctions"] <= 3].copy()
        ad.var["junction_id_index"] = np.arange(ad.shape[1])
        
        logger.info(f"Reduced data from {original_shape} to {ad.shape}")
        logger.info(f"Filtered to ATSEs with <= 3 junctions")
    
    # =============================================================================
    # Compute Centered PSI Values
    # =============================================================================
    
    with monitor.timer("PSI Computation"):
        logger.info("Computing centered PSI values...")
        
        junction_counts = ad.layers["cell_by_junction_matrix"]
        cluster_counts = ad.layers["cell_by_cluster_matrix"]
        
        # Ensure COO format
        if not isinstance(junction_counts, coo_matrix):
            junction_counts = junction_counts.tocoo()
        if not isinstance(cluster_counts, coo_matrix):
            cluster_counts = cluster_counts.tocoo()
        
        # Calculate centered PSI
        junc_ratio = wayp.calculate_centered_psi(junction_counts, cluster_counts)
        
        # Store as CSR for efficiency
        if isinstance(junc_ratio, coo_matrix):
            ad.layers["junc_ratio_centered"] = junc_ratio.tocsr()
        else:
            ad.layers["junc_ratio_centered"] = junc_ratio
        
        logger.info("✓ Computed centered junction ratio layer")
    
    # =============================================================================
    # PCA Analysis
    # =============================================================================
    
    with monitor.timer("PCA Analysis"):
        logger.info(f"Computing PCA with {N_PCA_COMPONENTS} components...")
        
        svd = TruncatedSVD(n_components=N_PCA_COMPONENTS, random_state=42)
        
        # Fit and transform
        U = svd.fit_transform(ad.layers["junc_ratio_centered"])
        S = svd.singular_values_
        U_by_S = U * S
        
        # Store results
        ad.obsm['X_pca'] = U_by_S
        ad.uns['pca_explained_variance_ratio'] = svd.explained_variance_ratio_
        
        logger.info(f"✓ PCA complete")
        logger.info(f"  Top 5 explained variance: {svd.explained_variance_ratio_[:5]}")
        logger.info(f"  Cumulative variance (10 PCs): {svd.explained_variance_ratio_[:10].sum():.2%}")
    
    # =============================================================================
    # Waypoint Identification
    # =============================================================================
    
    with monitor.timer("Waypoint Identification"):
        logger.info(f"Finding {N_WAYPOINTS} waypoints...")
        
        pca_components = ad.obsm["X_pca"]
        random_seed = np.random.randint(0, 10001)
        
        waypoints = wayp.max_min_sampling(
            pca_components, 
            N_WAYPOINTS, 
            num_components=N_DIM_COMPONENTS, 
            seed=random_seed
        )
        
        logger.info(f"Assigning {METACELL_SIZE} nearest cells to each waypoint...")
        metacell_dict = wayp.assign_nearest_cells(
            waypoints, 
            pca_components, 
            num_nearest=METACELL_SIZE
        )
        
        logger.info(f"✓ Created {N_WAYPOINTS} waypoints and metacells")
    
    # =============================================================================
    # Generate Model Initializations
    # =============================================================================
    
    with monitor.timer("Initialization Generation"):
        logger.info("Computing Phi and Psi initializations...")
        
        waypoints_dict = {N_WAYPOINTS: waypoints}
        metacell_dicts = {N_WAYPOINTS: metacell_dict}
        
        rho_hat = ad.layers["junc_ratio"]
        psi_initializations, phi_initializations = wayp.generate_initializations(
            rho_hat, 
            waypoints_dict, 
            metacell_dicts, 
            epsilon=0.001
        )
        
        # Store initializations
        for i, n_waypoints in enumerate(waypoints_dict.keys()):
            logger.info(f"Storing initializations for {n_waypoints} waypoints...")
            
            psi = psi_initializations[i]
            phi = phi_initializations[i]
            
            # Convert tensors to numpy if needed
            if isinstance(psi, torch.Tensor):
                psi = psi.cpu().numpy()
            if isinstance(phi, torch.Tensor):
                phi = phi.cpu().numpy()
            
            ad.varm[f'psi_init_{n_waypoints}_waypoints'] = psi
            ad.obsm[f'phi_init_{n_waypoints}_waypoints'] = phi
        
        logger.info("✓ Successfully stored initializations")
    
    # =============================================================================
    # Model Training
    # =============================================================================
    
    with monitor.timer("Model Setup"):
        logger.info("Initializing LeafletFA model...")
        
        leaflet_model = LeafletFA.LeafletFA(
            adata=ad, 
            K=MODEL_PARAMS["K"], 
            enable_cpu_optimization=MODEL_PARAMS.get("enable_cpu_optimization", True),
            junc_specific_prior=MODEL_PARAMS["junc_specific_prior"], 
            waypoints_use=MODEL_PARAMS["waypoints_use"], 
            input_conc_prior=MODEL_PARAMS["input_conc"], 
            delta_fixed=MODEL_PARAMS["delta_fixed"],
            num_epochs=MODEL_PARAMS["num_epochs"], 
            print_epochs=1, 
            ELBO_num_particles=MODEL_PARAMS["ELBO_num_particles"], 
            lr=MODEL_PARAMS["lr"], 
            gamma=MODEL_PARAMS["gamma"], 
            min_delta=MODEL_PARAMS["min_delta"],
            num_samples=MODEL_PARAMS["num_samples"], 
            patience=MODEL_PARAMS["patience"],
            output_dir=MODEL_PARAMS["output_dir"],
            log_wandb=False
        )
        
        logger.info(f"Model initialized with:")
        logger.info(f"  K factors: {MODEL_PARAMS['K']}")
        logger.info(f"  Epochs: {MODEL_PARAMS['num_epochs']}")
        logger.info(f"  Learning rate: {MODEL_PARAMS['lr']}")
        logger.info(f"  CPU optimization: {MODEL_PARAMS.get('enable_cpu_optimization', True)}")
    
    with monitor.timer("Data Extraction"):
        logger.info("Extracting sparse tensors from AnnData...")
        leaflet_model.from_anndata()
        
        # Log data statistics
        data_density = leaflet_model.data_density if hasattr(leaflet_model, 'data_density') else None
        if data_density:
            logger.info(f"  Data density: {data_density:.1%}")
            logger.info(f"  Non-zero elements: {leaflet_model.nnz:,}")
        
        if device.type == 'cuda':
            logger.info("Initializing Triton mask for GPU operations...")
            leaflet_model.initialize_triton_mask()
    
    with monitor.timer("Model Training"):
        logger.info("Starting model training...")
        training_start = time.time()
        
        # Train model
        leaflet_model.train(num_initializations=MODEL_PARAMS["num_inits"])
        
        training_time = time.time() - training_start
        logger.info(f"✓ Training completed in {monitor.format_time(training_time)}")
        
        # Log training metrics
        if hasattr(leaflet_model, 'best_elbo'):
            logger.info(f"  Best ELBO: {leaflet_model.best_elbo:.4e}")
        if hasattr(leaflet_model, 'latent_results'):
            final_losses = [r['losses'][-1] for r in leaflet_model.latent_results]
            logger.info(f"  Final losses: {final_losses}")
    
    with monitor.timer("Results Extraction"):
        logger.info("Extracting model variables...")
        leaflet_model.get_all_variables()
        
        # Log extracted variables
        logger.info(f"  Extracted K: {leaflet_model.K}")
        if hasattr(leaflet_model, 'pi'):
            logger.info(f"  Pi shape: {leaflet_model.pi.shape}")
        if hasattr(leaflet_model, 'assign_post'):
            logger.info(f"  Assignment shape: {leaflet_model.assign_post.shape}")
    
    # =============================================================================
    # Save Model and Results
    # =============================================================================
    
    with monitor.timer("Model Saving"):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full model
        full_model_path = os.path.join(OUTPUT_DIR, f"leaflet_model_full_{stamp}.pt")
        torch.save(leaflet_model, full_model_path)
        logger.info(f"✓ Full model saved to: {full_model_path}")
        
        # Save slim model
        slim_path = os.path.join(OUTPUT_DIR, f"leaflet_model_slim_{stamp}.pkl.gz")
        
        # Clean up heavy fields
        for attr in ("adata", "y", "total_counts", "triton_mask"):
            if hasattr(leaflet_model, attr):
                setattr(leaflet_model, attr, None)
        
        # Whitelist of essential attributes
        essential_attrs = [
            "ELBO_num_particles", "K", "a", "a_rate", "a_shape", "assign_post", 
            "b", "b_rate", "b_shape", "bb_conc", "best_elbo", "best_init", 
            "gamma", "alpha_pi", "input_conc_prior", "junc_specific_prior", 
            "losses", "phi_samples", "pi", "psi_learned", "psi_samples", 
            "dir_conc", "psis_loc", "psis_scale"
        ]
        
        # Get identifiers for alignment
        obs_names = np.array(ad.obs_names)
        var_names = np.array(ad.var_names)
        
        gc.collect()
        
        with gzip.open(slim_path, "wb") as f:
            # Save metadata
            pickle.dump({
                "_meta": {
                    "obs_names": obs_names,
                    "var_names": var_names,
                    "model_params": MODEL_PARAMS,
                    "training_time": training_time,
                    "data_shape": ad.shape,
                    "data_density": data_density
                }
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save model attributes
            for attr_name in essential_attrs:
                if not hasattr(leaflet_model, attr_name):
                    continue
                val = getattr(leaflet_model, attr_name)
                
                # Convert tensors to numpy
                if isinstance(val, torch.Tensor):
                    val = val.detach().cpu().numpy()
                elif isinstance(val, (list, tuple)) and len(val) > 0:
                    if isinstance(val[0], torch.Tensor):
                        val = [v.detach().cpu().numpy() for v in val]
                
                pickle.dump({attr_name: val}, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"✓ Slim model saved to: {slim_path}")
        logger.info(f"  Slim model size: {os.path.getsize(slim_path) / (1024**2):.2f} MB")
    
    # =============================================================================
    # Final Summary
    # =============================================================================
    
    # Print performance summary
    monitor.print_summary()
    
    # Save performance report
    report_path = os.path.join(OUTPUT_DIR, f"performance_report_{stamp}.json")
    monitor.save_report(report_path)
    
    # Save final summary
    summary_path = os.path.join(OUTPUT_DIR, f"training_summary_{stamp}.txt")
    with open(summary_path, 'w') as f:
        f.write("LEAFLETFA TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Runtime: {monitor.format_time(monitor.get_total_time())}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Data Shape: {ad.shape}\n")
        f.write(f"Data Density: {data_density:.1%}\n" if data_density else "")
        f.write(f"Model K: {MODEL_PARAMS['K']}\n")
        f.write(f"Epochs: {MODEL_PARAMS['num_epochs']}\n")
        f.write(f"Best ELBO: {leaflet_model.best_elbo:.4e}\n" if hasattr(leaflet_model, 'best_elbo') else "")
        f.write("\nOutput Files:\n")
        f.write(f"  - Full Model: {full_model_path}\n")
        f.write(f"  - Slim Model: {slim_path}\n")
        f.write(f"  - Performance Report: {report_path}\n")
    
    logger.info(f"✓ Training summary saved to: {summary_path}")
    
    print("\n" + "🎉 " * 20)
    print("TRAINING COMPLETE!")
    print(f"Total time: {monitor.format_time(monitor.get_total_time())}")
    print("🎉 " * 20)

# =============================================================================
# Script Entry Point
# =============================================================================

if __name__ == "__main__":
    main()

# =============================================================================
# SLURM Submission Commands
# =============================================================================

"""
conda activate LeafletSC
cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/Leaflet/leafletFAmodel/scVI_compare

To submit this script:
script=/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/LeafletFA_analysis/04_imputation_analysis.py

# CPU run with optimizations (recommended for large memory systems)
sbatch --job-name=leaflet_opt \
       --partition=bigmem \
       --mem=900G \
       --cpus-per-task=32 \
       --time=5-00:00:00 \
       --output=leaflet_opt_%j.out \
       --error=leaflet_opt_%j.err \
       --wrap="python $script"

# GPU run (if available)
sbatch --job-name=leaflet_gpu \
       --partition=gpu \
       --gres=gpu:1 \
       --mem=200G \
       --time=2-00:00:00 \
       --output=leaflet_gpu_%j.out \
       --error=leaflet_gpu_%j.err \
       --wrap="python $script"
"""