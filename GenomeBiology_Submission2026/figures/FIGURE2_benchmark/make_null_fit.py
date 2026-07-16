# Produce null_fit.pkl.gz so the FDR panel (6.3) loads from cache instead of refitting interactively.
import gzip, pickle, io, contextlib, numpy as np, torch, anndata as ad
from leafletfa import LeafletFA
NULL="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/SIMULATION/sim_heart_K5_binom_eff0.0_seed0.h5ad"
def to_np(x): return x.detach().cpu().numpy() if isinstance(x,torch.Tensor) else np.asarray(x)
b=ad.read_h5ad(NULL)
m=LeafletFA(adata=b,K=5,junc_specific_prior=True,waypoints_use=False,input_conc_prior=None,
            print_epochs=10**6,num_epochs=1200,patience=10**6,min_delta=0,min_delta_rel=1e-6,
            ELBO_num_particles=5,num_samples=50,lr=0.05,gamma=0.5,log_wandb=False,loss_plot=False)
with contextlib.redirect_stdout(io.StringIO()):
    m.from_anndata(); m.train(num_initializations=1,seeds=[0]); m.get_all_variables()
with gzip.open("null_fit.pkl.gz","wb") as f:
    pickle.dump(dict(all_samples={"psi":to_np(m.all_samples["psi"])},
                     psi=to_np(m.psi), assign_post=to_np(m.assign_post)), f)
print("wrote null_fit.pkl.gz")
