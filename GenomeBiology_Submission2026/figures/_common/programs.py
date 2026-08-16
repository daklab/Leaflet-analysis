"""Program naming for the final figures: SP1 is the most-used program by pi, SP2 the next, ...

Program indices inside a fit are arbitrary (they change with the seed). Every figure that shows
programs must map fit index -> SP rank through this module so the naming is one convention
everywhere. Rank is by the fitted mixing weight pi (trace_pi[-1]); ties are impossible in practice.
"""
import numpy as np


def sp_order(fit):
    """fit: path or loaded npz. Returns `order` such that order[r] is the fit index of SP{r+1}."""
    z = np.load(fit, allow_pickle=True) if isinstance(fit, str) else fit
    pi = np.asarray(z["trace_pi"])[-1] if "trace_pi" in z.files else np.asarray(z["phi"]).mean(0)
    return np.argsort(-pi)


def sp_rank(fit):
    """rank[k] = SP number (1-based) of fit index k."""
    o = sp_order(fit); r = np.empty(len(o), int); r[o] = np.arange(1, len(o) + 1); return r


def sp_labels(fit):
    """labels in FIT order: ['SP3', 'SP2', 'SP5', 'SP1', ...]"""
    return [f"SP{r}" for r in sp_rank(fit)]


def to_sp(mat, fit, axis=-1):
    """reorder a matrix whose `axis` runs over fit programs into SP order (SP1 first)."""
    return np.take(np.asarray(mat), sp_order(fit), axis=axis)
