"""
Memory-safe CSR primitives for scanning very large .h5ad matrices.

Written for the splice objects (30 GB, 0.6-1.5 billion nnz), where anything that
materialises the matrix is fatal. Everything here works on the raw h5py CSR groups
(`data` / `indices` / `indptr`) and never builds a dense array.

No plotting, no dataset knowledge — reusable by any figure or QC script.

Why raw h5py and not `anndata.read_h5ad(backed="r")`:
    backed mode's `.X.sum(axis=1)` materialises the whole matrix. These helpers read
    `data`/`indices` in row blocks and keep peak RSS at ~2 GB regardless of file size.

Measured on the mouse splice object (132,706 x 79,768, nnz 603,601,993):
    nnz_per_row              ~0.1 s   (no data read at all)
    one full scan            ~74 s    (~2 GB RSS at chunk_rows=20000)
"""

from __future__ import annotations

import numpy as np


# ── metadata ──────────────────────────────────────────────────────────────────

def csr_meta(h5, key):
    """(indptr, n_rows, n_cols, nnz) for a CSR group at `key`, e.g. 'layers/foo' or 'X'.

    Reads only indptr (n_rows+1 int64) — never data/indices.
    """
    grp = h5[key]
    enc = grp.attrs.get("encoding-type", b"")
    enc = enc.decode() if isinstance(enc, bytes) else enc
    if enc and enc != "csr_matrix":
        raise ValueError(f"{key}: expected csr_matrix, got {enc!r}")
    shape = grp.attrs.get("shape")
    indptr = grp["indptr"][:]
    n_rows = len(indptr) - 1
    n_cols = int(shape[1]) if shape is not None else int(grp["indices"][:].max()) + 1
    return indptr, n_rows, n_cols, int(grp["data"].shape[0])


def row_blocks(n_rows, chunk_rows):
    """Yield (start, stop) row-block boundaries."""
    for a in range(0, n_rows, chunk_rows):
        yield a, min(a + chunk_rows, n_rows)


# ── free: no data read ────────────────────────────────────────────────────────

def nnz_per_row(indptr):
    """Number of stored (non-zero) entries per row — i.e. features detected per cell.

    Free: derived entirely from indptr. This is the single biggest win in this module.
    """
    return np.diff(indptr).astype(np.int64)


# ── chunked scans ─────────────────────────────────────────────────────────────

def _reduceat_rows(data_chunk, indptr_chunk, start_offset):
    """Sum `data_chunk` per row, given this block's slice of indptr.

    TRAP guarded here: np.add.reduceat returns data[offset] (NOT 0) for an empty row,
    because the offset equals the next one. Rows with zero entries are explicitly zeroed.
    Mouse's min of 500 junctions/cell means this never fires there — which is precisely
    why it would go unnoticed on the next dataset.
    """
    local = indptr_chunk - start_offset
    counts = np.diff(indptr_chunk)
    if data_chunk.size == 0:
        return np.zeros(len(counts), dtype=np.float64)
    sums = np.add.reduceat(data_chunk.astype(np.float64), local[:-1])
    sums[counts == 0] = 0.0
    return sums


def scan_csr(h5, key, n_cols, chunk_rows=20_000, event_code=None, n_events=None,
             progress=None, row_mask=None):
    """One pass over a CSR matrix returning per-row and per-column aggregates.

    Deliberately does everything in a SINGLE pass — these files are I/O bound, so a
    second pass costs as much as the first.

    Returns dict with:
        row_sums          (n_rows,)  sum of values per row
        row_nnz           (n_rows,)  entries per row (free, from indptr)
        col_sums          (n_cols,)  sum of values per column
        col_nnz           (n_cols,)  rows in which each column is non-zero
      and, when `event_code` is given (length n_cols, mapping column -> group id):
        row_groups        (n_rows,)  DISTINCT groups present per row
        group_nnz         (n_events,) rows in which each group appears

    `row_mask` (bool, length n_rows) restricts which rows contribute to the COLUMN
    aggregates — use it to ask "which junctions are detected in this subset of cells?".
    Row-level outputs are still full length; rows outside the mask are left at 0.

    `event_code` handling uses a sort-unique on composite (row, group) keys.
    TRAP: group codes are NOT contiguous along column order (the mouse object has
    24,583 runs for 24,475 events), so the tempting "count change-points within a row"
    shortcut silently overcounts. Do not replace this with that.
    """
    indptr, n_rows, _, nnz = csr_meta(h5, key)
    grp = h5[key]
    d_ds, i_ds = grp["data"], grp["indices"]

    row_sums = np.zeros(n_rows, dtype=np.float64)
    col_sums = np.zeros(n_cols, dtype=np.float64)
    col_nnz = np.zeros(n_cols, dtype=np.int64)
    row_nnz = nnz_per_row(indptr)

    want_groups = event_code is not None
    if want_groups:
        event_code = np.asarray(event_code, dtype=np.int64)
        if len(event_code) != n_cols:
            raise ValueError(f"event_code length {len(event_code)} != n_cols {n_cols}")
        if n_events is None:
            n_events = int(event_code.max()) + 1
        # composite key = row * n_events + group must not overflow int64
        if n_rows * n_events >= 2 ** 62:
            raise ValueError("n_rows * n_events overflows int64 composite key")
        row_groups = np.zeros(n_rows, dtype=np.int64)
        group_nnz = np.zeros(n_events, dtype=np.int64)

    for a, b in row_blocks(n_rows, chunk_rows):
        s, e = int(indptr[a]), int(indptr[b])
        if e == s:
            continue
        idx = i_ds[s:e]
        dat = d_ds[s:e]

        # per-row sums (explicit slice assignment — see TRAP note in module docstring)
        row_sums[a:b] = _reduceat_rows(dat, indptr[a:b + 1], s)

        # per-column aggregates, optionally restricted to a subset of rows
        if row_mask is None:
            c_idx, c_dat = idx, dat
        else:
            keep_e = np.repeat(row_mask[a:b], np.diff(indptr[a:b + 1]))
            c_idx, c_dat = idx[keep_e], dat[keep_e]
        if c_idx.size:
            col_sums += np.bincount(c_idx, weights=c_dat.astype(np.float64),
                                    minlength=n_cols)
            col_nnz += np.bincount(c_idx, minlength=n_cols)

        if want_groups:
            rows_local = np.repeat(np.arange(a, b, dtype=np.int64), np.diff(indptr[a:b + 1]))
            key_ = rows_local * n_events + event_code[idx]
            key_.sort()
            first = np.empty(key_.shape, dtype=bool)
            first[0] = True
            np.not_equal(key_[1:], key_[:-1], out=first[1:])
            uk = key_[first]
            row_groups[a:b] += np.bincount(uk // n_events - a, minlength=b - a)
            group_nnz += np.bincount(uk % n_events, minlength=n_events)

        if progress is not None:
            progress(b, n_rows)

    out = dict(row_sums=row_sums, row_nnz=row_nnz, col_sums=col_sums, col_nnz=col_nnz)
    if want_groups:
        out["row_groups"] = row_groups
        out["group_nnz"] = group_nnz
    return out


def row_sums_only(h5, key, chunk_rows=50_000):
    """Per-row sums when column aggregates aren't needed (skips reading `indices`)."""
    indptr, n_rows, _, _ = csr_meta(h5, key)
    d_ds = h5[key]["data"]
    out = np.zeros(n_rows, dtype=np.float64)
    for a, b in row_blocks(n_rows, chunk_rows):
        s, e = int(indptr[a]), int(indptr[b])
        if e == s:
            continue
        out[a:b] = _reduceat_rows(d_ds[s:e], indptr[a:b + 1], s)
    return out


# ── verification helpers ──────────────────────────────────────────────────────

def verify_broadcast_layer(h5, junc_key, clus_key, event_code, n_rows=20, seed=0):
    """Check that `clus_key` is just `junc_key` summed within each event and broadcast back.

    If True, the cluster layer carries no information the junction layer lacks and can be
    skipped entirely — worth 17.5 GB of I/O on the mouse object.

    Returns (ok: bool, n_checked: int, first_mismatch: str | None).
    """
    ip_j, nr, nc, _ = csr_meta(h5, junc_key)
    ip_c, _, _, _ = csr_meta(h5, clus_key)
    jg, cg = h5[junc_key], h5[clus_key]
    rng = np.random.default_rng(seed)
    rows = rng.choice(nr, size=min(n_rows, nr), replace=False)
    event_code = np.asarray(event_code, dtype=np.int64)

    for r in rows:
        js, je = int(ip_j[r]), int(ip_j[r + 1])
        cs, ce = int(ip_c[r]), int(ip_c[r + 1])
        if je == js:
            continue
        j_idx, j_dat = jg["indices"][js:je], jg["data"][js:je]
        c_idx, c_dat = cg["indices"][cs:ce], cg["data"][cs:ce]
        totals = {}
        for col, val in zip(j_idx, j_dat):
            totals[event_code[col]] = totals.get(event_code[col], 0) + int(val)
        for col, val in zip(c_idx, c_dat):
            exp = totals.get(event_code[col])
            if exp is None or int(val) != exp:
                return False, len(rows), f"row {r} col {col}: cluster={int(val)} vs junction-sum={exp}"
    return True, len(rows), None
