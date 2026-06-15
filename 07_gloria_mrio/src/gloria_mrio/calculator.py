"""Batched footprint decomposition calculator (Cabernard, Pfister & Hellweg, 2019), with optional C++ kernel."""

import numpy as np
import zarr
from numcodecs import Blosc
import json
import os
import gc
import time
from typing import Dict, List, Optional
from tqdm import tqdm

from .config import Config
from .index_builder import IndexBuilder
from .mrio_loader import MRIOLoader
from .utils import inverse_with_progress, mem_gb

# Try to import C++ kernel (optional)
try:
    from . import _compute_kernel as _cpp
    HAS_CPP_KERNEL = True
except ImportError:
    HAS_CPP_KERNEL = False


class FootprintCalculator:
    """
    Supply chain footprint decomposition calculator (v3 Batched)
    """

    def __init__(self, mrio: MRIOLoader, index_builder: IndexBuilder,
                 satellite: Dict[str, np.ndarray], config: Config):
        self.mrio = mrio
        self.idx = index_builder
        self.config = config
        self.dtype = getattr(np, config.dtype)

        self.n_reg = config.n_regions
        self.n_sec = config.n_sectors
        self.n_tot = self.n_reg * self.n_sec

        self.Q = satellite['Q']
        self.Q_Y = satellite.get('Q_Y')
        self.n_indicators = self.Q.shape[0]

        self._compute_d()

    def _compute_d(self):
        """Compute emission coefficients d = Q / x"""
        print("  Computing emission coefficients d...")
        x_safe = np.where(self.mrio.x > 0, self.mrio.x, 1.0)
        Q_flat = self.Q.reshape(self.n_indicators, self.n_tot)
        self.d = Q_flat / x_safe[np.newaxis, :]
        self.d = np.nan_to_num(self.d, nan=0, posinf=0, neginf=0)
        self.d = np.maximum(self.d, 0).astype(self.dtype)
        print(f"    d shape: {self.d.shape}")

    def calculate(self, indicator: int, target_names: List[str],
                  output_dir: str) -> Dict[str, any]:
        """
        Execute footprint decomposition (v3 Batched — no FDreg loop)
        """
        print(f"\n  === Footprint Calculation (v3 Batched) ===")
        print(f"  C++ kernel: {'available' if HAS_CPP_KERNEL else 'not found, using numpy'}")
        print(f"  Indicator: {indicator}")
        print(f"  Target sectors: {target_names}")
        print(f"  Memory: {mem_gb():.2f} GB")

        t0 = time.time()

        index_t_s, index_t, index_o = self.idx.get_target_indices(target_names)
        n_target = len(index_t_s)
        n_o = len(index_o)
        n_fdreg = self.idx.n_fd_groups

        print(f"  n_target: {n_target}, n_other: {n_o}, n_fdreg: {n_fdreg}")

        print("  Computing L_oo inverse...")
        A_oo = self.mrio.A[np.ix_(index_o, index_o)]
        I_oo = np.eye(n_o, dtype=self.dtype)
        L_oo = inverse_with_progress(I_oo - A_oo, dtype=self.dtype, desc="    L_oo")
        del A_oo, I_oo
        gc.collect()
        print(f"  Memory after L_oo: {mem_gb():.2f} GB")

        print("  Precomputing Y for all FD regions...")
        t_y = time.time()

        # Y_all_fdreg[i, k] = total final demand from FDreg k for product i
        # Shape: (n_tot, n_fdreg)
        Y_all_fdreg = np.zeros((self.n_tot, n_fdreg), dtype=self.dtype)
        for k in range(n_fdreg):
            fdreg_regions = np.where(self.idx.Agg_FDreg[:, k] == 1)[0]
            Y_all_fdreg[:, k] = self.mrio.Y[:, fdreg_regions].sum(axis=1).astype(self.dtype)

        # Extract Y for index_o elements: (n_o, n_fdreg)
        Y_o_all = Y_all_fdreg[index_o, :]

        print(f"    Y precomputed in {time.time() - t_y:.1f}s")

        # Build sector mapping for index_o
        rows_by_s = [self.idx.index_matrix_P[:, s].astype(np.int32) for s in range(self.n_sec)]
        cols_by_s = rows_by_s

        all_s = np.arange(self.n_sec, dtype=np.int32)
        index_o_s = np.setdiff1d(all_s, index_t_s)

        # Map: for each sector, which positions in index_o belong to it?
        index_o_sectors = (index_o % self.n_sec).astype(np.int32)
        o_sector_map = {}
        for s in range(self.n_sec):
            positions = np.where(index_o_sectors == s)[0]
            if len(positions) > 0:
                o_sector_map[s] = positions

        # Region/sector of each element in index_o (for remaining scatter-add)
        o_regions = index_o // self.n_sec
        o_sectors = index_o % self.n_sec

        d_indicator = self.d[indicator, :]

        # Precompute target sector matrices (independent of FDreg)
        print("  Precomputing target sector matrices...")
        t_pre = time.time()

        target_dL = {}
        target_dL_A_Loo = {}

        for t_idx, tsec in enumerate(index_t_s):
            idx_tsec = cols_by_s[tsec]
            L_tsec = self.mrio.L[:, idx_tsec].astype(self.dtype)
            dL = d_indicator[:, np.newaxis] * L_tsec
            target_dL[tsec] = dL

            A_tsec_o = self.mrio.A[np.ix_(idx_tsec, index_o)].astype(self.dtype)
            target_dL_A_Loo[tsec] = (dL @ A_tsec_o) @ L_oo
            del L_tsec, A_tsec_o

        d_o = d_indicator[index_o].astype(self.dtype)

        print(f"    Precomputation done in {time.time() - t_pre:.1f}s")
        print(f"  Memory after precomp: {mem_gb():.2f} GB")

        os.makedirs(output_dir, exist_ok=True)

        # zstd + BITSHUFFLE: much better compression on float32 than default
        compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)

        target_path = os.path.join(output_dir, "target.zarr")
        z_target = zarr.open(
            target_path, mode='w', zarr_format=2,
            shape=(self.n_reg, self.n_sec + 1, n_target, self.n_sec + 1, n_fdreg, 2),
            chunks=(self.n_reg, self.n_sec + 1, 1, self.n_sec + 1, 1, 2),
            dtype='float32',
            compressor=compressor
        )

        remaining_path = os.path.join(output_dir, "remaining.zarr")
        z_remaining = zarr.open(
            remaining_path, mode='w', zarr_format=2,
            shape=(self.n_reg, self.n_sec + 1, self.n_sec + 1, n_fdreg),
            chunks=(self.n_reg, self.n_sec + 1, self.n_sec + 1, 1),
            dtype='float32',
            compressor=compressor
        )

        household_path = os.path.join(output_dir, "household.zarr")
        z_household = zarr.open(
            household_path, mode='w', zarr_format=2,
            shape=(self.n_reg, n_fdreg),
            dtype='float32',
            compressor=compressor
        )

        # =============================================
        # TARGET: Batched across ALL FDregs (NO FDreg loop)
        # =============================================
        # Batched across ALL FDregs (no FDreg loop)
        print(f"  Computing target footprints (batched, {n_target} target sectors)...")
        t_target = time.time()

        for t_idx, tsec in enumerate(tqdm(index_t_s, desc="  Target sectors")):
            idx_tsec = cols_by_s[tsec]

            # --- Direct (Slot 0) ---
            # dL: (n_tot, n_reg), Y_tsec_all: (n_reg, n_fdreg)
            # Batched matmul: (n_tot, n_reg) @ (n_reg, n_fdreg) -> (n_tot, n_fdreg)
            dL = target_dL[tsec]
            Y_tsec_all = Y_all_fdreg[idx_tsec, :]     # (n_reg, n_fdreg)
            direct_all = dL @ Y_tsec_all               # (n_tot, n_fdreg)

            # Reshape n_tot -> (n_reg, n_sec) and write
            direct_reshaped = np.maximum(
                direct_all.reshape(self.n_reg, self.n_sec, n_fdreg), 0
            )
            z_target[:, :self.n_sec, t_idx, tsec, :, 0] = direct_reshaped

            # --- Indirect (Slot 1) ---
            # Key restructuring: instead of looping 164 FDregs × 1 big matmul each,
            # loop 120 sectors × 1 small matmul each (batched across all FDregs)
            dL_A_Loo = target_dL_A_Loo[tsec]  # (n_tot, n_o)

            if HAS_CPP_KERNEL:
                # C++ kernel: parallelizes the sector loop with OpenMP
                indirect_data = _cpp.compute_indirect_batched(
                    dL_A_Loo, Y_o_all, index_o_sectors,
                    self.n_reg, self.n_sec, n_fdreg
                )
            else:
                # Pure numpy: loop over final-supply sectors, batch over FDregs
                indirect_data = np.zeros(
                    (self.n_reg, self.n_sec, self.n_sec, n_fdreg), dtype=self.dtype
                )
                for s2, cols in o_sector_map.items():
                    # (n_tot, n_cols) @ (n_cols, n_fdreg) -> (n_tot, n_fdreg)
                    chunk = dL_A_Loo[:, cols] @ Y_o_all[cols, :]
                    indirect_data[:, :, s2, :] = chunk.reshape(
                        self.n_reg, self.n_sec, n_fdreg
                    )

            z_target[:, :self.n_sec, t_idx, :self.n_sec, :, 1] = np.maximum(
                indirect_data, 0
            )
            del indirect_data

        print(f"    Target done in {time.time() - t_target:.1f}s")

        print("  Computing remaining footprints (batched)...")
        t_rem = time.time()

        remaining_data = np.zeros(
            (self.n_reg, self.n_sec + 1, self.n_sec + 1, n_fdreg), dtype=self.dtype
        )

        for tsec in index_o_s:
            if tsec not in o_sector_map:
                continue
            cols = o_sector_map[tsec]

            # (n_o, n_cols) element-wise-scaled, then @ (n_cols, n_fdreg)
            sub = L_oo[:, cols]                          # (n_o, n_cols)
            d_sub = d_o[:, np.newaxis] * sub             # (n_o, n_cols)
            Y_cols_all = Y_all_fdreg[index_o[cols], :]   # (n_cols, n_fdreg)
            tmp_all = d_sub @ Y_cols_all                 # (n_o, n_fdreg)

            # Scatter-add into (n_reg, n_sec, n_fdreg)
            # Each index_o element maps to a unique (reg, sec), so no collision
            remaining_data[o_regions, o_sectors, tsec, :] += tmp_all

        z_remaining[:] = np.maximum(remaining_data, 0)
        del remaining_data
        print(f"    Remaining done in {time.time() - t_rem:.1f}s")

        if self.Q_Y is not None:
            for k in range(n_fdreg):
                fdreg_regions = np.where(self.idx.Agg_FDreg[:, k] == 1)[0]
                z_household[fdreg_regions, k] = self.Q_Y[indicator, fdreg_regions]

        del L_oo, target_dL, target_dL_A_Loo, Y_all_fdreg, Y_o_all
        gc.collect()

        metadata = {
            'indicator': indicator,
            'target_names': target_names,
            'target_sectors': {name: self.config.target_sectors[name] for name in target_names},
            'target_sector_indices': index_t_s.tolist(),
            'region_grouping': {k: (v if isinstance(v, list) else 'all')
                                for k, v in self.config.region_grouping.items()},
            'fd_group_names': self.idx.fd_group_names,
            'n_regions': self.n_reg,
            'n_sectors': self.n_sec,
            'shapes': {
                'target': list(z_target.shape),
                'remaining': list(z_remaining.shape),
                'household': list(z_household.shape),
            },
            'dimensions': {
                'target': ['Preg', 'Psec', 'Tsec', 'FSsec', 'FDreg', 'Slot'],
                'remaining': ['Preg', 'Psec', 'FSsec', 'FDreg'],
                'household': ['Preg', 'FDreg'],
            },
            'slot_meaning': {'0': 'Direct', '1': 'Indirect'},
        }

        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        E_target = z_target[:].sum()
        E_remaining = z_remaining[:].sum()
        E_household = z_household[:].sum()
        E_total = E_target + E_remaining + E_household

        E_expected = (self.d[indicator, :] * self.mrio.x).sum()
        if self.Q_Y is not None:
            E_expected += self.Q_Y[indicator, :].sum()

        rel_error = abs(E_total - E_expected) / max(E_expected, 1e-12)
        elapsed = time.time() - t0

        print(f"\n  Verification:")
        print(f"    Target:    {E_target:.6e}")
        print(f"    Remaining: {E_remaining:.6e}")
        print(f"    Household: {E_household:.6e}")
        print(f"    Total:     {E_total:.6e}")
        print(f"    Expected:  {E_expected:.6e}")
        print(f"    Rel Error: {rel_error:.4%}")
        print(f"    Time: {elapsed:.1f}s")
        print(f"    Memory: {mem_gb():.2f} GB")

        return {
            'target': target_path,
            'remaining': remaining_path,
            'household': household_path,
            'metadata': metadata_path,
            'error': rel_error,
            'elapsed': elapsed,
            'totals': {
                'target': float(E_target),
                'remaining': float(E_remaining),
                'household': float(E_household),
                'total': float(E_total),
                'expected': float(E_expected),
            }
        }