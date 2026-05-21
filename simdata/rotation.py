"""
RotationData: A crude simulator for gene expression dynamics data with inter-cell signaling.

Specifically, a 2D grid of cells will be progressively rotated into a large number of dimensions to generate a time series of embeddings for each cell. The distances between the cells (and therefore the neighbors of each cell) will therefore be invariant over time. The "gene expression embedding" of each cell will only be a subset of its physical coordinates: for instance, if there are four types of cells and 1000 dimensions, then the embedding of Type A cells will be physical coordinates 0 through 249, the embedding of Type B cells will be physical coordinates 250 through 499, and so on. 


The map M (dim x dim) is applied independently to each cell's state vector each timestep:
  X_{t+1} = M @ X_t   (per cell)

M is a composition of:
  1. Cross-cell (ligand-receptor) Givens rotations — applied FIRST, mixing one in-embedding
     coordinate (receptor) with one out-of-embedding coordinate (ligand) within the same
     dim-dimensional state vector.
  2. Intra-type Givens rotations — applied SECOND, one random rotation per cell type acting
     only on that type's coordinate block.

The rotation angle for all rotations is theta = pi / (2 * nt), so the full trajectory
spans a quarter-circle and starts and ends in different places.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import anndata
from tqdm import tqdm
from sklearn.decomposition import PCA


def _givens_rotation(dim, i, j, theta):
    """
    Return a dim x dim Givens rotation matrix that rotates in the (i, j) plane by an angle randomly chosen between 0 and theta.
    """
    # random modulate the value of theta
    theta = theta * np.random.random()

    G = np.eye(dim)
    G[i, i] =  np.cos(theta)
    G[j, j] =  np.cos(theta)
    G[i, j] = -np.sin(theta)
    G[j, i] =  np.sin(theta)
    return G

class RotationData:
    def __init__(self, n_rows, n_cols, n_types, dim, nt, n_ligand_receptors, seed=0):
        """
        Parameters
        ----------
        n_rows : int
            How many rows of cells in the grid.
        n_cols : int
            How many cols of cells in the grid.
        n_types : int
            How many different cell types.
        dim : int
            Dimensionality of each cell's state vector (must be >= n_types * 2 so each
            type block has at least 2 coordinates to rotate within).
        nt : int
            Number of timesteps. Rotation angle per step is pi / (2 * nt).
        n_ligand_receptors : int
            Number of ligand-receptor pairs per adjacent-type connection.
        seed : int
            Random seed for reproducibility.
        """
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_types = n_types
        self.dim = dim
        self.nt = nt
        self.n_ligand_receptors = n_ligand_receptors
        self.n_cells = n_rows * n_cols
        self.rng = np.random.default_rng(seed)

        # Rotation angle per timestep (about 5 cycles)
        self.theta = 4 * np.sqrt(2) * np.pi / nt

        # pos[t, cell_idx, coord] — full physical state of each cell at each timestep
        self.pos = np.zeros((nt, self.n_cells, dim))

        self._initialize_pos()
        self._initialize_types()
        self._initialize_ligand_receptors()
        self._initialize_internal_connections()
        self._initialize_map()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _initialize_pos(self):
        """
        Arrange cells on an n_rows x n_cols grid with lattice constant 1, centered at 0.
        Each cell's initial state is grid_x * u_x + grid_y * u_y, where u_x and u_y are
        two independent random unit vectors in dim-dimensional space.
        """
        xs = np.arange(self.n_cols) - (self.n_cols - 1) / 2.0
        ys = np.arange(self.n_rows) - (self.n_rows - 1) / 2.0
        grid_x, grid_y = np.meshgrid(xs, ys)          # shape (n_rows, n_cols)
        grid_x = grid_x.ravel()                        # shape (n_cells,)
        grid_y = grid_y.ravel()

        # Draw two independent random unit vectors in dim-dimensional space
        u_x = self.rng.standard_normal(self.dim)
        u_x /= np.linalg.norm(u_x)
        u_y = self.rng.standard_normal(self.dim)
        u_y /= np.linalg.norm(u_y)

        # Each cell's state: scalar grid coords projected onto the random basis vectors
        self.pos[0] = np.outer(grid_x, u_x) + np.outer(grid_y, u_y)  # (n_cells, dim)

        # Also store the initial 2D grid positions for neighbor lookup (time-invariant)
        self._grid_x = grid_x
        self._grid_y = grid_y

    def get_neighbors(self, cell_idx):
        """
        Return a list of cell indices that are 4-connected (von Neumann) neighbors
        of cell_idx in the grid. Override the connectivity check here to change the rule.
        """
        row = cell_idx // self.n_cols
        col = cell_idx % self.n_cols

        candidates = [
            (row - 1, col),   # up
            (row + 1, col),   # down
            (row, col - 1),   # left
            (row, col + 1),   # right
        ]

        neighbors = []
        for r, c in candidates:
            if 0 <= r < self.n_rows and 0 <= c < self.n_cols:
                neighbors.append(r * self.n_cols + c)
        return neighbors

    def _initialize_types(self):
        """
        Divide the dim coordinates among n_types as evenly as possible.
        self.type_dims[k] = array of coordinate indices owned by type k.
        self.cell_types[i] = type index of cell i (randomly assigned).
        """
        # np.array_split handles uneven division gracefully
        all_coords = np.arange(self.dim)
        splits = np.array_split(all_coords, self.n_types)
        self.type_dims = splits   # list of arrays, one per type

        # Randomly assign each cell a type
        self.cell_types = self.rng.integers(0, self.n_types, size=self.n_cells)

    def _initialize_ligand_receptors(self):
        """
        For each adjacent pair of types (0↔1, 1↔2, …, (n_types-2)↔(n_types-1)),
        choose n_ligand_receptors pairs of (ligand_coord, receptor_coord) where:
          - ligand_coord  comes from the SOURCE type's block (out-of-embedding for TARGET)
          - receptor_coord comes from the TARGET type's block (in-embedding for TARGET)
        The coupling is mutual, so we store pairs in both directions.

        self.ligand_receptors is a list of dicts:
          {
            'source_type': int,   # type that "emits" the ligand
            'target_type': int,   # type that "receives" via receptor
            'ligand_coord': int,  # coordinate index in source type's block
            'receptor_coord': int # coordinate index in target type's block
          }
        """
        self.ligand_receptors = []

        for t in range(self.n_types - 1):
            type_a = t
            type_b = t + 1
            coords_a = self.type_dims[type_a]
            coords_b = self.type_dims[type_b]

            if len(coords_a) < self.n_ligand_receptors or len(coords_b) < self.n_ligand_receptors:
                raise ValueError(
                    f"Not enough coordinates in type blocks to support {self.n_ligand_receptors} "
                    f"ligand-receptor pairs between types {type_a} and {type_b}."
                )

            # Sample without replacement from each block
            lig_a  = self.rng.choice(coords_a, size=self.n_ligand_receptors, replace=False)
            rec_b  = self.rng.choice(coords_b, size=self.n_ligand_receptors, replace=False)
            lig_b  = self.rng.choice(coords_b, size=self.n_ligand_receptors, replace=False)
            rec_a  = self.rng.choice(coords_a, size=self.n_ligand_receptors, replace=False)

            for k in range(self.n_ligand_receptors):
                # A -> B direction: ligand from A's block, receptor from B's block
                self.ligand_receptors.append({
                    'source_type': type_a,
                    'target_type': type_b,
                    'ligand_coord':   int(lig_a[k]),
                    'receptor_coord': int(rec_b[k]),
                })
                # B -> A direction: ligand from B's block, receptor from A's block
                self.ligand_receptors.append({
                    'source_type': type_b,
                    'target_type': type_a,
                    'ligand_coord':   int(lig_b[k]),
                    'receptor_coord': int(rec_a[k]),
                })

    def _initialize_internal_connections(self):
        """
        Build self.internal_connections: a list of (i, j) coordinate pairs where i and j
        belong to the *same* type block, representing the intra-type rotation graph.

        The current implementation uses a ring graph over each type's coordinates:
          For type with coords [c0, c1, ..., ck], edges are
          (c0,c1), (c1,c2), ..., (c_{k-1},ck), (ck,c0).
        Repeated application of the resulting rotations gradually mixes all coordinates
        within the block, since the ring is a connected graph.

        To support an arbitrary graph in the future, replace this method with any
        other set of (i, j) pairs within each type's coords.
        """
        self.internal_connections = []
        for coords in self.type_dims:
            if len(coords) < 2:
                raise Exception('Not enough coordinates to rotate!')
            n = len(coords)
            for k in range(n):
                i = int(coords[k])
                j = int(coords[(k + 1) % n])
                self.internal_connections.append((i, j))

    def _initialize_map(self):
        """
        Build the (dim x dim) map M applied to each cell's state vector each timestep.

        We collect all coordinate pairs into two lists:

          self.external_connections : list of (i, j) pairs where i and j belong to
              *different* type blocks — one per ligand-receptor entry, with
              i = ligand_coord (source type's block), j = receptor_coord (target
              type's block). Applied FIRST so LR coupling feeds into intra-type evolution.

          self.internal_connections : list of (i, j) pairs where i and j belong to the
              *same* type block — derived from a ring graph over each type's coordinates:
              edges are (coords[0],coords[1]), ..., (coords[-1],coords[0]). Applied SECOND.

        M is the composition of one Givens rotation (angle theta) per pair:
            M = M_intra @ M_lr
        where M_lr   = product over external_connections,
              M_intra = product over internal_connections.
        """
        theta = self.theta

        # --- External connections: one (ligand, receptor) pair per LR entry ---
        self.external_connections = [
            (pair['ligand_coord'], pair['receptor_coord'])
            for pair in self.ligand_receptors
        ]

        # self.internal_connections is already built by _initialize_internal_connections

        # --- Compose M_lr: external connections applied first ---
        M_lr = np.eye(self.dim)
        for (i, j) in tqdm(self.external_connections, desc = 'Building external connections: '):
            M_lr = _givens_rotation(self.dim, i, j, theta) @ M_lr

        # --- Compose M_intra: internal connections applied second ---
        M_intra = np.eye(self.dim)
        for (i, j) in tqdm(self.internal_connections, desc = 'Building internal connections: '):
            M_intra = _givens_rotation(self.dim, i, j, theta) @ M_intra

        # Final map: external first, then internal
        self.map = M_intra @ M_lr

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def run(self):
        """
        Apply the map nt-1 times to generate temporal dynamics.
        Each cell's state vector is updated independently: X_{t+1} = M @ X_t.
        Results are stored in self.pos.
        """
        for t in tqdm(range(1, self.nt), desc = 'Running Simulation: '):
            # self.pos[t-1] has shape (n_cells, dim); apply M to each row
            self.pos[t] = self.pos[t - 1] @ self.map.T   # (n_cells, dim) @ (dim, dim)^T

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, path='rotation_data.h5ad'):
        """
        Save all relevant information as an AnnData object.

        AnnData layout:
          adata.obs         — cell metadata (cell index, cell type, grid row/col)
          adata.var         — variable (coordinate) metadata (coord index, owning type)
          adata.obsm['X_spatial'] — (n_cells, 2) initial 2D grid positions
          adata.uns['pos']  — full (nt, n_cells, dim) trajectory array
          adata.uns['map']  — the (dim, dim) rotation matrix M
          adata.uns['ligand_receptors'] — list of LR pair dicts
          adata.uns['params'] — simulation hyperparameters
          adata.X           — gene expression embedding at t=0:
                              for each cell, only the coordinates in its type block,
                              zero-padded to dim so the matrix is rectangular.
        """
        import pandas as pd

        # --- obs (cells) ---
        obs = pd.DataFrame({
            'cell_idx':  np.arange(self.n_cells),
            'cell_type': self.cell_types,
            'grid_row':  np.arange(self.n_cells) // self.n_cols,
            'grid_col':  np.arange(self.n_cells) % self.n_cols,
        }, index=[f'cell_{i}' for i in range(self.n_cells)])
        obs['cell_type'] = obs['cell_type'].astype('category')

        # --- var (coordinates = "genes") ---
        coord_to_type = np.empty(self.dim, dtype=int)
        for t, coords in enumerate(self.type_dims):
            coord_to_type[coords] = t
        var = pd.DataFrame({
            'coord_idx':   np.arange(self.dim),
            'owning_type': coord_to_type,
        }, index=[f'coord_{i}' for i in range(self.dim)])

        # --- X: embedding at t=0 (each cell sees only its type's coords) ---
        X = np.zeros((self.n_cells, self.dim), dtype=np.float32)
        for cell_i in range(self.n_cells):
            t = self.cell_types[cell_i]
            coords = self.type_dims[t]
            X[cell_i, coords] = self.pos[0, cell_i, coords]

        adata = anndata.AnnData(X=X, obs=obs, var=var)

        # --- obsm ---
        adata.obsm['X_spatial'] = np.stack([self._grid_x, self._grid_y], axis=1)

        # --- uns ---
        adata.uns['pos'] = self.pos                       # full trajectory
        adata.uns['map'] = self.map
        # Store ligand_receptors as a dict of arrays (AnnData can't store list-of-dicts)
        adata.uns['ligand_receptors'] = {
            'source_type':   np.array([p['source_type']   for p in self.ligand_receptors]),
            'target_type':   np.array([p['target_type']   for p in self.ligand_receptors]),
            'ligand_coord':  np.array([p['ligand_coord']  for p in self.ligand_receptors]),
            'receptor_coord':np.array([p['receptor_coord'] for p in self.ligand_receptors]),
        }
        adata.uns['params'] = {
            'n_rows': self.n_rows,
            'n_cols': self.n_cols,
            'n_types': self.n_types,
            'dim': self.dim,
            'nt': self.nt,
            'n_ligand_receptors': self.n_ligand_receptors,
            'theta': self.theta,
        }

        print(f"Saving to {path} ...")
        adata.write_h5ad(path)
        print(f"Saved: {path}")
        return adata

    # ------------------------------------------------------------------
    # Visualize
    # ------------------------------------------------------------------

    def _visualize_pca(self, type_colors):
        """
        PCA.png — Trajectory of each cell projected into 2D PCA space (computed from
                  the full (nt * n_cells, dim) trajectory). Points are colored by cell
                  type.
        """
        # Stack all timesteps: shape (nt * n_cells, dim)
        all_states = self.pos.reshape(-1, self.dim)
        pca = PCA(n_components=2)
        all_pca = pca.fit_transform(all_states)          # (nt * n_cells, 2)
        traj_pca = all_pca.reshape(self.nt, self.n_cells, 2)  # (nt, n_cells, 2)

        fig_pca, ax_pca = plt.subplots(figsize=(7, 6))
        ax_pca.set_title('Cell trajectories in PCA space', fontsize=13)
        ax_pca.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax_pca.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')

        for cell_i in range(self.n_cells):
            t = self.cell_types[cell_i]
            color = type_colors[t]
            traj = traj_pca[:, cell_i, :]    # (nt, 2)
            ax_pca.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.4, linewidth=0.8)
            ax_pca.scatter(traj[0, 0], traj[0, 1], color=color, s=20, zorder=3)

        legend_handles = [
            mpatches.Patch(color=type_colors[t], label=f'Type {t}')
            for t in range(self.n_types)
        ]
        ax_pca.legend(handles=legend_handles, loc='best', fontsize=9)
        fig_pca.tight_layout()
        fig_pca.savefig('PCA.png', dpi=150)
        plt.close(fig_pca)
        print("Saved: PCA.png")

    def _visualize_tissue(self, type_colors):
        """
        tissue.png — The 2D grid of cells colored by type, with edges drawn between
                     cells that have at least one ligand-receptor causal connection
                     (i.e. the cell types are adjacent in the LR chain).
        """
        # Determine which pairs of cell types have a causal LR connection
        connected_type_pairs = set()
        for pair in self.ligand_receptors:
            a, b = pair['source_type'], pair['target_type']
            connected_type_pairs.add((min(a, b), max(a, b)))

        fig_t, ax_t = plt.subplots(figsize=(max(5, self.n_cols), max(5, self.n_rows)))
        ax_t.set_title('Tissue grid with causal connections', fontsize=13)
        ax_t.set_aspect('equal')
        ax_t.set_xlim(self._grid_x.min() - 0.7, self._grid_x.max() + 0.7)
        ax_t.set_ylim(self._grid_y.min() - 0.7, self._grid_y.max() + 0.7)
        ax_t.axis('off')

        # Draw edges between cells with causal LR connections
        drawn_edges = set()
        for cell_i in range(self.n_cells):
            for cell_j in self.get_neighbors(cell_i):
                edge_key = (min(cell_i, cell_j), max(cell_i, cell_j))
                if edge_key in drawn_edges:
                    continue
                ti = self.cell_types[cell_i]
                tj = self.cell_types[cell_j]
                pair_key = (min(ti, tj), max(ti, tj))
                if pair_key in connected_type_pairs:
                    xi, yi = self._grid_x[cell_i], self._grid_y[cell_i]
                    xj, yj = self._grid_x[cell_j], self._grid_y[cell_j]
                    ax_t.plot([xi, xj], [yi, yj], color='gray', linewidth=1.5,
                              alpha=0.6, zorder=1)
                    drawn_edges.add(edge_key)

        # Draw cells
        for cell_i in range(self.n_cells):
            t = self.cell_types[cell_i]
            color = type_colors[t]
            ax_t.scatter(self._grid_x[cell_i], self._grid_y[cell_i],
                         color=color, s=300, zorder=2, edgecolors='white',
                         linewidths=1.5)
            ax_t.text(self._grid_x[cell_i], self._grid_y[cell_i],
                      str(t), ha='center', va='center', fontsize=7,
                      color='white', fontweight='bold', zorder=3)

        legend_handles = [
            mpatches.Patch(color=type_colors[t], label=f'Type {t}')
            for t in range(self.n_types)
        ]
        legend_handles.append(Line2D([0], [0], color='gray', linewidth=1.5,
                                     label='Causal LR connection'))
        ax_t.legend(handles=legend_handles, loc='upper right', fontsize=9)
        fig_t.tight_layout()
        fig_t.savefig('tissue.png', dpi=150)
        plt.close(fig_t)
        print("Saved: tissue.png")

    def _visualize_trajectories(self, type_colors, n_cells_sample=5, n_genes_sample=3, seed=0):
        """
        trajectories.png — For a random sample of cells, plot the value of a random
                           sample of that cell's own-type genes over time.

        Each subplot corresponds to one sampled cell. Within each subplot, each line
        is one gene (coordinate), colored by cell type and distinguished by line style.
        Only coordinates belonging to the cell's own type block are sampled, since
        those are the ones visible in the cell's embedding.
        """
        rng = np.random.default_rng(seed)

        n_cells_sample = min(n_cells_sample, self.n_cells)
        sampled_cells = rng.choice(self.n_cells, size=n_cells_sample, replace=False)
        timesteps = np.arange(self.nt)

        line_styles = ['-', '--', ':', '-.']

        fig, axes = plt.subplots(n_cells_sample, 1,
                                 figsize=(9, 2.5 * n_cells_sample),
                                 sharex=True)
        if n_cells_sample == 1:
            axes = [axes]

        fig.suptitle('Gene expression trajectories over time', fontsize=13)

        for ax, cell_i in zip(axes, sampled_cells):
            t = self.cell_types[cell_i]
            color = type_colors[t]
            own_coords = self.type_dims[t]
            n_genes_sample_actual = min(n_genes_sample, len(own_coords))
            sampled_genes = rng.choice(own_coords, size=n_genes_sample_actual, replace=False)

            for k, gene in enumerate(sampled_genes):
                values = self.pos[:, cell_i, gene]   # (nt,)
                ls = line_styles[k % len(line_styles)]
                ax.plot(timesteps, values, color=color, linestyle=ls, linewidth=1.2,
                        label=f'coord {gene}')

            ax.set_title(f'Cell {cell_i} (type {t})', fontsize=9)
            ax.set_ylabel('Value', fontsize=8)
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(True, linewidth=0.4, alpha=0.5)

        axes[-1].set_xlabel('Timestep', fontsize=9)
        fig.tight_layout()
        fig.savefig('trajectories.png', dpi=150)
        plt.close(fig)
        print("Saved: trajectories.png")

    def visualize(self):
        """
        Produce three figures:

        PCA.png          — Cell trajectories in 2D PCA space, colored by cell type.
        tissue.png       — 2D grid of cells with causal LR-connection edges.
        trajectories.png — Gene expression over time for a sample of cells and genes.
        """
        type_colors = plt.cm.tab10(np.linspace(0, 1, self.n_types))
        self._visualize_pca(type_colors)
        self._visualize_tissue(type_colors)
        self._visualize_trajectories(type_colors)
# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == '__main__':
    sim = RotationData(
        n_rows=10,
        n_cols=10,
        n_types=3,
        dim=100,
        nt=100,
        n_ligand_receptors=2,
        seed=42,
    )
    sim.run()
    sim.visualize()
    sim.save('rotation_data.h5ad')
