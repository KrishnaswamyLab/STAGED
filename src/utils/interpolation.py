import torch
from typing import Union, Optional

class HistoryInterpolator:
    """
    Handles interpolation of discrete historical gene expression data.
    Used to provide continuous-time history for Neural ODE applications.
    """

    def __init__(
        self,
        time_points: torch.Tensor,
        historical_data: torch.Tensor, # Shape: (n_time_points, n_cells, n_genes)
        interpolation_method: str = 'linear'
    ):
        """
        Initialize interpolator.

        Args:
            time_points: 1D Tensor of time points.
            historical_data: 3D Tensor of historical data (n_time_points, n_cells, n_genes).
            interpolation_method: 'linear'. 'cubic' is a placeholder for future.
        """
        if not torch.is_tensor(time_points) or not torch.is_tensor(historical_data):
            raise TypeError("time_points and historical_data must be PyTorch tensors.")
        if time_points.ndim != 1:
            raise ValueError("time_points must be a 1D tensor.")
        if historical_data.ndim != 3:
            raise ValueError("historical_data must be a 3D tensor (time, cells, genes).")
        if time_points.shape[0] != historical_data.shape[0]:
            raise ValueError("time_points and historical_data must have the same length along the time dimension.")

        # Sort data by time points
        sorted_indices = torch.argsort(time_points)
        self.time_points = time_points[sorted_indices].float()
        self.historical_data = historical_data[sorted_indices].float() # (time, cells, genes)
        
        self.method = interpolation_method
        if self.method not in ['linear']: # Add 'cubic' etc. when implemented
            print(f"Warning: interpolation_method '{self.method}' not recognized. Defaulting to 'linear'.")
            self.method = 'linear'

        self.min_time = self.time_points[0]
        self.max_time = self.time_points[-1]
        self.device = historical_data.device
        self.num_cells = historical_data.shape[1]
        self.num_genes = historical_data.shape[2]

    def _interpolate_single_trajectory(self, single_dim_trajectory: torch.Tensor, t_query: float) -> float:
        """
        Helper for 1D linear interpolation on a single trajectory (e.g., one gene for one cell).
        Args:
            single_dim_trajectory: 1D Tensor of shape (n_time_points,).
            t_query: The time point at which to interpolate.
        Returns:
            Interpolated value (float).
        """
        # Find the interval t_query falls into
        # torch.searchsorted returns the index such that if t_query were inserted, it would maintain order.
        # If t_query is less than all time_points, idx_right = 0.
        # If t_query is greater than all time_points, idx_right = len(time_points).
        idx_right = torch.searchsorted(self.time_points, t_query)

        if idx_right == 0: # Query time is before the first known time point
            return single_dim_trajectory[0].item()
        if idx_right == len(self.time_points): # Query time is after the last known time point
            return single_dim_trajectory[-1].item()

        # Query time is between time_points[idx_right - 1] and time_points[idx_right]
        t_left, t_right = self.time_points[idx_right - 1], self.time_points[idx_right]
        val_left, val_right = single_dim_trajectory[idx_right - 1], single_dim_trajectory[idx_right]

        if t_left == t_right: # Avoid division by zero if time points are identical
            return val_left.item()
            
        # Linear interpolation formula
        alpha = (t_query - t_left) / (t_right - t_left)
        return (val_left + alpha * (val_right - val_left)).item()

    def interpolate(self, t_query: float, cell_idx: int, gene_idx: Optional[int] = None) -> Union[float, torch.Tensor]:
        """
        Interpolate data at t_query for a specific cell_idx, and optionally a specific gene_idx.

        Args:
            t_query: The time point at which to interpolate (float).
            cell_idx: Index of the cell.
            gene_idx: Optional. Index of the gene. If None, returns a 1D tensor for all genes of the cell.

        Returns:
            Interpolated value (float if gene_idx is specified) or 
            1D tensor of shape (n_genes,) (if gene_idx is None).
        """
        if not (0 <= cell_idx < self.num_cells):
            raise IndexError(f"cell_idx {cell_idx} is out of bounds for num_cells {self.num_cells}")

        # Clamp t_query to the range of known time points for extrapolation (constant extrapolation)
        clamped_t_query = max(self.min_time.item(), min(float(t_query), self.max_time.item()))
            
        cell_data_all_genes = self.historical_data[:, cell_idx, :] # Shape: (n_time_points, n_genes)

        if gene_idx is not None:
            if not (0 <= gene_idx < self.num_genes):
                raise IndexError(f"gene_idx {gene_idx} is out of bounds for num_genes {self.num_genes}")
            single_gene_trajectory = cell_data_all_genes[:, gene_idx] # Shape: (n_time_points,)
            return self._interpolate_single_trajectory(single_gene_trajectory, clamped_t_query)
        else:
            # Interpolate each gene for the specified cell independently
            interpolated_vector = torch.zeros(self.num_genes, device=self.device)
            for g_idx in range(self.num_genes):
                gene_trajectory = cell_data_all_genes[:, g_idx]
                interpolated_vector[g_idx] = self._interpolate_single_trajectory(gene_trajectory, clamped_t_query)
            return interpolated_vector

def test_history_interpolator():
    print("Testing HistoryInterpolator...")
    times = torch.tensor([0., 1., 2., 3., 4.], dtype=torch.float32)
    # Data: (time, cells, genes)
    # Cell 0, Gene 0: Values [0, 10, 20, 30, 40]
    # Cell 0, Gene 1: Values [1, 11, 21, 31, 41]
    # Cell 1, Gene 0: Values [2, 12, 22, 32, 42]
    data = torch.zeros(5, 2, 2, dtype=torch.float32)
    data[:, 0, 0] = torch.arange(0, 50, 10, dtype=torch.float32)
    data[:, 0, 1] = torch.arange(1, 50, 10, dtype=torch.float32)
    data[:, 1, 0] = torch.arange(2, 50, 10, dtype=torch.float32)
    data[:, 1, 1] = torch.arange(3, 50, 10, dtype=torch.float32) # Cell 1, Gene 1: Values [3,13,23,33,43]


    interpolator = HistoryInterpolator(times, data)

    # Test single value interpolation
    val_c0_g0_t0_5 = interpolator.interpolate(t_query=0.5, cell_idx=0, gene_idx=0)
    expected_c0_g0_t0_5 = 5.0
    print(f"Cell 0, Gene 0 at t=0.5: Actual={val_c0_g0_t0_5}, Expected={expected_c0_g0_t0_5}")
    assert abs(val_c0_g0_t0_5 - expected_c0_g0_t0_5) < 1e-5

    val_c1_g1_t2_3 = interpolator.interpolate(t_query=2.3, cell_idx=1, gene_idx=1)
    # Expected for C1,G1: at t=2 is 23, at t=3 is 33. t=2.3 -> 23 + 0.3 * (33-23) = 23 + 0.3*10 = 23+3 = 26
    expected_c1_g1_t2_3 = 26.0
    print(f"Cell 1, Gene 1 at t=2.3: Actual={val_c1_g1_t2_3}, Expected={expected_c1_g1_t2_3}")
    assert abs(val_c1_g1_t2_3 - expected_c1_g1_t2_3) < 1e-5

    # Test vector interpolation (all genes for a cell)
    vec_c0_t1_5 = interpolator.interpolate(t_query=1.5, cell_idx=0)
    # Expected for C0 at t=1.5:
    # Gene 0: val at t=1 is 10, at t=2 is 20. Midpoint = 15
    # Gene 1: val at t=1 is 11, at t=2 is 21. Midpoint = 16
    expected_vec_c0_t1_5 = torch.tensor([15.0, 16.0], device=data.device)
    print(f"Cell 0, All Genes at t=1.5: Actual={vec_c0_t1_5}, Expected={expected_vec_c0_t1_5}")
    assert torch.allclose(vec_c0_t1_5, expected_vec_c0_t1_5)

    # Test edge cases (extrapolation)
    val_c0_g0_t_minus_1 = interpolator.interpolate(t_query=-1.0, cell_idx=0, gene_idx=0)
    expected_c0_g0_t_minus_1 = 0.0 # Should clamp to data[0,0,0]
    print(f"Cell 0, Gene 0 at t=-1.0 (extrapolation): Actual={val_c0_g0_t_minus_1}, Expected={expected_c0_g0_t_minus_1}")
    assert abs(val_c0_g0_t_minus_1 - expected_c0_g0_t_minus_1) < 1e-5

    val_c0_g0_t_10 = interpolator.interpolate(t_query=10.0, cell_idx=0, gene_idx=0)
    expected_c0_g0_t_10 = 40.0 # Should clamp to data[-1,0,0]
    print(f"Cell 0, Gene 0 at t=10.0 (extrapolation): Actual={val_c0_g0_t_10}, Expected={expected_c0_g0_t_10}")
    assert abs(val_c0_g0_t_10 - expected_c0_g0_t_10) < 1e-5
    
    print("HistoryInterpolator tests passed!")

if __name__ == '__main__':
    test_history_interpolator() 