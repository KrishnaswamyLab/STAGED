import torch
from typing import Callable


def ode_integration(func: Callable, 
                 y0: torch.Tensor, 
                 t: torch.Tensor, 
                 method: str = 'rk4',
                 rtol: float = 1e-7,
                 atol: float = 1e-9) -> torch.Tensor:
    """
    Fixed-step ODE integrator that matches time resolution exactly.
    
    Unlike torchode.odeint method which uses adaptive time-stepping,
    this integrator uses the exact time points provided in `t`.
    
    Args:
        func: Function dy/dt = func(t, y)
        y0: Initial condition tensor
        t: Time points tensor (must be evenly spaced for fixed step)
        method: Integration method ('euler', 'rk4', 'midpoint')
        rtol: Relative tolerance (kept for compatibility)
        atol: Absolute tolerance (kept for compatibility)
    
    Returns:
        Solution tensor with shape [len(t), *y0.shape]
    """
    device = y0.device
    dtype = y0.dtype
    
    # Ensure t is sorted and on correct device
    t = t.to(device=device, dtype=dtype)
    
    dt = t[1] - t[0]
    n_steps = len(t)
    
    # Preallocate solution tensor
    solution = torch.zeros(n_steps, *y0.shape, dtype=dtype, device=device)
    solution[0] = y0
    # Choose integration method
    if method == 'euler':
        step_fn = _euler_step
    elif method == 'rk4':
        step_fn = _rk4_step
    elif method == 'midpoint':
        step_fn = _midpoint_step
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Integration loop with fixed steps
    for i in range(1, n_steps):
        t_current = t[i-1]
        y_current = solution[i-1]  # Use previous solution value
        solution[i] = step_fn(func, t_current, y_current, dt)
    
    return solution


def _euler_step(func: Callable, t: torch.Tensor, y: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """Forward Euler step: y_{n+1} = y_n + dt * f(t_n, y_n)"""
    return y + dt * func(t, y)


def _midpoint_step(func: Callable, t: torch.Tensor, y: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """Midpoint method (RK2): y_{n+1} = y_n + dt * f(t_n + dt/2, y_n + dt/2 * f(t_n, y_n))"""
    k1 = func(t, y)
    k2 = func(t + dt/2, y + dt/2 * k1)
    return y + dt * k2


def _rk4_step(func: Callable, t: torch.Tensor, y: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """Fourth-order Runge-Kutta step"""
    k1 = func(t, y)
    k2 = func(t + dt/2, y + dt/2 * k1)
    k3 = func(t + dt/2, y + dt/2 * k2)
    k4 = func(t + dt, y + dt * k3)
    return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# Example usage and test
if __name__ == "__main__":
    # Test with a simple ODE: dy/dt = -y, y(0) = 1
    # Analytical solution: y(t) = exp(-t)
    
    def simple_ode(t, y):
        return -y
    
    # Time points
    t = torch.linspace(0, 1, 100)  # 21 evenly spaced points from 0 to 2
    y0 = torch.tensor([1.0])
    
    # Solve using our fixed-step integrator
    solution = ode_integration(simple_ode, y0, t, method='rk4')
    
    # Compare with analytical solution
    analytical = torch.exp(-t).unsqueeze(1)
    error = torch.abs(solution - analytical)
    
    print(f"Max error: {error.max().item():.6f}")
    print(f"Final value (numerical): {solution[-1].item():.6f}")
    print(f"Final value (analytical): {analytical[-1].item():.6f}")
