import torch
from tqdm import tqdm
from enum import Enum
from typing import Callable, Optional, Tuple, List, Union

class SolverType(Enum):
    """Enum for different ODE solver types."""
    EULER = "euler"
    HEUN = "heun"
    RK4 = "rk4"
    DOPRI5 = "dopri5"


class ODESolver:
    """Base class for ODE solvers used in Flow Matching."""
    
    def __init__(self, 
                 vector_field_fn: Callable,
                 t_0: float = 0.0, 
                 t_1: float = 1.0,
                 rtol: float = 1e-3,
                 atol: float = 1e-3,
                 device=None):
        """
        Args:
            vector_field_fn: Function that computes the vector field. 
                             Should accept (x, t) and return v(x, t).
            t_0: Initial time point for integration
            t_1: Final time point for integration
            rtol: Relative tolerance (for adaptive methods)
            atol: Absolute tolerance (for adaptive methods)
            device: Computation device
        """
        self.vector_field_fn = vector_field_fn
        self.t_0 = t_0
        self.t_1 = t_1
        self.rtol = rtol
        self.atol = atol
        self.device = device
    
    def step(self, x: torch.Tensor, t: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Take a single step using the ODE solver.
        
        Args:
            x: Current state
            t: Current time
            dt: Time step size
            
        Returns:
            Updated state after step
        """
        raise NotImplementedError("Subclasses must implement step method")
    
    def integrate(self, 
                  x_0: torch.Tensor, 
                  steps: int = 100, 
                  return_trajectory: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Integrate the ODE from t_0 to t_1 starting from x_0.
        
        For Flow Matching, this usually means integrating from noise (t=0) to data (t=1).
        
        Args:
            x_0: Initial state (usually noise)
            steps: Number of integration steps
            return_trajectory: Whether to return the entire trajectory
            
        Returns:
            Final state or tuple of (final_state, trajectory)
        """
        device = self.device if self.device is not None else x_0.device
        x = x_0.to(device)
        
        # Create evenly spaced time steps for integration
        ts = torch.linspace(self.t_0, self.t_1, steps + 1, device=device)
        dt = (self.t_1 - self.t_0) / steps
        
        # Initialize trajectory storage if needed
        trajectory = [x.detach().cpu()] if return_trajectory else None
        
        # Integration loop
        for i in tqdm(range(steps), desc="Integrating flow"):
            t = ts[i].expand(x.shape[0])
            x = self.step(x, t, dt)
            
            if return_trajectory:
                trajectory.append(x.detach().cpu())
        
        if return_trajectory:
            return x, trajectory
        else:
            return x


class EulerSolver(ODESolver):
    """Euler method for ODE integration.
    
    This is the simplest numerical method for solving ODEs:
    x_{t+dt} = x_t + dt * v(x_t, t)
    """
    
    def step(self, x: torch.Tensor, t: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Take a single Euler step.
        
        Args:
            x: Current state
            t: Current time
            dt: Time step size
            
        Returns:
            Updated state after Euler step
        """
        # Compute vector field at current point
        v = self.vector_field_fn(x, t)
        
        # Update using Euler's method
        return x + dt * v


class HeunSolver(ODESolver):
    """Heun's method (2nd order Runge-Kutta) for ODE integration.
    
    This method is more accurate than Euler by using a predictor-corrector approach:
    1. Predict using an Euler step
    2. Correct by averaging the vector fields at the start and predicted points
    """
    
    def step(self, x: torch.Tensor, t: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Take a single Heun's method step.
        
        Args:
            x: Current state
            t: Current time
            dt: Time step size
            
        Returns:
            Updated state after Heun step
        """
        # First stage - Euler step (predictor)
        v1 = self.vector_field_fn(x, t)
        x_pred = x + dt * v1
        
        # Second stage - correction
        t_next = t + dt
        v2 = self.vector_field_fn(x_pred, t_next)
        
        # Combine stages
        return x + 0.5 * dt * (v1 + v2)


class RK4Solver(ODESolver):
    """4th order Runge-Kutta method for ODE integration.
    
    This method uses four stages to achieve 4th order accuracy:
    k1 = v(x_t, t)
    k2 = v(x_t + 0.5*dt*k1, t + 0.5*dt)
    k3 = v(x_t + 0.5*dt*k2, t + 0.5*dt)
    k4 = v(x_t + dt*k3, t + dt)
    x_{t+dt} = x_t + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    """
    
    def step(self, x: torch.Tensor, t: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Take a single RK4 step.
        
        Args:
            x: Current state
            t: Current time
            dt: Time step size
            
        Returns:
            Updated state after RK4 step
        """
        # First stage
        k1 = self.vector_field_fn(x, t)
        
        # Second stage
        t_half = t + 0.5 * dt
        x_half = x + 0.5 * dt * k1
        k2 = self.vector_field_fn(x_half, t_half)
        
        # Third stage
        x_half2 = x + 0.5 * dt * k2
        k3 = self.vector_field_fn(x_half2, t_half)
        
        # Fourth stage
        t_next = t + dt
        x_next = x + dt * k3
        k4 = self.vector_field_fn(x_next, t_next)
        
        # Combine all four stages
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class DormandPrinceSolver(ODESolver):
    """5th order Dormand-Prince method (DOPRI5) with adaptive step size.
    
    This is an adaptive step-size method that uses embedded 4th and 5th order
    solutions to estimate the error and adjust the step size for efficiency.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Butcher tableau coefficients for Dormand-Prince method
        self.a = [
            [],
            [1/5],
            [3/40, 9/40],
            [44/45, -56/15, 32/9],
            [19372/6561, -25360/2187, 64448/6561, -212/729],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
        ]
        
        self.b1 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]  # 5th order
        self.b2 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]  # 4th order
        
        self.c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    
    def _compute_stages(self, x, t, dt):
        """Compute the intermediate stages for the Dormand-Prince method."""
        k = [self.vector_field_fn(x, t)]
        
        for i in range(1, 7):
            t_stage = t + dt * self.c[i]
            x_stage = x.clone()
            
            for j in range(i):
                x_stage = x_stage + dt * self.a[i-1][j] * k[j]
                
            k.append(self.vector_field_fn(x_stage, t_stage))
            
        return k
    
    def step(self, x: torch.Tensor, t: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Take an adaptive step using the Dormand-Prince method.
        
        Args:
            x: Current state
            t: Current time
            dt: Initial time step size (may be adjusted)
            
        Returns:
            Updated state after step
        """
        # Start with a copy of the initial dt
        current_dt = dt.clone()
        
        while True:
            # Compute stages
            k = self._compute_stages(x, t, current_dt)
            
            # 5th order solution
            x_next_5 = x.clone()
            for i in range(6):
                x_next_5 = x_next_5 + current_dt * self.b1[i] * k[i]
                
            # 4th order solution (for error estimate)
            x_next_4 = x.clone()
            for i in range(7):
                x_next_4 = x_next_4 + current_dt * self.b2[i] * k[i]
                
            # Compute error estimate
            error = torch.max(torch.abs(x_next_5 - x_next_4))
            error_ratio = error / (self.atol + self.rtol * torch.max(torch.abs(x)))
            
            # Check if error is acceptable
            if error_ratio <= 1.0:
                return x_next_5
                
            # Adjust step size (safety factor of 0.9)
            # Limit decrease to 0.1x and increase to 2x for stability
            new_dt = current_dt * min(max(0.9 * (1.0 / error_ratio) ** 0.2, 0.1), 2.0)
            current_dt = new_dt
            
            # Prevent getting stuck in a loop with very small steps
            if current_dt < dt * 1e-6:
                print("Warning: Step size became too small, using Euler method for this step")
                v = self.vector_field_fn(x, t)
                return x + dt * v


class FlowMatchingSolver:
    """Manages ODE solving for flow matching models.
    
    This class handles the interface between the neural network model that
    predicts the vector field and the ODE solvers that integrate the flow.
    """
    
    def __init__(self, model, flow_type="rectified_flow", solver_type="rk4", 
                 steps=50, rtol=1e-3, atol=1e-3, device=None):
        """
        Args:
            model: The UNet model that predicts the vector field
            flow_type: Type of flow - 'rectified_flow' or 'probability_flow'
            solver_type: Type of ODE solver to use
            steps: Number of integration steps
            rtol: Relative tolerance (for adaptive methods)
            atol: Absolute tolerance (for adaptive methods)
            device: Computation device
        """
        self.model = model
        self.flow_type = flow_type
        self.solver_type = SolverType(solver_type)
        self.steps = steps
        self.rtol = rtol
        self.atol = atol
        self.device = device
        
    def _vector_field_fn(self, x, t):
        """Vector field function that calls the model to predict the flow.
        
        Args:
            x: State tensor (B, C, H, W)
            t: Time tensor (B,)
            
        Returns:
            Vector field v(x, t) with shape (B, C, H, W)
        """
        # Ensure time tensor has correct shape for UNet
        if t.dim() == 0:
            t = t.expand(x.shape[0])
            
        # Get model prediction
        with torch.no_grad():
            v = self.model(x, t).sample
            
        return v
    
    def create_solver(self):
        """Create the appropriate ODE solver based on solver_type."""
        if self.solver_type == SolverType.EULER:
            return EulerSolver(
                vector_field_fn=self._vector_field_fn,
                t_0=0.0, t_1=1.0,
                rtol=self.rtol, atol=self.atol,
                device=self.device
            )
        elif self.solver_type == SolverType.HEUN:
            return HeunSolver(
                vector_field_fn=self._vector_field_fn,
                t_0=0.0, t_1=1.0,
                rtol=self.rtol, atol=self.atol,
                device=self.device
            )
        elif self.solver_type == SolverType.RK4:
            return RK4Solver(
                vector_field_fn=self._vector_field_fn,
                t_0=0.0, t_1=1.0,
                rtol=self.rtol, atol=self.atol,
                device=self.device
            )
        elif self.solver_type == SolverType.DOPRI5:
            return DormandPrinceSolver(
                vector_field_fn=self._vector_field_fn,
                t_0=0.0, t_1=1.0,
                rtol=self.rtol, atol=self.atol,
                device=self.device
            )
        else:
            raise ValueError(f"Unknown solver type: {self.solver_type}")
    
    def sample(self, noise, return_trajectory=False):
        """Generate samples by integrating the flow starting from random noise.
        
        Args:
            noise: Initial noise tensor (B, C, H, W)
            return_trajectory: Whether to return the entire trajectory
            
        Returns:
            Generated samples or (samples, trajectory)
        """
        # Create solver
        solver = self.create_solver()
        
        # Integrate ODE from noise (t=0) to data (t=1)
        return solver.integrate(
            x_0=noise,
            steps=self.steps,
            return_trajectory=return_trajectory
        )