"""
Ray Mask (RM) safeguard.

Maps every action radially towards the centre of the safe action set.
Implements the generalised ray mask (Paper Eq. 21) and all three
modifications proposed in Section V-C-3:
    1. Regularisation (shared with BP, Paper Eq. 16)
    2. Passthrough gradient
    3. Hyperbolic (tanh) mapping  (Paper Eq. 31)

Paper: Walter et al. (2025), Section V-C.
"""

import torch
import torch.nn as nn
import cvxpy as cp
import numpy as np
from cvxpylayers.torch import CvxpyLayer
from enum import Enum

from core.zonotope import Zonotope
from core.boundary_projection import BoundaryProjection


class RayMaskMode(Enum):
    LINEAR = "linear"        # original linear mapping  (Eq. after 21)
    HYPERBOLIC = "hyperbolic"  # tanh mapping (Eq. 31)


class SafeCentreApprox(Enum):
    ZONOTOPIC = "zonotopic"      # maximise inner zonotope (Eq. 26)
    ORTHOGONAL = "orthogonal"    # pierce-midpoint method   (Eq. 27-28)


class RayMask(nn.Module):
    """
    Generalised differentiable ray mask safeguard.

    Properties satisfied (Table 1 of paper):
        P1 (Safety):            Yes
        P2 (Subdifferentiable): Yes
        P3 (Full-rank Jacobian): Yes (almost everywhere), full rank d
        P4 (Rare interventions): No — all actions are mapped
        P5 (Computation):       1 LP (specified A_s) or conic+QP (derived)
    """

    EPS = 1e-6  # numerical stability for ||a - c_As||

    def __init__(
        self,
        safe_set: Zonotope,
        feasible_set: Zonotope,
        mode: RayMaskMode = RayMaskMode.LINEAR,
        centre_approx: SafeCentreApprox = SafeCentreApprox.ZONOTOPIC,
        cd: float = 0.0,
        passthrough: bool = False,
    ):
        """
        Args:
            safe_set:      Zonotope A_s
            feasible_set:  Zonotope A (feasible action set)
            mode:          LINEAR or HYPERBOLIC mapping function
            centre_approx: How to compute the safe centre c_As
            cd:            Regularisation coefficient (Paper Eq. 16)
            passthrough:   Replace Jacobian with identity (Paper V-C-3)
        """
        super().__init__()
        self.safe_set = safe_set
        self.feasible_set = feasible_set
        self.mode = mode
        self.centre_approx = centre_approx
        self.cd = cd
        self.passthrough = passthrough
        self.d = safe_set.d

        # Precompute safe centre for specified sets
        self._c_As = self._compute_safe_centre_zonotopic()
        self._build_lambda_layer()

    # ------------------------------------------------------------------
    # Safe centre approximation (Paper Section V-C-1, Fig. 4)
    # ------------------------------------------------------------------

    def _compute_safe_centre_zonotopic(self) -> torch.Tensor:
        """
        Zonotopic approximation: maximise generator lengths of an inner
        zonotope subject to containment in A_s. (Paper Eq. 26)
        For a specified zonotopic A_s the safe centre is simply c_As.
        """
        return self.safe_set.c.clone()

    def _compute_safe_centre_orthogonal(
        self, action: torch.Tensor, bp: BoundaryProjection
    ) -> torch.Tensor:
        """
        Orthogonal approximation: pierce A_s orthogonal to the boundary
        and use the midpoint. (Paper Eq. 27-28)
        Only valid for unsafe actions.
        """
        a_s_bp = bp.forward(action)                        # boundary point
        d_perp = a_s_bp - action
        d_perp_norm = d_perp / (torch.norm(d_perp) + self.EPS)

        # Solve max lambda s.t. a_s_bp + lambda * d_perp in A_s  (Eq. 27)
        lambda_perp = self._max_lambda_in_set(a_s_bp, d_perp_norm, self.safe_set)
        c_As = a_s_bp + (lambda_perp / 2.0) * d_perp_norm  # Eq. 28
        return c_As

    # ------------------------------------------------------------------
    # Distance computations (Paper Eq. 18-20, 25)
    # ------------------------------------------------------------------

    def _max_lambda_in_set(
        self, origin: torch.Tensor, direction: torch.Tensor, zono: Zonotope
    ) -> torch.Tensor:
        """
        Solve: max lambda  s.t.  origin + lambda * direction in zono
        via LP (Paper Eq. 25).
        """
        d = origin.shape[0]
        n = zono.n
        c_np = zono.c.numpy()
        G_np = zono.G.numpy()
        orig_np = origin.detach().numpy()
        dir_np = direction.detach().numpy()

        lam = cp.Variable(nonneg=True)
        gamma = cp.Variable(n)
        objective = cp.Maximize(lam)
        constraints = [
            orig_np + lam * dir_np == c_np + G_np @ gamma,
            cp.norm_inf(gamma) <= 1,
        ]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.CLARABEL)
        val = prob.value if prob.status in ["optimal", "optimal_inaccurate"] else 0.0
        return torch.tensor(max(val, 0.0), dtype=torch.float32)

    def _build_lambda_layer(self):
        """Pre-build CVXPY layer for lambda_As computation."""
        d = self.safe_set.d
        n = self.safe_set.n
        self._G_np = self.safe_set.G.numpy()
        self._c_np = self.safe_set.c.numpy()

    # ------------------------------------------------------------------
    # Mapping functions omega  (Paper Eq. 22-23, 30-32)
    # ------------------------------------------------------------------

    def _omega_linear(self, lambda_a, lambda_As, lambda_A):
        """omega_lin = lambda_a / lambda_A  (Paper Eq. 30)"""
        return lambda_a / (lambda_A + self.EPS)

    def _omega_hyperbolic(self, lambda_a, lambda_As, lambda_A):
        """omega_tanh = tanh(lambda_a / lambda_As) / tanh(lambda_A / lambda_As)
        (Paper Eq. 31)"""
        ratio = lambda_As + self.EPS
        return torch.tanh(lambda_a / ratio) / (torch.tanh(lambda_A / ratio) + self.EPS)

    # ------------------------------------------------------------------
    # Forward pass (Paper Eq. 21)
    # ------------------------------------------------------------------

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        """
        Apply generalised ray mask.

        g_RM(a) = c_As                          if ||a - c_As|| < eps
                  c_As + omega * lambda_As * d_a  otherwise

        Args:
            action: (..., d) action tensor

        Returns:
            safe_action: (..., d) safeguarded action
        """
        c_As = self._c_As
        diff = action - c_As
        lambda_a = torch.norm(diff, dim=-1, keepdim=True)  # Eq. 18

        # Degenerate case: action is at (or near) safe centre
        near_centre = (lambda_a.squeeze(-1) < self.EPS)

        # Unit direction towards action  d_a = (a - c_As) / ||a - c_As||
        d_a = diff / (lambda_a + self.EPS)  # (..., d)

        # Compute lambda_As and lambda_A per sample
        batch_shape = action.shape[:-1]
        flat_da = d_a.reshape(-1, self.d)
        flat_la = lambda_a.reshape(-1)

        safe_actions = []
        for i, (da_i, la_i) in enumerate(zip(flat_da, flat_la)):
            lAs = self._max_lambda_in_set(c_As, da_i, self.safe_set)     # Eq. 19
            lA  = self._max_lambda_in_set(c_As, da_i, self.feasible_set) # Eq. 20

            if self.mode == RayMaskMode.LINEAR:
                omega = self._omega_linear(la_i, lAs, lA)
            else:
                omega = self._omega_hyperbolic(la_i, lAs, lA)

            a_s = c_As + omega * lAs * da_i  # Eq. 21
            safe_actions.append(a_s)

        safe = torch.stack(safe_actions).reshape(*batch_shape, self.d)

        # Replace with c_As where action is too close to centre
        if near_centre.any():
            safe = torch.where(near_centre.unsqueeze(-1), c_As.expand_as(safe), safe)

        # Passthrough gradient override (Paper V-C-3 modification 2)
        if self.passthrough:
            safe = action + (safe - action).detach()

        return safe

    def loss(
        self,
        action: torch.Tensor,
        safe_action: torch.Tensor,
        reward_loss: torch.Tensor,
    ) -> torch.Tensor:
        """Augmented loss with optional regularisation (Paper Eq. 16)."""
        if self.cd > 0.0:
            reg = self.cd * torch.sum((safe_action - action) ** 2)
            return reward_loss + reg
        return reward_loss
