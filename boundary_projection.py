"""
Boundary Projection (BP) safeguard.

Maps any action to the closest safe action by solving a QP:
    min_{a_s} ||a - a_s||^2_2   s.t.  a_s in A_s

Backpropagation is done via CVXPYLayers (implicit function theorem).

Paper: Walter et al. (2025), Section V-B, Eq. (12).
"""

import torch
import torch.nn as nn
import cvxpy as cp
import numpy as np
from cvxpylayers.torch import CvxpyLayer

from core.zonotope import Zonotope


class BoundaryProjection(nn.Module):
    """
    Differentiable boundary projection onto a zonotopic safe action set.

    Properties satisfied (Table 1 of paper):
        P1 (Safety):           Yes
        P2 (Subdifferentiable): Yes
        P3 (Full-rank Jacobian): Only for a in A_s  (rank d-1 otherwise)
        P4 (Rare interventions): Yes — only unsafe actions are modified
        P5 (Computation):       1 QP (specified A_s) or 1 QP (derived A_s)
    """

    def __init__(self, safe_set: Zonotope, cd: float = 0.0):
        """
        Args:
            safe_set: Zonotope representing A_s
            cd:       Regularisation coefficient for gradient augmentation
                      (Paper Eq. 16). Set > 0 to recover gradient in the
                      projection direction (recommended for 1-D action spaces).
        """
        super().__init__()
        self.safe_set = safe_set
        self.cd = cd
        self._build_layer()

    def _build_layer(self):
        d = self.safe_set.d
        n = self.safe_set.n
        c_np = self.safe_set.c.numpy()
        G_np = self.safe_set.G.numpy()

        # CVXPY problem (Paper Eq. 44 canonical form)
        a_param = cp.Parameter(d)           # unsafe policy action
        a_s = cp.Variable(d)                # safe action
        gamma = cp.Variable(n)              # zonotope scaling factors

        objective = cp.Minimize(cp.sum_squares(a_param - a_s))
        constraints = [
            a_s == c_np + G_np @ gamma,     # zonotope membership
            cp.norm_inf(gamma) <= 1,        # ||gamma||_inf <= 1
        ]
        prob = cp.Problem(objective, constraints)
        self._layer = CvxpyLayer(prob, parameters=[a_param], variables=[a_s, gamma])

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action: (..., d) unsafe action tensor

        Returns:
            safe_action: (..., d) projected safe action
        """
        batch_shape = action.shape[:-1]
        flat = action.reshape(-1, action.shape[-1])

        safe_actions = []
        for a in flat:
            (a_s, _), = [self._layer(a)]
            safe_actions.append(a_s)
        safe = torch.stack(safe_actions, dim=0).reshape(*batch_shape, -1)

        return safe

    def loss(self, action: torch.Tensor, safe_action: torch.Tensor,
             reward_loss: torch.Tensor) -> torch.Tensor:
        """
        Augmented loss with regularisation term (Paper Eq. 16):
            l(a, s, a_s) = l_r(a_s, s) + c_d * ||a_s - a||^2_2

        The regularisation gradient (Paper Eq. 17) recovers the direction
        a_s - a that is otherwise lost in the projection Jacobian.

        Args:
            action:       raw policy action
            safe_action:  projected safe action
            reward_loss:  scalar reward loss l_r

        Returns:
            augmented scalar loss
        """
        if self.cd > 0.0:
            reg = self.cd * torch.sum((safe_action - action) ** 2)
            return reward_loss + reg
        return reward_loss

    def intervenes(self, action: torch.Tensor, tol: float = 1e-5) -> torch.Tensor:
        """Return boolean mask: True where action is unsafe (BP intervenes)."""
        safe = self.forward(action)
        return (torch.norm(safe - action, dim=-1) > tol)
