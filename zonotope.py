"""
Zonotope set representation.

A zonotope is defined as Z = {c + G*beta | ||beta||_inf <= 1}
with centre c in R^d and generator matrix G in R^(d x n).

Paper: Walter et al. (2025), Section III-C.
"""

import torch
import numpy as np


class Zonotope:
    def __init__(self, center: torch.Tensor, generators: torch.Tensor):
        """
        Args:
            center:     (d,) tensor — zonotope centre c
            generators: (d, n) tensor — generator matrix G
        """
        self.c = center.float()
        self.G = generators.float()
        self.d = center.shape[0]
        self.n = generators.shape[1]

    # ------------------------------------------------------------------
    # Set operations (Paper Eq. 3-6)
    # ------------------------------------------------------------------

    def minkowski_sum(self, other: "Zonotope") -> "Zonotope":
        """Z1 + Z2 = <c1+c2, [G1 G2]>  (Paper Eq. 4)"""
        new_c = self.c + other.c
        new_G = torch.cat([self.G, other.G], dim=1)
        return Zonotope(new_c, new_G)

    def linear_map(self, M: torch.Tensor) -> "Zonotope":
        """M * Z = <M*c, M*G>  (Paper Eq. 5)"""
        return Zonotope(M @ self.c, M @ self.G)

    def translate(self, v: torch.Tensor) -> "Zonotope":
        return Zonotope(self.c + v, self.G)

    def support_function(self, v: torch.Tensor) -> torch.Tensor:
        """rho_Z(v) = v^T c + ||G^T v||_1  (Paper Eq. 6)"""
        return v @ self.c + torch.norm(self.G.T @ v, p=1)

    # ------------------------------------------------------------------
    # Containment checks (Paper Eq. 7-8)
    # ------------------------------------------------------------------

    def contains_point(self, p: torch.Tensor, tol: float = 1e-6) -> bool:
        """
        Check p in Z via LP: min ||gamma||_inf s.t. p = c + G*gamma
        Returns True if min value <= 1.  (Paper Eq. 7)
        """
        import cvxpy as cp
        gamma = cp.Variable(self.n)
        objective = cp.Minimize(cp.norm_inf(gamma))
        constraints = [self.c.numpy() + self.G.numpy() @ gamma == p.numpy()]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.CLARABEL)
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return False
        return float(prob.value) <= 1.0 + tol

    def contains_zonotope(self, other: "Zonotope") -> bool:
        """
        Sufficient condition Z1 subset Z2 via LP.  (Paper Eq. 8)
        """
        import cvxpy as cp
        n1, n2 = other.n, self.n
        gamma = cp.Variable(n2)
        Lambda = cp.Variable((n2, n1))
        objective = cp.Minimize(cp.norm(cp.vstack([gamma, cp.reshape(Lambda, (n2 * n1, 1))]), "inf"))
        constraints = [
            other.G == self.G @ Lambda,
            self.c - other.c == self.G @ gamma,
        ]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.CLARABEL)
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return False
        return float(prob.value) <= 1.0

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def bounding_box(self):
        """Return axis-aligned bounding box [low, high]."""
        half_widths = torch.abs(self.G).sum(dim=1)
        return self.c - half_widths, self.c + half_widths

    @staticmethod
    def from_box(low: torch.Tensor, high: torch.Tensor) -> "Zonotope":
        """Construct zonotope from axis-aligned box."""
        c = (low + high) / 2.0
        diag = (high - low) / 2.0
        G = torch.diag(diag)
        return Zonotope(c, G)

    def __repr__(self):
        return f"Zonotope(d={self.d}, n={self.n}, c={self.c.tolist()})"
