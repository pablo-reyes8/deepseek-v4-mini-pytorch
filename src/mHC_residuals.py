# ============================================================
# Manifold-Constrained Hyper-Connection
# ============================================================

from src.mHC_residuals_utils import *
from src.transformer_modules.RMSNorm import RMSNorm

@dataclass
class ManifoldHyperConnectionConfig:
    d_model: int
    n_hc: int = 4

    sinkhorn_iters: int = 20
    eps: float = 1e-6
    use_log_sinkhorn: bool = False
    sinkhorn_fp32: bool = True

    # Dynamic parameterization.
    dynamic: bool = True
    init_alpha: float = 1e-3
    alpha_max: float = 1.0
    bounded_alpha: bool = True

    # Static initialization controls.
    static_a_stream0: float = 4.0
    static_a_other: float = -4.0

    static_b_diag: float = 4.0
    static_b_offdiag: float = -4.0

    static_c_stream0: float = 0.0
    static_c_other: float = -6.0

    init_std: float = 0.02

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {self.d_model}")

        if self.n_hc < 2:
            raise ValueError(f"n_hc must be >= 2 for mHC, got {self.n_hc}")

        if self.sinkhorn_iters <= 0:
            raise ValueError(f"sinkhorn_iters must be > 0, got {self.sinkhorn_iters}")

        if self.eps <= 0:
            raise ValueError(f"eps must be > 0, got {self.eps}")

        if self.init_alpha < 0:
            raise ValueError(f"init_alpha must be >= 0, got {self.init_alpha}")

        if self.alpha_max <= 0:
            raise ValueError(f"alpha_max must be > 0, got {self.alpha_max}")

        if self.bounded_alpha and self.init_alpha >= self.alpha_max:
            raise ValueError(
                "For bounded_alpha=True, init_alpha must be < alpha_max. "
                f"Got init_alpha={self.init_alpha}, alpha_max={self.alpha_max}"
            )

        if self.init_std <= 0:
            raise ValueError(f"init_std must be > 0, got {self.init_std}")


class ManifoldHyperConnection(nn.Module):
    """
    Manifold-Constrained Hyper-Connection.

    Canonical equation:

        X_{l+1} = B_l X_l + C_l F_l(A_l X_l)

    where:
        X_l: [B, T, n_hc, D]
        A_l: [B, T, 1, n_hc], values in (0, 1)
        B_l: [B, T, n_hc, n_hc], doubly stochastic
        C_l: [B, T, n_hc, 1], values in (0, 2)

    This version supports both APIs:

    1. Wrapper API:
        X_next = mhc(X, sublayer)

    2. Modular block API:
        A, B, C = mhc.compute_ABC(X)
        h = mhc.pre_mix(X, A=A)
        y = sublayer(norm(h))
        X = mhc.update(X, y, B_mat=B, C=C)
    """

    def __init__(self, config: ManifoldHyperConnectionConfig):
        super().__init__()
        config.validate()

        self.config = config
        self.d_model = config.d_model
        self.n_hc = config.n_hc
        self.sinkhorn_iters = config.sinkhorn_iters
        self.eps = config.eps
        self.dynamic = config.dynamic
        self.use_log_sinkhorn = config.use_log_sinkhorn
        self.sinkhorn_fp32 = config.sinkhorn_fp32
        self.bounded_alpha = config.bounded_alpha
        self.alpha_max = config.alpha_max

        flat_dim = config.n_hc * config.d_model

        self.param_norm = RMSNorm(flat_dim, eps=config.eps)

        self.dynamic_A = nn.Linear(flat_dim, config.n_hc, bias=True)
        self.dynamic_B = nn.Linear(flat_dim, config.n_hc * config.n_hc, bias=True)
        self.dynamic_C = nn.Linear(flat_dim, config.n_hc, bias=True)

        # Static parameters.
        self.static_A = nn.Parameter(torch.empty(config.n_hc))
        self.static_B = nn.Parameter(torch.empty(config.n_hc, config.n_hc))
        self.static_C = nn.Parameter(torch.empty(config.n_hc))

        # Raw scalar gates controlling dynamic contribution.
        # If bounded_alpha=True, effective alpha is:
        #   alpha = alpha_max * tanh(alpha_raw)
        # initialized so alpha ~= init_alpha.
        self.alpha_A_raw = nn.Parameter(torch.empty(()))
        self.alpha_B_raw = nn.Parameter(torch.empty(()))
        self.alpha_C_raw = nn.Parameter(torch.empty(()))

        self.reset_parameters()

    def _initial_alpha_raw(self) -> float:
        if not self.bounded_alpha:
            return float(self.config.init_alpha)

        ratio = self.config.init_alpha / self.config.alpha_max
        # Clamp only for numerical safety; validate already enforces ratio < 1.
        ratio = max(min(ratio, 1.0 - 1e-7), -1.0 + 1e-7)
        return float(math.atanh(ratio))

    def reset_parameters(self) -> None:
        # Dynamic generators start small/stable.
        for module in [self.dynamic_A, self.dynamic_B, self.dynamic_C]:
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            nn.init.zeros_(module.bias)

        with torch.no_grad():
            # A: select stream 0 initially.
            self.static_A.fill_(self.config.static_a_other)
            self.static_A[0] = self.config.static_a_stream0

            # B: approximately identity after Sinkhorn.
            self.static_B.fill_(self.config.static_b_offdiag)
            diag = torch.arange(self.n_hc, device=self.static_B.device)
            self.static_B[diag, diag] = self.config.static_b_diag

            # C: inject mostly into stream 0.
            self.static_C.fill_(self.config.static_c_other)
            self.static_C[0] = self.config.static_c_stream0

            init_raw = self._initial_alpha_raw()
            self.alpha_A_raw.fill_(init_raw)
            self.alpha_B_raw.fill_(init_raw)
            self.alpha_C_raw.fill_(init_raw)

    def _validate_X(self, X: torch.Tensor) -> Tuple[int, int]:
        if X.dim() != 4:
            raise ValueError(
                f"ManifoldHyperConnection expects X [B,T,n_hc,D], got {tuple(X.shape)}"
            )

        B, T, N, D = X.shape

        if N != self.n_hc:
            raise ValueError(f"Expected n_hc={self.n_hc}, got {N}")

        if D != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {D}")

        return B, T

    def _validate_y_sub(self, y_sub: torch.Tensor, B: int, T: int) -> None:
        if not isinstance(y_sub, torch.Tensor):
            raise TypeError("sublayer output must be a torch.Tensor")
        expected = (B, T, self.d_model)
        if y_sub.shape != expected:
            raise ValueError(
                "sublayer output must have shape [B,T,d_model]. "
                f"Expected {expected}, got {tuple(y_sub.shape)}"
            )

    def _effective_alpha(self, raw: torch.Tensor) -> torch.Tensor:
        if self.bounded_alpha:
            return self.alpha_max * torch.tanh(raw)
        return raw

    def get_alpha_values(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return effective dynamic gates alpha_A, alpha_B, alpha_C."""
        return (
            self._effective_alpha(self.alpha_A_raw),
            self._effective_alpha(self.alpha_B_raw),
            self._effective_alpha(self.alpha_C_raw),
        )

    def compute_ABC(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute constrained A, B, C.

        Args:
            X: [B,T,n_hc,D]

        Returns:
            A: [B,T,1,n_hc], values in (0,1)
            B: [B,T,n_hc,n_hc], approximately doubly stochastic
            C: [B,T,n_hc,1], values in (0,2)
        """
        Bsz, T = self._validate_X(X)

        X_flat = X.reshape(Bsz, T, self.n_hc * self.d_model)
        X_hat = self.param_norm(X_flat)

        if self.dynamic:
            A_dyn = self.dynamic_A(X_hat)
            B_dyn = self.dynamic_B(X_hat).view(Bsz, T, self.n_hc, self.n_hc)
            C_dyn = self.dynamic_C(X_hat)
        else:
            A_dyn = torch.zeros(Bsz, T, self.n_hc, device=X.device, dtype=X.dtype)
            B_dyn = torch.zeros(Bsz, T, self.n_hc, self.n_hc, device=X.device, dtype=X.dtype)
            C_dyn = torch.zeros(Bsz, T, self.n_hc, device=X.device, dtype=X.dtype)

        static_A = self.static_A.to(device=X.device, dtype=X.dtype)
        static_B = self.static_B.to(device=X.device, dtype=X.dtype)
        static_C = self.static_C.to(device=X.device, dtype=X.dtype)

        alpha_A, alpha_B, alpha_C = self.get_alpha_values()
        alpha_A = alpha_A.to(device=X.device, dtype=X.dtype)
        alpha_B = alpha_B.to(device=X.device, dtype=X.dtype)
        alpha_C = alpha_C.to(device=X.device, dtype=X.dtype)

        A_tilde = static_A[None, None, :] + alpha_A * A_dyn
        B_tilde = static_B[None, None, :, :] + alpha_B * B_dyn
        C_tilde = static_C[None, None, :] + alpha_C * C_dyn

        A = torch.sigmoid(A_tilde)[:, :, None, :]          # [B,T,1,n_hc]

        if self.use_log_sinkhorn:
            B_mat = log_sinkhorn(B_tilde, n_iters=self.sinkhorn_iters)
        else:
            B_mat = sinkhorn(
                B_tilde,
                n_iters=self.sinkhorn_iters,
                eps=self.eps,
                fp32=self.sinkhorn_fp32,
            )

        C = (2.0 * torch.sigmoid(C_tilde))[:, :, :, None]  # [B,T,n_hc,1]

        return A, B_mat, C

    def pre_mix(
        self,
        X: torch.Tensor,
        A: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Produce the d_model-dimensional sublayer input:

            x_sub = A(X) @ X

        Args:
            X: [B,T,n_hc,D]
            A: optional precomputed [B,T,1,n_hc]
            return_aux: if True, returns x_sub and {A,B,C}; useful when the
                caller wants to reuse B,C for update.

        Returns:
            x_sub: [B,T,D]
        """
        self._validate_X(X)

        if A is None:
            A, B_mat, C = self.compute_ABC(X)
        else:
            B_mat = None
            C = None
            expected = (*X.shape[:2], 1, self.n_hc)
            if A.shape != expected:
                raise ValueError(f"A must have shape {expected}, got {tuple(A.shape)}")

        x_sub = torch.einsum("btan,btnd->btad", A, X).squeeze(dim=2)

        if return_aux:
            aux = {"A": A}
            if B_mat is not None:
                aux["B"] = B_mat
            if C is not None:
                aux["C"] = C
            return x_sub, aux

        return x_sub

    def update(
        self,
        X: torch.Tensor,
        y_sub: torch.Tensor,
        B_mat: Optional[torch.Tensor] = None,
        C: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Apply residual mixing and output injection:

            X_next = B(X) @ X + C(X) * y_sub

        Args:
            X: [B,T,n_hc,D]
            y_sub: [B,T,D]
            B_mat: optional precomputed [B,T,n_hc,n_hc]
            C: optional precomputed [B,T,n_hc,1]
        """
        Bsz, T = self._validate_X(X)
        self._validate_y_sub(y_sub, Bsz, T)

        if B_mat is None or C is None:
            _, B_new, C_new = self.compute_ABC(X)
            if B_mat is None:
                B_mat = B_new
            if C is None:
                C = C_new

        expected_B = (Bsz, T, self.n_hc, self.n_hc)
        expected_C = (Bsz, T, self.n_hc, 1)

        if B_mat.shape != expected_B:
            raise ValueError(f"B_mat must have shape {expected_B}, got {tuple(B_mat.shape)}")

        if C.shape != expected_C:
            raise ValueError(f"C must have shape {expected_C}, got {tuple(C.shape)}")

        mixed_X = torch.einsum("btij,btjd->btid", B_mat, X)
        injected = C * y_sub[:, :, None, :]
        X_next = mixed_X + injected

        if return_aux:
            aux = {
                "B": B_mat,
                "C": C,
                "mixed_X": mixed_X,
                "injected": injected,
            }
            return X_next, aux

        return X_next

    def post_and_residual_mix(
        self,
        X: torch.Tensor,
        F_out: torch.Tensor,
        B_mat: Optional[torch.Tensor] = None,
        C: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Alias used by block-level code. Equivalent to update(X, F_out, ...).
        """
        return self.update(
            X=X,
            y_sub=F_out,
            B_mat=B_mat,
            C=C,
            return_aux=return_aux,
        )

    def forward(
        self,
        X: torch.Tensor,
        sublayer: Callable[[torch.Tensor], torch.Tensor],
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Wrapper API:

            A, B, C = compute_ABC(X)
            x_sub = pre_mix(X, A)
            y_sub = sublayer(x_sub)
            X_next = update(X, y_sub, B, C)
        """
        Bsz, T = self._validate_X(X)

        if not callable(sublayer):
            raise TypeError("sublayer must be callable")

        A, B_mat, C = self.compute_ABC(X)
        x_sub = self.pre_mix(X, A=A)

        y_sub = sublayer(x_sub)
        self._validate_y_sub(y_sub, Bsz, T)

        X_next, update_aux = self.update(
            X=X,
            y_sub=y_sub,
            B_mat=B_mat,
            C=C,
            return_aux=True,
        )

        if return_aux:
            alpha_A, alpha_B, alpha_C = self.get_alpha_values()
            aux = {
                "A": A,
                "B": B_mat,
                "C": C,
                "x_sub": x_sub,
                "y_sub": y_sub,
                "mixed_X": update_aux["mixed_X"],
                "injected": update_aux["injected"],
                "alpha_A": alpha_A,
                "alpha_B": alpha_B,
                "alpha_C": alpha_C,
            }
            return X_next, aux

        return X_next
