from dataclasses import dataclass


@dataclass
class ENFORCEConfig:
    """Configuration for the ENFORCE model.

    Parameters
    ----------
    input_neurons, hidden_neurons, output_neurons, hidden_layers :
        Network architecture.
    training_tolerance :
        Convergence criterion used during training (mean residual).
    inference_tolerance :
        Convergence criterion used at inference time (max residual).
    max_it :
        Maximum number of neural projection iterations.
    epoch_start_hard_constrained :
        Epoch at which the neural projection is first applied.  Set to 0 to
        project from the very first epoch.
    ada_np_auto_activation :
        If ``True``, skip the projection when it would increase the training
        loss (adaptive neural projection).
    ift_backward :
        If ``True``, use the Implicit Function Theorem backward pass instead of
        unrolled autograd.
    regularise_gram :
        If ``True``, add ``eps_chol * I`` to the gram matrix before Cholesky
        inversion.  For weighted projections (``W ≠ I``) this is always done;
        this flag extends it to the unweighted (``W = I``) case, where it is
        otherwise skipped to preserve the exact null-space projector.  Enable
        for ill-conditioned Jacobians despite ``W = I`` (e.g. bilinear
        constraints such as pooling).
    supervised :
        If ``True``, use supervised MSE loss.  If ``False``, use the
        self-supervised loss passed via ``ssl_loss``.
    soft_constrained :
        If ``True``, add a soft constraint penalty to the loss.
    weight_loss_displacement :
        Weight for the projection-displacement term in the loss.
    weight_loss_soft :
        Weight for the soft-constraint penalty term.
    verbose :
        Print detailed per-epoch projection statistics.
    random_seed :
        Random seed for reproducibility.
    """

    # ── Network architecture ─────────────────────────────────────────────────
    input_neurons: int = 1
    hidden_neurons: int = 64
    output_neurons: int = 1
    hidden_layers: int = 1

    # ── Newton projection ────────────────────────────────────────────────────
    training_tolerance: float = 1e-4
    inference_tolerance: float = 1e-6
    max_it: int = 100
    epoch_start_hard_constrained: int = 0
    ada_np_auto_activation: bool = True
    ift_backward: bool = False
    regularise_gram: bool = False

    # ── Loss ─────────────────────────────────────────────────────────────────
    supervised: bool = True
    soft_constrained: bool = False
    weight_loss_displacement: float = 0.5
    weight_loss_soft: float = 0.0

    # ── Misc ─────────────────────────────────────────────────────────────────
    verbose: bool = False
    random_seed: int = 42
