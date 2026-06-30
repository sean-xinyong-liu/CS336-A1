"""Loss functions used during language-model training."""

from torch import Tensor
from einops import rearrange, reduce

def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """Return mean cross-entropy over all batch-like dimensions.

    Args:
        inputs: Unnormalized logits with vocabulary size on the final dimension.
        targets: Integer class IDs matching ``inputs.shape[:-1]``.

    Returns:
        Scalar tensor containing the average negative log likelihood.
    """
    # inputs:  (..., num_classes)
    # targets: (...)

    logits_max = reduce(inputs, "... classes -> ... 1", "max")
    shifted_logits = inputs - logits_max

    target_indices = rearrange(targets, "... -> ... 1")
    target_logits = shifted_logits.gather(
        dim=-1,
        index=target_indices,
    )
    target_logits = rearrange(target_logits, "... 1 -> ...")

    log_sum_exp = reduce(
        shifted_logits.exp(),
        "... classes -> ...",
        "sum",
    ).log()

    return (log_sum_exp - target_logits).mean()