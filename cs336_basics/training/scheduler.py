"""Learning-rate schedules."""

import math


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Return the learning rate for a warmup plus cosine annealing schedule."""
    if it < warmup_iters:
        return it * max_learning_rate / warmup_iters
    if it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine_factor = 0.5 * (1 + math.cos(progress * math.pi))
        return min_learning_rate + cosine_factor * (max_learning_rate - min_learning_rate)
    return min_learning_rate
