from typing import List, Literal, Tuple

import numpy as np
from numpy.typing import NDArray

# Type aliases for better readability
IndexList = List[int]
CoverageArray = NDArray[np.float32]
SamplingMethod = Literal["random", "fixedsubset", "fixed"]


def fixed_sampler(n_input_t: int, min_cov: float, max_cov: float, coverage: CoverageArray, clear_tresh: float = 1e-3) -> Tuple[IndexList, int, bool]:
    """
    Sample time points with fixed cloud coverage criteria.

    Selects the least cloudy sample as target and tries to find input samples within
    specified cloud coverage range. Falls back to first n samples if criteria not met.

    Args:
        n_input_t: Number of input samples to select
        min_cov: Minimum cloud coverage threshold for input samples
        max_cov: Maximum cloud coverage threshold for input samples
        coverage: Array of cloud coverage values for each time point
        clear_tresh: Maximum coverage threshold for target sample

    Returns:
        Tuple containing:
        - List of input sample indices
        - Index of clearest (target) sample
        - Boolean indicating if coverage criteria were met
    """
    # Sort observation indices by cloud coverage (ascending)
    coverage_idx = np.argsort(coverage)
    cloudless_idx = coverage_idx[0]  # Take the (earliest) least cloudy sample

    # Select input samples within specified coverage range
    inputs_idx = [pdx for pdx, perc in enumerate(coverage) if min_cov <= perc <= max_cov][:n_input_t]

    # Fallback if not enough samples meet criteria
    if len(inputs_idx) < n_input_t:
        inputs_idx = [pdx for pdx in range(len(coverage)) if pdx != cloudless_idx][:n_input_t]
        coverage_match = False
    else:
        coverage_match = True

    # Verify target sample meets clearness criteria
    if coverage[cloudless_idx] > clear_tresh:
        coverage_match = False

    return inputs_idx, cloudless_idx, coverage_match


def fixedsubset_sampler(n_input_t: int, min_cov: float, max_cov: float, coverage: CoverageArray, clear_tresh: float = 1e-3, earliest_idx: int = 0, latext_idx: int = 30) -> Tuple[IndexList, int, bool]:
    """
    Sample time points from a subsequence with fixed coverage criteria.

    Applies fixed_sampler to a subsequence, falling back to full sequence if criteria
    aren't met in the subsequence.

    Args:
        n_input_t: Number of input samples to select
        min_cov: Minimum cloud coverage threshold
        max_cov: Maximum cloud coverage threshold
        coverage: Array of cloud coverage values
        clear_tresh: Maximum coverage threshold for target sample
        earliest_idx: Start index of subsequence
        latext_idx: End index of subsequence

    Returns:
        Tuple containing input indices, target index, and coverage match flag
    """
    # Apply fixed sampler to subsequence
    inputs_idx, cloudless_idx, coverage_match = fixed_sampler(n_input_t, min_cov, max_cov, coverage[earliest_idx:latext_idx], clear_tresh)

    # Adjust indices for subsequence offset
    inputs_idx = [idx + earliest_idx for idx in inputs_idx]
    cloudless_idx += earliest_idx

    # Fallback to full sequence if criteria not met
    if not coverage_match:
        inputs_idx, cloudless_idx, coverage_match = fixed_sampler(n_input_t, min_cov, max_cov, coverage, clear_tresh)

    return inputs_idx, cloudless_idx, coverage_match


def random_sampler(t_windows: List[List[int]], coverage: CoverageArray, clear_tresh: float = 1e-3) -> Tuple[IndexList, int, bool]:
    """
    Randomly sample time points with temporal coherence.

    Selects a random clear target and temporally adjacent inputs.

    Args:
        t_windows: List of temporal windows for sampling
        coverage: Array of cloud coverage values
        clear_tresh: Maximum coverage threshold for target sample

    Returns:
        Tuple containing input indices, target index, and coverage match flag
    """
    # Find clear samples (below threshold)
    is_clear = np.argwhere(np.array(coverage) < clear_tresh).flatten()

    # Select random clear sample or fallback to minimum coverage
    try:
        cloudless_idx = is_clear[np.random.randint(0, len(is_clear))]
    except (ValueError, IndexError):
        cloudless_idx = np.array(coverage).argmin()

    # Find temporal windows containing the target
    windows = [window for window in t_windows if cloudless_idx in window]

    # Select middle window for balanced temporal distribution
    inputs_idx = [input_t for input_t in windows[len(windows) // 2] if input_t != cloudless_idx]

    # Note: Coverage criteria not enforced in random mode
    return inputs_idx, cloudless_idx, True


def sampler(
    sampling: SamplingMethod,
    t_windows: List[List[int]],
    n_input_t: int,
    min_cov: float,
    max_cov: float,
    coverage: CoverageArray,
    clear_tresh: float = 1e-3,
    earliest_idx: int = 0,
    latext_idx: int = 30,
) -> Tuple[IndexList, int, bool]:
    """
    Main sampling function that routes to specific sampling strategies.

    Args:
        sampling: Sampling method ('random', 'fixedsubset', or 'fixed')
        t_windows: List of temporal windows (for random sampling)
        n_input_t: Number of input samples to select
        min_cov: Minimum cloud coverage threshold
        max_cov: Maximum cloud coverage threshold
        coverage: Array of cloud coverage values
        clear_tresh: Maximum coverage threshold for target sample
        earliest_idx: Start index for subsequence sampling
        latext_idx: End index for subsequence sampling

    Returns:
        Tuple containing input indices, target index, and coverage match flag
    """
    if sampling == "random":
        return random_sampler(t_windows, coverage, clear_tresh)
    elif sampling == "fixedsubset":
        return fixedsubset_sampler(n_input_t, min_cov, max_cov, coverage, clear_tresh, earliest_idx, latext_idx)
    else:  # Default to fixed sampling
        return fixed_sampler(n_input_t, min_cov, max_cov, coverage, clear_tresh)
