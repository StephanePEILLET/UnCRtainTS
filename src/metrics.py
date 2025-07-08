import numpy as np
import torch
from matplotlib import pyplot as plt


def binarize(arg, n_bins, floor=0, ceil=1):
    return np.digitize(arg, bins=np.linspace(floor, ceil, num=n_bins)[1:])


def compute_ece(vars, errors, n_samples, percent=5):
    # rank sample-averaged uncertainties ascendingly, and errors accordingly
    _, vars_indices = torch.sort(torch.Tensor(vars))
    errors = torch.Tensor(errors)
    errs_sort = errors[vars_indices]
    # incrementally remove 5% of errors, ranked by highest uncertainty
    bins = torch.linspace(0, n_samples, 100 // percent + 1, dtype=int)[1:]
    # get uncertainty-sorted cumulative errors, i.e. at x-tick 65% we report the average error for the 65% most certain predictions
    sorted_errors = np.array([torch.nanmean(errs_sort[:rdx]).cpu().numpy() for rdx in bins])

    return sorted_errors


def compute_uce_auce(writer, var, errors, n_samples, percent=5, l2=True, mode="val", step=0):
    n_bins = 100 // percent
    var, errors = torch.Tensor(var), torch.Tensor(errors)

    # metric: IN:  standard deviation & error
    #         OUT: either root mean variance & root mean squared error or mean standard deviation & mean absolute error
    def metric(arg):
        return torch.sqrt(torch.mean(arg**2)) if l2 else torch.mean(torch.abs(arg))

    m_str = "L2" if l2 else "L1"

    # group uncertainty values into n_bins
    var_idx = torch.Tensor(binarize(var, n_bins, floor=var.min(), ceil=var.max()))

    # compute bin-wise statistics, defaults to nan if no data contained in bin
    bk_var, bk_err = torch.empty(n_bins), torch.empty(n_bins)
    for bin_idx in range(n_bins):  # for each of the n_bins ...
        bk_var[bin_idx] = metric(var[var_idx == bin_idx].sqrt())  # note: taking the sqrt to wrap into metric function,
        bk_err[bin_idx] = metric(errors[var_idx == bin_idx])  # apply same metric function on error

    calib_err = torch.abs(bk_err - bk_var)  # calibration error: discrepancy of error vs uncertainty
    bk_weight = torch.histogram(var_idx, n_bins)[0] / n_samples  # fraction of total data per bin, for bin-weighting
    uce = torch.nansum(bk_weight * calib_err)  # calc. weighted UCE,
    auce = torch.nanmean(calib_err)  # calc. unweighted AUCE

    # plot bin-wise error versus bin-wise uncertainty
    fig, ax = plt.subplots()
    x_min, x_max = bk_var[~bk_var.isnan()].min(), bk_var[~bk_var.isnan()].max()
    y_min, y_max = 0, bk_err[~bk_err.isnan()].max()
    x_axis = np.linspace(x_min, x_max, num=n_bins)

    ax.plot(x_axis, x_axis)  # diagonal reference line
    ax.bar(
        x_axis,
        bk_err,
        width=x_axis[1] - x_axis[0],
        alpha=0.75,
        edgecolor="k",
        color="gray",
    )

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("Uncertainty")
    plt.ylabel(f"{m_str} Error")
    plt.legend(loc="upper left")
    plt.grid()
    fig.tight_layout()
    writer.add_figure(f"Img/{mode}/err_vs_var_{m_str}", fig, step)

    return uce, auce
