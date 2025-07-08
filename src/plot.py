import os

import numpy as np
from matplotlib import pyplot as plt


def discrete_matshow(data, n_colors=5, min=0, max=1):
    fig, ax = plt.subplots()
    # get discrete colormap
    cmap = plt.get_cmap("gray", n_colors + 1)
    ax.matshow(data, cmap=cmap, vmin=min, vmax=max)
    ax.axis("off")
    fig.tight_layout()
    return fig


def continuous_matshow(data, min=0, max=1):
    fig, ax = plt.subplots()
    # get discrete colormap
    cmap = plt.get_cmap("seismic")
    ax.matshow(data, cmap=cmap, vmin=min, vmax=max)
    ax.axis("off")
    # optionally: provide a colorbar and tick at integers
    # cax = plt.colorbar(mat, ticks=np.arange(min, max + 1))
    return fig


def plot_img(imgs, mod, plot_dir, s2_bands, file_id=None):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    try:
        imgs = imgs.cpu().numpy()
        for tdx, img in enumerate(imgs):  # iterate over temporal dimension
            time = "" if imgs.shape[0] == 1 else f"_t-{tdx}"
            if mod in ["pred", "in", "target", "s2"]:
                rgb = [3, 2, 1] if img.shape[0] == s2_bands else [5, 4, 3]
                img, val_min, val_max = img[rgb, ...], 0, 1
            elif mod == "s1":
                img, val_min, val_max = img[[0], ...], 0, 1
            elif mod == "mask":
                img, val_min, val_max = img[[0], ...], 0, 1
            elif mod == "err":
                img, val_min, val_max = img[[0], ...], 0, 0.01
            elif mod == "var":
                img, val_min, val_max = img[[0], ...], 0, 0.000025
            else:
                raise NotImplementedError
            if file_id is not None:  # export into file name
                img = img.clip(val_min, val_max)  # note: this only removes outliers, vmin/vmax below do the global rescaling (else doing instance-wise min/max scaling)
                plt.imsave(
                    os.path.join(plot_dir, f"img-{file_id}_{mod}{time}.png"),
                    np.moveaxis(img, 0, -1).squeeze(),
                    dpi=100,
                    cmap="gray",
                    vmin=val_min,
                    vmax=val_max,
                )
    except Exception:
        if isinstance(imgs, plt.Figure):  # the passed argument is a pre-rendered figure
            plt.savefig(os.path.join(plot_dir, f"img-{file_id}_{mod}.png"), dpi=100)
        else:
            raise NotImplementedError


def plot_discard(writer, sorted_errors, config, mode, step, is_se=True):
    metric = "SE" if is_se else "AE"

    fig, ax = plt.subplots()
    x_axis = np.arange(0.0, 1.0, 0.05)
    ax.scatter(
        x_axis,
        sorted_errors,
        c="b",
        alpha=1.0,
        marker=r".",
        label=f"{metric}, sorted by uncertainty",
    )

    # fit a linear regressor with slope b and intercept a
    sorted_errors[np.isnan(sorted_errors)] = np.nanmean(sorted_errors)
    b, a = np.polyfit(x_axis, sorted_errors, deg=1)
    x_seq = np.linspace(0, 1.0, num=1000)
    ax.plot(
        x_seq,
        a + b * x_seq,
        c="k",
        lw=1.5,
        alpha=0.75,
        label=f"linear fit, {round(a, 3)} + {round(b, 3)} * x",
    )
    plt.xlabel("Fraction of samples, sorted ascendingly by uncertainty")
    plt.ylabel("Error")
    plt.legend(loc="upper left")
    plt.grid()
    fig.tight_layout()
    writer.add_figure(f"Img/{mode}/discard_uncertain", fig, step)
    if mode == "test":  # export the final test split plots for print
        path_to = os.path.join(config.res_dir, config.experiment_name)
        print(f"Logging discard plots to path {path_to}")
        fig.savefig(
            os.path.join(path_to, f"plot_{mode}_{metric}_discard.png"),
            bbox_inches="tight",
            dpi=int(1e3),
        )
        fig.savefig(
            os.path.join(path_to, f"plot_{mode}_{metric}_discard.pdf"),
            bbox_inches="tight",
            dpi=int(1e3),
        )
