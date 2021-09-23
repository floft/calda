#!/usr/bin/env python3
"""
Synthetic 2D Gaussians (normal) datasets

Usage:
    python -m datasets.synthetic_datasets_normal
    ls datasets/synthetic/
"""
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

from absl import app
from absl import flags
from scipy import stats
from matplotlib.patches import Ellipse

from datasets.normalization import calc_normalization, apply_normalization
from datasets.synthetic_datasets import save_data_file, sine, display_xy

FLAGS = flags.FLAGS

flags.DEFINE_boolean("plots_for_paper", False, "Save plots only, i.e., not the data files")


def confidence_ellipse(ax, mean=None, cov=None, x=None, y=None, n_std=2.0,
        facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*. Or,
    optionally from mean/cov instead (if known/available) of estimated from
    samples drawn from these distributions.

    Based on:

    https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
    https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    # Sanity checks
    if x is not None and y is not None:
        based_on_xy = True

        if x.size != y.size:
            raise ValueError("x and y must be the same size")
    elif mean is not None and cov is not None:
        based_on_xy = False

        if len(cov) != 2 or len(mean) != 2:
            raise ValueError("mean/cov must be 2D")
    else:
        raise ValueError("Pass either mean/cov or x/y")

    # Estimate covariance from data, if not given
    if based_on_xy:
        cov = np.cov(x, y)

    # Calculate needed parameter
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])

    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from the squareroot of the
    # variance and multiplying with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # Calculate mean from the data if not given
    if based_on_xy:
        mean_x = np.mean(x)
    else:
        mean_x = mean[0]

    # Same for y
    scale_y = np.sqrt(cov[1, 1]) * n_std

    if based_on_xy:
        mean_y = np.mean(y)
    else:
        mean_y = mean[1]

    # Compute the ellipse
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse)


def get_translate_rotate(translate, rotate, dimensions, rotation_per_class):
    """ Generate how much / where to translate and rotate and the corresponding
    covariance matrix for the rotated multivariate normal distributions """
    translate_amount = translate/2  # half since +/-
    translation = np.random.uniform(
        -translate_amount, translate_amount, dimensions)

    rotate_amount = rotate/2*rotation_per_class
    rotation = np.random.uniform(-rotate_amount, rotate_amount)

    # No rotation of the Gaussian
    # rotation_conv = np.diag(np.ones((dimensions,)))
    # Simple rotation matrix to rotate the Gaussian distribution (if not
    # isotropic)
    rotation_conv = np.array([
        [np.cos(rotation), -np.sin(rotation)],
        [np.sin(rotation), np.cos(rotation)]
    ])

    return translation, rotation, rotation_conv


def get_near_psd(A):
    """
    Gets rid of "covariance is not positive-semidefinite" warning
    From: https://stackoverflow.com/a/63131309
    """
    C = (A + A.T)/2
    eigval, eigvec = np.linalg.eig(C)
    eigval[eigval < 0] = 0

    return eigvec.dot(np.diag(eigval)).dot(eigvec.T)


def make_domain(num_classes, center, radius, dimensions,
        inter_domain_translate=0, inter_domain_rotate=0,
        intra_domain_translate=0, intra_domain_rotate=0,
        initial_cov=None):
    """
    Make a domain, i.e. generate a list of mean/cov for each class

    Returns:
        [(class1 mean, class1 cov), (class2 mean, class2 cov), ...]
    """
    results = []

    # For now we only support 2D. There's some ways to do this in 3D, but
    # it looks like higher dimensions get a bit more complicated. For now
    # we'll start with 2D only.
    assert dimensions == 2, ">2D not yet implemented"

    # Evenly distribute classes around a circle
    rotation_per_class = 2*np.pi / num_classes

    # Calculate inter-domain shift
    inter_translate, inter_rotate, inter_rotate_conv = get_translate_rotate(
        inter_domain_translate, inter_domain_rotate, dimensions, rotation_per_class)

    # Start with a particular "shaped" Gaussian - for now, isotropic
    if initial_cov is None:
        initial_cov = np.diag(np.ones((dimensions,)))

    # Create distribution for each class
    for i in range(num_classes):
        # Intra-domain shift, i.e. shift each class differently
        intra_translate, intra_rotate, intra_rotate_conv = get_translate_rotate(
            intra_domain_translate, intra_domain_rotate, dimensions, rotation_per_class)

        # Adjust mean of distributions around the circle
        theta = i*rotation_per_class + inter_rotate + intra_rotate
        mean = np.array([
            center[0] + radius*np.cos(theta),
            center[1] + radius*np.sin(theta),
        ])

        # Initial covariance before rotation
        cov = initial_cov

        # Inter-domain shift, i.e. shift all classes the same
        mean += inter_translate + intra_translate

        # It's isotropic for now, so we don't need to rotate.
        # cov = inter_rotate_conv.dot(cov)
        # cov = intra_rotate_conv.dot(cov)

        # Get close positive-semidefinite to this matrix - needed because of
        # the rotation matrices
        # print("Before:", cov)
        # print("After:", get_near_psd(cov))
        # cov = get_near_psd(cov)

        results.append((mean, cov))

    return results


def get_distribution_data(domain, num_points):
    """ Create the multivariate normal distribution and draw a number of samples
    from the distribution """
    dists = []
    data = []
    labels = []

    for i, (mean, cov) in enumerate(domain):
        dist = stats.multivariate_normal(mean, cov)
        dists.append(dist)

        if num_points > 0:
            data.append(dist.rvs(num_points))
            labels.append([i]*num_points)

    return dists, data, labels


def plot_domains(domains, num_points, draw_lines, from_data, normalized,
        fig=None, ax=None, title=None):
    """ Visualize a set of multivariate normal distributions, one for each
    class of each domain. Optionally estimate from a sample of points drawn
    from these distributions rather from the mean/covariance directly, and
    optionally from the normalized samples instead. Optionally draw lines
    from each domain's class distributions to the center/mean of that domain
    for added clarity. """
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    if not FLAGS.plots_for_paper:
        ax.axvline(c='grey', lw=1)
        ax.axhline(c='grey', lw=1)

    # https://matplotlib.org/stable/tutorials/colors/colors.html
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    # For each domain
    for i, domain in enumerate(domains):
        means = np.array([mean for mean, cov in domain])
        domain_mean = np.mean(means, axis=0)

        # Create distributions and draw samples (if desired) from these
        # distributions
        _, data, _ = get_distribution_data(domain, num_points)

        # If desired, normalize the points
        if normalized:
            # Compute stats over entire data, not over just one class
            all_data = np.vstack(data)
            norm = calc_normalization(all_data, "meanstd")

            # Then, normalize each class's data
            normalized_data = []

            for d in data:
                normalized_data.append(apply_normalization(d, norm))

            data = normalized_data

        # Calculate means from data if desired
        if from_data:
            means = [np.mean(d, axis=0) for d in data]
            domain_mean = np.mean(means, axis=0)

        # For each class in that domain
        for j, (mean, cov) in enumerate(domain):
            # Get class color
            color = colors[j%len(colors)]

            # Label for each class
            if i == 0:
                label = "Class {}".format(j)
            else:
                label = None

            # Plot some data from the distribution
            if num_points > 0:
                x1 = data[j][:, 0]
                x2 = data[j][:, 1]
                ax.scatter(x1, x2, s=1.0, alpha=0.5, c=color)

            # Make sure they're numpy arrays
            mean = np.array(mean)
            cov = np.array(cov)

            # Plot the ellipse for the multivariate normal distribution
            if from_data:
                assert num_points > 0, "if from_data, set num_points > 0"
                confidence_ellipse(ax, x=x1, y=x2,
                    edgecolor=color, alpha=0.5, facecolor=color,
                    label=label)

                # Also overwrite mean, so we use the means from the data
                mean = means[j]
            else:
                confidence_ellipse(ax, mean=mean, cov=cov,
                    edgecolor=color, alpha=0.5, facecolor=color,
                    label=label)

            # Annotate each center by which source/target it's from, note
            # target is at the end
            txt = "$T$" if i == len(domains)-1 else "$S_{"+str(i)+"}$"
            ax.annotate(txt, (mean[0], mean[1]))

            if draw_lines:
                # Line from each class center to the domain mean - makes it easier
                # to see how domains are translated/rotated
                ax.plot([domain_mean[0], mean[0]], [domain_mean[1], mean[1]],
                    'r-', alpha=0.5)

    if not FLAGS.plots_for_paper:
        if title is None:
            title = "MS-DA Distributions, n={}".format(len(domains)-1)

        if from_data and normalized:
            title += ", from normalized data"
        elif from_data:
            title += ", from data"
        elif normalized:
            title += ", normalized"

        ax.set_title(title)

    ax.legend()

    if fig is None and ax is None:
        plt.tight_layout()


def generate_msda_problems(num_sources, num_classes,
        center=(0, 0), radius=5, inter_domain_translate=10, inter_domain_rotate=1,
        intra_domain_translate=5, intra_domain_rotate=0.5,
        dimensions=2, num_points=0, seed=42, draw_lines=False,
        from_data=False, normalize=False, fig=None, ax=None, title=None):
    """ Generate a single MS-DA problem, for a given value of n, L, radius,
    amount of translation/rotation for inter/intra-domain shifts, etc. Plot
    the result. """
    # Make repeatable
    random.seed(seed)
    np.random.seed(seed)

    # Create domains
    target = make_domain(num_classes, center, radius, dimensions)
    sources = []

    for i in range(num_sources):
        sources.append(make_domain(num_classes, center, radius, dimensions,
            inter_domain_translate, inter_domain_rotate,
            intra_domain_translate, intra_domain_rotate))

    # Plot domains
    assert dimensions == 2, "Can only plot domains for 2D data at the moment"
    # Note: we depend on the target being last for labeling
    plot_domains(sources + [target], num_points, draw_lines, from_data,
        normalize, fig, ax, title)

    return sources, target


def save_plot(name, out="normal_plots", extension="png"):
    """ Rather than displaying the plot, save it to a file """
    if not os.path.exists(out):
        os.makedirs(out)

    plt.savefig(os.path.join(out, name+"."+extension), bbox_inches='tight')
    plt.close()


def all_problems(num_sources, num_classes, draw_lines=False, center=(25, 25),
        inter_scaling=[1], intra_scaling=[1], raw=False,
        base_params=None, display=False, suffix=None):
    """ Generate all the MS-DA problems for a given value of n and L, varying
    inter-domain and/or intra-domain scaling, putting all of these results into
    one plot. Both the theoretical distribution (exact from mean/covariance) and
    the distribution estimated from 1k normalized points are displayed, to make
    clear what may happen when we normalize the data. """
    subplot_index = 0
    rows = len(inter_scaling) * len(intra_scaling)
    cols = 2
    # num_plots = rows * cols

    if FLAGS.plots_for_paper:
        subplots = False
        plot_ext = "pdf"
    else:
        subplots = True
        plot_ext = "png"

    if subplots:
        fig, axs = plt.subplots(rows, cols, figsize=(10, 5*rows))
        fig.suptitle("MS-DA Distributions, n={}".format(num_sources))

        # For some reason Matplotlib doesn't include the extra dimension if
        # it's only 1 row. To make the code below work either way, add dim back.
        if rows == 1:
            axs = [axs]

        # Save space
        fig.tight_layout()

        # top - make title above plots
        # wspace - between columns
        # hspace - between rows
        plt.subplots_adjust(top=0.93, wspace=0.05*cols, hspace=0.07*rows)

    # Which are we varying?
    if inter_scaling == [1] and intra_scaling == [1]:
        variation = "none"
    elif inter_scaling == [1]:
        variation = "intra"
    elif intra_scaling == [1]:
        variation = "inter"
    else:
        variation = "both"

    # Output suffix
    if suffix is None:
        suffix = ""
    else:
        suffix = "_" + suffix

    # Base inter/intra-domain shifts
    if base_params is None:
        base_params = [10, 1, 5, 0.5]

    for inter_scale in inter_scaling:
        for intra_scale in intra_scaling:
            assert len(base_params) == 4, "wrong number of base_params"
            inter_domain_translate = base_params[0]*inter_scale
            inter_domain_rotate = base_params[1]*inter_scale
            intra_domain_translate = base_params[2]*intra_scale
            intra_domain_rotate = base_params[3]*intra_scale

            # Get title
            if variation == "none":
                title = ""
            elif variation == "intra":
                title = "intra-scale {}".format(intra_scale)
            elif variation == "inter":
                title = "inter-scale {}".format(inter_scale)
            else:
                title = "inter-scale {}, intra-scale {}".format(inter_scale, intra_scale)

            name = "normal_n{}_l{}_inter{}_intra{}{}".format(
                num_sources, num_classes, inter_scale, intra_scale,
                suffix)

            # The left column is the un-normalized
            if subplots:
                ax = axs[subplot_index//cols][subplot_index%cols]
                subplot_index += 1
            else:
                fig = None
                ax = None

            generate_msda_problems(num_sources, num_classes,
                draw_lines=draw_lines, num_points=100, center=center,
                inter_domain_translate=inter_domain_translate,
                inter_domain_rotate=inter_domain_rotate,
                intra_domain_translate=intra_domain_translate,
                intra_domain_rotate=intra_domain_rotate,
                fig=fig, ax=ax, title=title)

            # Check that from data looks the same
            # generate_msda_problems(num_sources, num_classes,
            #     draw_lines=draw_lines, num_points=1000, from_data=True)

            # The right column is the from_data/normalized
            if subplots:
                ax = axs[subplot_index//cols][subplot_index%cols]
                subplot_index += 1
            else:
                fig = None
                ax = None

            # We only want the normalized plots when we generate subfigures
            if subplots:
                # Also, check results if we normalize - note only applies when we use this
                # data directly rather than generating sines of two frequencies (x1 and x2).
                sources, target = generate_msda_problems(num_sources, num_classes,
                    draw_lines=draw_lines, num_points=1000, from_data=True, normalize=True,
                    center=center,
                    inter_domain_translate=inter_domain_translate,
                    inter_domain_rotate=inter_domain_rotate,
                    intra_domain_translate=intra_domain_translate,
                    intra_domain_rotate=intra_domain_rotate,
                    fig=fig, ax=ax, title=title)

                # Generate actual data
                if not FLAGS.plots_for_paper:
                    if raw:
                        save_data(sources, target, num_classes, name, raw=True,
                            display=display)
                    else:
                        save_data(sources, target, num_classes, name+"_sine",
                            display=display)
            else:
                save_plot(name, extension=plot_ext)

    # Save plot
    if subplots:
        save_plot("normal_n{}_l{}_{}{}".format(num_sources, num_classes,
            variation, suffix), extension=plot_ext)


def sanity_check_freqs(freqs, sample_freq):
    """ Check that no frequencies are negative or over Nyquist sampling rate """
    for i, freq in enumerate(freqs):
        for j, f in enumerate(freq):
            if f < 0 or f > sample_freq/2:
                raise ValueError(
                    "Found frequency {}, which is either <0 or >{}".format(
                        f, sample_freq/2))


def generate_sine(data):
    """ Generate a bunch of univariate time series signals (length matches that
    of data) with two sine waves of particular frequencies each

    Input format: [(ex1 freq1, ex1 freq2), (ex2 freq1, ex2 freq2), ...]
    Outputs: x, y
        - x is the time dimension, i.e. from 0 to duration (in seconds)
        - y is the amplitude at each point in time, i.e. the signal to be used
          for classification, having shape [time_steps, num_examples] where
          time_steps = duration * sample_freq
    """
    # Config
    duration = 0.2  # seconds
    sample_freq = 250  # Hz
    amp_noise = 0.0  # slight amplitude noise
    freq_noise = 0.0  # slight frequency noise
    phase_shift = 0.0  # random phase shift

    # Generate time series data
    freqs = data  # [num examples, 2 frequencies]
    sanity_check_freqs(freqs, sample_freq)
    amps = [1.0, 1.0]
    x, y = sine(f=freqs, amps=amps, maxt=duration,
        length=duration*sample_freq,
        freq_noise=freq_noise, phase_shift=phase_shift)

    if amp_noise is not None:
        y += np.random.normal(0.0, amp_noise, (y.shape[0], len(data)))

    return x, y


def save_single_domain_data(domain, num_points, filename, raw, display=False):
    """
    Generate either raw or time series data for each one domain and save that
    data (and optionally display a sample)

    If raw=True, then write the raw data (2D points) rather than the time series
    signals. Num points indicates how many examples to generate.
    """
    _, data_by_class, labels_by_class = get_distribution_data(domain, num_points)
    data = np.vstack(data_by_class)
    labels = np.hstack(labels_by_class)

    # Convert to sine waves
    if not raw:
        # Debugging
        if display:
            num = 10  # how many from each class to display
            for i in range(len(data_by_class)):
                x, y = generate_sine(data_by_class[i])
                display_xy(x[:, :num], y[:, :num], show=False)
                title = "Class {}".format(i)
                print(title, data_by_class[i][:num])
                plt.title(title)
            plt.show()

        # Generate data for all classes combined
        x, y = generate_sine(data)

        # Transpose so we have [examples, time_steps]
        data = np.array(y, dtype=np.float32).T

        # Expand so we have 1 feature: [examples, time_steps, num_features]
        data = np.expand_dims(data, axis=-1)

    # Write to file
    save_data_file(data, labels, filename)


def save_data(sources, target, num_classes, name,
        num_train_points=10000, num_test_points=1000,
        out="datasets/synthetic", raw=False, display=False):
    """ Save data for each source domain and the target domain """
    if not os.path.exists(out):
        os.makedirs(out)

    # We want this many points total, but they're generated per-class
    num_train_points = num_train_points // num_classes
    num_test_points = num_test_points // num_classes

    for i, source in enumerate(sources):
        save_single_domain_data(source, num_train_points,
            '{}/{}_s{}_TRAIN'.format(out, name, i), raw, display)
        save_single_domain_data(source, num_test_points,
            '{}/{}_s{}_TEST'.format(out, name, i), raw, display)

    save_single_domain_data(target, num_train_points,
        '{}/{}_t_TRAIN'.format(out, name), raw, display)
    save_single_domain_data(target, num_test_points,
        '{}/{}_t_TEST'.format(out, name), raw, display)


def main(argv):
    # Test inter/intra translate/rotate each separately
    # n=4, L=3
    all_problems(num_sources=4, num_classes=3, draw_lines=True,
        base_params=[5, 0, 0, 0], inter_scaling=[0, 1, 2],
        suffix="5,0,0,0")
    all_problems(num_sources=4, num_classes=3, draw_lines=True,
        base_params=[0, 0.5, 0, 0], inter_scaling=[0, 1, 2],
        suffix="0,0.5,0,0")
    all_problems(num_sources=4, num_classes=3, draw_lines=True,
        base_params=[0, 0, 5, 0], intra_scaling=[0, 1, 2],
        suffix="0,0,5,0")
    all_problems(num_sources=4, num_classes=3, draw_lines=True,
        base_params=[0, 0, 0, 0.5], intra_scaling=[0, 1, 2],
        suffix="0,0,0,0.5")

    # n=12, L=3 -- so we have n=2, 4, 6, 8, 10 and 10 has 3 diff. sets available
    all_problems(num_sources=12, num_classes=3, draw_lines=True,
        base_params=[5, 0, 0, 0], inter_scaling=[0, 1, 2],
        suffix="5,0,0,0")
    all_problems(num_sources=12, num_classes=3, draw_lines=True,
        base_params=[0, 0.5, 0, 0], inter_scaling=[0, 1, 2],
        suffix="0,0.5,0,0")
    all_problems(num_sources=12, num_classes=3, draw_lines=True,
        base_params=[0, 0, 5, 0], intra_scaling=[0, 1, 2],
        suffix="0,0,5,0")
    all_problems(num_sources=12, num_classes=3, draw_lines=True,
        base_params=[0, 0, 0, 0.5], intra_scaling=[0, 1, 2],
        suffix="0,0,0,0.5")


if __name__ == "__main__":
    app.run(main)
