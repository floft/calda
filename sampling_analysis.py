#!/usr/bin/env python3
"""
Compare hard/random sampling
"""
import collections
import numpy as np
import matplotlib.pyplot as plt

from absl import app
from absl import flags
from matplotlib.ticker import MaxNLocator

from analysis import pretty_dataset_name, nice_method_names, gen_jitter
from hyperparameter_tuning_analysis import get_average_accuracy

FLAGS = flags.FLAGS


def print_results(datasets, methods, prefixes, only_n=None):
    for dataset in datasets:
        for method in methods:
            print("  Dataset:", dataset)
            print("    Method:", method)

            print("      Average accuracy for each run:")
            for prefix in prefixes:
                accuracy, stdev = get_average_accuracy("results", [prefix],
                    dataset, method, only_n=only_n)
                print("        ", prefix, accuracy, stdev)
            print()


def get_results(datasets, methods, prefixes, only_n=None):
    # results[dataset][key][method] = ()
    results = collections.defaultdict(
        lambda: collections.defaultdict(
            lambda: collections.defaultdict(tuple)
        )
    )

    for dataset in datasets:
        dataset_name = pretty_dataset_name(dataset)

        for method in methods:
            method_name = nice_method_names[method]

            for key, prefix in prefixes:
                accuracy, stdev = get_average_accuracy("results", [prefix],
                    dataset, method, only_n=only_n)

                # Don't add if it wasn't found
                if accuracy != -1 and stdev != -1:
                    # Save with nice names
                    assert method_name not in results[dataset_name][key], \
                        "duplicate found: " + method_name + " already in " + str(results[dataset_name][key])
                    results[dataset_name][key][method_name] = (accuracy, stdev)

    return results


def get_csv(results, output_filename):
    # Output CSV rather than printing results
    def f1(v):
        """ Format mean and stdev properly """
        return "{:.1f} $\\pm$ {:.1f}".format(v[0]*100, v[1]*100)

    def f2(v):
        """ Format single float properly """
        return "{:.1f}".format(v*100)

    with open(output_filename, "w") as f:
        f.write("Dataset;n;P&N Multiplier;CALDA-XS,R;CALDA-XS,H;Gap\n")
        csv_results = collections.defaultdict(list)
        for n in results.keys():
            for d in results[n].keys():
                for multiplier in results[n][d].keys():
                    r = results[n][d][multiplier]["CALDA-XS,R"]
                    h = results[n][d][multiplier]["CALDA-XS,H"]
                    gap = h[0] - r[0]

                    row = [d, n, multiplier, f1(r), f1(h), f2(gap)]
                    row_str = ";".join([str(x) for x in row])

                    # Also keep raw data
                    csv_results[n].append([d, n, multiplier, r, h, gap])

                    f.write(row_str + "\n")

                f.write(";;;;;\n")

    return csv_results


def best_fit(x, y, label="Least Sq.", alpha=1.0):
    # Best-fit line
    bestfit, stats = np.polynomial.polynomial.Polynomial.fit(x, y, deg=1, full=True)
    # resid, rank, sv, rcond = stats

    x = np.linspace(min(x), max(x))
    y = bestfit(x)
    plt.plot(x, y, "-", label=label, alpha=alpha)


def plot(results, save_plot_filename=None, figsize=(5, 3), ncol=1):
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=100)

    # ax.set_ylim(yrange)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    markers = ["o", "v", "^", "<", ">", "s", "p", "*", "D", "P", "X", "h",
       "1", "2", "3", "4", "+", "x", "d", "H", "|", "_"]

    xs = []
    ys = []

    jitter = gen_jitter(len(results), amount=0.05)

    for i, n in enumerate(results.keys()):
        avg = n == "Avg"

        # Get multiplier for x and gap for y
        x = [v[2] for v in results[n]]
        y = [v[5]*100 for v in results[n]]  # accuracy

        # Jitter slightly
        x_jittered = [x[j] + jitter[i] for j in range(len(x))]

        # line_type = "-" if avg else "--"
        label = "Avg" if avg else "$n={}$".format(n)
        # plt.plot(x, y, markers[i]+line_type, label=label, alpha=0.8)

        # Exclude average
        if not avg:
            plt.scatter(x_jittered, y, label=label, marker=markers[i])

            # For best-fit line
            xs += x
            ys += y

            # best_fit(x, y, label)

    best_fit(xs, ys)

    ax.set_xlabel("P&N Multiplier")
    ax.set_ylabel("Hard Sampling Accuracy Gain ($H - R$ %)")

    # Put legend outside the graph http://stackoverflow.com/a/4701285
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=ncol)

    if save_plot_filename is not None:
        plt.savefig(save_plot_filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def all_results(prefixes, datasets, methods, n):
    """ Get results for each value of n in addition to on average """
    results = {
        "Avg": get_results(datasets, methods, prefixes)
    }

    for only_n in n:
        results[only_n] = get_results(datasets, methods, prefixes, only_n)

    return results


def main(argv):
    n = [2, 8, 14, 20, 26]  # for WISDM AR

    prefixes_b128 = [
        (1, "sample-b128-p5-n10"),
        (2, "sample-b128-p10-n20"),
        (4, "sample-b128-p20-n40"),
        (6, "sample-b128-p30-n60"),
        (8, "sample-b128-p40-n80"),
    ]

    prefixes_b64 = [
        (1, "sample-b64-p5-n10"),
        (2, "sample-b64-p10-n20"),
        (4, "sample-b64-p20-n40"),
        (6, "sample-b64-p30-n60"),
        (8, "sample-b64-p40-n80"),
    ]

    datasets = ["wisdm_ar"]
    methods = ["calda_xs_r", "calda_xs_h"]

    def output_results(prefixes, output_filename_prefix):
        """ Generate plot and CSV file """
        results = all_results(prefixes, datasets, methods, n)
        csv_results = get_csv(results, output_filename_prefix + ".csv")
        plot(csv_results, output_filename_prefix + ".pdf")

    output_results(prefixes_b64, "sampling_analysis_b64")
    output_results(prefixes_b128, "sampling_analysis_b128")


if __name__ == "__main__":
    app.run(main)
