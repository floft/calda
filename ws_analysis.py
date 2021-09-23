#!/usr/bin/env python3
"""
Plots for how sensitive weak supervision is to noise
"""
import collections

from absl import app
from absl import flags

from sampling_analysis import get_results

FLAGS = flags.FLAGS


def get_csv(results, output_filename, full=True):
    # Output CSV rather than printing results
    def f1(v):
        """ Format mean and stdev properly """
        if full:
            if v[0] == -1 and v[1] == -1:
                return ";"

            return "{};{}".format(v[0]*100, v[1]*100)
        else:
            if v[0] == -1 and v[1] == -1:
                return ""

            return "{:.1f} $\\pm$ {:.1f}".format(v[0]*100, v[1]*100)

    def f2(v):
        """ Format single float properly """
        if full:
            return "{}".format(v*100)
        else:
            return "{:.1f}".format(v*100)

    methods = [
        "CoDATS",
        "CALDA-XS,H",
        "CALDA-Any,R",
        "CoDATS-WS",
        "CALDA-XS,H,WS",
        "CALDA-Any,R,WS",
    ]

    with open(output_filename, "w") as f:
        columns = ["Dataset", "n", "Noise"]

        # Stdev is separate column in full results
        if full:
            for m in methods:
                columns += [m, "+/-"]
        else:
            columns += methods

        columns = ";".join(columns)
        f.write(columns + "\n")

        csv_results = collections.defaultdict(list)
        for n in results.keys():
            for d in results[n].keys():
                # key is the noise amount
                for key in results[n][d].keys():
                    method_results = []

                    for m in methods:
                        if m in results[n][d][key]:
                            method_results += [results[n][d][key][m]]
                        else:
                            method_results += [(-1, -1)]

                    row = [d, n, key] + [f1(m) for m in method_results]
                    row_str = ";".join([str(x) for x in row])

                    # Also keep raw data
                    csv_results[n].append([d, n, key] + method_results)

                    f.write(row_str + "\n")

                f.write(";;;;;\n")

    return csv_results


def all_results(prefixes, datasets, methods, n):
    """ Get results for each value of n in addition to on average """
    results = {
        "Avg": get_results(datasets, methods, prefixes)
    }

    for dataset in datasets:
        for only_n in n[dataset]:
            # TODO handle more than one dataset
            results[only_n] = get_results([dataset], methods, prefixes, only_n)

    return results


def main(argv):
    n = {
        "ucihar": [2, 8, 14, 20, 26],
        "ucihhar": [2, 3, 4, 5, 6],
        "wisdm_ar": [2, 8, 14, 20, 26],
        "wisdm_at": [2, 12, 22, 32, 42],
    }
    prefixes_ar = [
        (0, "alltune1"),  # for CALDA-XS,H on WISDM AR
        # (0, "alltune1_bounds2"),  # for upper/lower bounds on WISDM AR
        (0, "alltune2"),  # for CALDA-XS,H on WISDM AT
        (0, "allin1"),  # for CALDA-Any-R

        # Weak supervision experiments -- "wsar" for WISDM AR
        (0, "weak2"),
        (0.05, "wsar0.05"),
        (0.1, "wsar0.1"),
        (0.2, "wsar0.2"),
        (0.4, "wsar0.4"),
    ]
    prefixes_at = [
        (0, "alltune1"),  # for CALDA-XS,H on WISDM AR
        # (0, "alltune1_bounds2"),  # for upper/lower bounds on WISDM AR
        (0, "alltune2"),  # for CALDA-XS,H on WISDM AT
        (0, "allin1"),  # for CALDA-Any-R

        # Weak supervision experiments -- "wsat" for WISDM AT
        (0, "weak2"),
        (0.05, "wsat0.05"),
        (0.1, "wsat0.1"),
        (0.2, "wsat0.2"),
        (0.4, "wsat0.4"),
    ]

    non_ws_methods = [
        "codats", "calda_xs_h", "calda_any_r",
    ]

    methods = ["codats_ws", "calda_xs_h_ws", "calda_any_r_ws"]
    methods += non_ws_methods

    def output_results(prefixes, output_filename_prefix, datasets):
        """ Generate plot and CSV file """
        results = all_results(prefixes, datasets, methods, n)
        csv_results = get_csv(results, output_filename_prefix + ".csv")
        return csv_results

    output_results(prefixes_ar, "ws_analysis_ar", ["wisdm_ar"])
    output_results(prefixes_at, "ws_analysis_at", ["wisdm_at"])


if __name__ == "__main__":
    app.run(main)
