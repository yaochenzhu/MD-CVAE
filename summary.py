import os
import glob
import argparse

import numpy as np
import pandas as pd

def get_info(path):
    infos = path.split(os.path.sep)
    split = int(infos[-2])
    dataset = infos[-3]
    try:
        method = infos[-5]
    except:
        method = "md-cvae"
    return method, dataset, split

def summarize():
    ### Parse the console arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all",
        help="specify the dataset for experiment")
    args = parser.parse_args()

    metrics = ["recalls", "NDCGs"]
    agg_tables = []

    for metric in metrics:
        agg_subtables = []
        vbae_temp = os.path.join("models", "*", "*")
        vbae_table_paths = glob.glob(os.path.join(
            vbae_temp, "{}.csv".format(metric)))

        baseline_temp = os.path.join("baselines", "*", vbae_temp)
        baseline_table_paths = glob.glob(os.path.join(
            baseline_temp, "{}.csv".format(metric)))

        table_paths = vbae_table_paths + baseline_table_paths

        for table_path in table_paths:
            method, dataset, split = get_info(table_path)
            if dataset != args.dataset and args.dataset != "all":
                continue
            table = pd.read_csv(table_path)
            if metric == "recalls":
                table = table.set_index("k").loc[[20, 40]].reset_index()
            elif metric == "NDCGs":
                table = table.set_index("k").loc[[100]].reset_index()

            table.insert(0, "split", split)
            table.insert(0, "dataset", dataset)
            table.insert(0, "method", method)
            agg_subtables.append(table)

        cur_subtable = pd.concat(agg_subtables)
        cur_subtable["k"] = cur_subtable["k"].apply(lambda x:metric[:-1]+"@"+str(x))
        cur_subtable.rename(columns={"k":"metric", metric:"values"}, inplace=True)
        cur_subtable.set_index(["method", "dataset", "split", "metric"], inplace=True)
        agg_tables.append(cur_subtable)

    save_root = "results"
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    agg_path = os.path.join(save_root, "{}_results_agg.csv".format(args.dataset))

    agg_table = pd.concat(agg_tables)
    agg_table.reset_index(inplace=True)
    agg_table.to_csv(agg_path, index=False)

    avg_path = os.path.join(save_root, "{}_results_avg.csv".format(args.dataset))
    ordered_methods_all = ["md-cvae", "sym_cvae", "fm", "ctr", "cdl", "cvae", 
                           "mvae", "covae", "condvae", "dicer", "dave"]
    ordered_methods = []
    for method in ordered_methods_all:
        if method in agg_table["method"].values:
            ordered_methods.append(method)

    agg_table["method"] = agg_table["method"].astype("category")
    agg_table["method"].cat.reorder_categories(ordered_methods, inplace=True)

    ordered_metrics = ["recall@20", "recall@40", "NDCG@100"]
    agg_table["metric"] = agg_table["metric"].astype("category")
    agg_table["metric"].cat.reorder_categories(ordered_metrics, inplace=True)

    ### Create a pivot table, average over different splits
    avg_table = agg_table.pivot_table(
        index="method",
        columns=["dataset", "metric"],
        values="values",
        aggfunc=[np.mean, np.std]
    )

    ### Save the pivot table
    avg_table.sort_index(inplace=True)
    print(avg_table)
    avg_table.to_csv(avg_path, float_format='%.3f')

    ### Save the Latex Code
    latex_path = os.path.join(save_root, "latex.txt")
    avg_table.to_latex(latex_path, float_format='%.3f')

if __name__ == '__main__':
    summarize()