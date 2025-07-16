import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import matplotlib.font_manager as fm
fm.fontManager.addfont('times.ttf')
plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams.update({
    "font.size": 16,              # 控制全局字体大小（包括坐标轴、刻度、图例等）
    "axes.titlesize": 15,         # 子图标题字体
    "axes.labelsize": 16,         # 坐标轴标签
    "xtick.labelsize": 14,        # x轴刻度
    "ytick.labelsize": 14,        # y轴刻度
    "legend.fontsize": 14,        # 图例字体
})


root_dir = "results"
datasets = ["Fashion", "CIFAR10", "MNIST"]
alphas = ["alpha=0.1", "alpha=0.5", "alpha=1.0"]
output_dir = os.path.join(root_dir, "cost")
os.makedirs(output_dir, exist_ok=True)

time_all = {ds: defaultdict(dict) for ds in datasets}
energy_all = {ds: defaultdict(dict) for ds in datasets}

for dataset in datasets:
    for alpha in alphas:
        path = os.path.join(root_dir, dataset, alpha, "num=100", "local_rounds=1")
        if not os.path.exists(path):
            continue
        for fname in os.listdir(path):
            if fname.endswith("_result.json"):
                method = fname.split("_")[0]
                full_path = os.path.join(path, fname)
                try:
                    with open(full_path, "r") as f:
                        result = json.load(f)
                    total_time = sum(result.get("total_time", []))
                    total_energy = sum(result.get("total_energy", []))
                    time_all[dataset][method][alpha] = total_time
                    energy_all[dataset][method][alpha] = total_energy
                except Exception as e:
                    print(f"Failed to read {full_path}: {e}")

def plot_bar(metric_data, metric_name, ds):
    plt.figure(figsize=(10, 6))
    methods = list(metric_data[ds].keys())
    methods = sorted(methods, key=lambda m: (m != "ppo", m))

    n_methods = len(methods)
    n_alphas = len(alphas)
    bar_width = 0.1  # 细点的柱宽
    x = np.arange(n_alphas) * 0.8

    for i, method in enumerate(methods):
        values = [metric_data[ds][method].get(alpha, 0) for alpha in alphas]
        if method == "ppo":
            label = "fedppo"
        elif method == "pg":
            label = "CSBWA"
        else:
            label = method

        plt.bar(x + i * bar_width, values, width=bar_width, label=label)

    plt.xticks(x + bar_width * (n_methods - 1) / 2, [a.split("=")[1] for a in alphas])
    plt.xlabel("Alpha")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} Comparison on {ds}")
    plt.legend()
    plt.grid(axis="y", linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{ds}_{metric_name.replace(' ', '_')}.png")  
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def plot_combined_bar(metric_data):
    order = ["CIFAR10", "Fashion", "MNIST"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 7), sharey=False)  # sharey=False ✅

    methods_set = set()
    for ds in order:
        methods_set.update(metric_data[ds].keys())
    methods = sorted(methods_set, key=lambda m: (m != "ppo", m))

    bar_width = 0.1
    x = np.arange(len(alphas)) * 0.8

    handles_all = []
    labels_all = []

    for row, (current_data, current_name) in enumerate([(time_all, "Total Time"), (energy_all, "Total Energy")]):
        for col, ds in enumerate(order):
            ax = axes[row][col]
            for i, method in enumerate(methods):
                values = [current_data[ds].get(method, {}).get(alpha, 0) for alpha in alphas]

                if method == "ppo":
                    label = "FedHPPPO"
                elif method == "pg":
                    label = "CSBWA"
                elif method == "fedavg":
                    label = "FedAvg"
                elif method == "fedcs":
                    label = "FedCS"
                elif method == "greedy":
                    label = "Greedy"
                else:
                    label = method

                bars = ax.bar(x + i * bar_width, values, width=bar_width, label=label)
                if row == 0 and col == 0:
                    handles_all.append(bars[0])
                    labels_all.append(label)

            if row == 0:
                ax.set_title(ds)

            ax.set_xticks(x + bar_width * (len(methods) - 1) / 2)
            ax.set_xticklabels([a.split("=")[1] for a in alphas])
            ax.set_xlabel("Alpha")
            if col == 0:
                ax.set_ylabel(current_name)
            ax.grid(axis="y", linestyle='--', alpha=0.7)

    for ax in axes.flatten():
        if ax.get_legend():
            ax.legend_.remove()

    fig.legend(handles_all, labels_all, loc='upper center', ncol=len(labels_all), 
               fontsize='small', bbox_to_anchor=(0.5, 1.05))
    plt.subplots_adjust(top=0.88, hspace=0.4)

    save_pdf = os.path.join(output_dir, "cost.pdf")
    save_svg = os.path.join(output_dir, "cost.svg")
    # if metric_name == "Total Time":
    #     save_pdf = os.path.join(output_dir, "T.pdf")
    #     save_svg = os.path.join(output_dir, "T.svg")
    # elif metric_name == "Total Energy":
    #     save_pdf = os.path.join(output_dir, "E.pdf")
    #     save_svg = os.path.join(output_dir, "E.svg")   

    plt.savefig(save_pdf, bbox_inches='tight', dpi=500)
    plt.savefig(save_svg, bbox_inches='tight', dpi=500)
    plt.close()
    print(f"Saved: {save_pdf}")


def plot_combined_bar_column(time_data, energy_data):
    """
    绘制一个 3行2列 的柱状图子图网格。
    每行表示一个 dataset（CIFAR10、Fashion、MNIST），
    左图为 Total Time，右图为 Total Energy。
    """
    datasets_order = ["CIFAR10", "Fashion", "MNIST"]
    fig, axes = plt.subplots(3, 2, figsize=(9, 9))  # 横放单栏 or 竖放双栏

    methods_set = set()
    for ds in datasets_order:
        methods_set.update(time_data[ds].keys())
    methods = sorted(methods_set, key=lambda m: (m != "ppo", m))

    bar_width = 0.1
    x = np.arange(len(alphas)) * 0.8  # 横轴位置

    handles_all = []
    labels_all = []

    for row, ds in enumerate(datasets_order):
        for col, (data, metric_name) in enumerate([(time_data, "Total Time"), (energy_data, "Total Energy")]):
            ax = axes[row][col]
            for i, method in enumerate(methods):
                if method == "ppo2": continue
                values = [data[ds].get(method, {}).get(alpha, 0) for alpha in alphas]
                offset = i * bar_width
                label = {
                    "ppo": "FedHPPO",
                    "pg": "CSBWA",
                    "fedavg": "FedAvg",
                    "fedcs": "FedCS",
                    "greedy": "Greedy"
                }.get(method, method)

                bars = ax.bar(x + offset, values, width=bar_width, label=label)
                if row == 0 and col == 0:
                    handles_all.append(bars[0])
                    labels_all.append(label)

            ax.set_title(f"{ds}", fontsize=16)
            ax.set_xticks(x + bar_width * (len(methods) - 1) / 2)
            ax.set_xticklabels([a.split("=")[1] for a in alphas])
            ax.set_xlabel("dirichlet alpha")
            if col == 0:
                ax.set_ylabel(metric_name)
            if col == 1:
                ax.set_ylabel(metric_name)

            ax.grid(axis="y", linestyle='--', alpha=0.6)

    # 去除子图重复图例，在顶部统一加图例
    for ax in axes.flatten():
        if ax.get_legend():
            ax.legend_.remove()

    fig.legend(handles_all, labels_all, loc='upper center', ncol=len(labels_all),
               fontsize=14, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    save_pdf = os.path.join(output_dir, "cost_column.pdf")
    save_svg = os.path.join(output_dir, "cost_column.svg")
    plt.savefig(save_pdf, bbox_inches='tight', dpi=500)
    plt.savefig(save_svg, bbox_inches='tight', dpi=500)
    plt.close()
    print(f"Saved: {save_pdf}")

def plot_alpha_comparison_row(time_data, energy_data, alpha_key="alpha=0.5"):
    datasets_order = ["CIFAR10", "Fashion", "MNIST"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 一行两个子图

    methods_set = set()
    for ds in datasets_order:
        methods_set.update(time_data[ds].keys())
    methods = sorted(methods_set, key=lambda m: (m != "ppo", m))  # fedppo 优先

    bar_width = 0.12
    x = np.arange(len(datasets_order))  # 横坐标为数据集索引

    handles_all = []
    labels_all = []

    for col, (data, metric_name) in enumerate([(time_data, "Total Time"), (energy_data, "Total Energy")]):
        ax = axes[col]
        for i, method in enumerate(methods):
            values = [data[ds].get(method, {}).get(alpha_key, 0) for ds in datasets_order]
            offset = i * bar_width
            label = {
                "ppo": "FedHPPO",
                "pg": "CSBWA",
                "fedavg": "FedAvg",
                "fedcs": "FedCS",
                "greedy": "Greedy"
            }.get(method, method)

            bars = ax.bar(x + offset, values, width=bar_width, label=label)
            if col == 0:
                handles_all.append(bars[0])
                labels_all.append(label)

        ax.set_xticks(x + bar_width * (len(methods) - 1) / 2)
        ax.set_xticklabels(datasets_order)
        ax.set_title(metric_name)
        ax.set_ylabel(metric_name)
        # ax.set_xlabel("Dataset")
        ax.grid(axis="y", linestyle='--', alpha=0.6)

    fig.legend(handles_all, labels_all, loc='upper center', ncol=len(labels_all), fontsize=14, bbox_to_anchor=(0.5, 1.05))
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_pdf = os.path.join(output_dir, "cost_row.pdf")
    save_svg = os.path.join(output_dir, "cost_row.svg")
    plt.savefig(save_pdf, bbox_inches='tight', dpi=500)
    plt.savefig(save_svg, bbox_inches='tight', dpi=500)
    plt.close()
    print(f"Saved: {save_pdf}")




def main(mode='all'):
    if mode == 'all':
        for ds in datasets:
            plot_bar(time_all, "Total Time", ds)
            plot_bar(energy_all, "Total Energy", ds)
    elif mode == 'combined':
        plot_combined_bar(time_all)
    else:
        print("Invalid mode. Choose 'all' or 'combined'.")
    

if __name__ == "__main__":
    # main(mode='all')
    plot_combined_bar_column(time_all, energy_all)    
    # plot_alpha_comparison_row(time_all, energy_all)   

