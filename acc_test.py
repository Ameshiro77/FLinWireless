import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm

# 加载 Times New Roman 字体
fm.fontManager.addfont('times.ttf')
plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
    'legend.title_fontsize': 14
})

datasets = ['CIFAR10', 'Fashion', 'MNIST']
dir_vals = ['0.5', '1.0', '0.0']  # IID 放最右边
data_dir = './data/dist'
col_names = [f'class{i}' for i in range(10)]

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10), sharex=False)

for i, dataset in enumerate(datasets):
    for j, dir_val in enumerate(dir_vals):
        filename = f"{dataset}_{dir_val}.csv"
        filepath = os.path.join(data_dir, filename)

        df = pd.read_csv(filepath, skiprows=1)
        class_dist = df[col_names].iloc[:10].multiply(df['Amount'].iloc[:10], axis=0)

        ax = axes[i, j]
        class_dist.plot.barh(stacked=True, ax=ax, legend=False)

        display_name = 'FashionMNIST' if dataset == 'Fashion' else dataset
        if dir_val == '0.0':
            ax.set_title(f'{display_name} (IID)')
        else:
            ax.set_title(f'{display_name} (dir={dir_val})')

        if j == 0:
            ax.set_ylabel('Client ID')
        else:
            ax.set_ylabel('')

        if i == 2:
            ax.set_xlabel('Sample Count')
        else:
            ax.set_xlabel('')

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, title='Classes', loc='upper center', ncol=10, bbox_to_anchor=(0.5, 1.04))

plt.tight_layout()
plt.subplots_adjust(top=0.88)

output_path_pdf = os.path.join(data_dir, 'dist.pdf')
output_path_svg = os.path.join(data_dir, 'dist.svg')
plt.savefig(output_path_pdf, dpi=400, bbox_inches='tight')
plt.savefig(output_path_svg, dpi=400, bbox_inches='tight')
plt.show()
