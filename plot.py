import matplotlib.pyplot as plt
import numpy as np
import json
import os

# 指定存放结果文件的目录
results_dir = './results'

# 读取指定目录下的所有 result json 文件
json_files = [f for f in os.listdir(results_dir) if f.endswith('_result.json')]

# 初始化数据容器
accuracies = {}
times = {}
energies = {}

# 加载所有 json 文件数据
for json_file in json_files:
    file_path = os.path.join(results_dir, json_file)  # 获取文件的完整路径
    with open(file_path, 'r') as f:
        result = json.load(f)
        
        # 提取每个文件的 round 和对应的 global_accuracy, total_time, total_energy
        name = json_file.replace('_result.json', '')  # 使用文件名作为区分标识
        accuracies[name] = result['global_accuracy']
        
        # 对每个文件的 total_time 和 total_energy 进行求和
        times[name] = sum(result['total_time'])  # 假设total_time存在并且是一个列表
        energies[name] = sum(result['total_energy'])  # 假设total_energy存在并且是一个列表

# 1. 比较 global_accuracy 的对比折线图
plt.figure(figsize=(12, 6))
for name, accuracy in accuracies.items():
    plt.plot(result['current_round'], accuracy, label=name)

plt.xlabel('Round')
plt.ylabel('Global Accuracy')
plt.title('Global Accuracy over Rounds')
plt.grid(True)
plt.legend()

# 保存图表到 results 目录
plt.savefig(os.path.join(results_dir, 'global_accuracy_comparison.png'))
plt.close()

# 2. 比较 total_time 的对比柱状图，进行求和
plt.figure(figsize=(12, 6))
# 设置宽度为 0.3，避免柱状图铺满整个图表
plt.bar(times.keys(), times.values(), color='g', width=0.1)

plt.xlabel('Result Files')
plt.ylabel('Total Time (s)')
plt.title('Total Time (Sum) over Rounds')

# 保存图表到 results 目录
plt.savefig(os.path.join(results_dir, 'total_time_comparison.png'))
plt.close()

# 3. 比较 total_energy 的对比柱状图，进行求和
plt.figure(figsize=(12, 6))
# 设置宽度为 0.3，避免柱状图铺满整个图表
plt.bar(energies.keys(), energies.values(), color='r', width=0.1)

plt.xlabel('Result Files')
plt.ylabel('Total Energy (J)')
plt.title('Total Energy (Sum) over Rounds')

# 保存图表到 results 目录
plt.savefig(os.path.join(results_dir, 'total_energy_comparison.png'))
plt.close()

print("Charts have been saved to the 'results' directory.")
