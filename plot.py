import matplotlib.pyplot as plt
import numpy as np
import json
import os


def smooth_data(y, window_size=5):
    """对数据应用滑动窗口平均（Moving Average）"""
    if len(y) < window_size:
        return y  # 数据点太少，不处理
    window = np.ones(window_size) / window_size
    return np.convolve(y, window, mode='valid')  # 'valid' 模式会减少数据点

def save_figs(results_root_dir, smooth_window=5):
    # 遍历所有数据集目录（如 CIFAR10）
    for dataset_name in os.listdir(results_root_dir):
        dataset_dir = os.path.join(results_root_dir, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue

        # 递归查找所有子目录中的 _result.json 文件
        for root, _, files in os.walk(dataset_dir):
            json_files = [f for f in files if f.endswith('_result.json')]
            if not json_files:
                continue

            # 提取 alpha 和 num 参数（从路径中）
            path_parts = root.split(os.sep)
            alpha_part = next((p for p in path_parts if p.startswith('alpha=')), None)
            num_part = next((p for p in path_parts if p.startswith('num=')), None)

            # 初始化数据结构
            acc_data = {}
            time_data = {}
            energy_data = {}

            # 读取所有JSON文件
            for json_file in json_files:
                file_path = os.path.join(root, json_file)
                with open(file_path, 'r') as f:
                    result = json.load(f)
                
                method_name = json_file.replace(f'_{dataset_name}_result.json', '')
                
                # 存储准确率数据（应用滑动平均）
                acc_values = result['global_accuracy']
                smoothed_acc = smooth_data(acc_values, smooth_window)
                acc_data[method_name] = {
                    'rounds': result['current_round'][:len(smoothed_acc)],  # 调整 rounds 长度
                    'values': smoothed_acc
                }
                
                # 存储总时间和总能耗
                time_data[method_name] = sum(result['total_time'])
                energy_data[method_name] = sum(result['total_energy'])

            # 生成图表文件名前缀（包含alpha和num信息）
            prefix = ""
            if alpha_part and num_part:
                prefix = f"{alpha_part}_{num_part}_"
            elif alpha_part:
                prefix = f"{alpha_part}_"
            elif num_part:
                prefix = f"{num_part}_"

            # 1. 绘制平滑后的准确率对比折线图
            plt.figure(figsize=(12, 6))
            for method_name, data in acc_data.items():
                plt.plot(data['rounds'], data['values'], label=method_name)
            plt.xlabel('Round')
            plt.ylabel('Global Accuracy (Smoothed)')
            plt.title(f'{dataset_name} - Smoothed Accuracy Comparison ({alpha_part}, {num_part})')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            plt.savefig(os.path.join(root, f'accuracy_comparison.png'))
            plt.close()

            # 2. 绘制总时间对比柱状图（保持不变）
            plt.figure(figsize=(12, 6))
            plt.bar(time_data.keys(), time_data.values())
            plt.xlabel('Methods')
            plt.ylabel('Total Time (s)')
            plt.title(f'{dataset_name} - Total Time Comparison ({alpha_part}, {num_part})')
            plt.xticks(rotation=45)
            plt.savefig(os.path.join(root, f'time_comparison.png'), bbox_inches='tight')
            plt.close()

            # 3. 绘制总能耗对比柱状图（保持不变）
            plt.figure(figsize=(12, 6))
            plt.bar(energy_data.keys(), energy_data.values(), color='orange')
            plt.xlabel('Methods')
            plt.ylabel('Total Energy (J)')
            plt.title(f'{dataset_name} - Total Energy Comparison ({alpha_part}, {num_part})')
            plt.xticks(rotation=45)
            plt.savefig(os.path.join(root, f'energy_comparison.png'), bbox_inches='tight')
            plt.close()

            print(f"图表已保存到 {root} ({prefix})")
            
            # 4. 写入每种方法的最高准确率到文本文件
            best_acc_lines = []
            for method_name, data in acc_data.items():
                max_acc = max(data['values'])
                best_acc_lines.append(f"{method_name}: {max_acc:.4f}")
            best_acc_text = "\n".join(best_acc_lines)

            with open(os.path.join(root, 'best_accuracy.txt'), 'w') as f:
                f.write(f"Best Smoothed Accuracy Comparison ({alpha_part}, {num_part})\n")
                f.write(best_acc_text + "\n")


if __name__ == '__main__':
    results_dir = './results'
    save_figs(results_dir,smooth_window=1)
