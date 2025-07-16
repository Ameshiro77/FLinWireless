import matplotlib.pyplot as plt
import numpy as np
import json
import os

def smooth_data(y, window_size=5):
    """对数据应用滑动窗口平均（Moving Average）"""
    if len(y) < window_size:
        return y
    window = np.ones(window_size) / window_size
    return np.convolve(y, window, mode='valid')

def save_figs(results_root_dir, smooth_window=2):
    for dataset_name in os.listdir(results_root_dir):
        dataset_dir = os.path.join(results_root_dir, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue

        for root, _, files in os.walk(dataset_dir):
            json_files = [f for f in files if f.endswith('_result.json')]
            if not json_files:
                continue

            path_parts = root.split(os.sep)
            alpha_part = next((p for p in path_parts if p.startswith('alpha=')), None)
            num_part = next((p for p in path_parts if p.startswith('num=')), None)

            acc_data = {}
            time_data = {}
            energy_data = {}

            for json_file in json_files:
                file_path = os.path.join(root, json_file)
                with open(file_path, 'r') as f:
                    result = json.load(f)
                
                method_name = json_file.replace(f'_{dataset_name}_result.json', '')
                
                acc_values = result['global_accuracy']
                smoothed_acc = smooth_data(acc_values, smooth_window)
                acc_data[method_name] = {
                    'rounds': result['current_round'][:len(smoothed_acc)],
                    'values': smoothed_acc
                }
                
                time_data[method_name] = sum(result['total_time'])
                energy_data[method_name] = sum(result['total_energy'])

            # 1. 绘制平滑后的准确率对比折线图（保留）
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

            # 2. 绘制总时间对比柱状图（保留）
            plt.figure(figsize=(12, 6))
            colors = ['green' if 'ppo' in method_name.lower() else 'blue' for method_name in time_data.keys()]
            plt.bar(time_data.keys(), time_data.values(), color=colors)
            plt.xlabel('Methods')
            plt.ylabel('Total Time (s)')
            plt.title(f'{dataset_name} - Total Time Comparison ({alpha_part}, {num_part})')
            plt.xticks(rotation=45)
            plt.savefig(os.path.join(root, f'time_comparison.png'), bbox_inches='tight')
            plt.close()

            # 3. 绘制总能耗对比柱状图（保留）
            plt.figure(figsize=(12, 6))
            colors = ['green' if 'ppo' in method_name.lower() else 'orange' for method_name in energy_data.keys()]
            plt.bar(energy_data.keys(), energy_data.values(), color=colors)
            plt.xlabel('Methods')
            plt.ylabel('Total Energy (J)')
            plt.title(f'{dataset_name} - Total Energy Comparison ({alpha_part}, {num_part})')
            plt.xticks(rotation=45)
            plt.savefig(os.path.join(root, f'energy_comparison.png'), bbox_inches='tight')
            plt.close()

            # 4. 新增：绘制时间和能耗的组合对比图
            plt.figure(figsize=(12, 6))
            
            # 准备数据
            methods = list(time_data.keys())
            time_values = list(time_data.values())
            energy_values = list(energy_data.values())
            
            # 创建双Y轴
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # 左侧Y轴（时间）
            color = 'tab:blue'
            bars1 = ax1.bar([x + 0.2 for x in range(len(methods))], time_values, 
                          width=0.4, color=color, alpha=0.6, label='Time (s)')
            ax1.set_xlabel('Methods')
            ax1.set_ylabel('Total Time (s)', color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            
            # 右侧Y轴（能耗）
            ax2 = ax1.twinx()
            color = 'tab:orange'
            bars2 = ax2.bar([x - 0.2 for x in range(len(methods))], energy_values, 
                           width=0.4, color=color, alpha=0.6, label='Energy (J)')
            ax2.set_ylabel('Total Energy (J)', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            
            # 设置X轴标签
            plt.xticks(range(len(methods)), methods)
            plt.xticks(rotation=45)
            
            # 添加图例
            lines = [bars1[0], bars2[0]]
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper right')
            
            plt.title(f'{dataset_name} - Time and Energy Comparison ({alpha_part}, {num_part})')
            plt.savefig(os.path.join(root, f'time_and_energy.png'), bbox_inches='tight')
            plt.close()

            # 5. 写入最高准确率
            best_acc_lines = []
            for json_file in json_files:
                file_path = os.path.join(root, json_file)
                with open(file_path, 'r') as f:
                    result = json.load(f)
                method_name = json_file.replace(f'_{dataset_name}_result.json', '')
                max_acc = max(result['global_accuracy'])
                best_acc_lines.append(f"{method_name}: {max_acc:.4f}")

            with open(os.path.join(root, 'best_accuracy.txt'), 'w') as f:
                f.write(f"Best Accuracy Comparison ({alpha_part}, {num_part})\n")
                f.write("\n".join(best_acc_lines) + "\n")

            print(f"图表已保存到 {root}")

if __name__ == '__main__':
    results_dir = './results'
    save_figs(results_dir, smooth_window=2)