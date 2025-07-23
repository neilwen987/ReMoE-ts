import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_log_file(log_file_path):
    """Parse training log file and extract iteration and loss values"""
    iterations = []
    loss_values = []
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Match validation loss lines, but exclude final results on test set and validation set
            if 'validation loss at iteration' in line and 'on validation set' not in line and 'on test set' not in line:
                # Use regex to extract iteration and loss values
                match = re.search(r'validation loss at iteration (\d+) \| lm loss value: ([\d.E+-]+)', line)
                if match:
                    iteration = int(match.group(1))
                    loss_value = float(match.group(2))
                    
                    iterations.append(iteration)
                    loss_values.append(loss_value)
    
    return iterations, loss_values

def plot_loss_curve(iterations, loss_values, save_path=None):
    """绘制loss曲线图"""
    plt.figure(figsize=(12, 8))
    
    # 绘制曲线
    plt.plot(iterations, loss_values, 'b-', linewidth=2, marker='o', markersize=4, alpha=0.7)
    
    # 设置图表样式
    plt.title('eval loss curve', fontsize=16, fontweight='bold')
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('LM Loss Value', fontsize=14)
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 设置坐标轴
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # 添加最小值标注
    min_loss_idx = np.argmin(loss_values)
    min_iteration = iterations[min_loss_idx]
    min_loss = loss_values[min_loss_idx]
    
    plt.annotate(f'Loss: {min_loss:.4f}\n iteration: {min_iteration}',
                xy=(min_iteration, min_loss),
                xytext=(min_iteration + 2000, min_loss + 0.1),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 紧凑布局
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    # 显示图表
    plt.show()
    
    return plt

def main():
    # 日志文件路径
    path = 'MOE-SPTopk-182M-0722-134613'
    log_file = f"logs/{path}/train.log"
    
    # 检查文件是否存在
    if not Path(log_file).exists():
        print(f"错误: 找不到日志文件 {log_file}")
        return
    
    # 解析日志文件
    print("正在解析日志文件...")
    iterations, loss_values = parse_log_file(log_file)
    
    if not iterations:
        print("错误: 在日志文件中没有找到loss数据")
        return
    
    print(f"找到 {len(iterations)} 个数据点")
    print(f"迭代范围: {min(iterations)} - {max(iterations)}")
    print(f"Loss范围: {min(loss_values):.4f} - {max(loss_values):.4f}")
    
    # 绘制曲线图
    print("正在绘制曲线图...")
    save_path = f"logs/{path}/loss_curve.png"
    plot_loss_curve(iterations, loss_values, save_path)
    
    # 打印一些统计信息
    print(f"\n统计信息:")
    print(f"初始Loss: {loss_values[0]:.4f} (迭代 {iterations[0]})")
    print(f"最终Loss: {loss_values[-1]:.4f} (迭代 {iterations[-1]})")
    print(f"最低Loss: {min(loss_values):.4f} (迭代 {iterations[np.argmin(loss_values)]})")
    print(f"Loss改善: {loss_values[0] - loss_values[-1]:.4f} ({((loss_values[0] - loss_values[-1]) / loss_values[0] * 100):.1f}%)")

if __name__ == "__main__":
    main() 