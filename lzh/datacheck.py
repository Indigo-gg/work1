import os
import cv2
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def loader(train_file, max_images=None, show_plot=True):
    """
    读取train.txt中的图片路径，统计图片尺寸分布
    
    参数:
        train_file: str, train.txt文件路径
        max_images: int, 最多处理的图片数量，None表示处理所有
        show_plot: bool, 是否显示尺寸分布图表
    
    返回:
        size_counts: dict, 尺寸到数量的映射
        unique_sizes: list, 唯一尺寸列表（按出现次数降序排列）
    """
    # 确保文件存在
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"文件不存在: {train_file}")
    
    print(f"开始读取{train_file}中的图片尺寸...")
    
    # 读取图片路径
    with open(train_file, 'r') as f:
        lines = f.readlines()
        # 假设每行格式为"图片路径 标签"，提取路径部分
        img_paths = [line.strip().split()[0] for line in lines]
        
        # 限制处理图片数量
        if max_images:
            img_paths = img_paths[:max_images]
            print(f"限制处理{max_images}张图片")
    
    # 统计尺寸
    size_counts = Counter()
    invalid_images = 0
    
    for i, path in enumerate(img_paths):
        try:
            # 读取图片
            img = cv2.imread(path)
            if img is None:
                invalid_images += 1
                continue
                
            # 获取尺寸 (高度, 宽度)
            h, w = img.shape[:2]
            size = (h, w)
            size_counts[size] += 1
            
            # 打印进度
            if (i + 1) % 100 == 0 or (i + 1) == len(img_paths):
                print(f"已处理{i+1}/{len(img_paths)}张图片，当前尺寸: {size}")
                
        except Exception as e:
            print(f"处理图片{path}时出错: {str(e)}")
            invalid_images += 1
    
    # 转换为按出现次数降序排列的列表
    unique_sizes = sorted(size_counts.items(), key=lambda x: x[1], reverse=True)
    
    # 打印统计结果
    print(f"\n图片尺寸统计结果:")
    print(f"总图片数: {len(img_paths)}")
    print(f"有效图片数: {len(img_paths) - invalid_images}")
    print(f"无效图片数: {invalid_images}")
    print(f"唯一尺寸数量: {len(unique_sizes)}")
    
    for size, count in unique_sizes[:10]:  # 打印前10大尺寸
        print(f"尺寸 {size}: {count}张, 占比 {count/len(img_paths)*100:.2f}%")
    
    if len(unique_sizes) > 10:
        others = sum(count for size, count in unique_sizes[10:])
        print(f"其他尺寸: {others}张, 占比 {others/len(img_paths)*100:.2f}%")

    
    # 添加中文字体配置
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    # 解决负号显示问题
    plt.rcParams["axes.unicode_minus"] = False
    
    # 绘制尺寸分布图表
    if show_plot and len(unique_sizes) > 0:
        plt.figure(figsize=(12, 8))
        
        # 提取尺寸和数量
        sizes = [f"{h}x{w}" for (h, w), count in unique_sizes[:20]]  # 最多显示20种尺寸
        counts = [count for (h, w), count in unique_sizes[:20]]
        
        # 绘制柱状图
        sns.barplot(x=sizes, y=counts)
        plt.title('图片尺寸分布')
        plt.xlabel('尺寸 (高度x宽度)')
        plt.ylabel('数量')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 保存图表
        plot_path = os.path.join(os.path.dirname(train_file), 'image_size_distribution.png')
        plt.savefig(plot_path)
        print(f"尺寸分布图表已保存至: {plot_path}")
        plt.close()
    
    return dict(unique_sizes), unique_sizes

# 使用示例
if __name__ == "__main__":
    train_file = "/home/wangz/lzh/dataset/train_list.txt"  # 替换为实际路径
    size_counts, unique_sizes = loader(train_file, max_images=44000, show_plot=True)
    
    # 保存统计结果为JSON
    import json
    stats_path = os.path.join(os.path.dirname(train_file), 'image_size_stats.json')
    with open(stats_path, 'w') as f:
        # 转换尺寸元组为字符串以便JSON序列化
        stats = {f"{h}x{w}": count for (h, w), count in unique_sizes}
        json.dump(stats, f, indent=2)
    print(f"尺寸统计数据已保存至: {stats_path}")
