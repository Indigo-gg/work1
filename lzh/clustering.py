import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 假设已从代码生成的npz文件中加载数据
data = np.load('./fitted_coefficients/dataset_20250620_021115/coefficients.npz')
train_coeffs = data['train_coeffs']  # 形状: (N, 3), 列顺序: [b2, yi, yf]
train_labels = data['train_labels']  # 形状: (N,)

# 提取b2和yi作为特征
b2 = train_coeffs[:, 0]  # 衰减指数
yi = train_coeffs[:, 1]   # 起始幅值
labels = train_labels     # 标签

# 定义颜色映射 (根据MODEL_LABELS调整)
model_colors = {
    0: '#2ecc71',  # real - 绿色
    1: '#e74c3c',  # ProGAN - 红色
    2: '#3498db',  # MMDGAN - 蓝色
    3: '#9b59b6',  # pProGAN - 紫色
    4: '#f1c40f',  # StyleGAN - 黄色
    5: '#1abc9c',  # VanillaVAE - 蓝绿色
    6: '#d35400',  # BetaVAE - 橙色
    7: '#34495e',  # ADM - 深灰蓝
    8: '#7f8c8d',  # DDPM - 灰色
    9: '#e84393',  # SD1.5 - 粉色
    10: '#00cec9', # SD2.1 - 青绿色
}

# 创建颜色向量
colors = [model_colors[label] for label in labels]

# 绘制聚类图
plt.figure(figsize=(14, 10), dpi=120)
scatter = plt.scatter(
    b2, 
    yi, 
    c=colors, 
    alpha=0.7, 
    s=40, 
    edgecolor='white', 
    linewidth=0.3
)

# 添加图例
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Real',          markerfacecolor=model_colors[0],  markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='ProGAN',        markerfacecolor=model_colors[1],  markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='MMDGAN',        markerfacecolor=model_colors[2],  markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='pProGAN',       markerfacecolor=model_colors[3],  markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='StyleGAN',      markerfacecolor=model_colors[4],  markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='VanillaVAE',    markerfacecolor=model_colors[5],  markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='BetaVAE',       markerfacecolor=model_colors[6],  markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='ADM',           markerfacecolor=model_colors[7],  markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='DDPM',          markerfacecolor=model_colors[8],  markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='SD1.5',         markerfacecolor=model_colors[9],  markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='SD2.1',         markerfacecolor=model_colors[10], markersize=10),
]

plt.legend(
    handles=legend_elements,
    title="Model Types",
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    borderaxespad=0.,
    framealpha=0.9
)

# 坐标轴和标题
plt.xlabel('Decay Exponent (b2)', fontsize=12)
plt.ylabel('Initial Magnitude (yi)', fontsize=12)
plt.title('Clustering of 11 Image Types by Fourier Spectrum Characteristics', pad=20, fontsize=14)

# 优化显示
plt.grid(True, linestyle='--', alpha=0.4)
plt.gca().set_facecolor('#f8f9fa')  # 浅背景
plt.xlim(min(b2)-0.5, max(b2)+0.5)  # 动态调整坐标范围
plt.ylim(min(yi)-10, max(yi)+10)

# 保存高清图
plt.savefig(
    'all_models_clustering.png', 
    bbox_inches='tight', 
    dpi=300, 
    transparent=False
)
plt.show()