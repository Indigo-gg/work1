import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# 查找所有支持中文的字体
chinese_fonts = [f.name for f in fm.fontManager.ttflist if 'hei' in f.name.lower() or 'song' in f.name.lower() or 'noto' in f.name.lower()]

if chinese_fonts:
    plt.rcParams['font.sans-serif'] = [chinese_fonts[0]]  # 使用第一个找到的中文字体
    plt.rcParams['axes.unicode_minus'] = False
else:
    print("警告：未找到中文字体，可能无法正确显示中文！")