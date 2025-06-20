import os
import numpy as np
import cv2
from scipy.fft import fft2, fftshift
from scipy.optimize import minimize
from tqdm import tqdm
import json
from multiprocessing import Pool
import time
from datetime import datetime

# 配置参数
class Config:
    TRAIN_FILE = './dataset/train_list.txt'
    TEST_FILE = './dataset/test_list.txt'
    MODEL_LABELS = {
        0: 'real',
        1: 'ProGAN',
        2: 'MMDGAN',
        3: 'pProGAN',
        4: 'StyleGAN',
        5: 'VanillaVAE',
        6: 'BetaVAE',
        7: 'ADM',
        8: 'DDPM',
        9: 'SD1.5',
        10: 'SD2.1'
    }
    UNKNOWN_LABEL_START = 11
    R_THRESH = 0.50
    R_THRESH_FIT = 0.85  # 与MATLAB的rthreshfit一致
    N_BINS = 200
    SMOOTH_WIN = 5
    OUTPUT_DIR = './fitted_coefficients'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# 傅里叶频谱处理工具类
class FourierAnalyzer:
    @staticmethod
    def radial_spectrum(img, r_thresh):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32) / 255.0
        F = fftshift(fft2(img))
        magnitude = np.abs(F)
        ny, nx = magnitude.shape
        center_x, center_y = nx // 2, ny // 2
        y, x = np.indices(magnitude.shape)
        x = x - center_x
        y = y - center_y
        r = np.sqrt(x**2 + y**2)
        max_r = np.sqrt(center_x**2 + center_y**2)
        mask = r > r_thresh * max_r
        r_selected = r[mask]
        mag_selected = magnitude[mask]
        dc = magnitude[center_y, center_x]  # DC分量
        return r_selected, mag_selected, dc

    @staticmethod
    def bin_spectrum(r, mag, n_bins):
        bins = np.linspace(np.min(r), np.max(r), n_bins + 1)
        bin_indices = np.digitize(r, bins) - 1
        bin_mag = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        for i in range(len(r)):
            if 0 <= bin_indices[i] < n_bins:
                bin_mag[bin_indices[i]] += mag[i]
                bin_counts[bin_indices[i]] += 1
        bin_mag = np.where(bin_counts > 0, bin_mag / bin_counts, 0)
        bin_mag = np.convolve(bin_mag, np.ones(Config.SMOOTH_WIN)/Config.SMOOTH_WIN, mode='same')
        bin_centers = (bins[:-1] + bins[1:]) / 2
        return bin_centers, bin_mag

    @staticmethod
    def power_law_fit(x, y, yi, pnorm=2):
        def objective(c):
            return np.linalg.norm(y - yi * (x / x[0])**c[0], pnorm)
        res = minimize(objective, x0=[-2], bounds=[(-5, 0)])
        return res.x[0]

    @staticmethod
    def get_fit_coefficients(img):
        """修改后返回与MATLAB一致的系数: [衰减指数, 起始点幅值, 终点幅值]"""
        r, mag, dc = FourierAnalyzer.radial_spectrum(img, Config.R_THRESH)
        x_binned, y_binned = FourierAnalyzer.bin_spectrum(r, mag, Config.N_BINS)
        
        # 确定拟合起始点（与MATLAB的nstart逻辑一致）
        n_start = int(Config.N_BINS * (Config.R_THRESH_FIT - Config.R_THRESH) / (1 - Config.R_THRESH))
        x_fit = x_binned[n_start:] / np.max(x_binned)
        y_fit = y_binned[n_start:]
        yi = y_fit[0]  # 起始点幅值
        yf = y_fit[-1]  # 终点幅值
        
        # 幂律拟合，获取衰减指数b2
        b2 = FourierAnalyzer.power_law_fit(x_fit, y_fit, yi)
        
        # 与MATLAB一致，返回 [b2, yi, yf]
        return np.array([b2, yi, yf])

def process_image_with_label(input_line):
    try:
        img_path, label = input_line.strip().split()
        label = int(label)
        img = cv2.imread(img_path)
        if img is None:
            return None, label
        coeffs = FourierAnalyzer.get_fit_coefficients(img)
        return coeffs, label
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return None, -1

def generate_dataset():
    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_subdir = os.path.join(Config.OUTPUT_DIR, f"dataset_{timestamp}")
    os.makedirs(output_subdir, exist_ok=True)
    
    metadata = {
        'train': [],
        'test': [],
        'config': {
            'TRAIN_FILE': Config.TRAIN_FILE,
            'TEST_FILE': Config.TEST_FILE,
            'MODEL_LABELS': Config.MODEL_LABELS,
            'UNKNOWN_LABEL_START': Config.UNKNOWN_LABEL_START,
            'R_THRESH': Config.R_THRESH,
            'R_THRESH_FIT': Config.R_THRESH_FIT,
            'N_BINS': Config.N_BINS,
            'SMOOTH_WIN': Config.SMOOTH_WIN
        }
    }
    
    known_models = list(Config.MODEL_LABELS.values())[1:]
    unknown_models = []
    
    print("Processing training set...")
    with open(Config.TRAIN_FILE, 'r') as f:
        train_lines = f.readlines()
    with Pool() as p:
        results = list(tqdm(p.imap(process_image_with_label, train_lines), total=len(train_lines)))
    for coeffs, label in results:
        if coeffs is None:
            continue
        metadata['train'].append((coeffs.tolist(), label))  # 转换为list以便JSON序列化
        if label >= Config.UNKNOWN_LABEL_START:
            model_name = f"unknown_{label}"
            if model_name not in unknown_models:
                unknown_models.append(model_name)
    
    print("Processing test set...")
    with open(Config.TEST_FILE, 'r') as f:
        test_lines = f.readlines()
    with Pool() as p:
        results = list(tqdm(p.imap(process_image_with_label, test_lines), total=len(test_lines)))
    for coeffs, label in results:
        if coeffs is None:
            continue
        metadata['test'].append((coeffs.tolist(), label))  # 转换为list以便JSON序列化
        if label >= Config.UNKNOWN_LABEL_START:
            model_name = f"unknown_{label}"
            if model_name not in unknown_models:
                unknown_models.append(model_name)
    
    # 转换为numpy数组
    train_coeffs = np.array([item[0] for item in metadata['train']])
    train_labels = np.array([item[1] for item in metadata['train']])
    test_coeffs = np.array([item[0] for item in metadata['test']])
    test_labels = np.array([item[1] for item in metadata['test']])
    
    # 更新metadata
    metadata.update({
        'train_samples': len(train_labels),
        'test_samples': len(test_labels),
        'known_models': known_models,
        'unknown_models': unknown_models,
        'generated_time': timestamp
    })
    
    print("Saving results...")
    # 保存NPZ文件
    np.savez(
        os.path.join(output_subdir, 'coefficients.npz'),
        train_coeffs=train_coeffs,
        train_labels=train_labels,
        test_coeffs=test_coeffs,
        test_labels=test_labels
    )
    
    # 保存JSON文件
    with open(os.path.join(output_subdir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Results saved to: {output_subdir}")

if __name__ == '__main__':
    generate_dataset()