import os
import numpy as np
import cv2
from scipy.fft import fft2, fftshift
from scipy.optimize import minimize
from tqdm import tqdm
import json
from multiprocessing import Pool
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
    R_THRESH_FIT = 0.85
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
        dc = magnitude[center_y, center_x]
        return r_selected, mag_selected, dc

    @staticmethod
    def bin_spectrum(r, mag, n_bins):
        """分箱计算，包含空箱处理"""
        if len(r) == 0:
            return np.array([]), np.array([])  # 无有效数据时返回空数组
        
        bins = np.linspace(np.min(r), np.max(r), n_bins + 1)
        bin_indices = np.digitize(r, bins) - 1
        bin_mag = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        
        for i in range(len(r)):
            if 0 <= bin_indices[i] < n_bins:
                bin_mag[bin_indices[i]] += mag[i]
                bin_counts[bin_indices[i]] += 1
        
        # 安全除法计算分箱均值
        epsilon = 1e-10  # 防止除零
        bin_mag = np.where(bin_counts > 0, bin_mag / (bin_counts + epsilon), 0)
        
        # 处理空箱：使用相邻分箱均值填充
        if n_bins > 0:
            bin_mag = FourierAnalyzer._fill_empty_bins(bin_mag)
        
        # 平滑处理
        bin_mag = np.convolve(bin_mag, np.ones(Config.SMOOTH_WIN)/Config.SMOOTH_WIN, mode='same')
        bin_centers = (bins[:-1] + bins[1:]) / 2
        return bin_centers, bin_mag
    
    @staticmethod
    def _fill_empty_bins(bin_mag):
        """使用相邻分箱均值填充空箱"""
        if len(bin_mag) == 0:
            return bin_mag
            
        # 找到空箱索引
        empty_indices = np.where(bin_mag == 0)[0]
        if len(empty_indices) == 0:
            return bin_mag
            
        # 复制原始数组，避免修改输入
        filled_mag = bin_mag.copy()
        
        for idx in empty_indices:
            # 寻找左右有效分箱
            left_idx = idx - 1
            while left_idx >= 0 and filled_mag[left_idx] == 0:
                left_idx -= 1
                
            right_idx = idx + 1
            while right_idx < len(filled_mag) and filled_mag[right_idx] == 0:
                right_idx += 1
                
            # 使用左右有效分箱的均值填充
            if left_idx >= 0 and right_idx < len(filled_mag):
                filled_mag[idx] = (filled_mag[left_idx] + filled_mag[right_idx]) / 2
            elif left_idx >= 0:  # 只有左有效分箱
                filled_mag[idx] = filled_mag[left_idx]
            elif right_idx < len(filled_mag):  # 只有右有效分箱
                filled_mag[idx] = filled_mag[right_idx]
            # 左右都无效时保持为0（极少情况）
            
        return filled_mag

    @staticmethod
    def power_law_fit(x, y, yi, pnorm=2):
        def objective(c):
            return np.linalg.norm(y - yi * (x / x[0])**c[0], pnorm)
        res = minimize(objective, x0=[-2], bounds=[(-5, 0)])
        return res.x[0]

    @staticmethod
    def get_fit_coefficients(img):
        r, mag, dc = FourierAnalyzer.radial_spectrum(img, Config.R_THRESH)
        x_binned, y_binned = FourierAnalyzer.bin_spectrum(r, mag, Config.N_BINS)
        
        n_start = int(Config.N_BINS * (Config.R_THRESH_FIT - Config.R_THRESH) / (1 - Config.R_THRESH))
        if len(x_binned) > n_start:
            x_fit = x_binned[n_start:] / np.max(x_binned)
            y_fit = y_binned[n_start:]
            yi = y_fit[0]
            yf = y_fit[-1]
            b2 = FourierAnalyzer.power_law_fit(x_fit, y_fit, yi)
            return np.array([b2, yi, yf])
        return np.array([0, 0, 0])  # 无有效数据时返回默认值

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
    metadata = {'train': [], 'test': []}
    known_models = list(Config.MODEL_LABELS.values())[1:]
    unknown_models = []
    
    print("Processing training set...")
    with open(Config.TRAIN_FILE, 'r') as f:
        train_lines = f.readlines()
    with Pool() as p:
        results = list(tqdm(p.imap(process_image_with_label, train_lines), total=len(train_lines)))
    for coeffs, label in results:
        if coeffs is None or np.all(coeffs == 0):
            continue
        metadata['train'].append((coeffs, label))
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
        if coeffs is None or np.all(coeffs == 0):
            continue
        metadata['test'].append((coeffs, label))
        if label >= Config.UNKNOWN_LABEL_START:
            model_name = f"unknown_{label}"
            if model_name not in unknown_models:
                unknown_models.append(model_name)
    
    train_coeffs = np.array([item[0] for item in metadata['train']])
    train_labels = np.array([item[1] for item in metadata['train']])
    test_coeffs = np.array([item[0] for item in metadata['test']])
    test_labels = np.array([item[1] for item in metadata['test']])
    
    print("Saving results...")
    time_label = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    # 在文件名中添加时间标签
    npz_filename = f"fitted_coefficients_{time_label}.npz"
    json_filename = f"metadata_{time_label}.json"
    
    np.savez(
        os.path.join(Config.OUTPUT_DIR, npz_filename),
        train_coeffs=train_coeffs,
        train_labels=train_labels,
        test_coeffs=test_coeffs,
        test_labels=test_labels
    )
    
    with open(os.path.join(Config.OUTPUT_DIR, json_filename), 'w') as f:
        json.dump({
            'train_samples': len(train_labels),
            'test_samples': len(test_labels),
            'known_models': known_models,
            'unknown_models': unknown_models,
            'label_mapping': Config.MODEL_LABELS,
            'creation_time': time_label  # 在JSON中记录创建时间
        }, f, indent=2)

if __name__ == '__main__':
    generate_dataset()