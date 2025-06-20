import os
import numpy as np
import cv2
import json
from joblib import dump, load
from datetime import datetime
import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse

class DatasetBuilder:
    def __init__(self, train_file='./dataset/train_list.txt', test_file='./dataset/test_list.txt',
                 output_dir='./fitted_coefficients', image_size=(256, 256), n_workers=8):
        """初始化数据集构建器"""
        self.train_file = train_file
        self.test_file = test_file
        self.output_dir = output_dir
        self.image_size = image_size
        self.n_workers = n_workers
        self.time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dataset_dir = os.path.join(output_dir, f"dataset_new_{self.time_tag}")
        os.makedirs(self.dataset_dir, exist_ok=True)
        
        # 特征存储路径
        self.train_features_path = os.path.join(self.dataset_dir, "train_features.npy")
        self.train_labels_path = os.path.join(self.dataset_dir, "train_labels.npy")
        self.test_features_path = os.path.join(self.dataset_dir, "test_features.npy")
        self.test_labels_path = os.path.join(self.dataset_dir, "test_labels.npy")
        self.metadata_path = os.path.join(self.dataset_dir, "metadata.json")
        
        print(f"数据集将保存至: {self.dataset_dir}")
    
    def load_data_from_txt(self, txt_file):
        """从txt文件加载图像路径和标签"""
        image_paths = []
        labels = []
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    path, label = line.split()
                    image_paths.append(path)
                    labels.append(int(label))
                except:
                    print(f"跳过无效行: {line}")
        
        return np.array(image_paths), np.array(labels)
    
    def is_valid_image(self, path):
        """检查图像是否有效"""
        try:
            img = cv2.imread(path)
            return img is not None and img.size > 0
        except Exception as e:
            print(f"图像验证失败: {path}, 错误: {str(e)}")
            return False
    
    def extract_spectrum_features(self, image_path):
        """提取单张图像的频谱特征"""
        try:
            # 检查图像有效性
            if not self.is_valid_image(image_path):
                print(f"无效图像: {image_path}")
                return None
            
            img = cv2.imread(image_path)
            
            # 转为灰度图并调整大小
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            gray = cv2.resize(gray, self.image_size)
            
            # 二维FFT
            fft_result = np.fft.fftshift(np.fft.fft2(gray))
            magnitude = np.abs(fft_result)
            ny, nx = magnitude.shape
            center_x, center_y = nx // 2, ny // 2
            
            # 径向频谱分析
            y, x = np.indices(magnitude.shape)
            x = x - center_x
            y = y - center_y
            r = np.sqrt(x**2 + y**2)
            max_r = np.sqrt(center_x**2 + center_y**2)
            mask = r > 0.5 * max_r  # 忽略中心低频部分
            
            r_selected = r[mask]
            mag_selected = magnitude[mask]
            dc_component = magnitude[center_y, center_x]  # DC分量
            
            # 频谱分箱
            n_bins = 200
            bins = np.linspace(np.min(r_selected), np.max(r_selected), n_bins + 1)
            bin_indices = np.digitize(r_selected, bins) - 1
            bin_mag = np.zeros(n_bins)
            bin_counts = np.zeros(n_bins)
            
            for i in range(len(r_selected)):
                if 0 <= bin_indices[i] < n_bins:
                    bin_mag[bin_indices[i]] += mag_selected[i]
                    bin_counts[bin_indices[i]] += 1
            
            # 安全处理除零问题
            valid_bins = bin_counts > 0
            if not np.any(valid_bins):
                print(f"警告：所有分箱计数为0，图像可能异常: {image_path}")
                return None
            
            bin_mag[valid_bins] = bin_mag[valid_bins] / bin_counts[valid_bins]
            bin_mag[~valid_bins] = 0  # 处理无效分箱
            
            # 平滑处理
            bin_mag = np.convolve(bin_mag, np.ones(5)/5, mode='same')
            
            # 幂律拟合
            n_start = int(n_bins * 0.3)
            x_fit = bin_mag[n_start:]
            y_fit = np.arange(len(x_fit))
            if np.sum(x_fit) > 0:
                x_fit = x_fit / np.max(x_fit)
                # 简化拟合
                log_x = np.log(x_fit[x_fit > 0])
                log_y = np.log(y_fit[x_fit > 0] + 1e-10)
                if len(log_x) >= 2:
                    slope, _ = np.polyfit(log_x, log_y, 1)
                else:
                    slope = 0
            else:
                slope = 0
            
            # 频谱熵
            spectrum_entropy = self._calculate_entropy(bin_mag)
            
            # 对称性特征
            symmetry = self._calculate_symmetry(magnitude, center_x, center_y)
            
            # 组合特征
            features = np.array([
                slope, dc_component, np.mean(bin_mag), spectrum_entropy, symmetry
            ])
            return features
        
        except Exception as e:
            print(f"处理图像 {image_path} 时发生异常: {str(e)}")
            return None
    
    def _calculate_entropy(self, spectrum):
        p = spectrum / np.sum(spectrum + 1e-10)
        p = p[p > 0]
        return -np.sum(p * np.log(p + 1e-10)) if len(p) > 0 else 0
    
    def _calculate_symmetry(self, magnitude, center_x, center_y):
        q1 = magnitude[:center_y, :center_x]
        q2 = magnitude[:center_y, center_x:]
        q3 = magnitude[center_y:, center_x:]
        q4 = magnitude[center_y:, :center_x]
        sum_q1 = np.sum(q1)
        sum_q3 = np.sum(q3)
        sum_q2 = np.sum(q2)
        sum_q4 = np.sum(q4)
        sym1 = np.abs(sum_q1 - sum_q3) / (sum_q1 + sum_q3 + 1e-10)
        sym2 = np.abs(sum_q2 - sum_q4) / (sum_q2 + sum_q4 + 1e-10)
        return (sym1 + sym2) / 2
    
    def batch_extract_features(self, image_paths, batch_size=100):
        """批量提取特征"""
        all_features = []
        n_images = len(image_paths)
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(tqdm.tqdm(executor.map(self.extract_spectrum_features, image_paths), 
                                   total=n_images, desc="提取特征"))
        
        valid_features = []
        valid_indices = []
        for i, feat in enumerate(results):
            if feat is not None:
                valid_features.append(feat)
                valid_indices.append(i)
        
        if not valid_features:
            raise ValueError("未成功提取任何特征")
        
        return np.array(valid_features), np.array(valid_indices)
    
    def build_dataset(self, force_extract=False):
        """构建数据集：从txt读取并提取特征，保存处理结果"""
        # 加载路径和标签
        print("加载训练数据...")
        X_train_paths, y_train = self.load_data_from_txt(self.train_file)
        print(f"训练集原始大小: {len(X_train_paths)}")
        
        print("加载测试数据...")
        X_test_paths, y_test = self.load_data_from_txt(self.test_file)
        print(f"测试集原始大小: {len(X_test_paths)}")
        
        # 提取特征
        print("提取训练集特征...")
        X_train, train_indices = self.batch_extract_features(X_train_paths)
        y_train = y_train[train_indices]
        print(f"成功提取 {len(X_train)} 个训练样本的特征 ({len(train_indices)}/{len(X_train_paths)})")
        
        print("提取测试集特征...")
        X_test, test_indices = self.batch_extract_features(X_test_paths)
        y_test = y_test[test_indices]
        print(f"成功提取 {len(X_test)} 个测试样本的特征 ({len(test_indices)}/{len(X_test_paths)})")
        
        # 保存特征和标签
        np.save(self.train_features_path, X_train)
        np.save(self.train_labels_path, y_train)
        np.save(self.test_features_path, X_test)
        np.save(self.test_labels_path, y_test)
        
        # 保存元数据
        metadata = {
            'timestamp': self.time_tag,
            'params': {
                'image_size': self.image_size,
                'n_workers': self.n_workers,
                'feature_dim': X_train.shape[1] if len(X_train) > 0 else 0
            },
            'dataset_stats': {
                'train_size': len(X_train),
                'test_size': len(X_test),
                'num_classes': len(np.unique(y_train)) if len(y_train) > 0 else 0,
                'class_distribution_train': {str(c): int(np.sum(y_train == c)) for c in np.unique(y_train)} if len(y_train) > 0 else {},
                'class_distribution_test': {str(c): int(np.sum(y_test == c)) for c in np.unique(y_test)} if len(y_test) > 0 else {}
            }
        }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"特征和元数据已保存至: {self.dataset_dir}")
        return {
            "X_train": X_train, 
            "y_train": y_train, 
            "X_test": X_test, 
            "y_test": y_test,
            "dataset_dir": self.dataset_dir
        }

if __name__ == "__main__":
    # 命令行参数
    parser = argparse.ArgumentParser(description="图像频谱数据集构建器")
    parser.add_argument("--train_file", default="./dataset/train_list.txt", help="训练集txt文件")
    parser.add_argument("--test_file", default="./dataset/test_list.txt", help="测试集txt文件")
    parser.add_argument("--output_dir", default="./fitted_coefficients", help="输出目录")
    parser.add_argument("--image_size", default="256,256", help="图像大小，格式为W,H")
    parser.add_argument("--n_workers", type=int, default=8, help="特征提取线程数")
    parser.add_argument("--force_extract", action='store_true', help="强制重新提取特征")
    args = parser.parse_args()
    
    # 解析图像大小
    w, h = map(int, args.image_size.split(','))
    image_size = (w, h)
    
    # 初始化数据集构建器
    builder = DatasetBuilder(
        train_file=args.train_file,
        test_file=args.test_file,
        output_dir=args.output_dir,
        image_size=image_size,
        n_workers=args.n_workers
    )
    
    # 构建数据集
    dataset = builder.build_dataset(force_extract=args.force_extract)
    
    print(f"数据集构建完成:")
    print(f"  训练集大小: {len(dataset['X_train'])}")
    print(f"  测试集大小: {len(dataset['X_test'])}")
    print(f"  特征维度: {dataset['X_train'].shape[1]}")
    print(f"数据集保存在: {dataset['dataset_dir']}")