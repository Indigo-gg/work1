import os
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from joblib import dump, load
from datetime import datetime
import argparse

class ClassifierTrainer:
    def __init__(self, dataset_dir, output_dir='./fitted_coefficients', n_workers=8):
        """初始化分类器训练器"""
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.n_workers = n_workers
        self.time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 特征文件路径
        self.train_features_path = os.path.join(dataset_dir, "train_features.npy")
        self.train_labels_path = os.path.join(dataset_dir, "train_labels.npy")
        self.test_features_path = os.path.join(dataset_dir, "test_features.npy")
        self.test_labels_path = os.path.join(dataset_dir, "test_labels.npy")
        self.metadata_path = os.path.join(dataset_dir, "metadata.json")
        
        # 模型相关路径
        self.results_dir = os.path.join(output_dir, f"results_{self.time_tag}")
        os.makedirs(self.results_dir, exist_ok=True)
        self.model_path = os.path.join(self.results_dir, "xgb_model.joblib")
        self.scaler_path = os.path.join(self.results_dir, "scaler.joblib")
        self.eval_path = os.path.join(self.results_dir, "evaluation.json")
        self.cm_path = os.path.join(self.results_dir, "confusion_matrix.png")
        
        # 初始化组件
        self.scaler = None
        self.model = None
        self.class_names = []
        
        # 加载元数据
        self.metadata = self._load_metadata()
        print(f"从 {dataset_dir} 加载数据集")
        print(f"特征维度: {self.metadata['params']['feature_dim']}")
        print(f"训练集大小: {self.metadata['dataset_stats']['train_size']}")
        print(f"测试集大小: {self.metadata['dataset_stats']['test_size']}")
    
    def _load_metadata(self):
        """加载数据集元数据"""
        with open(self.metadata_path, 'r') as f:
            return json.load(f)
    
    def load_dataset(self):
        """加载预处理好的数据集"""
        print("加载训练集特征...")
        X_train = np.load(self.train_features_path)
        y_train = np.load(self.train_labels_path)
        
        print("加载测试集特征...")
        X_test = np.load(self.test_features_path)
        y_test = np.load(self.test_labels_path)
        
        print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
        return X_train, y_train, X_test, y_test
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """使用XGBoost训练模型（多GPU优化）"""
        # 特征标准化
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 初始化XGBoost（多GPU配置）
        self.model = xgb.XGBClassifier(
            tree_method='hist',
            device='cuda',
            n_gpus=-1,         # 使用所有GPU
            objective='multi:softmax',
            num_class=len(np.unique(y_train)),
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            min_child_weight=1,
            verbosity=2
        )
        
        # 训练模型
        print("开始训练XGBoost模型...")
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            early_stopping_rounds=10,
            verbose=20
        )
        
        # 评估模型
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"测试集准确率: {accuracy:.4f}")
        
        # 生成分类报告
        report = classification_report(y_test, y_pred)
        print("分类报告:\n", report)
        
        # 生成混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        self._plot_confusion_matrix(cm)
        
        # 保存评估结果
        eval_results = {
            "accuracy": float(accuracy),
            "classification_report": report,
            "timestamp": self.time_tag,
            "model_params": self.model.get_params()
        }
        with open(self.eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"评估结果已保存至: {self.eval_path}")
        
        return accuracy
    
    def _plot_confusion_matrix(self, cm):
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("预测标签")
        plt.ylabel("真实标签")
        plt.title("混淆矩阵")
        plt.tight_layout()
        plt.savefig(self.cm_path, dpi=300)
        print(f"混淆矩阵已保存至: {self.cm_path}")
    
    def save_model(self):
        """保存模型和标准化器"""
        if self.model is None or self.scaler is None:
            raise ValueError("模型未训练，无法保存")
        
        dump(self.model, self.model_path)
        dump(self.scaler, self.scaler_path)
        print(f"模型已保存至: {self.model_path}")
        print(f"标准化器已保存至: {self.scaler_path}")

if __name__ == "__main__":
    # 命令行参数
    parser = argparse.ArgumentParser(description="图像频谱分类器训练器")
    parser.add_argument("--dataset_dir", required=True, help="数据集目录路径")
    parser.add_argument("--output_dir", default="./fitted_coefficients", help="输出目录")
    parser.add_argument("--n_workers", type=int, default=8, help="特征提取线程数")
    args = parser.parse_args()
    
    # 设置CUDA环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    os.environ['XGBOOST_CUDA_ALLOCATOR'] = 'cudaMallocAsync'
    
    # 初始化分类器训练器
    trainer = ClassifierTrainer(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        n_workers=args.n_workers
    )
    
    # 加载数据集
    X_train, y_train, X_test, y_test = trainer.load_dataset()
    
    # 训练模型
    accuracy = trainer.train_xgboost(X_train, y_train, X_test, y_test)
    
    # 保存模型
    trainer.save_model()
    
    print(f"训练完成，测试集准确率: {accuracy:.4f}")
    print(f"模型保存在: {trainer.results_dir}")