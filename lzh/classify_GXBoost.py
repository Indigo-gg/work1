import os
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from joblib import dump, load
from datetime import datetime

class XGBoostMultiGPUTrainer:
    def __init__(self, config_path='./fitted_coefficients/dataset_20250620_021115/metadata.json', 
                 data_path='./fitted_coefficients/dataset_20250620_021115/coefficients.npz',
                 output_root_dir='./fitted_coefficients/'):
        """初始化XGBoost多GPU分类器训练器"""
        self.config_path = config_path
        self.data_path = data_path
        self.output_root_dir = output_root_dir
        self.metadata = self._load_metadata()
        self.class_names = self._get_class_names()
        
        # 创建带有时间标签的结果目录
        self.time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(self.output_root_dir, f'results_XGBoost_{self.time_tag}')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 定义固定文件名
        self.model_save_path = os.path.join(self.results_dir, 'xgb_classifier.joblib')
        self.confusion_matrix_path = os.path.join(self.results_dir, 'confusion_matrix.png')
        self.evaluation_path = os.path.join(self.results_dir, 'evaluation.json')
        
        # 初始化模型和scaler
        self.model = None
        self.scaler = None
    
    def _load_metadata(self):
        """加载元数据"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"元数据文件不存在: {self.config_path}")
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def _get_class_names(self):
        """获取类别名称列表"""
        known_models = self.metadata['known_models']
        unknown_models = self.metadata['unknown_models']
        return ['real'] + known_models + unknown_models
    
    def load_data(self):
        """加载拟合系数和标签"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
        data = np.load(self.data_path)
        return (
            data['train_coeffs'].astype(np.float32),  # 转换为float32
            data['train_labels'],
            data['test_coeffs'].astype(np.float32),   # 转换为float32
            data['test_labels']
        )
    
    def preprocess_data(self, train_features, test_features):
        """数据预处理：仅标准化（XGBoost自动处理GPU数据传输）"""
        self.scaler = StandardScaler()
        train_features_scaled = self.scaler.fit_transform(train_features)
        test_features_scaled = self.scaler.transform(test_features)
        return train_features_scaled, test_features_scaled
    
    def optimize_hyperparameters(self, train_features, train_labels):
        """优化XGBoost超参数（多GPU模式）"""
        print("开始优化XGBoost超参数（多GPU模式）...")
        
        xgb = XGBClassifier(
            # GPU核心配置
            tree_method='gpu_hist',       # GPU直方图算法
            gpu_id=0,                    # 起始GPU ID
            n_gpus=-1,                   # 使用所有可用GPU
            predictor='gpu_predictor',   # GPU预测器
            
            # 算法参数（针对低维数据优化）
            objective='multi:softmax',
            num_class=len(self.class_names),
            n_estimators=100,            # 树的数量
            max_depth=3,                 # 低维数据使用浅树
            learning_rate=0.1,
            gamma=0,
            min_child_weight=1,
            subsample=1.0,               # 全量采样
            colsample_bytree=1.0,        # 全量特征
            
            # 性能参数
            batch_size=len(train_labels),# 一次性加载全量数据
            verbosity=2,                 # 详细日志
            nthread=-1                   # 使用所有CPU核心
        )
        
        # 超参数搜索空间（简化版）
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [2, 3, 4],
            'learning_rate': [0.05, 0.1, 0.2]
        }
        
        grid_search = GridSearchCV(
            xgb, param_grid, cv=3, scoring='accuracy', 
            n_jobs=1, verbose=2  # 单进程避免GPU冲突
        )
        grid_search.fit(train_features, train_labels)
        
        print(f"最佳超参数: {grid_search.best_params_}")
        print(f"交叉验证最佳准确率: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_
    
    def train(self):
        """训练模型（多GPU模式）"""
        print("开始多GPU训练流程...")
        print(f"本次运行结果将保存至: {self.results_dir}")
        
        # 1. 加载数据
        train_coeffs, train_labels, test_coeffs, test_labels = self.load_data()
        print(f"加载数据完成: 训练集大小={len(train_labels)}, 测试集大小={len(test_labels)}")
        
        # 2. 数据预处理（仅标准化，无需手动转换到GPU）
        train_scaled, test_scaled = self.preprocess_data(train_coeffs, test_coeffs)
        print("数据预处理完成（CPU模式，XGBoost自动传输至GPU）")
        
        # 3. 超参数优化
        self.model = self.optimize_hyperparameters(train_scaled, train_labels)
        
        # 4. 训练最终模型
        print("训练最终模型...")
        self.model.fit(train_scaled, train_labels)
        
        # 5. 保存模型
        self.save_model()
        
        # 6. 返回测试数据
        return test_scaled, test_labels
    
    def test(self, test_features, test_labels):
        """测试模型"""
        if self.model is None:
            raise ValueError("模型未训练，请先调用train()方法")
        
        print("开始测试流程...")
        
        # 评估模型
        evaluation_results = self.evaluate_model(test_features, test_labels)
        
        # 保存评估结果
        self._save_evaluation_results(evaluation_results)
        
        print("测试流程完成")
        return evaluation_results
    
    def evaluate_model(self, test_features, test_labels):
        """评估模型性能"""
        # 预测（XGBoost自动使用GPU）
        test_pred = self.model.predict(test_features)
        
        # 计算准确率
        test_accuracy = accuracy_score(test_labels, test_pred)
        print(f"测试集准确率: {test_accuracy:.4f}")
        
        # 生成分类报告
        report = classification_report(
            test_labels, test_pred, target_names=self.class_names
        )
        print("分类报告:\n", report)
        
        # 生成混淆矩阵
        cm = confusion_matrix(test_labels, test_pred)
        self._plot_confusion_matrix(cm)
        
        return {
            'test_accuracy': float(test_accuracy),  # 转换为JSON可序列化类型
            'classification_report': report,
            'timestamp': self.time_tag
        }
    
    def _plot_confusion_matrix(self, cm):
        """绘制混淆矩阵"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(self.confusion_matrix_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"混淆矩阵已保存至: {self.confusion_matrix_path}")
    
    def save_model(self):
        """保存训练好的XGBoost模型"""
        if self.model is None:
            raise ValueError("没有可保存的模型，请先训练模型")
        
        # 保存模型和scaler
        dump({'model': self.model, 'scaler': self.scaler}, self.model_save_path)
        print(f"XGBoost模型已保存至: {self.model_save_path}")
    
    def load_saved_model(self):
        """加载已保存的模型"""
        if not os.path.exists(self.model_save_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_save_path}")
            
        saved_data = load(self.model_save_path)
        self.model = saved_data['model']
        self.scaler = saved_data['scaler']
        print(f"已从 {self.model_save_path} 加载模型和scaler")
    
    def _save_evaluation_results(self, results):
        """保存评估结果"""
        with open(self.evaluation_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"评估结果已保存至: {self.evaluation_path}")
    
    def train_and_evaluate(self):
        """完整的训练和评估流程"""
        test_features, test_labels = self.train()
        return self.test(test_features, test_labels)

if __name__ == '__main__':
    # 设置CUDA环境变量（使用所有8张GPU）
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    os.environ['XGBOOST_CUDA_ALLOCATOR'] = 'cudaMallocAsync'
    
    # 初始化训练器
    trainer = XGBoostMultiGPUTrainer()
    
    # 执行完整训练和评估流程
    trainer.train_and_evaluate()