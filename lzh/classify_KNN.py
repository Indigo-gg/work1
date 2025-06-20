import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from joblib import dump, load
from datetime import datetime

class KNNClassifierTrainer:
    def __init__(self, config_path='./fitted_coefficients/dataset_20250620_021115/metadata.json', 
                 data_path='./fitted_coefficients/dataset_20250620_021115/coefficients.npz',
                 output_root_dir='./fitted_coefficients/'):
        """初始化KNN分类器训练器"""
        self.config_path = config_path
        self.data_path = data_path
        self.output_root_dir = output_root_dir
        self.metadata = self._load_metadata()
        self.class_names = self._get_class_names()
        
        # 创建带有时间标签的结果目录
        self.time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(self.output_root_dir, f'Results_{self.time_tag}')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 定义不包含时间标签的固定文件名
        self.model_save_path = os.path.join(self.results_dir, 'knn_classifier.joblib')
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
            data['train_coeffs'], 
            data['train_labels'], 
            data['test_coeffs'], 
            data['test_labels']
        )
    
    def preprocess_data(self, train_features, test_features):
        """数据预处理：标准化特征"""
        self.scaler = StandardScaler()
        train_features_scaled = self.scaler.fit_transform(train_features)
        test_features_scaled = self.scaler.transform(test_features)
        return train_features_scaled, test_features_scaled
    
    def optimize_hyperparameters(self, train_features, train_labels):
        """优化KNN超参数"""
        print("开始优化KNN超参数...")
        knn = KNeighborsClassifier()
        
        param_grid = {
            'n_neighbors': range(1, 31),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
        
        grid_search = GridSearchCV(
            knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        grid_search.fit(train_features, train_labels)
        
        print(f"最佳超参数: {grid_search.best_params_}")
        print(f"交叉验证最佳准确率: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_
    
    def train(self):
        """训练模型"""
        print("开始训练流程...")
        print(f"本次运行结果将保存至: {self.results_dir}")
        
        # 1. 加载数据
        train_coeffs, train_labels, test_coeffs, test_labels = self.load_data()
        print(f"加载数据完成: 训练集大小={len(train_labels)}, 测试集大小={len(test_labels)}")
        
        # 2. 数据预处理
        train_scaled, test_scaled = self.preprocess_data(train_coeffs, test_coeffs)
        print("数据预处理完成")
        
        # 3. 超参数优化
        self.model = self.optimize_hyperparameters(train_scaled, train_labels)
        
        # 4. 训练最终模型
        print("训练最终模型...")
        self.model.fit(train_scaled, train_labels)
        
        # 5. 保存模型
        self.save_model()
        
        # 6. 返回训练数据用于后续评估
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
        test_pred = self.model.predict(test_features)
        test_accuracy = accuracy_score(test_labels, test_pred)
        
        print(f"测试集准确率: {test_accuracy:.4f}")
        
        report = classification_report(
            test_labels, test_pred, target_names=self.class_names
        )
        print("分类报告:\n", report)
        
        cm = confusion_matrix(test_labels, test_pred)
        self._plot_confusion_matrix(cm)
        
        return {
            'test_accuracy': test_accuracy,
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
        """保存训练好的KNN模型"""
        if self.model is None:
            raise ValueError("没有可保存的模型，请先训练模型")
            
        dump({'model': self.model, 'scaler': self.scaler}, self.model_save_path)
        print(f"KNN模型已保存至: {self.model_save_path}")
    
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
        """完整的训练和评估流程（保持向后兼容）"""
        test_features, test_labels = self.train()
        return self.test(test_features, test_labels)

if __name__ == '__main__':
    # 示例用法
    trainer = KNNClassifierTrainer()
    
    # 方式1：完整流程
    # trainer.train_and_evaluate()
    
    # 方式2：分离流程
    test_features, test_labels = trainer.train()
    results = trainer.test(test_features, test_labels)
    
    # 方式3：加载已有模型测试
    # trainer.load_saved_model()
    # results = trainer.test(test_features, test_labels)