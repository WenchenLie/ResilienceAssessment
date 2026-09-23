import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.linalg import cholesky
import seaborn as sns
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


class EDPMatrixExpander:
    """
    工程需求参数矩阵扩充类
    用于基于IDA分析结果进行EDP矩阵的概率性扩充
    """
    
    def __init__(self, random_state=None):
        """初始化

        Args:
            random_state (int, optional): 随机种子
        """
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
        
        # 存储模型参数
        self.regression_params = {}  # 每个EDP的回归参数
        self.correlation_matrix = None  # EDP间的相关性矩阵
        self.edp_names = None  # EDP名称列表
        self.im_name = None  # IM名称
        
    def load_data(self, csv_file_path: Path | str, im_column: int=0):
        """从CSV文件加载数据

        Args:
            csv_file_path (Path | str): csv文件路径
            im_column (int, optional): IM列的索引，从0开始，默认为第一列

        Returns:
            _type_: _description_
        """
        self.data = pd.read_csv(csv_file_path)
        self.im_name = self.data.columns[im_column]
        self.edp_names = [col for col in self.data.columns if col != self.im_name]
        # print(f"数据加载成功:")
        # print(f"  - IM参数: {self.im_name}")
        # print(f"  - EDP参数 ({len(self.edp_names)}个): {self.edp_names}")
        # print(f"  - 总数据点数: {len(self.data)}")
        return self.data
    
    def fit_models(self):
        """拟合单变量概率模型和联合概率模型"""
        # print("\n开始拟合概率模型...")
        
        # 准备对数变换的数据
        ln_im: np.ndarray = np.log(self.data[self.im_name].values)
        
        self.regression_params = {}
        standardized_residuals = []
        
        # 对每个EDP进行单变量模型拟合
        for i, edp in enumerate(self.edp_names):
            ln_edp = np.log(self.data[edp].values)
            
            # 使用线性回归拟合 ln(EDP) ~ ln(IM)
            reg = LinearRegression()
            reg.fit(ln_im.reshape(-1, 1), ln_edp)
            
            # 计算预测值和残差
            ln_edp_pred = reg.predict(ln_im.reshape(-1, 1))
            residuals = ln_edp - ln_edp_pred
            sigma = np.std(residuals)
            
            # 存储模型参数
            self.regression_params[edp] = {
                'alpha': np.exp(reg.intercept_),
                'beta': reg.coef_[0],
                'sigma': sigma,
                'intercept': reg.intercept_
            }
            
            # 计算标准化残差
            z_scores = residuals / sigma
            standardized_residuals.append(z_scores)
            
            # print(f"  {edp}: ln(EDP) = {reg.intercept_:.4f} + {reg.coef_[0]:.4f} * ln(IM), σ = {sigma:.4f}")
        
        # 计算EDP间的相关性矩阵
        standardized_residuals_matrix = np.column_stack(standardized_residuals)
        self.correlation_matrix = np.corrcoef(standardized_residuals_matrix.T)
        
        # print(f"\nEDP间相关性矩阵:")
        corr_df = pd.DataFrame(self.correlation_matrix, 
                              index=self.edp_names, 
                              columns=self.edp_names)
        # print(corr_df.round(3))
        return self.regression_params, self.correlation_matrix
    
    def generate_samples(self,
                         target_im_values: np.ndarray,
                         samples_per_im: int=1000
        ) -> pd.DataFrame:
        """生成扩充样本

        Args:
            target_im_values (np.ndarray): 目标IM值数组
            samples_per_im (int, optional): 每个IM水平生成的样本数量

        Returns:
            pd.DataFrame: 扩充后的数据Dataframe
        """
        # print(f"\n开始生成扩充样本...")
        # print(f"  - 目标IM数量: {len(target_im_values)}")
        # print(f"  - 每个IM的样本数: {samples_per_im}")
        # print(f"  - 总扩充样本数: {len(target_im_values) * samples_per_im}")
        
        self.samples_per_im = samples_per_im
        # 准备结果存储
        expanded_samples = []
        
        # 对每个目标IM值生成样本
        for im_val in target_im_values:
            ln_im = np.log(im_val)
            
            # 生成相关的随机变量
            try:
                # 使用Cholesky分解生成相关的随机变量
                L = cholesky(self.correlation_matrix, lower=True)
                correlated_z = np.dot(L, np.random.randn(len(self.edp_names), samples_per_im))
            except np.linalg.LinAlgError:
                # 如果相关性矩阵不是正定的，使用近似方法
                print("警告: 相关性矩阵不是正定的，使用特征值分解近似")
                eigvals, eigvecs = np.linalg.eig(self.correlation_matrix)
                eigvals[eigvals < 0] = 0  # 将负特征值设为0
                L = np.dot(eigvecs, np.diag(np.sqrt(eigvals)))
                correlated_z = np.dot(L, np.random.randn(len(self.edp_names), samples_per_im))
            
            # 为每个EDP生成样本
            im_samples = np.full((samples_per_im, 1), im_val)
            
            for i, edp in enumerate(self.edp_names):
                params = self.regression_params[edp]
                # ln(EDP) = intercept + beta * ln(IM) + z * sigma
                ln_edp_samples = params['intercept'] + params['beta'] * ln_im + correlated_z[i, :] * params['sigma']
                edp_samples = np.exp(ln_edp_samples).reshape(-1, 1)
                im_samples = np.hstack([im_samples, edp_samples])
            
            expanded_samples.append(im_samples)
        
        # 合并所有样本
        all_samples = np.vstack(expanded_samples)
        self.expanded_data = pd.DataFrame(all_samples, columns=[self.im_name] + self.edp_names)

        return self.expanded_data
    
    def get_edp(self, idx_im: int, idx_iter: int):
        """从扩充的矩阵中获取第idx_im个IM的第idx_iter个edp(从0开始)

        Args:
            idx_im (int): 第idx_im个IM
            idx_iter (int): 第idx_iter个edp
        """
        return self.expanded_data.iloc[idx_im * self.samples_per_im + idx_iter, 1:]
    
    def validate_models(self, original_data: pd.DataFrame, expanded_data: pd.DataFrame, n_bins: int=10):
        """验证模型拟合效果

        Args:
            original_data (pd.DataFrame): 原始数据
            expanded_data (pd.DataFrame): 扩充后数据
            n_bins (int, optional): IM值分箱数量，默认为10
        """
        print("\n开始模型验证...")
        
        # 1. 分位数验证
        im_min, im_max = original_data[self.im_name].min(), original_data[self.im_name].max()
        im_bins = np.linspace(im_min, im_max, n_bins + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, edp in enumerate(self.edp_names[:4]):  # 只显示前4个EDP的验证
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # 计算每个IM区间的分位数
            original_quantiles = []
            expanded_quantiles = []
            im_centers = []
            
            for j in range(len(im_bins) - 1):
                im_low, im_high = im_bins[j], im_bins[j+1]
                im_center = (im_low + im_high) / 2
                
                # 原始数据分位数
                orig_mask = (original_data[self.im_name] >= im_low) & (original_data[self.im_name] < im_high)
                if orig_mask.sum() > 0:
                    original_quantiles.append(np.percentile(original_data[edp][orig_mask], [16, 50, 84]))
                else:
                    original_quantiles.append([np.nan, np.nan, np.nan])
                
                # 扩充数据分位数
                exp_mask = (expanded_data[self.im_name] >= im_low) & (expanded_data[self.im_name] < im_high)
                if exp_mask.sum() > 0:
                    expanded_quantiles.append(np.percentile(expanded_data[edp][exp_mask], [16, 50, 84]))
                else:
                    expanded_quantiles.append([np.nan, np.nan, np.nan])
                
                im_centers.append(im_center)
            
            original_quantiles = np.array(original_quantiles)
            expanded_quantiles = np.array(expanded_quantiles)
            
            # 绘制分位数比较
            ax.plot(im_centers, original_quantiles[:, 1], 'ro-', label='Median of original data', markersize=6)
            ax.fill_between(im_centers, original_quantiles[:, 0], original_quantiles[:, 2], 
                           alpha=0.3, color='red', label='16%-84% of original data')
            
            ax.plot(im_centers, expanded_quantiles[:, 1], 'bo-', label='Median of expanded data', markersize=4)
            ax.fill_between(im_centers, expanded_quantiles[:, 0], expanded_quantiles[:, 2], 
                           alpha=0.3, color='blue', label='16%-84% of expanded data')
            
            ax.set_xlabel(self.im_name)
            ax.set_ylabel(edp)
            ax.set_title(f'{edp} - Percentile Verification')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 2. 相关性验证
        if len(self.edp_names) >= 2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 原始数据相关性散点图
            ax1.scatter(original_data[self.edp_names[0]], original_data[self.edp_names[1]], 
                       alpha=0.6, s=30, color='red', label='Original data')
            ax1.set_xlabel(self.edp_names[0])
            ax1.set_ylabel(self.edp_names[1])
            ax1.set_title('Correlation of original data')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 扩充数据相关性散点图（抽样显示，避免点太多）
            sample_size = min(1000, len(expanded_data))
            sampled_data = expanded_data.sample(sample_size, random_state=self.random_state)
            ax2.scatter(sampled_data[self.edp_names[0]], sampled_data[self.edp_names[1]], 
                       alpha=0.3, s=10, color='blue', label='Expanded data')
            ax2.set_xlabel(self.edp_names[0])
            ax2.set_ylabel(self.edp_names[1])
            ax2.set_title('Correlation of expanded data')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        print("模型验证完成!")
    
    def plot_model_fit(self):
        """绘制模型拟合效果图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        ln_im = np.log(self.data[self.im_name].values)
        
        for i, edp in enumerate(self.edp_names[:4]):  # 只显示前4个EDP
            if i >= len(axes):
                break
                
            ax = axes[i]
            ln_edp = np.log(self.data[edp].values)
            params = self.regression_params[edp]
            
            # 绘制原始数据散点图
            ax.scatter(self.data[self.im_name], self.data[edp], alpha=0.6, s=30, label='Original data')
            
            # 绘制拟合的中位值曲线
            im_range = np.linspace(self.data[self.im_name].min(), self.data[self.im_name].max(), 100)
            ln_im_range = np.log(im_range)
            ln_edp_median = params['intercept'] + params['beta'] * ln_im_range
            edp_median = np.exp(ln_edp_median)
            ax.plot(im_range, edp_median, 'r-', linewidth=2, label='Median of fitted data')
            
            # 绘制±1σ范围
            ln_edp_plus_sigma = ln_edp_median + params['sigma']
            ln_edp_minus_sigma = ln_edp_median - params['sigma']
            edp_plus_sigma = np.exp(ln_edp_plus_sigma)
            edp_minus_sigma = np.exp(ln_edp_minus_sigma)
            ax.fill_between(im_range, edp_minus_sigma, edp_plus_sigma, alpha=0.3, 
                           color='red', label='±sigma')
            
            ax.set_xlabel(self.im_name)
            ax.set_ylabel(edp)
            ax.set_title(f'{edp} - Model fitting')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            ax.set_xscale('log')
        
        plt.tight_layout()
        plt.show()

def main():
    expander = EDPMatrixExpander(random_state=42)
    csv_file_path = "H:/results_RockingMRFwithVED2/IDA/MRF_4_frag/EDP_matrix.csv"
    original_data = expander.load_data(csv_file_path, im_column=0)

    # 步骤2: 拟合概率模型
    regression_params, corr_matrix = expander.fit_models()
    
    # 步骤3: 绘制模型拟合图
    expander.plot_model_fit()
    
    # 步骤4: 定义目标IM范围并生成扩充样本
    im_min, im_max = original_data[expander.im_name].min(), original_data[expander.im_name].max()
    target_im_values = np.linspace(im_min, im_max, 20)  # 在原始IM范围内生成20个目标点

    expander.generate_samples(target_im_values, samples_per_im=500)
    
    print(expander.get_edp(1, 4))
    
    expander.validate_models(original_data, expander.expanded_data)


if __name__ == "__main__":
    main()