import json
import numpy as np
from scipy import stats
import os
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from typing import List, Dict, Tuple
from sklearn.manifold import TSNE
import networkx as nx
import random
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import gaussian_kde

class Relationator:
    def __init__(self, config=None, config_path=None):
        """Initialize Relationator with either config dict or config file path
        
        Args:
            config: Configuration dictionary
            config_path: Path to configuration YAML file
        """
        if config_path is not None:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        if config is None:
            raise ValueError("Either config or config_path must be provided")
            
        self.config = config
        self.name = config.get("name", "relation")
        self.task = config.get("task", "t2i")
        self.relation_config = config.get("relation", {})
        self.models = self.relation_config.get("models", ["test"])
        self.res = self.relation_config.get("res", "demo")
        self.metric = self.relation_config.get("metric", "pearson_corr")
        self.plot = self.relation_config.get("plot", False)
        self.heatmap = self.relation_config.get("heatmap", False)
        self.tsne = self.relation_config.get("tsne", False)
        self.tradeoff = self.relation_config.get("tradeoff", False)
        self.quadrant_analysis = self.relation_config.get("quadrant_analysis", True)
        self.node_colors = self.relation_config.get("node_colors", {})
        self.insight_thresholds = self.relation_config.get("insight_thresholds", {
            "synergy_density": 0.4,     # 协同区密度阈值
            "bottleneck_density": 0.4,  # 瓶颈区密度阈值
            "dominance_ratio": 0.4,     # 主导比例阈值
            "tradeoff_corr": 0.6        # 零和博弈判定的相关系数阈值
        })
        self.dominance_threshold = self.relation_config.get("dominance_threshold", 0.3)
        self.bottleneck_threshold = self.relation_config.get("bottleneck_threshold", 0.5)
        
        # 从配置文件读取阈值，如果没有则使用默认值
        self.thresholds = self.relation_config.get("thresholds", {
            "synergy": 0.8,      # 协同区阈值
            "bottleneck": 0.5    # 瓶颈区阈值
        })
        
        self.base_dir = Path("data/relation") / "-".join(self.models) / self.res / self.metric
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.data = None

    def calculate_pearson_correlation(self, scores1, scores2):
        """Calculate Pearson correlation coefficient between two sets of scores."""
        try:
            if np.std(scores1) == 0 or np.std(scores2) == 0:
                return 0.0
            return stats.pearsonr(scores1, scores2)[0]
        except:
            return 0.0

    def calculate_spearman_correlation(self, scores1, scores2):
        """Calculate Spearman correlation coefficient between two sets of scores."""
        try:
            return stats.spearmanr(scores1, scores2)[0]
        except:
            return 0.0

    def calculate_pareto_frontier(self, scores1, scores2):
        """Calculate Pareto frontier score between two sets of scores."""
        try:
            pareto_points = []
            for i in range(len(scores1)):
                dominated = False
                for j in range(len(scores1)):
                    if i != j and scores1[j] >= scores1[i] and scores2[j] >= scores2[i]:
                        dominated = True
                        break
                if not dominated:
                    pareto_points.append((scores1[i], scores2[i]))
            
            if len(pareto_points) < 2:
                return 0.0
                
            pareto_points.sort()
            x, y = zip(*pareto_points)
            area = np.trapz(y, x)
            max_area = np.max(scores1) * np.max(scores2)
            return area / max_area if max_area > 0 else 0.0
        except:
            return 0.0

    def get_correlation_method(self):
        """Get the correlation calculation method based on metric config."""
        methods = {
            'pearson_corr': self.calculate_pearson_correlation,
            'spearman_corr': self.calculate_spearman_correlation,
            'paretoF': self.calculate_pareto_frontier
        }
        return methods[self.metric]

    def plot_scatter(self, data: Dict, metrics: List[str] = None, save_dir: str = None) -> Figure:
        """Generate enhanced scatter plots with quadrant analysis."""
        dimension_pairs = list(data.keys())
        n_pairs = len(dimension_pairs)
        n_cols = min(2, n_pairs)
        n_rows = (n_pairs + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(8*n_cols, 6*n_rows))
        
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        
        for idx, dim_pair in enumerate(dimension_pairs):
            ax = fig.add_subplot(n_rows, n_cols, idx + 1)
            metric1, metric2 = dim_pair.split('_')
            
            scores1 = np.array([v[0] for v in data[dim_pair].values()])
            scores2 = np.array([v[1] for v in data[dim_pair].values()])
            
            # 使用配置的阈值，synergy阈值x和y相同
            synergy_thresh = self.thresholds["synergy"]      # 0.8
            bottleneck_thresh = self.thresholds["bottleneck"] # 0.5
            
            # Plotting
            sns.scatterplot(x=scores1, y=scores2, ax=ax)
            try:
                sns.regplot(x=scores1, y=scores2, scatter=False, ax=ax, 
                          color='red', line_kws={'linestyle': '--'})
                correlation = self.get_correlation_method()(scores1, scores2)
                ax.text(0.05, 0.95, f'{self.metric}: {correlation:.3f}', 
                       transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
            except:
                pass
            
            # Quadrant visualization
            ax.axhline(synergy_thresh, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(synergy_thresh, color='gray', linestyle='--', alpha=0.5)
            
            # 1. Synergy zone (右上角矩形，得分都大于0.8)
            ax.fill_between([synergy_thresh, 1], synergy_thresh, 1, color='green', alpha=0.1)
            
            # 2. Bottleneck zone (左下角矩形，得分都小于0.5)
            ax.fill_between([0, bottleneck_thresh], 0, bottleneck_thresh, color='red', alpha=0.1)
            
            # 3. Tradeoff zones (三个区域)
            # 左上角区域：[0,bottleneck_thresh] × [bottleneck_thresh, 1]
            ax.fill_between([0, bottleneck_thresh], bottleneck_thresh, 1, color='yellow', alpha=0.1)
            
            # 右下角区域：[synergy_thresh, 1] × [0, synergy_thresh]
            ax.fill_between([synergy_thresh, 1], 0, synergy_thresh, color='yellow', alpha=0.1)
            
            # 中间区域：[bottleneck_thresh, synergy_thresh] × [0, 1]
            ax.fill_between([bottleneck_thresh, synergy_thresh], 0, 1, color='yellow', alpha=0.1)
            
            # Legend
            legend_elements = [
                Patch(facecolor='green', alpha=0.1, label='Synergy Zone'),
                Patch(facecolor='yellow', alpha=0.1, label='Tradeoff Zone'),
                Patch(facecolor='red', alpha=0.1, label='Bottleneck Zone')
            ]
            ax.legend(handles=legend_elements)
            
            ax.set_xlabel(metric1)
            ax.set_ylabel(metric2)
            ax.set_title(f'{dim_pair}')
            ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        if save_dir:
            models_str = "-".join(self.models)
            plot_name = f'{models_str}-{self.res}-{self.metric}-scatter.png'
            fig.savefig(save_path / plot_name, dpi=300, bbox_inches='tight')
        return fig

    def calculate_quadrant_metrics(self, scores1, scores2):
        """Calculate quadrant analysis metrics."""
        metrics = {}
        
        # 使用配置的阈值，synergy阈值x和y相同
        synergy_thresh = self.thresholds["synergy"]      # 0.8
        bottleneck_thresh = self.thresholds["bottleneck"] # 0.5
        
        # Region masks - 三个互不相交的区域
        # 1. Synergy: 右上角的矩形区域 (>0.8)
        synergy_mask = (scores1 > synergy_thresh) & (scores2 > synergy_thresh)
        
        # 2. Bottleneck: 左下角的区域 (<0.5)
        bottleneck_mask = (scores1 < bottleneck_thresh) & (scores2 < bottleneck_thresh)
        
        # 3. Tradeoff: 所有其他区域
        tradeoff_mask = ~(synergy_mask | bottleneck_mask)
        
        # Synergy metrics
        metrics['synergy_density'] = np.mean(synergy_mask)
        metrics['synergy_gain'] = (np.mean(scores1[synergy_mask]) + 
                                  np.mean(scores2[synergy_mask])) / np.sqrt(2)
        
        # Bottleneck metrics
        metrics['bottleneck_density'] = np.mean(bottleneck_mask)
        bottleneck_scores = np.stack([scores1[bottleneck_mask], scores2[bottleneck_mask]])
        metrics['defect_coupling'] = 1 - (np.max(bottleneck_scores, axis=0) - 
                                     np.min(bottleneck_scores, axis=0)).mean()
        
        # Tradeoff metrics
        tradeoff_corr = self.calculate_pearson_correlation(scores1[tradeoff_mask], scores2[tradeoff_mask])
        metrics['tradeoff_strength'] = abs(tradeoff_corr) * (
            np.var(scores1[tradeoff_mask]) + np.var(scores2[tradeoff_mask]))
        
        return metrics

    def determine_relationship_code(self, dim_pair):
        """按照优先级顺序判断关系模式"""
        scores1 = np.array([v[0] for v in self.data[dim_pair].values()])
        scores2 = np.array([v[1] for v in self.data[dim_pair].values()])
        
        # 1. 计算各区域密度
        synergy_mask = (scores1 > self.thresholds["synergy"]) & \
                    (scores2 > self.thresholds["synergy"])
        bottleneck_mask = (scores1 < self.thresholds["bottleneck"]) & \
                        (scores2 < self.thresholds["bottleneck"])
        tradeoff_mask = ~(synergy_mask | bottleneck_mask)
        
        synergy_density = np.mean(synergy_mask)
        bottleneck_density = np.mean(bottleneck_mask)
        
        # 2. 按优先级判断模式
        # 优先级1：正向协同
        if synergy_density >= self.insight_thresholds["synergy_density"]:
            return 0  # 协同
        
        # 优先级2：系统瓶颈
        if bottleneck_density >= self.insight_thresholds["bottleneck_density"]:
            return 3  # 瓶颈
        
        # 优先级3：维度主导
        if np.sum(tradeoff_mask) >= 10:  # 确保有足够的点进行分析
            # 只分析博弈区域的点
            tradeoff_scores1 = scores1[tradeoff_mask]
            tradeoff_scores2 = scores2[tradeoff_mask]
            
            # 拟合回归线
            reg = LinearRegression().fit(tradeoff_scores1.reshape(-1,1), tradeoff_scores2)
            pred_y = reg.predict(tradeoff_scores1.reshape(-1,1))
            
            # 计算点在回归线上下的数量
            above_count = np.sum(tradeoff_scores2 > pred_y)
            below_count = np.sum(tradeoff_scores2 <= pred_y)
            
            # 计算大数/小数的比例，判断主导关系
            ratio = max(above_count, below_count) / min(above_count, below_count)
            dominance_ratio = 1 / self.insight_thresholds["dominance_ratio"]  # 例如 1/0.5 = 2
            if ratio > dominance_ratio:  # 使用配置文件中的阈值
                if above_count > below_count:
                    return -2  # Y主导X（上方点数多）
                else:
                    return 2   # X主导Y（下方点数多）
        
        # 优先级4：零和博弈
        # 只考虑相关系数
        corr = self.get_correlation_method()(scores1, scores2)
        if corr < self.insight_thresholds["tradeoff_corr"]:  # 例如 -0.6
            return 1  # 零和博弈
        
        # 如果没有匹配任何模式
        return -1  # 未定义

    def generate_insights(self, individual_metrics, correlation_matrix, quadrant_analysis, relationship_codes):
        """Generate insights from analysis results."""
        insights = []
        
        # 遍历所有可能的指标对
        for i, metric1 in enumerate(individual_metrics):
            for j, metric2 in enumerate(individual_metrics):
                if i >= j:  # 跳过对角线和重复的对
                    continue
            
                pair = f"{metric1}_{metric2}"
                if pair not in relationship_codes:
                    continue
            
                code = relationship_codes[pair]
            
                # 根据不同的关系代码生成洞察
                if code == 3:  # 瓶颈关系
                    insights.append({
                        'type': 'bottleneck',
                        'metrics': (metric1, metric2),
                        'description': f"发现瓶颈关系：{metric1} 和 {metric2} 在低分区域存在显著耦合",
                        'suggestion': "建议关注这两个指标的共同改进"
                    })
                elif code == 2 or code == -2:  # 主导关系
                    dominant = metric1 if code == 2 else metric2
                    dependent = metric2 if code == 2 else metric1
                    insights.append({
                        'type': 'dominance',
                        'metrics': (metric1, metric2),
                        'description': f"发现主导关系：{dominant} 对 {dependent} 具有显著影响力",
                        'suggestion': f"建议优先优化 {dominant} 以带动 {dependent} 的提升"
                    })
                elif code == 1:  # 权衡关系
                    insights.append({
                        'type': 'tradeoff',
                        'metrics': (metric1, metric2),
                        'description': f"发现权衡关系：{metric1} 和 {metric2} 存在负相关",
                        'suggestion': "需要在这两个指标之间寻找平衡点"
                    })
                elif code == 0:  # 协同关系
                    insights.append({
                        'type': 'synergy',
                        'metrics': (metric1, metric2),
                        'description': f"发现协同关系：{metric1} 和 {metric2} 可以共同提升",
                        'suggestion': "可以同时优化这两个指标"
                    })
                elif code == -1:  # 无明显关系
                    insights.append({
                        'type': 'undefined',
                        'metrics': (metric1, metric2),
                        'description': f"{metric1} 和 {metric2} 之间没有明显的关系模式",
                        'suggestion': "需要进一步分析这两个指标之间的关系"
                    })
        
        return insights

    def plot_heatmap(self, code_matrix: np.ndarray, individual_metrics: List[str], save_dir: Path = None) -> Figure:
        """Generate heatmap visualization for relationship codes."""
        # 创建颜色映射 - 高饱和度的热力图风格
        colors = {
            2: '#1a53ff',   # 深蓝色 - 主导
            -1: '#f0f0f0',  # 浅灰色 - 未定义关系
            0: '#00cc44',   # 翠绿色 - 协同
            1: '#00a0dc',   # 亮蓝色 - 权衡
            -2: '#ff3333',  # 鲜红色 - 被主导
            3: '#9933ff'    # 亮紫色 - 瓶颈
        }
        
        # 创建自定义colormap，添加白色用于左下角和对角线
        cmap = plt.cm.colors.ListedColormap(['white'] + [colors[i] for i in [2, -1, 0, 1, -2, 3]])
        bounds = [-3] + [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        
        # 创建图形
        plt.figure(figsize=(12, 10))
        
        # 创建一个新的矩阵，左下角和对角线设为白色
        display_matrix = code_matrix.copy()
        for i in range(len(individual_metrics)):
            for j in range(i + 1):  # 包括对角线
                display_matrix[i,j] = -3
        
        # 绘制热力图
        im = plt.imshow(display_matrix, cmap=cmap, norm=norm)
        
        # 添加关系类型标签（只包含右上角的关系）
        labels = {
            2: 'Dominated',
            -1: 'Undefined',
            0: 'Synergy',
            1: 'Trade-off',
            -2: 'Dominant',
            3: 'Bottleneck'
        }
        
        # 创建colorbar（只显示右上角的关系颜色）
        cbar = plt.colorbar(im, ticks=[2, -1, 0, 1, -2, 3])
        cbar.set_ticklabels([labels[i] for i in [2, -1, 0, 1, -2, 3]])
        
        # 设置坐标轴
        plt.xticks(range(len(individual_metrics)), individual_metrics, rotation=45, ha='right')
        plt.yticks(range(len(individual_metrics)), individual_metrics)
        
        # 读取CSV文件获取左下角的数值
        csv_path = self.base_dir / "relationship_codes.csv"
        csv_data = []
        with open(csv_path, 'r') as f:
            next(f)  # 跳过标题行
            for line in f:
                csv_data.append(line.strip().split(',')[1:])  # 跳过第一列
        
        # 添加数值标签
        for i in range(len(individual_metrics)):
            for j in range(len(individual_metrics)):
                if i == j:  # 对角线
                    text = csv_data[i][j]
                    color = 'black'
                elif j > i:  # 右上角：关系代码
                    val = code_matrix[i, j]
                    text = str(int(val))
                    color = 'white' if val in [2, -2, 3, 1] else 'black'
                else:  # 左下角：CSV中的数值
                    text = csv_data[i][j]
                    if text != "NULL":  # 只有在不是NULL时才处理格式
                        text = text.strip('[]').replace('-', '\n')
                    color = 'black'
                
                plt.text(j, i, text,
                        ha="center", va="center", color=color,
                        fontweight='bold', fontsize=12)
        
        plt.title("Relationship Type Heatmap", pad=20, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # 保存图形
        if save_dir:
            models_str = "-".join(self.models)
            plot_name = f'{models_str}-{self.res}-{self.metric}-heatmap.png'
            plt.savefig(save_dir / plot_name, dpi=300, bbox_inches='tight')
        
        return plt.gcf()

    def build_relation(self):
        """Main analysis pipeline."""
        # Load data
        result_path = Path(f"data/result/{self.res}.json")
        with open(result_path, 'r') as f:
            self.data = json.load(f)
        
        metrics = list(self.data.keys())
        n_metrics = len(metrics)
        
        # Calculate correlations
        correlation_matrix = np.zeros((n_metrics, n_metrics))
        method = self.get_correlation_method()
        
        for i, dim_pair in enumerate(metrics):
            scores1 = np.array([v[0] for v in self.data[dim_pair].values()])
            scores2 = np.array([v[1] for v in self.data[dim_pair].values()])
            correlation_matrix[i][i] = method(scores1, scores2)
        
        # Generate plots
        if self.plot:
            self.plot_scatter(self.data, save_dir=self.base_dir)
        
        # Save results
        output = {
            "metrics": metrics,
            "correlation_matrix": correlation_matrix.tolist(),
            "quadrant_analysis": {},
            "relationship_codes": {}
        }
        
        # Relationship coding
        individual_metrics = sorted(set(m for pair in metrics for m in pair.split('_')))
        n = len(individual_metrics)
        code_matrix = np.zeros((n, n), dtype=int)
        relationship_codes = {}  # 存储所有关系代码
        
        # 构建关系代码矩阵 - 只处理上三角矩阵
        for i in range(n):
            for j in range(i + 1, n):  # 只处理上三角部分
                metric1 = individual_metrics[i]
                metric2 = individual_metrics[j]
                pair = f"{metric1}_{metric2}"
                reverse_pair = f"{metric2}_{metric1}"
                
                # 确定关系代码
                code = -1  # 默认为未定义关系
                if pair in metrics:
                    code = self.determine_relationship_code(pair)
                elif reverse_pair in metrics:
                    code = self.determine_relationship_code(reverse_pair)
                    # 如果是从reverse_pair得到的code，需要调整主导关系的方向
                    if code == 2:
                        code = -2
                    elif code == -2:
                        code = 2
                
                # 设置上三角位置
                code_matrix[i][j] = code
                relationship_codes[pair] = code
                
                # 设置下三角位置（对称）
                if code == 2:
                    code_matrix[j][i] = -2
                    relationship_codes[reverse_pair] = -2
                elif code == -2:
                    code_matrix[j][i] = 2
                    relationship_codes[reverse_pair] = 2
                else:  # 对于非主导关系，直接复制相同的值
                    code_matrix[j][i] = code
                    relationship_codes[reverse_pair] = code
        
        # 保存关系代码到输出
        output["relationship_codes"] = relationship_codes
        
        # 计算每个维度对在特定配对中的平均分
        pair_means = {}  # 存储每个维度对的平均分
        for dim_pair in metrics:
            dim1, dim2 = dim_pair.split('_')
            scores1 = np.array([v[0] for v in self.data[dim_pair].values()])
            scores2 = np.array([v[1] for v in self.data[dim_pair].values()])
            
            # 计算这个配对中两个维度的平均分
            pair_means[dim_pair] = {
                dim1: np.mean(scores1),
                dim2: np.mean(scores2)
            }
        
        # 计算每个维度在所有配对中的所有数据点的总平均分
        dimension_all_scores = {}  # 存储每个维度的所有得分
        for dim_pair in metrics:
            dim1, dim2 = dim_pair.split('_')
            scores1 = [v[0] for v in self.data[dim_pair].values()]
            scores2 = [v[1] for v in self.data[dim_pair].values()]
            
            if dim1 not in dimension_all_scores:
                dimension_all_scores[dim1] = []
            if dim2 not in dimension_all_scores:
                dimension_all_scores[dim2] = []
            
            dimension_all_scores[dim1].extend(scores1)
            dimension_all_scores[dim2].extend(scores2)
        
        # 计算总平均分
        dimension_means = {
            dim: np.mean(scores) 
            for dim, scores in dimension_all_scores.items()
        }
        
        # Save relationship codes to CSV with means in lower triangle
        code_csv_path = self.base_dir / "relationship_codes.csv"
        with open(code_csv_path, 'w') as f:
            f.write("," + ",".join(individual_metrics) + "\n")
            for i, metric1 in enumerate(individual_metrics):
                f.write(f"{metric1},")
                for j, metric2 in enumerate(individual_metrics):
                    if i == j:  # 对角线：该维度的总平均分
                        value = f"{dimension_means[metric1]:.3f}"
                    elif j > i:  # 右上角：关系代码
                        value = str(code_matrix[i][j])
                    else:  # 左下角：该配对中的平均分
                        pair = f"{metric2}_{metric1}"  # 注意顺序
                        reverse_pair = f"{metric1}_{metric2}"
                        
                        # 检查是否存在这个维度对
                        if pair in pair_means:
                            value = f"[{pair_means[pair][metric1]:.3f}-{pair_means[pair][metric2]:.3f}]"
                        elif reverse_pair in pair_means:
                            value = f"[{pair_means[reverse_pair][metric1]:.3f}-{pair_means[reverse_pair][metric2]:.3f}]"
                        else:
                            value = "NULL"  # 如果找不到维度对，显示NULL
                    
                    f.write(value + ("," if j < len(individual_metrics)-1 else "\n"))
        
        # 在保存CSV之后添加热力图绘制
        if self.heatmap:
            self.plot_heatmap(code_matrix, individual_metrics, save_dir=self.base_dir)
        
        # Generate insights using individual_metrics
        insights = self.generate_insights(individual_metrics, correlation_matrix, 
                                        output["quadrant_analysis"],
                                        relationship_codes)
        output["insights"] = insights
        
        # Save results
        json_path = self.base_dir / "result.json"
        with open(json_path, 'w') as f:
            json.dump(output, f, indent=4)
        
        return output

    def __call__(self):
        return self.build_relation()

if __name__ == "__main__":
    # Example usage
    analyzer = Relationator(config_path="config.yaml")
    result = analyzer()