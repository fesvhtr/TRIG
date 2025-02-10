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

class Relationator:
    def __init__(self, config=None, config_path=None):
        """Initialize Relationator with either config dict or config file path
        
        Args:
            config: Configuration dictionary
            config_path: Path to configuration YAML file
        """
        if config_path is not None:
            # Load config from YAML file
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
        self.plot = self.relation_config.get("plot", False)  # Add plot configuration
        self.heatmap = self.relation_config.get("heatmap", False)  # Add heatmap configuration
        
    def calculate_pearson_correlation(self, scores1, scores2):
        """Calculate Pearson correlation coefficient between two sets of scores.
        
        Args:
            scores1: First array of scores
            scores2: Second array of scores
            
        Returns:
            float: Pearson correlation coefficient
        """
        try:
            # Check if either array is constant
            if np.std(scores1) == 0 or np.std(scores2) == 0:
                print(f"Warning: Constant array detected, correlation coefficient undefined")
                return 0.0
            correlation, _ = stats.pearsonr(scores1, scores2)
            return correlation
        except Exception as e:
            print(f"Error calculating Pearson correlation: {e}")
            return 0.0
            
    def calculate_spearman_correlation(self, scores1, scores2):
        """Calculate Spearman correlation coefficient between two sets of scores.
        
        Args:
            scores1: First array of scores
            scores2: Second array of scores
            
        Returns:
            float: Spearman correlation coefficient
        """
        try:
            correlation, _ = stats.spearmanr(scores1, scores2)
            return correlation
        except Exception as e:
            print(f"Error calculating Spearman correlation: {e}")
            return 0.0
            
    def calculate_pareto_frontier(self, scores1, scores2):
        """Calculate Pareto frontier score between two sets of scores.
        
        Args:
            scores1: First array of scores
            scores2: Second array of scores
            
        Returns:
            float: Pareto frontier score
        """
        try:
            # Convert to numpy arrays
            scores1_arr = np.array(scores1)
            scores2_arr = np.array(scores2)
            
            # Find Pareto optimal points
            pareto_points = []
            for i in range(len(scores1)):
                dominated = False
                for j in range(len(scores1)):
                    if i != j:
                        if (scores1[j] >= scores1[i] and scores2[j] >= scores2[i] and 
                            (scores1[j] > scores1[i] or scores2[j] > scores2[i])):
                            dominated = True
                            break
                if not dominated:
                    pareto_points.append((scores1[i], scores2[i]))
            
            # Calculate area under Pareto frontier as a score
            if len(pareto_points) < 2:
                return 0.0
            
            # Sort points by x-coordinate
            pareto_points.sort()
            x_coords, y_coords = zip(*pareto_points)
            
            # Calculate area under curve (normalized)
            area = np.trapz(y_coords, x_coords)
            max_area = max(scores1_arr) * max(scores2_arr)
            pareto_score = area / max_area if max_area > 0 else 0.0
            
            return pareto_score
        
        except Exception as e:
            print(f"Error calculating Pareto frontier score: {e}")
            return 0.0

    def get_correlation_method(self):
        """Get the correlation calculation method based on metric config.
        
        Returns:
            function: Correlation calculation function
        """
        correlation_methods = {
            'pearson_corr': self.calculate_pearson_correlation,
            'spearman_corr': self.calculate_spearman_correlation,
            'paretoF': self.calculate_pareto_frontier
        }
        
        method = correlation_methods.get(self.metric)
        if method is None:
            raise ValueError(f"Unsupported correlation method: {self.metric}")
        return method
        
    def plot_scatter(self, data: Dict, metrics: List[str] = None, save_dir: str = None) -> Figure:
        """Generate scatter plots for each dimension pair, showing relationship between its two metrics.
        
        Args:
            data: Dictionary containing metric scores
            metrics: List of metrics to plot. If None, use all metrics
            save_dir: Directory to save plots. If None, don't save
            
        Returns:
            matplotlib figure
        """
        # Get dimension pairs
        dimension_pairs = list(data.keys())
        n_pairs = len(dimension_pairs)
        
        # Calculate grid size
        n_cols = min(2, n_pairs)  # Maximum 2 columns
        n_rows = (n_pairs + n_cols - 1) // n_cols  # Ceiling division
        
        # Create figure with subplots
        fig = plt.figure(figsize=(8*n_cols, 6*n_rows))
        
        # Create save directory if needed
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        
        # Generate plots for each dimension pair
        for idx, dim_pair in enumerate(dimension_pairs):
            # Create subplot
            ax = fig.add_subplot(n_rows, n_cols, idx + 1)
            
            # Split dimension pair name to get individual metrics
            # Example: "IQ-O_TA-R" -> ["IQ-O", "TA-R"]
            metrics = dim_pair.split("_")
            if len(metrics) != 2:
                print(f"Warning: Unexpected dimension pair format: {dim_pair}")
                continue
            
            metric1, metric2 = metrics
            
            # Collect scores for both metrics
            scores1 = []  # First metric scores (e.g., IQ-O)
            scores2 = []  # Second metric scores (e.g., TA-R)
            
            # Get all samples
            for key in data[dim_pair].keys():
                # Each sample has two scores
                score_pair = data[dim_pair][key]
                scores1.append(score_pair[0])  # First score for metric1
                scores2.append(score_pair[1])  # Second score for metric2
            
            # Convert to numpy arrays
            scores1 = np.array(scores1)
            scores2 = np.array(scores2)
            
            # Check if we have non-constant data
            if np.std(scores1) > 0 or np.std(scores2) > 0:
                # Create scatter plot
                sns.scatterplot(x=scores1, y=scores2, ax=ax)
                
                # Add trend line
                try:
                    sns.regplot(x=scores1, y=scores2, scatter=False, ax=ax, 
                              color='red', line_kws={'linestyle': '--'})
                    
                    # Calculate and display correlation
                    correlation_method = self.get_correlation_method()
                    correlation = correlation_method(scores1, scores2)
                    ax.text(0.05, 0.95, f'{self.metric}: {correlation:.3f}', 
                           transform=ax.transAxes, 
                           bbox=dict(facecolor='white', alpha=0.8))
                except Exception as e:
                    print(f"Warning: Could not add trend line for {dim_pair}: {e}")
            else:
                ax.scatter(scores1, scores2)
                print(f"Warning: Constant scores for {dim_pair}")
            
            # Customize plot
            ax.set_xlabel(metric1)  # e.g., "IQ-O"
            ax.set_ylabel(metric2)  # e.g., "TA-R"
            ax.set_title(f'{dim_pair}')
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            if self.metric == 'paretoF':
                # Add Pareto frontier line
                pareto_points = []
                for i in range(len(scores1)):
                    dominated = False
                    for j in range(len(scores1)):
                        if i != j:
                            if (scores1[j] >= scores1[i] and scores2[j] >= scores2[i] and 
                                (scores1[j] > scores1[i] or scores2[j] > scores2[i])):
                                dominated = True
                                break
                        if not dominated:
                            pareto_points.append((scores1[i], scores2[i]))
                
                if pareto_points:
                    # Sort points and plot frontier
                    pareto_points.sort()
                    pareto_x, pareto_y = zip(*pareto_points)
                    ax.plot(pareto_x, pareto_y, 'r--', label='Pareto Frontier')
                    ax.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot if directory is specified
        if save_dir:
            models_str = "-".join(self.models)
            plot_name = f'{models_str}-{self.res}-{self.metric}-plot.png'
            fig.savefig(save_path / plot_name, dpi=300, bbox_inches='tight')
        
        return fig

    def plot_heatmap(self, csv_path: Path, save_dir: Path) -> Figure:
        """Generate a heatmap visualization of the correlation matrix.
        
        Args:
            csv_path: Path to the CSV file containing correlation matrix
            save_dir: Directory to save the heatmap
            
        Returns:
            matplotlib figure
        """
        # Read CSV data
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        
        # Parse header and metrics
        metrics = lines[0].strip().split(',')[1:]  # Skip first empty cell
        
        # Parse correlation matrix
        matrix = []
        for line in lines[1:]:
            # Convert empty strings to 0 and strings to floats
            row = [float(x) if x.strip() else 0 for x in line.strip().split(',')[1:]]
            matrix.append(row)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(
            matrix,
            xticklabels=metrics,
            yticklabels=metrics,
            cmap='YlGnBu',  # Changed from 'RdBu_r' to 'YlGnBu'
            center=None,    # Remove center as we're using a sequential colormap
            vmin=-1,
            vmax=1,
            annot=True,
            fmt='.2f',
            square=True,
            cbar_kws={'label': f'{self.metric} correlation'}
        )
        
        # Customize plot
        plt.title('Correlation Matrix Heatmap')
        plt.tight_layout()
        
        # Save plot
        models_str = "-".join(self.models)
        plot_name = f'{models_str}-{self.res}-{self.metric}-heatmap.png'
        plt.savefig(save_dir / plot_name, dpi=300, bbox_inches='tight')
        
        return plt.gcf()

    def build_relation(self):
        """Build relation between different metrics"""
        # 1. Load result data
        result_path = Path(f"data/result/{self.res}.json")
        if not result_path.exists():
            raise FileNotFoundError(f"Result file not found: {result_path}")
            
        with open(result_path, 'r') as f:
            data = json.load(f)
            
        # 2. Extract metrics and calculate correlation
        metrics = list(data.keys())  # Get all metric names like "IQ-O_TA-R", "IQ-O_TA-S"
        n_metrics = len(metrics)
        
        # Create correlation matrix
        correlation_matrix = np.zeros((n_metrics, n_metrics))
        
        # Get correlation calculation method
        correlation_method = self.get_correlation_method()
        
        # Calculate correlation for each metric pair
        for i in range(n_metrics):
            metric_pair = metrics[i]
            dim1, dim2 = metric_pair.split('_')  # Split into two dimensions
            
            # Extract scores for both dimensions
            scores1 = []  # First dimension scores
            scores2 = []  # Second dimension scores
            
            # Get all samples for this metric pair
            for key in data[metric_pair].keys():
                score_pair = data[metric_pair][key]
                scores1.append(score_pair[0])  # First score for dim1
                scores2.append(score_pair[1])  # Second score for dim2
            
            # Calculate correlation if we have valid data
            if len(scores1) > 0 and len(scores2) > 0:
                scores1_arr = np.array(scores1, dtype=float)
                scores2_arr = np.array(scores2, dtype=float)
                
                if np.all(np.isfinite(scores1_arr)) and np.all(np.isfinite(scores2_arr)):
                    correlation = correlation_method(scores1_arr, scores2_arr)
                    correlation_matrix[i][i] = correlation
                else:
                    print(f"Warning: Invalid values found for {metric_pair}")
        
        # Generate scatter plots only if plot is True
        if self.plot:
            scatter_save_dir = Path("data/relation/plots")
            self.plot_scatter(data, metrics=metrics, save_dir=scatter_save_dir)
        
        # 3. Save results
        output = {
            "metrics": metrics,
            "correlation_matrix": correlation_matrix.tolist()
        }
        
        # Create subdirectories for different file types
        json_dir = Path("data/relation/json")
        csv_dir = Path("data/relation/csv")
        json_dir.mkdir(parents=True, exist_ok=True)
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the JSON output
        models_str = "-".join(self.models)
        json_filename = f"{models_str}-{self.res}-{self.metric}.json"
        json_path = json_dir / json_filename
        
        with open(json_path, 'w') as f:
            json.dump(output, f, indent=4)
        
        # Extract unique individual metrics from dimension pairs
        individual_metrics = set()
        for dim_pair in metrics:
            metric1, metric2 = dim_pair.split('_')
            individual_metrics.add(metric1)
            individual_metrics.add(metric2)
        individual_metrics = sorted(list(individual_metrics))

        # Generate CSV filename and save
        csv_filename = f"{models_str}-{self.res}-{self.metric}.csv"
        csv_path = csv_dir / csv_filename

        # Create CSV content with upper triangular matrix
        with open(csv_path, 'w') as f:
            # Write header row
            f.write("," + ",".join(individual_metrics) + "\n")
            
            # Write each row
            for i, metric1 in enumerate(individual_metrics):
                row = [metric1]  # First column is metric name
                for j, metric2 in enumerate(individual_metrics):
                    if j < i:  # Lower triangle
                        # Look for the opposite pair and negate its correlation
                        pair = f"{metric2}_{metric1}"  # Reverse the pair order
                        if pair in metrics:
                            idx = metrics.index(pair)
                            correlation = -correlation_matrix[idx][idx]  # Negate the correlation
                            row.append(f"{correlation:.3f}")
                        else:
                            row.append("")  # No correlation found
                    elif metric1 == metric2:  # Diagonal
                        row.append("1.000")
                    else:  # Upper triangle
                        # Look for this pair in the metrics list
                        pair = f"{metric1}_{metric2}"
                        if pair in metrics:
                            idx = metrics.index(pair)
                            correlation = correlation_matrix[idx][idx]
                            row.append(f"{correlation:.3f}")
                        else:
                            row.append("")  # No correlation found
                
                f.write(",".join(row) + "\n")

        # Generate heatmap if enabled
        if self.heatmap:
            heatmap_save_dir = Path("data/relation/heatmap")
            heatmap_save_dir.mkdir(parents=True, exist_ok=True)
            self.plot_heatmap(csv_path, heatmap_save_dir)

        return output

    def __call__(self):
        return self.build_relation()
