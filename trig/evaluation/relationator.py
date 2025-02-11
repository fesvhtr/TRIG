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
from matplotlib.patches import FancyArrowPatch

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
        self.tsne = self.relation_config.get("tsne", False)  # Add t-SNE configuration
        self.tradeoff = self.relation_config.get("tradeoff", False)  # Add trade-off configuration
        self.node_colors = self.relation_config.get("node_colors", {})
        
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
            # Example: "IQ-O_TA-R" -> ["IQ-O", "TA-R"]  (using underscore, not hyphen)
            metrics = dim_pair.split('_')  # Changed from split('-') to split('_')
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

    def plot_tsne(self, correlation_matrix: np.ndarray, metrics: List[str], save_dir: Path) -> Figure:
        """Generate t-SNE visualization of the correlation matrix.
        
        Args:
            correlation_matrix: Matrix of correlations between metrics
            metrics: List of metric names
            save_dir: Directory to save the plot
            
        Returns:
            matplotlib figure
        """
        # Extract individual metric names from pairs
        individual_metrics = set()
        for metric_pair in metrics:
            m1, m2 = metric_pair.split('_')
            individual_metrics.add(m1)
            individual_metrics.add(m2)
        individual_metrics = sorted(list(individual_metrics))
        
        # Create similarity matrix for individual metrics
        n = len(individual_metrics)
        similarity_matrix = np.zeros((n, n))
        metric_to_idx = {m: i for i, m in enumerate(individual_metrics)}
        
        # Fill similarity matrix using correlation values
        for i, metric_pair in enumerate(metrics):
            m1, m2 = metric_pair.split('_')
            idx1, idx2 = metric_to_idx[m1], metric_to_idx[m2]
            similarity_matrix[idx1, idx2] = correlation_matrix[i, i]
            similarity_matrix[idx2, idx1] = correlation_matrix[i, i]  # Symmetric
        
        # Set diagonal to 1
        np.fill_diagonal(similarity_matrix, 1.0)
        
        # Apply t-SNE
        n_samples = len(individual_metrics)
        perplexity = min(30, n_samples - 1)  # Ensure perplexity < n_samples
        tsne = TSNE(
            n_components=2,
            metric='precomputed',
            random_state=42,
            perplexity=perplexity,  # Adjust perplexity based on sample size
            n_iter=1000,  # Increase iterations for better convergence
            learning_rate='auto'  # Let TSNE choose the best learning rate
        )
        
        # Convert similarity to distance: distance = 1 - similarity
        distances = 1 - np.abs(similarity_matrix)
        embedding = tsne.fit_transform(distances)
        
        # Create plot
        plt.figure(figsize=(10, 10))
        
        # Plot points
        plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.8)
        
        # Add labels
        for i, metric in enumerate(individual_metrics):
            plt.annotate(
                metric,
                (embedding[i, 0], embedding[i, 1]),
                xytext=(5, 5),
                textcoords='offset points',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
            )
        
        # Customize plot
        plt.title('Metric Relationship Visualization (t-SNE)')
        plt.xlabel('t-SNE Position X')
        plt.ylabel('t-SNE Position Y')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save plot
        models_str = "-".join(self.models)
        plot_name = f'{models_str}-{self.res}-{self.metric}-tsne.png'
        plt.savefig(save_dir / plot_name, dpi=300, bbox_inches='tight')
        
        return plt.gcf()

    def calculate_tradeoff(self, scores1, scores2):
        """Calculate trade-off relationship based on correlation value."""
        try:
            # Calculate correlation using the configured method
            correlation_method = self.get_correlation_method()
            correlation = correlation_method(scores1, scores2)
            
            # Use correlation value to determine relationship type
            if correlation > 0:  # Positive correlation indicates dominating relationship
                return {
                    "relationship_type": "dominating",
                    "strength": abs(correlation),
                    "direction": 1
                }
            else:  # Negative correlation indicates contradicting relationship
                return {
                    "relationship_type": "contradicting",
                    "strength": abs(correlation),
                    "direction": -1
                }
            
        except Exception as e:
            print(f"Error calculating trade-off: {e}")
            return {
                "relationship_type": "error",
                "strength": 0,
                "direction": 0
            }

    def detect_dominance_cycles(self, G):
        """Detect cycles in dominance relationships."""
        cycles = []
        for node in G.nodes():
            try:
                # Get all cycles containing this node
                cycle = nx.find_cycle(G, node, orientation="original")
                
                # Check if this is a new cycle and all edges are dominating
                if cycle not in cycles:
                    # Get edge data for each edge in the cycle
                    is_dominating_cycle = True
                    for u, v, _ in cycle:
                        edge_data = G.get_edge_data(u, v)
                        if edge_data["relationship_type"] != "dominating":
                            is_dominating_cycle = False
                            break
                    
                    if is_dominating_cycle:
                        cycles.append(cycle)
                    
            except nx.NetworkXNoCycle:
                continue
        return cycles

    def calculate_effective_dominance(self, G, cycles):
        """Calculate effective dominance considering cycles."""
        effective_dominance = {}
        for node in G.nodes():
            # Count direct dominance
            dominating = sum(1 for _, _, d in G.edges(node, data=True) 
                            if d["relationship_type"] == "dominating")
            dominated = sum(1 for _, _, d in G.edges(node, data=True) 
                           if d["relationship_type"] == "dominating" and 
                           node == list(d.keys())[1])
            
            # Reduce dominance score for nodes in cycles
            cycle_penalty = 0
            for cycle in cycles:
                if any(node in edge for edge in cycle):
                    cycle_penalty += 0.5  # Reduce dominance effect for cyclic relationships
            
            effective_dominance[node] = dominating - dominated - cycle_penalty
        return effective_dominance

    def plot_tradeoff(self, data: Dict, save_dir: Path) -> Figure:
        """Generate trade-off visualization between metrics."""
        # Load the pre-calculated correlation values from CSV
        models_str = "-".join(self.models)
        csv_filename = f"{models_str}-{self.res}-{self.metric}.csv"
        csv_path = Path("data/relation/csv") / csv_filename
        
        # Read correlation matrix from CSV
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            metrics = lines[0].strip().split(',')[1:]  # Get metric names
            matrix = {}
            
            # Only read upper triangular part of the matrix
            for i, line in enumerate(lines[1:], 1):
                values = line.strip().split(',')
                metric1 = values[0]
                for j, val in enumerate(values[1:], 1):
                    if val.strip() and j >= i:  # Only read upper triangular part
                        metric2 = metrics[j-1]
                        matrix[f"{metric1}_{metric2}"] = float(val)

        # Create network graph
        G = nx.DiGraph()  # Use directed graph for arrows
        
        # Set fixed node size for all nodes
        node_sizes = {metric: 1000 for metric in metrics}
        
        # Add nodes
        for metric in metrics:
            G.add_node(metric)
        
        # Add edges based on correlations
        for pair, correlation in matrix.items():
            if correlation != 1.0:  # Skip diagonal values
                m1, m2 = pair.split('_')
                if correlation > 0:
                    G.add_edge(m1, m2, relationship_type="dominating")
                else:  # correlation < 0
                    G.add_edge(m1, m2, relationship_type="contradicting")

        # Create weighted graph for layout
        G_layout = nx.Graph()
        for metric in metrics:
            G_layout.add_node(metric)
        
        # Add edges with higher weights for positive correlations
        for pair, correlation in matrix.items():
            if correlation != 1.0:  # Skip diagonal values
                m1, m2 = pair.split('_')
                if correlation > 0:
                    # Higher weight means stronger attraction
                    G_layout.add_edge(m1, m2, weight=2.0)
                else:
                    # Lower weight means weaker attraction
                    G_layout.add_edge(m1, m2, weight=0.5)

        # Calculate layout with weighted edges
        pos = nx.spring_layout(G_layout, k=0.5, weight='weight', iterations=50)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Generate colors for nodes
        node_colors = {}
        for metric in metrics:
            if metric in self.node_colors:
                node_colors[metric] = self.node_colors[metric]
            else:
                hue = random.random()
                node_colors[metric] = plt.cm.hsv(hue)
        
        # Create legend elements
        legend_elements = [
            Line2D([0], [0], color='green', 
                   label='Dominating', linestyle='-',
                   linewidth=2),  # Simple green line for dominating
            Line2D([0], [0], color='orange',
                   label='Contradicting', linestyle='-',
                   linewidth=2)   # Simple orange line for contradicting
        ]

        # Define arrow properties
        ARROW_WIDTH = 0.02
        ARROW_HEAD_WIDTH = 0.05
        ARROW_HEAD_LENGTH = 0.1
        ARROW_ALPHA = 0.8
        ARROW_LINE_WIDTH = 120

        # Draw edges
        for (m1, m2, data) in G.edges(data=True):
            # Calculate positions
            dx = pos[m2][0] - pos[m1][0]
            dy = pos[m2][1] - pos[m1][1]
            angle = np.arctan2(dy, dx)
            
            # Calculate exact circle radii
            source_radius = np.sqrt(node_sizes[m1] / np.pi) / 30
            target_radius = np.sqrt(node_sizes[m2] / np.pi) / 30
            
            # Calculate exact points on circle edges
            start_x = pos[m1][0] + source_radius * np.cos(angle)
            start_y = pos[m1][1] + source_radius * np.sin(angle)
            end_x = pos[m2][0] - target_radius * np.cos(angle)
            end_y = pos[m2][1] - target_radius * np.sin(angle)
            
            if data["relationship_type"] == "dominating":
                # Use annotation for dominating relationship with inward arrows
                plt.annotate("",
                            xy=(end_x, end_y), xycoords='data',
                            xytext=(start_x, start_y), textcoords='data',
                            arrowprops=dict(arrowstyle='-',
                                          color='green',
                                          lw=ARROW_WIDTH*ARROW_LINE_WIDTH,
                                          alpha=ARROW_ALPHA,
                                          shrinkA=0, shrinkB=0),
                            zorder=1)
            else:  # contradicting relationship
                plt.annotate("",
                            xy=(end_x, end_y), xycoords='data',
                            xytext=(start_x, start_y), textcoords='data',
                            arrowprops=dict(arrowstyle="<->",
                                          color="orange",
                                          lw=ARROW_WIDTH*ARROW_LINE_WIDTH,
                                          alpha=ARROW_ALPHA,
                                          shrinkA=0, shrinkB=0),
                            zorder=1)

        # Draw nodes
        for metric in G.nodes():
            plt.scatter(pos[metric][0], pos[metric][1],
                       s=node_sizes[metric],
                       c=[node_colors[metric]],
                       alpha=0.8,
                       zorder=2)
        
        # Draw labels
        for metric in G.nodes():
            plt.annotate(metric,
                        (pos[metric][0], pos[metric][1]),
                        xytext=(0, 0),
                        textcoords='offset points',
                        ha='center',
                        va='center',
                        zorder=3)
        
        # Add legend
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Customize plot
        plt.title("Metric Trade-off Relationships")
        plt.axis('equal')
        
        # Save plot
        plot_name = f'{models_str}-{self.res}-{self.metric}-tradeoff.png'
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
                row = [metric1]
                for j, metric2 in enumerate(individual_metrics):
                    if j < i:  # Lower triangle
                        pair = f"{metric2}_{metric1}"
                        if pair in metrics:
                            idx = metrics.index(pair)
                            correlation = correlation_matrix[idx][idx]
                            
                            # Handle correlation based on metric type
                            if self.metric in ['pearson_corr', 'spearman_corr']:
                                # For Pearson and Spearman correlations, negate the value
                                correlation = -correlation
                            elif self.metric == 'paretoF':
                                # For Pareto frontier score, use 1 - correlation
                                # This maintains the [0,1] range while showing opposite relationship
                                correlation = 1 - correlation
                            
                            row.append(f"{correlation:.3f}")
                        else:
                            row.append("")
                    elif metric1 == metric2:  # Diagonal
                        row.append("1.000")
                    else:  # Upper triangle
                        pair = f"{metric1}_{metric2}"
                        if pair in metrics:
                            idx = metrics.index(pair)
                            correlation = correlation_matrix[idx][idx]
                            row.append(f"{correlation:.3f}")
                        else:
                            row.append("")
                
                f.write(",".join(row) + "\n")

        # Generate heatmap if enabled
        if self.heatmap:
            heatmap_save_dir = Path("data/relation/heatmap")
            heatmap_save_dir.mkdir(parents=True, exist_ok=True)
            self.plot_heatmap(csv_path, heatmap_save_dir)

        # Generate t-SNE visualization if enabled
        if self.tsne:
            tsne_save_dir = Path("data/relation/tsne")
            tsne_save_dir.mkdir(parents=True, exist_ok=True)
            self.plot_tsne(correlation_matrix, metrics, tsne_save_dir)

        # Generate trade-off analysis if enabled
        if self.tradeoff:
            tradeoff_save_dir = Path("data/relation/tradeoff")
            tradeoff_save_dir.mkdir(parents=True, exist_ok=True)
            self.plot_tradeoff(data, tradeoff_save_dir)

        return output

    def __call__(self):
        return self.build_relation()
