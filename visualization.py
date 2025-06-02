import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import MDS
import os
from pathlib import Path
import seaborn as sns

class RepoFileVisualizer:
    """
    A comprehensive visualization tool for repository file analysis.
    
    This class creates interactive visualizations showing file relationships
    based on directory hierarchy, with visual encoding for file complexity
    and development activity.
    """
    
    def __init__(self, csv_path):
        """Initialize the visualizer with CSV data."""
        self.df = self.load_and_validate_data(csv_path)
        self.distance_matrix = None
        self.positions = None
        
    def load_and_validate_data(self, csv_path):
        """Load CSV data and perform validation checks."""
        try:
            df = pd.read_csv(csv_path)
            required_cols = ['path', 'lines', 'commits']
            
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV must contain columns: {required_cols}")
                
            # Clean and validate data
            df = df.dropna()
            df = df[df['lines'] > 0]  # Remove files with zero lines
            df = df[df['commits'] >= 0]  # Ensure non-negative commits
            
            print(f"Loaded {len(df)} valid file records")
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def calculate_path_distance(self, path1, path2):
        """
        Calculate hierarchical distance between two file paths.
        
        Distance calculation:
        - Same directory: distance = 1
        - Subdirectory relationship: distance = depth difference + 1
        - Different branches: distance = sum of depths from common ancestor
        """
        try:
            # Normalize paths and split into components
            p1_parts = Path(path1).parts
            p2_parts = Path(path2).parts
            
            # Find common path length
            common_length = 0
            for i in range(min(len(p1_parts), len(p2_parts))):
                if p1_parts[i] == p2_parts[i]:
                    common_length += 1
                else:
                    break
            
            # Calculate distance based on hierarchy
            if p1_parts == p2_parts:
                return 0  # Same file
            elif common_length == len(p1_parts) - 1 and len(p2_parts) == len(p1_parts):
                return 1  # Same directory
            else:
                # Distance = steps to common ancestor + steps from common ancestor
                dist1 = len(p1_parts) - common_length
                dist2 = len(p2_parts) - common_length
                return dist1 + dist2
                
        except Exception as e:
            print(f"Error calculating distance between {path1} and {path2}: {e}")
            return float('inf')
    
    def build_distance_matrix(self):
        """Construct distance matrix for all file pairs."""
        n_files = len(self.df)
        distance_matrix = np.zeros((n_files, n_files))
        
        paths = self.df['path'].tolist()
        
        for i in range(n_files):
            for j in range(i + 1, n_files):
                dist = self.calculate_path_distance(paths[i], paths[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
                
        self.distance_matrix = distance_matrix
        print(f"Built distance matrix for {n_files} files")
        return distance_matrix
    
    def calculate_positions(self, random_state=42):
        """Use MDS to calculate 2D positions preserving distance relationships."""
        if self.distance_matrix is None:
            self.build_distance_matrix()
        
        # Apply MDS with precomputed distance matrix
        mds = MDS(n_components=2, dissimilarity='precomputed', 
                  random_state=random_state, max_iter=1000)
        
        try:
            self.positions = mds.fit_transform(self.distance_matrix)
            print(f"MDS stress: {mds.stress_:.4f}")
            return self.positions
        except Exception as e:
            print(f"MDS failed: {e}")
            # Fallback to random positioning
            n_files = len(self.df)
            np.random.seed(random_state)
            self.positions = np.random.randn(n_files, 2) * 10
            return self.positions
    
    def create_visualization(self, figsize=(15, 12), save_path=None):
        """Generate the comprehensive file visualization."""
        if self.positions is None:
            self.calculate_positions()
        
        # Set up the plot with professional styling
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=figsize, dpi=100)
        
        # Prepare data for visualization
        x_coords = self.positions[:, 0]
        y_coords = self.positions[:, 1]
        lines = self.df['lines'].values
        commits = self.df['commits'].values
        
        # Normalize sizes (circle areas) based on lines of code
        size_min, size_max = 50, 1000
        sizes = size_min + (size_max - size_min) * (lines - lines.min()) / (lines.max() - lines.min())
        
        # Create green-to-red colormap for commits
        colors = ['#2E8B57', '#32CD32', '#FFFF00', '#FF8C00', '#FF4500', '#DC143C']
        n_bins = 100
        custom_cmap = LinearSegmentedColormap.from_list('commits', colors, N=n_bins)
        
        # Create the scatter plot
        scatter = ax.scatter(x_coords, y_coords, s=sizes, c=commits, 
                           cmap=custom_cmap, alpha=0.7, edgecolors='black', 
                           linewidth=0.5)
        
        # Add colorbar with proper labeling
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Number of Commits', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)
        
        # Customize the plot appearance
        ax.set_xlabel('Repository Structure Dimension 1', fontsize=14, fontweight='bold')
        ax.set_ylabel('Repository Structure Dimension 2', fontsize=14, fontweight='bold')
        ax.set_title('Repository File Visualization\n' + 
                    'Position: Directory Hierarchy • Size: Lines of Code • Color: Commit Frequency',
                    fontsize=16, fontweight='bold', pad=20)
        
        # Add grid and styling
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        
        # Create legend for sizes
        legend_sizes = [lines.min(), np.percentile(lines, 50), lines.max()]
        legend_labels = [f'{int(size)} lines' for size in legend_sizes]
        legend_handles = []
        
        for size, label in zip(legend_sizes, legend_labels):
            normalized_size = size_min + (size_max - size_min) * (size - lines.min()) / (lines.max() - lines.min())
            handle = plt.scatter([], [], s=normalized_size, c='gray', alpha=0.6, 
                               edgecolors='black', linewidth=0.5)
            legend_handles.append(handle)
        
        legend1 = ax.legend(legend_handles, legend_labels, 
                           title='File Size (Lines of Code)', 
                           loc='upper left', bbox_to_anchor=(0.02, 0.98),
                           fontsize=10, title_fontsize=11)
        legend1.get_title().set_fontweight('bold')
        
        # Add summary statistics as text
        stats_text = f"""Files: {len(self.df)}
Avg Lines: {lines.mean():.0f}
Avg Commits: {commits.mean():.1f}
Max Distance: {self.distance_matrix.max():.0f}"""
        
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
        
        return fig, ax
    
    def print_analysis_summary(self):
        """Print comprehensive analysis of the repository structure."""
        if self.distance_matrix is None:
            self.build_distance_matrix()
            
        print("\n" + "="*60)
        print("REPOSITORY ANALYSIS SUMMARY")
        print("="*60)
        
        # Basic statistics
        print(f"Total Files Analyzed: {len(self.df)}")
        print(f"Average Lines per File: {self.df['lines'].mean():.1f}")
        print(f"Total Lines of Code: {self.df['lines'].sum():,}")
        print(f"Average Commits per File: {self.df['commits'].mean():.1f}")
        print(f"Total Commits: {self.df['commits'].sum():,}")
        
        # Distance analysis
        non_zero_distances = self.distance_matrix[self.distance_matrix > 0]
        print(f"\nDirectory Structure Analysis:")
        print(f"Average File Distance: {non_zero_distances.mean():.2f}")
        print(f"Maximum File Distance: {non_zero_distances.max():.0f}")
        
        # Identify clusters and outliers
        lines = self.df['lines'].values
        commits = self.df['commits'].values
        
        large_files = self.df[self.df['lines'] > np.percentile(lines, 90)]
        active_files = self.df[self.df['commits'] > np.percentile(commits, 90)]
        
        print(f"\nTop 10% Largest Files ({len(large_files)} files):")
        for _, file in large_files.nlargest(5, 'lines').iterrows():
            print(f"  {file['path']}: {file['lines']} lines, {file['commits']} commits")
            
        print(f"\nTop 10% Most Active Files ({len(active_files)} files):")
        for _, file in active_files.nlargest(5, 'commits').iterrows():
            print(f"  {file['path']}: {file['commits']} commits, {file['lines']} lines")


def main():
    """Main execution function with example usage."""
    # Example CSV creation for demonstration
    sample_data = """path,lines,commits
src/main.py,450,23
src/utils/helper.py,120,8
src/utils/data_processor.py,280,15
src/models/base_model.py,350,12
src/models/advanced_model.py,520,18
tests/test_main.py,180,6
tests/utils/test_helper.py,95,4
docs/README.md,80,3
config/settings.py,60,5
src/api/endpoints.py,380,20"""
    
    # Save sample data for demonstration
    with open('sample_repo_data.csv', 'w') as f:
        f.write(sample_data)
    
    try:
        # Initialize and run the visualizer
        visualizer = RepoFileVisualizer('sample_repo_data.csv')
        
        # Generate comprehensive analysis
        visualizer.print_analysis_summary()
        
        # Create the visualization
        print("\nGenerating visualization...")
        visualizer.create_visualization(save_path='repo_visualization.png')
        
        print("\nVisualization complete! Check 'repo_visualization.png' for the saved plot.")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
