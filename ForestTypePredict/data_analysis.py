"""
Comprehensive Data Analysis & Visualization Generator
Generates all required plots and statistics for Workshop 4 Data Preparation section
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

class ForestDataAnalyzer:
    """
    Comprehensive data analysis and visualization for Forest Cover dataset
    """
    
    def __init__(self, data_path: str):
        """
        Initialize analyzer with data path
        
        Args:
            data_path: Path to train.csv
        """
        print("=" * 70)
        print("FOREST COVER DATA ANALYSIS & VISUALIZATION")
        print("=" * 70)
        
        self.data_path = Path(data_path)
        self.output_dir = Path("analysis_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data
        print(f"\n📂 Loading data from: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(self.df)} samples with {len(self.df.columns)} columns")
        
        # Separate features and target
        self.X = self.df.drop(['Id', 'Cover_Type'], axis=1, errors='ignore')
        self.y = self.df['Cover_Type'] if 'Cover_Type' in self.df.columns else None
        
        # Define feature groups
        self.numerical_features = [
            'Elevation', 'Aspect', 'Slope',
            'Horizontal_Distance_To_Hydrology',
            'Vertical_Distance_To_Hydrology',
            'Horizontal_Distance_To_Roadways',
            'Horizontal_Distance_To_Fire_Points',
            'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm'
        ]
        
        self.wilderness_features = [f'Wilderness_Area{i}' for i in range(1, 5)]
        self.soil_features = [f'Soil_Type{i}' for i in range(1, 41)]
        
        print(f"  Numerical features: {len(self.numerical_features)}")
        print(f"  Wilderness features: {len(self.wilderness_features)}")
        print(f"  Soil features: {len(self.soil_features)}")
    
    def generate_feature_distributions(self, save: bool = True):
        """
        Generate comprehensive feature distribution plots
        """
        print("\n📊 Generating feature distribution plots...")
        
        # Create figure with subplots for key numerical features
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        # Plot distributions for all numerical features
        for idx, feature in enumerate(self.numerical_features):
            ax = axes[idx]
            
            # Histogram with KDE
            data = self.X[feature].dropna()
            
            ax.hist(data, bins=50, alpha=0.7, color='steelblue', 
                   edgecolor='black', density=True, label='Histogram')
            
            # Add KDE
            kde_xs = np.linspace(data.min(), data.max(), 200)
            kde = stats.gaussian_kde(data)
            ax.plot(kde_xs, kde(kde_xs), 'r-', linewidth=2, label='KDE')
            
            # Add mean line
            mean_val = data.mean()
            ax.axvline(mean_val, color='green', linestyle='--', 
                      linewidth=2, label=f'Mean: {mean_val:.1f}')
            
            # Styling
            ax.set_title(f'{feature}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Value', fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Add statistics box
            stats_text = f'μ={data.mean():.1f}\nσ={data.std():.1f}\nMin={data.min():.0f}\nMax={data.max():.0f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=7, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Hide extra subplots
        for idx in range(len(self.numerical_features), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Distribution of Key Numerical Features', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / "feature_distributions.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: {output_path}")
        
        plt.show()
        
        return fig
    
    def generate_class_distribution(self, save: bool = True):
        """
        Generate class distribution visualization
        """
        if self.y is None:
            print("⚠️  No target variable found, skipping class distribution")
            return
        
        print("\n📊 Generating class distribution plot...")
        
        cover_types = {
            1: 'Spruce/Fir',
            2: 'Lodgepole Pine',
            3: 'Ponderosa Pine',
            4: 'Cottonwood/Willow',
            5: 'Aspen',
            6: 'Douglas-fir',
            7: 'Krummholz'
        }
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar plot
        class_counts = self.y.value_counts().sort_index()
        colors = plt.cm.Set3(np.linspace(0, 1, 7))
        
        axes[0].bar(range(1, 8), class_counts.values, color=colors, 
                   edgecolor='black', linewidth=1.5)
        axes[0].set_xlabel('Cover Type', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Count', fontsize=11, fontweight='bold')
        axes[0].set_title('Forest Cover Type Distribution', 
                         fontsize=12, fontweight='bold')
        axes[0].set_xticks(range(1, 8))
        axes[0].set_xticklabels([cover_types[i] for i in range(1, 8)], 
                               rotation=45, ha='right')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(class_counts.values):
            axes[0].text(i+1, v + 50, str(v), ha='center', 
                        fontweight='bold', fontsize=9)
        
        # Pie chart
        axes[1].pie(class_counts.values, labels=[cover_types[i] for i in range(1, 8)],
                   autopct='%1.1f%%', colors=colors, startangle=90,
                   textprops={'fontsize': 9, 'fontweight': 'bold'})
        axes[1].set_title('Cover Type Proportions', 
                         fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / "class_distribution.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: {output_path}")
        
        plt.show()
        
        return fig
    
    def generate_correlation_matrix(self, save: bool = True):
        """
        Generate correlation matrix for numerical features
        """
        print("\n📊 Generating correlation matrix...")
        
        # Calculate correlation matrix
        corr_matrix = self.X[self.numerical_features].corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(250, 10, as_cmap=True)
        
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   annot=True, fmt='.2f', annot_kws={'fontsize': 8})
        
        ax.set_title('Feature Correlation Matrix', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / "correlation_matrix.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: {output_path}")
        
        plt.show()
        
        # Print high correlations
        print("\n  High Correlations (|r| > 0.5):")
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.5:
                    high_corr.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
        
        if high_corr:
            for item in sorted(high_corr, key=lambda x: abs(x['Correlation']), reverse=True):
                print(f"    {item['Feature 1']} ↔ {item['Feature 2']}: {item['Correlation']:.3f}")
        else:
            print("    No correlations > 0.5 found")
        
        return fig, corr_matrix
    
    def generate_elevation_analysis(self, save: bool = True):
        """
        Detailed elevation analysis with chaos threshold detection
        """
        print("\n📊 Generating elevation analysis...")
        
        elevation = self.X['Elevation']
        thresholds = [2400, 2800, 3200]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Elevation distribution with thresholds
        axes[0, 0].hist(elevation, bins=50, alpha=0.7, color='forestgreen', 
                       edgecolor='black', density=True)
        
        # Add KDE
        kde_xs = np.linspace(elevation.min(), elevation.max(), 200)
        kde = stats.gaussian_kde(elevation)
        axes[0, 0].plot(kde_xs, kde(kde_xs), 'b-', linewidth=2, label='KDE')
        
        # Add chaos thresholds
        for threshold in thresholds:
            axes[0, 0].axvline(threshold, color='red', linestyle='--', 
                              linewidth=2, alpha=0.7)
            axes[0, 0].axvspan(threshold - 50, threshold + 50, 
                              alpha=0.2, color='red', label='Chaos Zone' if threshold == 2400 else '')
        
        axes[0, 0].set_xlabel('Elevation (m)', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Density', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Elevation Distribution with Chaos Thresholds', 
                            fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Elevation zones
        zones = pd.cut(elevation, bins=[0, 2400, 2800, 3200, np.inf],
                      labels=['Foothill', 'Montane', 'Subalpine', 'Alpine'])
        zone_counts = zones.value_counts()
        
        colors_zones = ['#90EE90', '#228B22', '#006400', '#004d00']
        axes[0, 1].bar(range(len(zone_counts)), zone_counts.values, 
                      color=colors_zones, edgecolor='black', linewidth=1.5)
        axes[0, 1].set_xticks(range(len(zone_counts)))
        axes[0, 1].set_xticklabels(zone_counts.index, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Count', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Samples per Elevation Zone', 
                            fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Add percentages
        for i, v in enumerate(zone_counts.values):
            pct = 100 * v / len(elevation)
            axes[0, 1].text(i, v + 100, f'{v}\n({pct:.1f}%)', 
                           ha='center', fontweight='bold', fontsize=9)
        
        # 3. Chaos zone proximity
        near_threshold = np.zeros(len(elevation), dtype=bool)
        for threshold in thresholds:
            near_threshold |= (np.abs(elevation - threshold) <= 50)
        
        chaos_stats = pd.Series(['In Chaos Zone' if x else 'Normal' 
                                for x in near_threshold]).value_counts()
        
        axes[1, 0].pie(chaos_stats.values, labels=chaos_stats.index,
                      autopct='%1.1f%%', colors=['#ff6b6b', '#51cf66'],
                      startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
        axes[1, 0].set_title('Chaos Zone Proximity Distribution', 
                            fontsize=12, fontweight='bold')
        
        # 4. Cover type by elevation (if target available)
        if self.y is not None:
            for cover_type in sorted(self.y.unique()):
                subset = elevation[self.y == cover_type]
                axes[1, 1].hist(subset, bins=30, alpha=0.5, 
                               label=f'Type {cover_type}', density=True)
            
            axes[1, 1].set_xlabel('Elevation (m)', fontsize=11, fontweight='bold')
            axes[1, 1].set_ylabel('Density', fontsize=11, fontweight='bold')
            axes[1, 1].set_title('Elevation Distribution by Cover Type', 
                                fontsize=12, fontweight='bold')
            axes[1, 1].legend(fontsize=8, ncol=2)
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Target variable not available', 
                           ha='center', va='center', fontsize=12)
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / "elevation_analysis.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: {output_path}")
        
        plt.show()
        
        # Print statistics
        print("\n  Elevation Statistics:")
        print(f"    Mean: {elevation.mean():.1f} m")
        print(f"    Std: {elevation.std():.1f} m")
        print(f"    Range: {elevation.min():.0f} - {elevation.max():.0f} m")
        print(f"    In chaos zones: {near_threshold.sum()} ({100*near_threshold.sum()/len(elevation):.2f}%)")
        
        return fig
    
    def generate_soil_sparsity_analysis(self, save: bool = True):
        """
        Analyze soil type sparsity
        """
        print("\n📊 Generating soil type sparsity analysis...")
        
        soil_data = self.X[self.soil_features]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Soil type frequencies
        soil_sums = soil_data.sum().sort_values(ascending=False)
        
        axes[0, 0].bar(range(len(soil_sums)), soil_sums.values, 
                      color='sienna', edgecolor='black', linewidth=0.5)
        axes[0, 0].set_xlabel('Soil Type (sorted by frequency)', 
                             fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Count', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Soil Type Frequency Distribution', 
                            fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 2. Sparsity distribution
        sparsity = (soil_data == 0).mean() * 100
        
        axes[0, 1].hist(sparsity, bins=20, color='coral', 
                       edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(sparsity.mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {sparsity.mean():.1f}%')
        axes[0, 1].set_xlabel('Sparsity (%)', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Soil Feature Sparsity Distribution', 
                            fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Top 10 most common soil types
        top10 = soil_sums.head(10)
        
        axes[1, 0].barh(range(len(top10)), top10.values, 
                       color='brown', edgecolor='black', linewidth=1)
        axes[1, 0].set_yticks(range(len(top10)))
        axes[1, 0].set_yticklabels(top10.index)
        axes[1, 0].set_xlabel('Count', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Top 10 Most Common Soil Types', 
                            fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        axes[1, 0].invert_yaxis()
        
        # 4. Sparsity impact
        rare_soils = (soil_sums < 100).sum()
        common_soils = (soil_sums >= 100).sum()
        
        axes[1, 1].pie([rare_soils, common_soils], 
                      labels=[f'Rare (n<100): {rare_soils}', 
                             f'Common (n≥100): {common_soils}'],
                      autopct='%1.1f%%', colors=['#ff6b6b', '#51cf66'],
                      startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
        axes[1, 1].set_title('Soil Type Rarity Distribution', 
                            fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / "soil_sparsity_analysis.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: {output_path}")
        
        plt.show()
        
        # Print statistics
        print(f"\n  Soil Type Statistics:")
        print(f"    Total soil types: {len(self.soil_features)}")
        print(f"    Mean sparsity: {sparsity.mean():.2f}%")
        print(f"    Rare types (n<100): {rare_soils}")
        print(f"    Common types (n≥100): {common_soils}")
        
        return fig
    
    def generate_summary_statistics(self, save: bool = True):
        """
        Generate comprehensive summary statistics JSON
        """
        print("\n📊 Generating summary statistics...")
        
        summary = {
            "dataset_overview": {
                "total_samples": int(len(self.df)),
                "total_features": int(len(self.X.columns)),
                "numerical_features": len(self.numerical_features),
                "wilderness_features": len(self.wilderness_features),
                "soil_features": len(self.soil_features),
                "target_classes": int(self.y.nunique()) if self.y is not None else None
            },
            "data_quality": {
                "missing_values": int(self.X.isnull().sum().sum()),
                "missing_percentage": float(self.X.isnull().sum().sum() / self.X.size * 100),
                "duplicate_rows": int(self.df.duplicated().sum())
            },
            "numerical_features_stats": {}
        }
        
        # Add numerical feature statistics
        for feature in self.numerical_features:
            data = self.X[feature].dropna()
            summary["numerical_features_stats"][feature] = {
                "mean": float(data.mean()),
                "std": float(data.std()),
                "min": float(data.min()),
                "max": float(data.max()),
                "q25": float(data.quantile(0.25)),
                "median": float(data.median()),
                "q75": float(data.quantile(0.75)),
                "skewness": float(data.skew()),
                "kurtosis": float(data.kurtosis())
            }
        
        # Add class distribution if available
        if self.y is not None:
            summary["class_distribution"] = self.y.value_counts().to_dict()
        
        # Save to JSON
        if save:
            output_path = self.output_dir / "summary_statistics.json"
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"  ✓ Saved: {output_path}")
        
        # Print key statistics
        print("\n  Key Statistics:")
        print(f"    Total samples: {summary['dataset_overview']['total_samples']:,}")
        print(f"    Total features: {summary['dataset_overview']['total_features']}")
        print(f"    Missing values: {summary['data_quality']['missing_values']} ({summary['data_quality']['missing_percentage']:.2f}%)")
        print(f"    Duplicate rows: {summary['data_quality']['duplicate_rows']}")
        
        return summary
    
    def run_complete_analysis(self):
        """
        Run all analysis and generate all visualizations
        """
        print("\n" + "=" * 70)
        print("RUNNING COMPLETE DATA ANALYSIS")
        print("=" * 70)
        
        # Generate all plots
        self.generate_feature_distributions()
        self.generate_class_distribution()
        self.generate_correlation_matrix()
        self.generate_elevation_analysis()
        self.generate_soil_sparsity_analysis()
        summary = self.generate_summary_statistics()
        
        print("\n" + "=" * 70)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"\nAll outputs saved to: {self.output_dir.absolute()}")
        print("\nGenerated files:")
        print("  📸 feature_distributions.png")
        print("  📸 class_distribution.png")
        print("  📸 correlation_matrix.png")
        print("  📸 elevation_analysis.png")
        print("  📸 soil_sparsity_analysis.png")
        print("  📄 summary_statistics.json")
        
        return summary


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Path to your training data (matching your project structure)
    import os
    
    # Get project root directory
    project_root = Path(__file__).parent
    DATA_PATH = project_root / "ForestTypePredict" / "src" / "data" / "raw" / "train.csv"
    
    # Alternative: If running from root, use direct path
    # DATA_PATH = "ForestTypePredict/src/data/raw/train.csv"
    
    # Check if file exists
    if not DATA_PATH.exists():
        print(f"❌ Data file not found at: {DATA_PATH}")
        print(f"   Current directory: {os.getcwd()}")
        print("\n🔍 Searching for train.csv in project...")
        
        # Try alternative paths
        alternative_paths = [
            Path("ForestTypePredict/src/data/raw/train.csv"),
            Path("src/data/raw/train.csv"),
            Path("data/raw/train.csv")
        ]
        
        for alt_path in alternative_paths:
            if alt_path.exists():
                DATA_PATH = alt_path
                print(f"✓ Found at: {DATA_PATH}")
                break
        else:
            print("\n⚠️  Please provide the correct path to train.csv")
            exit(1)
    
    print(f"\n✓ Using data from: {DATA_PATH}")
    
    # Create analyzer
    analyzer = ForestDataAnalyzer(str(DATA_PATH))
    
    # Run complete analysis
    summary = analyzer.run_complete_analysis()
    
    print("\n✅ Ready for LaTeX integration!")
    print("   Copy the PNG files to your LaTeX figures folder")
    print(f"\n📂 Output directory: {analyzer.output_dir.absolute()}")