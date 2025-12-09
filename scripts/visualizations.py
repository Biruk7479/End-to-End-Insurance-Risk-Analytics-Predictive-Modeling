"""
Visualization Module
Handles all visualization tasks for EDA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class InsuranceVisualizer:
    """Class for creating insurance data visualizations"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize InsuranceVisualizer
        
        Args:
            df: Input DataFrame
        """
        self.df = df
        
    def plot_distribution(self, column: str, bins: int = 30, 
                         figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot distribution of a numerical column
        
        Args:
            column: Column name
            bins: Number of bins for histogram
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        axes[0].hist(self.df[column].dropna(), bins=bins, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel(column)
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Distribution of {column}')
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(self.df[column].dropna(), vert=True)
        axes[1].set_ylabel(column)
        axes[1].set_title(f'Box Plot of {column}')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def plot_categorical(self, column: str, top_n: int = 10,
                        figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot categorical variable distribution
        
        Args:
            column: Column name
            top_n: Number of top categories to show
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        value_counts = self.df[column].value_counts().head(top_n)
        
        ax.bar(range(len(value_counts)), value_counts.values, alpha=0.7)
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
        ax.set_xlabel(column)
        ax.set_ylabel('Count')
        ax.set_title(f'Top {top_n} Categories in {column}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def plot_correlation_matrix(self, columns: Optional[List[str]] = None,
                               figsize: Tuple[int, int] = (14, 10)) -> None:
        """
        Plot correlation matrix heatmap
        
        Args:
            columns: Columns to include. If None, use all numerical
            figsize: Figure size
        """
        if columns is None:
            df_corr = self.df.select_dtypes(include=[np.number])
        else:
            df_corr = self.df[columns]
            
        correlation = df_corr.corr()
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, ax=ax,
                   cbar_kws={"shrink": 0.8})
        ax.set_title('Correlation Matrix', fontsize=16, pad=20)
        
        plt.tight_layout()
        plt.show()
        
    def plot_loss_ratio_by_category(self, category_col: str,
                                    figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot loss ratio by category
        
        Args:
            category_col: Category column name
            figsize: Figure size
        """
        if 'LossRatio' not in self.df.columns:
            logger.warning("LossRatio column not found")
            return
            
        fig, ax = plt.subplots(figsize=figsize)
        
        loss_by_category = self.df.groupby(category_col)['LossRatio'].agg(['mean', 'median'])
        loss_by_category = loss_by_category.sort_values('mean', ascending=False).head(15)
        
        x = range(len(loss_by_category))
        ax.bar(x, loss_by_category['mean'], alpha=0.7, label='Mean')
        ax.plot(x, loss_by_category['median'], 'ro-', label='Median')
        
        ax.set_xticks(x)
        ax.set_xticklabels(loss_by_category.index, rotation=45, ha='right')
        ax.set_xlabel(category_col)
        ax.set_ylabel('Loss Ratio')
        ax.set_title(f'Loss Ratio by {category_col}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def plot_claims_vs_premium(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot scatter plot of TotalClaims vs TotalPremium
        
        Args:
            figsize: Figure size
        """
        if 'TotalClaims' not in self.df.columns or 'TotalPremium' not in self.df.columns:
            logger.warning("TotalClaims or TotalPremium column not found")
            return
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sample data if too large
        df_plot = self.df.sample(min(10000, len(self.df)))
        
        ax.scatter(df_plot['TotalPremium'], df_plot['TotalClaims'], 
                  alpha=0.5, s=20)
        
        # Add diagonal line (break-even line)
        max_val = max(df_plot['TotalPremium'].max(), df_plot['TotalClaims'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', label='Break-even line')
        
        ax.set_xlabel('Total Premium')
        ax.set_ylabel('Total Claims')
        ax.set_title('Total Claims vs Total Premium')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def plot_temporal_trends(self, date_col: str = 'TransactionMonth',
                            figsize: Tuple[int, int] = (14, 6)) -> None:
        """
        Plot temporal trends in claims and premiums
        
        Args:
            date_col: Date column name
            figsize: Figure size
        """
        if date_col not in self.df.columns:
            logger.warning(f"{date_col} column not found")
            return
            
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Ensure date column is datetime
        df_temp = self.df.copy()
        df_temp[date_col] = pd.to_datetime(df_temp[date_col])
        
        # Group by month
        monthly_data = df_temp.groupby(df_temp[date_col].dt.to_period('M')).agg({
            'TotalPremium': 'sum',
            'TotalClaims': 'sum',
            'PolicyID': 'count'
        })
        
        monthly_data.index = monthly_data.index.to_timestamp()
        
        # Premium and Claims over time
        axes[0].plot(monthly_data.index, monthly_data['TotalPremium'], 
                    marker='o', label='Total Premium', linewidth=2)
        axes[0].plot(monthly_data.index, monthly_data['TotalClaims'], 
                    marker='s', label='Total Claims', linewidth=2)
        axes[0].set_ylabel('Amount')
        axes[0].set_title('Monthly Premium and Claims Trends')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Policy count over time
        axes[1].plot(monthly_data.index, monthly_data['PolicyID'], 
                    marker='o', color='green', linewidth=2)
        axes[1].set_xlabel('Month')
        axes[1].set_ylabel('Policy Count')
        axes[1].set_title('Monthly Policy Count')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def plot_geographic_analysis(self, geo_col: str = 'Province',
                                figsize: Tuple[int, int] = (14, 8)) -> None:
        """
        Plot geographic analysis
        
        Args:
            geo_col: Geographic column name
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Premium by geography
        premium_by_geo = self.df.groupby(geo_col)['TotalPremium'].sum().sort_values(ascending=False)
        axes[0, 0].bar(range(len(premium_by_geo)), premium_by_geo.values, alpha=0.7)
        axes[0, 0].set_xticks(range(len(premium_by_geo)))
        axes[0, 0].set_xticklabels(premium_by_geo.index, rotation=45, ha='right')
        axes[0, 0].set_title(f'Total Premium by {geo_col}')
        axes[0, 0].set_ylabel('Total Premium')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Claims by geography
        claims_by_geo = self.df.groupby(geo_col)['TotalClaims'].sum().sort_values(ascending=False)
        axes[0, 1].bar(range(len(claims_by_geo)), claims_by_geo.values, 
                      alpha=0.7, color='orange')
        axes[0, 1].set_xticks(range(len(claims_by_geo)))
        axes[0, 1].set_xticklabels(claims_by_geo.index, rotation=45, ha='right')
        axes[0, 1].set_title(f'Total Claims by {geo_col}')
        axes[0, 1].set_ylabel('Total Claims')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Loss Ratio by geography
        if 'LossRatio' in self.df.columns:
            loss_by_geo = self.df.groupby(geo_col)['LossRatio'].mean().sort_values(ascending=False)
            axes[1, 0].bar(range(len(loss_by_geo)), loss_by_geo.values, 
                          alpha=0.7, color='red')
            axes[1, 0].set_xticks(range(len(loss_by_geo)))
            axes[1, 0].set_xticklabels(loss_by_geo.index, rotation=45, ha='right')
            axes[1, 0].set_title(f'Average Loss Ratio by {geo_col}')
            axes[1, 0].set_ylabel('Loss Ratio')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Policy count by geography
        count_by_geo = self.df.groupby(geo_col).size().sort_values(ascending=False)
        axes[1, 1].bar(range(len(count_by_geo)), count_by_geo.values, 
                      alpha=0.7, color='green')
        axes[1, 1].set_xticks(range(len(count_by_geo)))
        axes[1, 1].set_xticklabels(count_by_geo.index, rotation=45, ha='right')
        axes[1, 1].set_title(f'Policy Count by {geo_col}')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def create_missing_values_plot(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Create visualization of missing values
        
        Args:
            figsize: Figure size
        """
        missing = self.df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if len(missing) == 0:
            logger.info("No missing values found")
            return
            
        fig, ax = plt.subplots(figsize=figsize)
        
        missing_pct = (missing / len(self.df)) * 100
        
        ax.barh(range(len(missing)), missing_pct.values, alpha=0.7)
        ax.set_yticks(range(len(missing)))
        ax.set_yticklabels(missing.index)
        ax.set_xlabel('Percentage Missing (%)')
        ax.set_title('Missing Values by Column')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.show()


def create_eda_report_plots(df: pd.DataFrame, output_dir: str = 'reports/figures') -> None:
    """
    Create comprehensive EDA plots and save them
    
    Args:
        df: Input DataFrame
        output_dir: Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = InsuranceVisualizer(df)
    
    logger.info("Creating EDA visualizations...")
    
    # Add your key visualizations here
    # visualizer.plot_distribution('TotalPremium')
    # visualizer.plot_correlation_matrix()
    # etc.
    
    logger.info(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    pass
