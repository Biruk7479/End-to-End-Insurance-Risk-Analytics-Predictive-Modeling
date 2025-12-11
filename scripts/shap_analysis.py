"""
SHAP (SHapley Additive exPlanations) Analysis Module
Provides model interpretability using SHAP values
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from typing import Optional, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """Class for SHAP-based model interpretability analysis"""
    
    def __init__(self, model, model_type: str = 'tree'):
        """
        Initialize SHAP Analyzer
        
        Args:
            model: Trained model (tree-based or linear)
            model_type: Type of model ('tree', 'linear', or 'kernel')
        """
        self.model = model
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        
    def create_explainer(self, X_train: pd.DataFrame):
        """
        Create SHAP explainer based on model type
        
        Args:
            X_train: Training data for background distribution
        """
        logger.info(f"Creating {self.model_type} SHAP explainer...")
        
        if self.model_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_type == 'linear':
            self.explainer = shap.LinearExplainer(self.model, X_train)
        else:  # kernel
            # Use a subset for computational efficiency
            background = shap.sample(X_train, min(100, len(X_train)))
            self.explainer = shap.KernelExplainer(self.model.predict, background)
        
        logger.info("SHAP explainer created successfully")
        
    def calculate_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate SHAP values for given data
        
        Args:
            X: Data to explain
            
        Returns:
            SHAP values array
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer first.")
        
        logger.info(f"Calculating SHAP values for {len(X)} samples...")
        
        self.feature_names = X.columns.tolist() if hasattr(X, 'columns') else None
        self.shap_values = self.explainer.shap_values(X)
        
        logger.info("SHAP values calculated successfully")
        return self.shap_values
    
    def plot_summary(self, X: Optional[pd.DataFrame] = None,
                    max_display: int = 10,
                    save_path: Optional[str] = None):
        """
        Create SHAP summary plot showing feature importance
        
        Args:
            X: Data used for SHAP calculation
            max_display: Maximum number of features to display
            save_path: Path to save plot (if None, display only)
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call calculate_shap_values first.")
        
        plt.figure(figsize=(12, 8))
        
        shap.summary_plot(
            self.shap_values,
            X,
            max_display=max_display,
            show=False
        )
        
        plt.title("SHAP Feature Importance Summary", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Summary plot saved to {save_path}")
        
        plt.show()
    
    def plot_bar(self, X: Optional[pd.DataFrame] = None,
                max_display: int = 10,
                save_path: Optional[str] = None):
        """
        Create SHAP bar plot showing mean absolute SHAP values
        
        Args:
            X: Data used for SHAP calculation
            max_display: Maximum number of features to display
            save_path: Path to save plot
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated.")
        
        plt.figure(figsize=(12, 8))
        
        shap.summary_plot(
            self.shap_values,
            X,
            plot_type="bar",
            max_display=max_display,
            show=False
        )
        
        plt.title("SHAP Feature Importance (Mean |SHAP value|)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Bar plot saved to {save_path}")
        
        plt.show()
    
    def plot_waterfall(self, X: pd.DataFrame, 
                      sample_index: int = 0,
                      save_path: Optional[str] = None):
        """
        Create waterfall plot for a single prediction
        
        Args:
            X: Data used for SHAP calculation
            sample_index: Index of sample to explain
            save_path: Path to save plot
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated.")
        
        plt.figure(figsize=(12, 8))
        
        # Create explanation object
        if self.explainer is not None:
            explanation = shap.Explanation(
                values=self.shap_values[sample_index],
                base_values=self.explainer.expected_value,
                data=X.iloc[sample_index].values,
                feature_names=X.columns.tolist()
            )
            
            shap.waterfall_plot(explanation, show=False)
        
        plt.title(f"SHAP Waterfall Plot - Sample {sample_index}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Waterfall plot saved to {save_path}")
        
        plt.show()
    
    def plot_force(self, X: pd.DataFrame,
                  sample_index: int = 0,
                  matplotlib: bool = True,
                  save_path: Optional[str] = None):
        """
        Create force plot for a single prediction
        
        Args:
            X: Data used for SHAP calculation
            sample_index: Index of sample to explain
            matplotlib: Use matplotlib (True) or JS visualization (False)
            save_path: Path to save plot
        """
        if self.shap_values is None or self.explainer is None:
            raise ValueError("SHAP values and explainer required.")
        
        if matplotlib:
            plt.figure(figsize=(20, 3))
            shap.force_plot(
                self.explainer.expected_value,
                self.shap_values[sample_index],
                X.iloc[sample_index],
                matplotlib=True,
                show=False
            )
            
            plt.title(f"SHAP Force Plot - Sample {sample_index}", fontsize=12, fontweight='bold')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Force plot saved to {save_path}")
            
            plt.show()
        else:
            # Interactive JS visualization
            force_plot = shap.force_plot(
                self.explainer.expected_value,
                self.shap_values[sample_index],
                X.iloc[sample_index]
            )
            return force_plot
    
    def plot_dependence(self, feature: str,
                       X: pd.DataFrame,
                       interaction_feature: Optional[str] = None,
                       save_path: Optional[str] = None):
        """
        Create dependence plot for a specific feature
        
        Args:
            feature: Feature name to plot
            X: Data used for SHAP calculation
            interaction_feature: Feature to use for coloring (auto if None)
            save_path: Path to save plot
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated.")
        
        plt.figure(figsize=(10, 6))
        
        shap.dependence_plot(
            feature,
            self.shap_values,
            X,
            interaction_index=interaction_feature,
            show=False
        )
        
        plt.title(f"SHAP Dependence Plot - {feature}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Dependence plot saved to {save_path}")
        
        plt.show()
    
    def get_feature_importance_df(self, X: pd.DataFrame,
                                 top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance as DataFrame
        
        Args:
            X: Data used for SHAP calculation
            top_n: Number of top features to return
            
        Returns:
            DataFrame with features and their mean absolute SHAP values
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated.")
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        # Create DataFrame
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)
        
        return feature_importance.head(top_n)
    
    def analyze_top_features(self, X: pd.DataFrame,
                            top_n: int = 10) -> Dict:
        """
        Comprehensive analysis of top features
        
        Args:
            X: Data used for SHAP calculation
            top_n: Number of top features to analyze
            
        Returns:
            Dictionary with analysis results
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated.")
        
        # Get top features
        feature_importance = self.get_feature_importance_df(X, top_n)
        
        results = {
            'feature_importance': feature_importance,
            'top_features': feature_importance['feature'].tolist(),
            'shap_statistics': {}
        }
        
        # Calculate statistics for each top feature
        for feature in results['top_features']:
            feature_idx = X.columns.get_loc(feature)
            feature_shap = self.shap_values[:, feature_idx]
            
            results['shap_statistics'][feature] = {
                'mean': float(np.mean(feature_shap)),
                'std': float(np.std(feature_shap)),
                'min': float(np.min(feature_shap)),
                'max': float(np.max(feature_shap)),
                'median': float(np.median(feature_shap))
            }
        
        return results
    
    def generate_interpretation_report(self, X: pd.DataFrame,
                                      top_n: int = 10) -> str:
        """
        Generate text report interpreting SHAP results
        
        Args:
            X: Data used for SHAP calculation
            top_n: Number of top features to analyze
            
        Returns:
            String containing interpretation report
        """
        analysis = self.analyze_top_features(X, top_n)
        
        report = "# SHAP Analysis Interpretation Report\n\n"
        report += "## Top Feature Importance\n\n"
        
        for idx, row in analysis['feature_importance'].iterrows():
            feature = row['feature']
            importance = row['mean_abs_shap']
            stats = analysis['shap_statistics'][feature]
            
            report += f"### {idx + 1}. {feature}\n"
            report += f"- Mean |SHAP|: {importance:.4f}\n"
            report += f"- Average impact: {stats['mean']:.4f}\n"
            report += f"- Impact range: [{stats['min']:.4f}, {stats['max']:.4f}]\n"
            
            # Interpret direction
            if stats['mean'] > 0:
                report += f"- Overall effect: Increases prediction (positive correlation)\n"
            else:
                report += f"- Overall effect: Decreases prediction (negative correlation)\n"
            
            report += "\n"
        
        return report


def perform_shap_analysis(model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                         model_type: str = 'tree',
                         top_n: int = 10,
                         save_plots: bool = True,
                         output_dir: str = '../reports/figures') -> Dict:
    """
    Perform complete SHAP analysis on a model
    
    Args:
        model: Trained model
        X_train: Training data
        X_test: Test data
        model_type: Type of model ('tree', 'linear', or 'kernel')
        top_n: Number of top features to analyze
        save_plots: Whether to save plots
        output_dir: Directory to save plots
        
    Returns:
        Dictionary with analysis results
    """
    # Initialize analyzer
    analyzer = SHAPAnalyzer(model, model_type)
    
    # Create explainer
    analyzer.create_explainer(X_train)
    
    # Calculate SHAP values
    analyzer.calculate_shap_values(X_test)
    
    # Generate plots
    if save_plots:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        analyzer.plot_summary(X_test, max_display=top_n,
                            save_path=f"{output_dir}/shap_summary.png")
        analyzer.plot_bar(X_test, max_display=top_n,
                        save_path=f"{output_dir}/shap_bar.png")
    else:
        analyzer.plot_summary(X_test, max_display=top_n)
        analyzer.plot_bar(X_test, max_display=top_n)
    
    # Get analysis results
    analysis = analyzer.analyze_top_features(X_test, top_n)
    analysis['report'] = analyzer.generate_interpretation_report(X_test, top_n)
    
    return analysis


if __name__ == "__main__":
    # Example usage
    pass
