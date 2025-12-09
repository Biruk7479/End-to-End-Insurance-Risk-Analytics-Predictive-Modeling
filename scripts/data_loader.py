"""
Data Loading Module
Handles loading and initial validation of insurance data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Class for loading and validating insurance data"""
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize DataLoader
        
        Args:
            data_path: Path to the data file
        """
        self.data_path = data_path
        self.df: Optional[pd.DataFrame] = None
        
    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from CSV or TXT file
        
        Args:
            file_path: Path to data file (overrides init path)
            
        Returns:
            Loaded DataFrame
        """
        path = file_path or self.data_path
        
        if path is None:
            raise ValueError("No data path provided")
            
        logger.info(f"Loading data from {path}")
        
        try:
            # Try loading with different separators
            if path.endswith('.txt'):
                self.df = pd.read_csv(path, sep='|', low_memory=False)
            else:
                self.df = pd.read_csv(path, low_memory=False)
                
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
            
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get basic information about the loaded data
        
        Returns:
            Dictionary with data information
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'duplicates': self.df.duplicated().sum()
        }
        
        return info
        
    def validate_required_columns(self, required_columns: list) -> bool:
        """
        Validate that required columns exist in the data
        
        Args:
            required_columns: List of required column names
            
        Returns:
            True if all columns exist
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        missing = set(required_columns) - set(self.df.columns)
        
        if missing:
            logger.warning(f"Missing required columns: {missing}")
            return False
            
        logger.info("All required columns present")
        return True
        
    def get_summary_stats(self) -> pd.DataFrame:
        """
        Get summary statistics for numerical columns
        
        Returns:
            DataFrame with summary statistics
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        return self.df.describe(include='all')
        
    def get_column_types(self) -> Dict[str, list]:
        """
        Categorize columns by data type
        
        Returns:
            Dictionary with columns categorized by type
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = self.df.select_dtypes(include=['datetime64']).columns.tolist()
        
        return {
            'numerical': numerical_cols,
            'categorical': categorical_cols,
            'datetime': datetime_cols
        }


def load_insurance_data(file_path: str) -> pd.DataFrame:
    """
    Convenience function to load insurance data
    
    Args:
        file_path: Path to data file
        
    Returns:
        Loaded DataFrame
    """
    loader = DataLoader(file_path)
    return loader.load_data()


if __name__ == "__main__":
    # Example usage
    # loader = DataLoader("data/insurance_data.txt")
    # df = loader.load_data()
    # print(loader.get_data_info())
    pass
