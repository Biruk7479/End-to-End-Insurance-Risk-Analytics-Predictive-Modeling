"""
Data Preprocessing Module
Handles data cleaning, transformation, and feature engineering
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Class for preprocessing insurance data"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize DataPreprocessor
        
        Args:
            df: Input DataFrame
        """
        self.df = df.copy()
        self.original_shape = df.shape
        
    def handle_missing_values(self, strategy: str = 'drop', 
                              threshold: float = 0.5) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            strategy: 'drop', 'fill_mean', 'fill_median', 'fill_mode'
            threshold: For 'drop' strategy, drop columns with > threshold missing
            
        Returns:
            DataFrame with missing values handled
        """
        logger.info(f"Handling missing values with strategy: {strategy}")
        
        if strategy == 'drop':
            # Drop columns with too many missing values
            missing_pct = self.df.isnull().sum() / len(self.df)
            cols_to_drop = missing_pct[missing_pct > threshold].index
            logger.info(f"Dropping {len(cols_to_drop)} columns with >{threshold*100}% missing")
            self.df = self.df.drop(columns=cols_to_drop)
            
            # Drop rows with any remaining missing values
            before = len(self.df)
            self.df = self.df.dropna()
            logger.info(f"Dropped {before - len(self.df)} rows with missing values")
            
        elif strategy == 'fill_mean':
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numerical_cols] = self.df[numerical_cols].fillna(
                self.df[numerical_cols].mean()
            )
            
        elif strategy == 'fill_median':
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numerical_cols] = self.df[numerical_cols].fillna(
                self.df[numerical_cols].median()
            )
            
        elif strategy == 'fill_mode':
            for col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0] 
                                                    if not self.df[col].mode().empty 
                                                    else self.df[col])
        
        return self.df
        
    def convert_data_types(self) -> pd.DataFrame:
        """
        Convert columns to appropriate data types
        
        Returns:
            DataFrame with converted types
        """
        logger.info("Converting data types")
        
        # Convert date columns
        date_columns = ['TransactionMonth', 'VehicleIntroDate']
        for col in date_columns:
            if col in self.df.columns:
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    logger.info(f"Converted {col} to datetime")
                except Exception as e:
                    logger.warning(f"Could not convert {col} to datetime: {e}")
        
        # Convert boolean-like columns
        boolean_cols = ['IsVATRegistered', 'CrossBorder', 'NewVehicle', 
                       'WrittenOff', 'Rebuilt', 'Converted']
        for col in boolean_cols:
            if col in self.df.columns:
                try:
                    self.df[col] = self.df[col].astype(bool)
                    logger.info(f"Converted {col} to boolean")
                except Exception as e:
                    logger.warning(f"Could not convert {col} to boolean: {e}")
        
        return self.df
        
    def remove_duplicates(self) -> pd.DataFrame:
        """
        Remove duplicate rows
        
        Returns:
            DataFrame without duplicates
        """
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = before - len(self.df)
        
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
        else:
            logger.info("No duplicate rows found")
            
        return self.df
        
    def handle_outliers(self, columns: Optional[List[str]] = None, 
                       method: str = 'iqr', threshold: float = 3.0) -> pd.DataFrame:
        """
        Handle outliers in numerical columns
        
        Args:
            columns: List of columns to check. If None, check all numerical
            method: 'iqr' or 'zscore'
            threshold: IQR multiplier or Z-score threshold
            
        Returns:
            DataFrame with outliers handled
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
        logger.info(f"Handling outliers using {method} method")
        
        for col in columns:
            if col not in self.df.columns:
                continue
                
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = ((self.df[col] < lower_bound) | 
                           (self.df[col] > upper_bound)).sum()
                
                if outliers > 0:
                    logger.info(f"{col}: {outliers} outliers detected")
                    # Cap outliers instead of removing
                    self.df[col] = self.df[col].clip(lower_bound, upper_bound)
                    
            elif method == 'zscore':
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / 
                                 self.df[col].std())
                outliers = (z_scores > threshold).sum()
                
                if outliers > 0:
                    logger.info(f"{col}: {outliers} outliers detected")
                    
        return self.df
        
    def create_features(self) -> pd.DataFrame:
        """
        Create new features for analysis
        
        Returns:
            DataFrame with new features
        """
        logger.info("Creating new features")
        
        # Loss Ratio
        if 'TotalClaims' in self.df.columns and 'TotalPremium' in self.df.columns:
            self.df['LossRatio'] = np.where(
                self.df['TotalPremium'] > 0,
                self.df['TotalClaims'] / self.df['TotalPremium'],
                0
            )
            logger.info("Created LossRatio feature")
        
        # Profit Margin
        if 'TotalClaims' in self.df.columns and 'TotalPremium' in self.df.columns:
            self.df['ProfitMargin'] = self.df['TotalPremium'] - self.df['TotalClaims']
            logger.info("Created ProfitMargin feature")
        
        # Claim Flag (binary: has claim or not)
        if 'TotalClaims' in self.df.columns:
            self.df['HasClaim'] = (self.df['TotalClaims'] > 0).astype(int)
            logger.info("Created HasClaim feature")
        
        # Vehicle Age
        if 'RegistrationYear' in self.df.columns:
            current_year = pd.Timestamp.now().year
            self.df['VehicleAge'] = current_year - self.df['RegistrationYear']
            logger.info("Created VehicleAge feature")
        
        return self.df
        
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get summary of preprocessing steps
        
        Returns:
            Dictionary with preprocessing summary
        """
        return {
            'original_shape': self.original_shape,
            'current_shape': self.df.shape,
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'columns_removed': self.original_shape[1] - self.df.shape[1],
            'missing_values': self.df.isnull().sum().sum()
        }


def preprocess_insurance_data(df: pd.DataFrame, 
                              handle_missing: bool = True,
                              remove_dups: bool = True,
                              create_feats: bool = True) -> pd.DataFrame:
    """
    Convenience function to preprocess insurance data
    
    Args:
        df: Input DataFrame
        handle_missing: Whether to handle missing values
        remove_dups: Whether to remove duplicates
        create_feats: Whether to create new features
        
    Returns:
        Preprocessed DataFrame
    """
    preprocessor = DataPreprocessor(df)
    
    if remove_dups:
        preprocessor.remove_duplicates()
        
    preprocessor.convert_data_types()
    
    if handle_missing:
        preprocessor.handle_missing_values(strategy='drop', threshold=0.5)
        
    if create_feats:
        preprocessor.create_features()
        
    logger.info("Preprocessing complete")
    logger.info(f"Summary: {preprocessor.get_preprocessing_summary()}")
    
    return preprocessor.df


if __name__ == "__main__":
    # Example usage
    pass
