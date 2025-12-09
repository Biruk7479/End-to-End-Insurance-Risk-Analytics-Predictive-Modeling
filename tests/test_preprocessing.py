"""
Unit tests for preprocessing module
"""

import pytest
import pandas as pd
import numpy as np
from scripts.preprocessing import DataPreprocessor, preprocess_insurance_data


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class"""
    
    def test_preprocessor_init(self):
        """Test DataPreprocessor initialization"""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        preprocessor = DataPreprocessor(df)
        assert preprocessor.df.shape == (3, 2)
        assert preprocessor.original_shape == (3, 2)
        
    def test_remove_duplicates(self):
        """Test duplicate removal"""
        df = pd.DataFrame({
            'A': [1, 2, 2, 3],
            'B': [4, 5, 5, 6]
        })
        preprocessor = DataPreprocessor(df)
        result = preprocessor.remove_duplicates()
        assert len(result) == 3
        
    def test_create_features(self):
        """Test feature creation"""
        df = pd.DataFrame({
            'TotalPremium': [1000, 2000, 3000],
            'TotalClaims': [500, 1000, 0],
            'RegistrationYear': [2015, 2018, 2020]
        })
        preprocessor = DataPreprocessor(df)
        result = preprocessor.create_features()
        
        assert 'LossRatio' in result.columns
        assert 'ProfitMargin' in result.columns
        assert 'HasClaim' in result.columns
        assert 'VehicleAge' in result.columns


if __name__ == "__main__":
    pytest.main([__file__])
