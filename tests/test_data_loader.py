"""
Unit tests for data_loader module
"""

import pytest
import pandas as pd
import numpy as np
from scripts.data_loader import DataLoader, load_insurance_data


class TestDataLoader:
    """Test cases for DataLoader class"""
    
    def test_data_loader_init(self):
        """Test DataLoader initialization"""
        loader = DataLoader("test_path.csv")
        assert loader.data_path == "test_path.csv"
        assert loader.df is None
        
    def test_get_column_types_empty(self):
        """Test get_column_types with no data loaded"""
        loader = DataLoader()
        with pytest.raises(ValueError):
            loader.get_column_types()
            
    def test_validate_required_columns_no_data(self):
        """Test validate_required_columns with no data loaded"""
        loader = DataLoader()
        with pytest.raises(ValueError):
            loader.validate_required_columns(['col1', 'col2'])


def test_load_insurance_data():
    """Test convenience function"""
    # This would require a test data file
    pass


if __name__ == "__main__":
    pytest.main([__file__])
