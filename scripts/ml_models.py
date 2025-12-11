"""
Machine Learning Modeling Module for Insurance Analytics
Implements predictive models for claim severity and premium optimization
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
import joblib
from typing import Dict, Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InsuranceModel:
    """Base class for insurance predictive models"""
    
    def __init__(self, model_type: str = 'regression'):
        """
        Initialize InsuranceModel
        
        Args:
            model_type: Type of model ('regression' or 'classification')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.target_name = None
        
    def prepare_features(self, df: pd.DataFrame, 
                        target_col: str,
                        categorical_cols: Optional[List[str]] = None,
                        numerical_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for modeling
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            categorical_cols: List of categorical columns
            numerical_cols: List of numerical columns
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        df_model = df.copy()
        
        # Auto-detect if not specified
        if categorical_cols is None:
            categorical_cols = df_model.select_dtypes(include=['object']).columns.tolist()
        if numerical_cols is None:
            numerical_cols = df_model.select_dtypes(include=[np.number]).columns.tolist()
            numerical_cols = [col for col in numerical_cols if col != target_col]
        
        # Remove target from feature lists
        categorical_cols = [col for col in categorical_cols if col != target_col and col in df_model.columns]
        numerical_cols = [col for col in numerical_cols if col != target_col and col in df_model.columns]
        
        # Handle missing values
        for col in numerical_cols:
            df_model[col].fillna(df_model[col].median(), inplace=True)
        
        for col in categorical_cols:
            df_model[col].fillna('Unknown', inplace=True)
        
        # Encode categorical variables
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_model[col] = self.label_encoders[col].fit_transform(df_model[col].astype(str))
            else:
                df_model[col] = self.label_encoders[col].transform(df_model[col].astype(str))
        
        # Select features
        feature_cols = numerical_cols + categorical_cols
        X = df_model[feature_cols]
        y = df_model[target_col]
        
        self.feature_names = feature_cols
        self.target_name = target_col
        
        return X, y
    
    def train_test_split_data(self, X: pd.DataFrame, y: pd.Series, 
                              test_size: float = 0.2, 
                              random_state: int = 42) -> Tuple:
        """
        Split data into train and test sets
        
        Args:
            X: Features
            y: Target
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple:
        """
        Scale features using StandardScaler
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            Tuple of (X_train_scaled, X_test_scaled)
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Evaluate regression model
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with evaluation metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
    
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray,
                               y_pred_proba: Optional[np.ndarray] = None) -> Dict:
        """
        Evaluate classification model
        
        Args:
            y_true: True values
            y_pred: Predicted values
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = {
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
        
        if y_pred_proba is not None:
            results['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        return results
    
    def save_model(self, filepath: str):
        """Save model to file"""
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file"""
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")


class ClaimSeverityModel(InsuranceModel):
    """Model for predicting claim severity (Total Claims amount)"""
    
    def __init__(self):
        super().__init__(model_type='regression')
        self.models = {}
        
    def train_linear_regression(self, X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
        """Train Linear Regression model"""
        logger.info("Training Linear Regression...")
        model = LinearRegression()
        model.fit(X_train, y_train)
        self.models['linear_regression'] = model
        return model
    
    def train_decision_tree(self, X_train: np.ndarray, y_train: np.ndarray,
                           max_depth: int = 10) -> DecisionTreeRegressor:
        """Train Decision Tree model"""
        logger.info("Training Decision Tree...")
        model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        self.models['decision_tree'] = model
        return model
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           n_estimators: int = 100, max_depth: int = 15) -> RandomForestRegressor:
        """Train Random Forest model"""
        logger.info("Training Random Forest...")
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        return model
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                     n_estimators: int = 100, max_depth: int = 6) -> xgb.XGBRegressor:
        """Train XGBoost model"""
        logger.info("Training XGBoost...")
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['xgboost'] = model
        return model
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Train all models and return them"""
        self.train_linear_regression(X_train, y_train)
        self.train_decision_tree(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        return self.models
    
    def evaluate_all_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate all trained models"""
        results = {}
        
        for model_name, model in self.models.items():
            y_pred = model.predict(X_test)
            metrics = self.evaluate_regression(y_test, y_pred)
            results[model_name] = metrics
            logger.info(f"\n{model_name} - RMSE: {metrics['RMSE']:.2f}, R2: {metrics['R2']:.4f}")
        
        return results
    
    def get_feature_importance(self, model_name: str = 'random_forest',
                              top_n: int = 10) -> pd.DataFrame:
        """Get feature importance from tree-based model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return feature_importance.head(top_n)
        else:
            raise ValueError(f"Model {model_name} does not have feature_importances_")


class PremiumOptimizationModel(InsuranceModel):
    """Model for predicting optimal premium"""
    
    def __init__(self):
        super().__init__(model_type='regression')
        self.claim_probability_model = None
        self.claim_severity_model = None
        self.models = {}
        
    def train_claim_probability_model(self, X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
        """Train model to predict probability of claim occurrence"""
        logger.info("Training Claim Probability Model...")
        
        # Convert target to binary (has claim vs no claim)
        y_binary = (y_train > 0).astype(int)
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_binary)
        
        self.claim_probability_model = model
        return model
    
    def train_premium_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Train models to predict premium"""
        logger.info("Training Premium Prediction Models...")
        
        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        self.models['linear_regression'] = lr_model
        
        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model
        
        return self.models
    
    def predict_risk_based_premium(self, X: np.ndarray,
                                   claim_severity_prediction: np.ndarray,
                                   expense_loading: float = 0.15,
                                   profit_margin: float = 0.10) -> np.ndarray:
        """
        Calculate risk-based premium using formula:
        Premium = (P(Claim) × Expected Claim Severity) × (1 + Expense + Profit)
        
        Args:
            X: Features
            claim_severity_prediction: Predicted claim amounts
            expense_loading: Expense loading factor (default 15%)
            profit_margin: Profit margin factor (default 10%)
            
        Returns:
            Array of predicted premiums
        """
        if self.claim_probability_model is None:
            raise ValueError("Claim probability model not trained")
        
        # Get probability of claim
        claim_prob = self.claim_probability_model.predict_proba(X)[:, 1]
        
        # Calculate base premium
        expected_claims = claim_prob * claim_severity_prediction
        
        # Add expense loading and profit margin
        premium = expected_claims * (1 + expense_loading + profit_margin)
        
        return premium


def build_claim_severity_model(df: pd.DataFrame, 
                               target_col: str = 'TotalClaims',
                               feature_cols: Optional[List[str]] = None) -> Tuple[ClaimSeverityModel, Dict]:
    """
    Build and evaluate claim severity prediction model
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        feature_cols: List of feature columns (if None, auto-select)
        
    Returns:
        Tuple of (trained model, evaluation results)
    """
    # Filter for policies with claims
    df_with_claims = df[df[target_col] > 0].copy()
    
    logger.info(f"Building model on {len(df_with_claims)} policies with claims")
    
    # Initialize model
    model = ClaimSeverityModel()
    
    # Prepare features
    if feature_cols is None:
        # Auto-select relevant features
        numerical_features = ['SumInsured', 'CalculatedPremiumPerTerm', 'Kilowatts',
                            'Cubiccapacity', 'VehicleAge', 'RegistrationYear']
        categorical_features = ['Province', 'VehicleType', 'Make', 'CoverType',
                              'Gender', 'MaritalStatus']
        
        # Filter existing columns
        numerical_features = [col for col in numerical_features if col in df_with_claims.columns]
        categorical_features = [col for col in categorical_features if col in df_with_claims.columns]
    else:
        numerical_features = None
        categorical_features = None
    
    X, y = model.prepare_features(df_with_claims, target_col, categorical_features, numerical_features)
    
    # Split data
    X_train, X_test, y_train, y_test = model.train_test_split_data(X, y)
    
    # Scale features
    X_train_scaled, X_test_scaled = model.scale_features(X_train, X_test)
    
    # Train all models
    model.train_all_models(X_train_scaled, y_train)
    
    # Evaluate
    results = model.evaluate_all_models(X_test_scaled, y_test)
    
    return model, results


if __name__ == "__main__":
    # Example usage
    pass
