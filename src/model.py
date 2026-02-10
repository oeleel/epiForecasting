"""
XGBoost model wrapper for flu forecasting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import joblib
import json
import os
from datetime import datetime
from src import config


class FluForecastingModel:
    """XGBoost-based flu forecasting model"""

    def __init__(self, model_params: Optional[Dict] = None, use_monotonic: bool = True,
                 quantile: Optional[float] = None):
        """
        Initialize the flu forecasting model

        Args:
            model_params: XGBoost parameters (uses config defaults if None)
            use_monotonic: Whether to apply monotonic constraints from config
            quantile: If specified, use quantile regression with this quantile level (0-1).
                     None = standard squared error regression.
                     0.5 = median regression (replaces point prediction).
        """
        self.model_params = model_params or config.XGBOOST_PARAMS.copy()
        self.model = None
        self.feature_columns = None
        self.label_encoders = {}
        self.training_info = {}
        self.use_monotonic = use_monotonic
        self.monotonic_constraints = getattr(config, 'MONOTONIC_FEATURES', {})
        self.quantile = quantile

        # Configure for quantile regression if quantile is specified
        if self.quantile is not None:
            self.model_params = self.model_params.copy()
            self.model_params['objective'] = 'reg:quantileerror'
            self.model_params['quantile_alpha'] = self.quantile
        
    def prepare_features(self, data: pd.DataFrame, target_col: str = 'value') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for training/prediction
        
        Args:
            data: DataFrame with features
            target_col: Name of target column
            
        Returns:
            Tuple of (features_df, target_series)
        """
        # Separate features and target
        feature_cols = [col for col in data.columns if col not in ['date', 'location', target_col]]
        self.feature_columns = feature_cols
        
        X = data[feature_cols].copy()
        y = data[target_col].copy()
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                # Handle unseen categories during prediction
                X[col] = X[col].astype(str)
                
                # Replace unseen categories with the most frequent category from training
                unseen_mask = ~X[col].isin(self.label_encoders[col].classes_)
                if unseen_mask.any():
                    # Use the first class (most frequent) as default
                    default_value = self.label_encoders[col].classes_[0]
                    X.loc[unseen_mask, col] = default_value
                
                X[col] = self.label_encoders[col].transform(X[col])
        
        # Fill any remaining NaN values
        X = X.fillna(0)
        
        return X, y
    
    def _build_monotonic_constraints(self, feature_cols: List[str]) -> str:
        """
        Build monotonic constraints string for XGBoost based on feature columns.
        
        Args:
            feature_cols: List of feature column names
            
        Returns:
            Tuple of monotonic constraints (e.g., "(1,0,-1,0,...)")
        """
        constraints = []
        for col in feature_cols:
            # Get constraint from config (default 0 = no constraint)
            constraint = self.monotonic_constraints.get(col, 0)
            constraints.append(str(constraint))
        return "(" + ",".join(constraints) + ")"
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None) -> Dict:
        """
        Train the XGBoost model
        
        Args:
            X: Training features
            y: Training targets
            validation_data: Optional validation data tuple (X_val, y_val)
            
        Returns:
            Dictionary with training results
        """
        # Build monotonic constraints if enabled
        train_params = self.model_params.copy()
        if self.use_monotonic and self.monotonic_constraints:
            constraints_str = self._build_monotonic_constraints(list(X.columns))
            train_params['monotone_constraints'] = constraints_str
        
        # Initialize model
        self.model = xgb.XGBRegressor(**train_params)
        
        # Prepare validation data if provided
        eval_set = None
        if validation_data is not None:
            X_val, y_val = validation_data
            eval_set = [(X_val, y_val)]
        
        # Train model
        if eval_set is not None:
            # Use early stopping with validation data
            self.model.fit(
                X, y,
                eval_set=eval_set,
                verbose=False
            )
        else:
            # Remove early stopping if no validation data
            model_params_no_early_stop = self.model_params.copy()
            if 'early_stopping_rounds' in model_params_no_early_stop:
                del model_params_no_early_stop['early_stopping_rounds']
            
            # Reinitialize model without early stopping
            self.model = xgb.XGBRegressor(**model_params_no_early_stop)
            self.model.fit(X, y, verbose=False)
        
        # Store training info
        self.training_info = {
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'feature_names': list(X.columns),
            'training_date': datetime.now().isoformat(),
            'model_params': self.model_params.copy(),
            'quantile': self.quantile
        }
        
        # Calculate training metrics
        train_pred = self.model.predict(X)
        train_mae = np.mean(np.abs(train_pred - y))
        train_rmse = np.sqrt(np.mean((train_pred - y) ** 2))
        
        results = {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'n_features': X.shape[1],
            'n_samples': X.shape[0]
        }
        
        if validation_data is not None:
            val_pred = self.model.predict(X_val)
            val_mae = np.mean(np.abs(val_pred - y_val))
            val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
            results.update({
                'val_mae': val_mae,
                'val_rmse': val_rmse
            })
        
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure same feature columns as training
        X_pred = X[self.feature_columns].copy()
        
        # Handle categorical features
        categorical_cols = X_pred.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col in self.label_encoders:
                X_pred[col] = X_pred[col].astype(str)
                
                # Replace unseen categories with the most frequent category from training
                unseen_mask = ~X_pred[col].isin(self.label_encoders[col].classes_)
                if unseen_mask.any():
                    # Use the first class (most frequent) as default
                    default_value = self.label_encoders[col].classes_[0]
                    X_pred.loc[unseen_mask, col] = default_value
                
                X_pred[col] = self.label_encoders[col].transform(X_pred[col])
        
        # Fill NaN values
        X_pred = X_pred.fillna(0)
        
        return self.model.predict(X_pred)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model
        
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model must be trained before getting feature importance")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: str) -> str:
        """
        Save the trained model and metadata
        
        Args:
            filepath: Path to save the model
            
        Returns:
            Path to saved model file
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        model_file = filepath if filepath.endswith('.json') else f"{filepath}.json"
        self.model.save_model(model_file)
        
        # Save metadata
        metadata_file = model_file.replace('.json', '_metadata.json')
        metadata = {
            'feature_columns': self.feature_columns,
            'label_encoders': {col: encoder.classes_.tolist() for col, encoder in self.label_encoders.items()},
            'training_info': self.training_info,
            'model_params': self.model_params
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return model_file
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model and metadata
        
        Args:
            filepath: Path to the model file
        """
        # Load model
        model_file = filepath if filepath.endswith('.json') else f"{filepath}.json"
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_file)
        
        # Load metadata
        metadata_file = model_file.replace('.json', '_metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            self.feature_columns = metadata['feature_columns']
            self.training_info = metadata['training_info']
            self.model_params = metadata['model_params']
            
            # Reconstruct label encoders
            for col, classes in metadata['label_encoders'].items():
                encoder = LabelEncoder()
                encoder.classes_ = np.array(classes)
                self.label_encoders[col] = encoder
        else:
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    def get_model_info(self) -> Dict:
        """
        Get information about the trained model
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {'status': 'not_trained'}
        
        info = {
            'status': 'trained',
            'n_features': len(self.feature_columns) if self.feature_columns else 0,
            'feature_columns': self.feature_columns,
            'training_info': self.training_info,
            'model_params': self.model_params
        }
        
        return info


class ResidualCorrectionModel:
    """
    Two-stage model that trains a primary model and a secondary model on residuals.
    The secondary model learns to correct systematic errors from the primary model.
    """
    
    def __init__(self, primary_params: Optional[Dict] = None, 
                 residual_params: Optional[Dict] = None):
        """
        Initialize residual correction model.
        
        Args:
            primary_params: XGBoost parameters for primary model
            residual_params: XGBoost parameters for residual model (typically simpler)
        """
        self.primary_params = primary_params or config.XGBOOST_PARAMS.copy()
        
        # Residual model should be simpler to avoid overfitting to noise
        self.residual_params = residual_params or {
            'objective': 'reg:squarederror',
            'max_depth': 3,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        self.primary_model = FluForecastingModel(self.primary_params, use_monotonic=True)
        self.residual_model = FluForecastingModel(self.residual_params, use_monotonic=False)
        self.is_trained = False
        self.training_info = {}
    
    def train(self, X: pd.DataFrame, y: pd.Series,
              validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None) -> Dict:
        """
        Train both primary and residual models.
        
        Args:
            X: Training features
            y: Training targets
            validation_data: Optional validation data tuple (X_val, y_val)
            
        Returns:
            Dictionary with training results
        """
        print("Training primary model...")
        primary_results = self.primary_model.train(X, y, validation_data)
        
        # Get residuals from primary model
        print("Computing residuals...")
        train_predictions = self.primary_model.predict(X)
        residuals = y - train_predictions
        
        # Train residual model
        print("Training residual correction model...")
        
        # For residual model, we don't use validation early stopping
        # since we want it to learn the systematic errors
        residual_results = self.residual_model.train(X, residuals, None)
        
        # Compute combined model metrics
        combined_predictions = train_predictions + self.residual_model.predict(X)
        combined_mae = np.mean(np.abs(y - combined_predictions))
        combined_rmse = np.sqrt(np.mean((y - combined_predictions) ** 2))
        
        self.training_info = {
            'training_date': datetime.now().isoformat(),
            'primary_train_mae': primary_results.get('train_mae'),
            'primary_train_rmse': primary_results.get('train_rmse'),
            'residual_train_mae': residual_results.get('train_mae'),
            'combined_train_mae': combined_mae,
            'combined_train_rmse': combined_rmse,
            'improvement': primary_results.get('train_mae', 0) - combined_mae
        }
        
        # Evaluate on validation if provided
        if validation_data is not None:
            X_val, y_val = validation_data
            val_pred_primary = self.primary_model.predict(X_val)
            val_pred_residual = self.residual_model.predict(X_val)
            val_pred_combined = val_pred_primary + val_pred_residual
            
            val_mae_primary = np.mean(np.abs(y_val - val_pred_primary))
            val_mae_combined = np.mean(np.abs(y_val - val_pred_combined))
            
            self.training_info['val_mae_primary'] = val_mae_primary
            self.training_info['val_mae_combined'] = val_mae_combined
            self.training_info['val_improvement'] = val_mae_primary - val_mae_combined
            
            print(f"Validation MAE - Primary: {val_mae_primary:.2f}, Combined: {val_mae_combined:.2f}")
        
        self.is_trained = True
        print(f"Training complete. Improvement: {self.training_info['improvement']:.2f}")
        
        return self.training_info
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using combined primary + residual correction.
        
        Args:
            X: Features for prediction
            
        Returns:
            Combined predictions array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        primary_pred = self.primary_model.predict(X)
        residual_correction = self.residual_model.predict(X)
        
        # Clip residual correction to avoid extreme adjustments
        max_correction = np.abs(primary_pred) * 0.5  # Max 50% correction
        residual_correction = np.clip(residual_correction, -max_correction, max_correction)
        
        return primary_pred + residual_correction
    
    def get_feature_importance(self) -> Dict[str, pd.DataFrame]:
        """
        Get feature importance from both models.
        
        Returns:
            Dictionary with 'primary' and 'residual' importance DataFrames
        """
        return {
            'primary': self.primary_model.get_feature_importance(),
            'residual': self.residual_model.get_feature_importance()
        }
    
    def save_model(self, filepath: str) -> str:
        """
        Save both models and metadata.
        
        Args:
            filepath: Base path for saving
            
        Returns:
            Path to saved model directory
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create directory
        model_dir = filepath if not filepath.endswith('.json') else filepath.replace('.json', '')
        os.makedirs(model_dir, exist_ok=True)
        
        # Save primary and residual models
        self.primary_model.save_model(os.path.join(model_dir, 'primary_model'))
        self.residual_model.save_model(os.path.join(model_dir, 'residual_model'))
        
        # Save combined metadata
        metadata = {
            'training_info': self.training_info,
            'primary_params': self.primary_params,
            'residual_params': self.residual_params
        }
        
        with open(os.path.join(model_dir, 'combined_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return model_dir
    
    def load_model(self, filepath: str) -> None:
        """
        Load both models and metadata.
        
        Args:
            filepath: Path to saved model directory
        """
        model_dir = filepath if not filepath.endswith('.json') else filepath.replace('.json', '')
        
        # Load primary and residual models
        self.primary_model.load_model(os.path.join(model_dir, 'primary_model'))
        self.residual_model.load_model(os.path.join(model_dir, 'residual_model'))
        
        # Load combined metadata
        metadata_path = os.path.join(model_dir, 'combined_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.training_info = metadata.get('training_info', {})
            self.primary_params = metadata.get('primary_params', {})
            self.residual_params = metadata.get('residual_params', {})
        
        self.is_trained = True
