import pandas as pd
import numpy as np
import time
import traceback
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, mean_squared_error, mean_absolute_error,
    r2_score, precision_score, recall_score, f1_score
)
from xgboost import XGBRegressor, XGBClassifier
import joblib
import os
from typing import Dict, List, Tuple, Any
from imagekit_service import imagekit_service

logger = logging.getLogger(__name__)

class MLService:
    def __init__(self):
        self.models = {}
        self.label_encoders = {}
        self.feature_columns = []
        self.trained_models = {}
        
        # Model registries
        self.CLASSIFIER_REGISTRY = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'knn': KNeighborsClassifier(n_neighbors=5),
            'naive_bayes': GaussianNB(),
            'random_forest_classifier': RandomForestClassifier(n_estimators=100, random_state=42),
            'xgboost_classifier': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
            'lda_classifier': LinearDiscriminantAnalysis()
        }
        
        self.REGRESSOR_REGISTRY = {
            'linear_regression': LinearRegression(),
            'xgboost_regressor': XGBRegressor(random_state=42),
            'decision_tree_regressor': DecisionTreeRegressor(random_state=42),
            'random_forest_regressor': RandomForestRegressor(n_estimators=100, random_state=42)
        }
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare data for training"""
        # Use sensor_readings as features (assuming it's a list column)
        if 'sensor_readings' in df.columns:
            # Convert sensor_readings list to individual columns
            sensor_data = df['sensor_readings'].apply(pd.Series)
            sensor_data.columns = [f'Sensor_{i+1}' for i in range(len(sensor_data.columns))]
            X = pd.concat([df[['Time']], sensor_data], axis=1)
        else:
            # Fallback to Time only if no sensor_readings
            X = df[['Time']]
        
        # Prepare targets
        le = LabelEncoder()
        y_class = le.fit_transform(df['Type'])
        y_reg = df['Concentration']
        
        self.label_encoders['type'] = le
        self.feature_columns = list(X.columns)
        
        return X, y_class, y_reg
    
    def train_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train all classification and regression models"""
        try:
            logger.info("Starting model training")
            
            # Prepare data
            X, y_class, y_reg = self.prepare_data(df)
            
            # Split data
            X_train, X_test, y_class_train, y_class_test = train_test_split(
                X, y_class, test_size=0.2, random_state=42, stratify=y_class
            )
            X_train, X_test, y_reg_train, y_reg_test = train_test_split(
                X, y_reg, test_size=0.2, random_state=42
            )
            
            results = {
                'classifiers': {},
                'regressors': {},
                'training_time': 0,
                'feature_columns': self.feature_columns,
                'unique_types': self.label_encoders['type'].classes_.tolist()
            }
            
            start_time = time.time()
            
            # Train classifiers
            for name, model in self.CLASSIFIER_REGISTRY.items():
                try:
                    logger.info(f"Training {name}")
                    model_start = time.time()
                    
                    # Train model
                    model.fit(X_train, y_class_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_class_test, y_pred)
                    precision = precision_score(y_class_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_class_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_class_test, y_pred, average='weighted', zero_division=0)
                    
                    training_time = time.time() - model_start
                    
                    results['classifiers'][name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'training_time': training_time
                    }
                    
                    # Store trained model in memory
                    self.trained_models[name] = model
                    
                    # Save model to ImageKit
                    model_url = imagekit_service.save_model(name, model, 'classification')
                    if model_url:
                        logger.info(f"Model {name} saved to ImageKit: {model_url}")
                    else:
                        logger.warning(f"Failed to save model {name} to ImageKit")
                    
                except Exception as e:
                    logger.error(f"Error training {name}: {str(e)}")
                    results['classifiers'][name] = {'error': str(e)}
            
            # Train regressors
            for name, model in self.REGRESSOR_REGISTRY.items():
                try:
                    logger.info(f"Training {name}")
                    model_start = time.time()
                    
                    # Train model
                    model.fit(X_train, y_reg_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_reg_test, y_pred)
                    mae = mean_absolute_error(y_reg_test, y_pred)
                    r2 = r2_score(y_reg_test, y_pred)
                    
                    training_time = time.time() - model_start
                    
                    results['regressors'][name] = {
                        'mse': mse,
                        'mae': mae,
                        'r2_score': r2,
                        'training_time': training_time
                    }
                    
                    # Store trained model in memory
                    self.trained_models[name] = model
                    
                    # Save model to ImageKit
                    model_url = imagekit_service.save_model(name, model, 'regression')
                    if model_url:
                        logger.info(f"Model {name} saved to ImageKit: {model_url}")
                    else:
                        logger.warning(f"Failed to save model {name} to ImageKit")
                    
                except Exception as e:
                    logger.error(f"Error training {name}: {str(e)}")
                    results['regressors'][name] = {'error': str(e)}
            
            results['training_time'] = time.time() - start_time
            logger.info(f"Training completed in {results['training_time']:.2f} seconds")
            
            return results
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            logger.error(traceback.format_exc())
            raise e
    
    def predict(self, data: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """Make predictions using a trained model"""
        try:
            # Check if model is in memory, if not try to load from ImageKit
            if model_name not in self.trained_models:
                # Determine model type
                model_type = 'classification' if model_name in self.CLASSIFIER_REGISTRY else 'regression'
                
                # Try to load from ImageKit
                if not self.load_model_from_storage(model_name, model_type):
                    raise ValueError(f"Model {model_name} not found in memory or ImageKit storage")
            
            # Prepare input data
            if 'sensor_readings' in data:
                sensor_data = pd.Series(data['sensor_readings'])
                X = pd.DataFrame({
                    'Time': [data['Time']],
                    **{f'Sensor_{i+1}': [val] for i, val in enumerate(sensor_data)}
                })
            else:
                X = pd.DataFrame({'Time': [data['Time']]})
            
            # Ensure columns match training data
            for col in self.feature_columns:
                if col not in X.columns:
                    X[col] = 0
            
            X = X[self.feature_columns]
            
            model = self.trained_models[model_name]
            
            if model_name in self.CLASSIFIER_REGISTRY:
                # Classification prediction
                y_pred = model.predict(X)[0]
                predicted_type = self.label_encoders['type'].inverse_transform([y_pred])[0]
                
                # Get prediction probabilities for confidence
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)[0]
                    confidence = float(np.max(proba))
                else:
                    confidence = None
                
                return {
                    'predicted_type': predicted_type,
                    'confidence': confidence,
                    'model_type': 'classification'
                }
            else:
                # Regression prediction
                y_pred = model.predict(X)[0]
                return {
                    'predicted_concentration': float(y_pred),
                    'model_type': 'regression'
                }
                
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise e
    
    def load_model_from_storage(self, model_name: str, model_type: str) -> bool:
        """Load a model from ImageKit storage"""
        try:
            model = imagekit_service.load_model(model_name, model_type)
            if model:
                self.trained_models[model_name] = model
                logger.info(f"Model {model_name} loaded from ImageKit")
                return True
            else:
                logger.warning(f"Model {model_name} not found in ImageKit")
                return False
        except Exception as e:
            logger.error(f"Error loading model {model_name} from ImageKit: {str(e)}")
            return False
    
    def load_all_models(self) -> Dict[str, bool]:
        """Load all available models from ImageKit storage"""
        results = {}
        
        # Load classifiers
        for name in self.CLASSIFIER_REGISTRY.keys():
            results[name] = self.load_model_from_storage(name, 'classification')
        
        # Load regressors
        for name in self.REGRESSOR_REGISTRY.keys():
            results[name] = self.load_model_from_storage(name, 'regression')
        
        return results
    
    def clear_models(self):
        """Clear all models from memory and ImageKit"""
        try:
            # Clear from memory
            self.trained_models.clear()
            
            # Delete from ImageKit
            for name in self.CLASSIFIER_REGISTRY.keys():
                imagekit_service.delete_model(name, 'classification')
            
            for name in self.REGRESSOR_REGISTRY.keys():
                imagekit_service.delete_model(name, 'regression')
                
            logger.info("All models cleared from memory and ImageKit")
        except Exception as e:
            logger.error(f"Error clearing models: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about trained models"""
        return {
            'trained_models': list(self.trained_models.keys()),
            'feature_columns': self.feature_columns,
            'unique_types': self.label_encoders.get('type', {}).classes_.tolist() if 'type' in self.label_encoders else []
        }

# Global ML service instance
ml_service = MLService()
