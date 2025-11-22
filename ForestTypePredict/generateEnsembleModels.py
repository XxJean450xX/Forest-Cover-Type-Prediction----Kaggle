"""
Generate Ensemble Models
Train Random Forest, XGBoost, and LightGBM models and save as ensemble_models.pkl
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any, Tuple
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    classification_report, 
    confusion_matrix
)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠️  LightGBM not installed. Install with: pip install lightgbm")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleModelTrainer:
    """
    Train and ensemble multiple models for forest cover prediction
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize trainer
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.model_scores = {}
        self.ensemble_weights = [0.3, 0.4, 0.3]  # RF, XGB, LGB
        self.is_fitted = False
        
    def build_models(self) -> Dict[str, Any]:
        """
        Build base models with optimized hyperparameters
        
        Returns:
            Dictionary of models
        """
        models = {}
        
        # Random Forest
        logger.info("Building Random Forest model...")
        models['RandomForest'] = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0
        )
        
        # XGBoost
        if HAS_XGBOOST:
            logger.info("Building XGBoost model...")
            models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0
            )
        else:
            logger.warning("XGBoost not available, skipping...")
        
        # LightGBM
        if HAS_LIGHTGBM:
            logger.info("Building LightGBM model...")
            models['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=400,
                num_leaves=64,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
        else:
            logger.warning("LightGBM not available, skipping...")
        
        return models
    
    def train_model(self, model, X_train, y_train, model_name: str) -> Tuple[Any, Dict]:
        """
        Train a single model and collect metrics
        
        Args:
            model: Model instance
            X_train: Training features
            y_train: Training labels
            model_name: Name of the model
            
        Returns:
            Trained model and metrics dictionary
        """
        logger.info(f"Training {model_name}...")
        
        # Track training time
        start_time = time.time()
        
        # Train model
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Predictions
        y_pred = model.predict(X_train)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_train, y_pred),
            'f1_macro': f1_score(y_train, y_pred, average='macro'),
            'f1_weighted': f1_score(y_train, y_pred, average='weighted'),
            'training_time': training_time
        }
        
        logger.info(f"  ✓ {model_name} - Accuracy: {metrics['accuracy']:.4f}, "
                   f"F1 (macro): {metrics['f1_macro']:.4f}, "
                   f"Time: {training_time:.2f}s")
        
        return model, metrics
    
    def cross_validate_model(self, model, X_train, y_train, 
                            model_name: str, cv_folds: int = 5) -> Dict:
        """
        Perform cross-validation on a model
        
        Args:
            model: Model instance
            X_train: Training features
            y_train: Training labels
            model_name: Name of the model
            cv_folds: Number of CV folds
            
        Returns:
            CV metrics dictionary
        """
        logger.info(f"Cross-validating {model_name} ({cv_folds} folds)...")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                            random_state=self.random_state)
        
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=cv, scoring='accuracy', n_jobs=-1
        )
        
        cv_metrics = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
        
        logger.info(f"  ✓ {model_name} CV - Mean: {cv_metrics['cv_mean']:.4f} "
                   f"(±{cv_metrics['cv_std']:.4f})")
        
        return cv_metrics
    
    def fit(self, X_train, y_train, perform_cv: bool = True) -> 'EnsembleModelTrainer':
        """
        Train all models in the ensemble
        
        Args:
            X_train: Training features
            y_train: Training labels
            perform_cv: Whether to perform cross-validation
            
        Returns:
            Self for chaining
        """
        logger.info("=" * 70)
        logger.info("ENSEMBLE MODEL TRAINING")
        logger.info("=" * 70)
        
        # Build models
        models = self.build_models()
        
        # Train each model
        for model_name, model in models.items():
            # Train
            trained_model, metrics = self.train_model(
                model, X_train, y_train, model_name
            )
            
            # Cross-validate (optional)
            if perform_cv:
                cv_metrics = self.cross_validate_model(
                    model, X_train, y_train, model_name
                )
                metrics.update(cv_metrics)
            
            # Store
            self.models[model_name] = trained_model
            self.model_scores[model_name] = metrics
        
        self.is_fitted = True
        
        logger.info("=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 70)
        
        return self
    
    def predict_proba(self, X) -> np.ndarray:
        """
        Get probability predictions from ensemble
        
        Args:
            X: Features
            
        Returns:
            Probability array of shape (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        # Get predictions from each model
        predictions = []
        weights = []
        
        model_list = ['RandomForest', 'XGBoost', 'LightGBM']
        
        for i, model_name in enumerate(model_list):
            if model_name in self.models:
                pred = self.models[model_name].predict_proba(X)
                predictions.append(pred)
                weights.append(self.ensemble_weights[i])
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Weighted average
        ensemble_proba = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            ensemble_proba += weight * pred
        
        return ensemble_proba
    
    def predict(self, X) -> np.ndarray:
        """
        Get class predictions from ensemble
        
        Args:
            X: Features (DataFrame or ndarray)
            
        Returns:
            Predicted classes (0-6, add 1 for original labels 1-7)
        """
        # Convert DataFrame to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)  # Returns 0-6
    
    def get_feature_importance(self, feature_names=None) -> pd.DataFrame:
        """
        Get feature importance from all models
        
        Args:
            feature_names: Optional list of feature names
            
        Returns:
            DataFrame with feature importance
        """
        importances = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances[model_name] = model.feature_importances_
        
        if not importances:
            return None
        
        df = pd.DataFrame(importances)
        
        if feature_names is not None and len(feature_names) == len(df):
            df.index = feature_names
        
        return df
    
    def save(self, filepath: str):
        """
        Save ensemble to pickle file
        
        Args:
            filepath: Path to save file
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted ensemble.")
        
        ensemble_data = {
            'models': self.models,
            'model_scores': self.model_scores,
            'ensemble_weights': self.ensemble_weights,
            'is_fitted': self.is_fitted,
            'random_state': self.random_state,
            'metadata': {
                'n_models': len(self.models),
                'model_names': list(self.models.keys()),
                'timestamp': pd.Timestamp.now().isoformat(),
                'version': '1.0.0'
            }
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(ensemble_data, f)
        
        logger.info(f"✓ Ensemble saved to: {filepath}")
        logger.info(f"  File size: {filepath.stat().st_size / 1024:.2f} KB")
    
    @staticmethod
    def load(filepath: str) -> 'EnsembleModelTrainer':
        """
        Load ensemble from pickle file
        
        Args:
            filepath: Path to pickle file
            
        Returns:
            Loaded ensemble
        """
        with open(filepath, 'rb') as f:
            ensemble_data = pickle.load(f)
        
        trainer = EnsembleModelTrainer(
            random_state=ensemble_data['random_state']
        )
        trainer.models = ensemble_data['models']
        trainer.model_scores = ensemble_data['model_scores']
        trainer.ensemble_weights = ensemble_data['ensemble_weights']
        trainer.is_fitted = ensemble_data['is_fitted']
        
        logger.info(f"✓ Ensemble loaded from: {filepath}")
        logger.info(f"  Models: {ensemble_data['metadata']['model_names']}")
        logger.info(f"  Version: {ensemble_data['metadata']['version']}")
        
        return trainer
    
    def summary(self):
        """Print ensemble summary"""
        print("=" * 70)
        print("ENSEMBLE MODEL SUMMARY")
        print("=" * 70)
        
        print(f"\nStatus: {'✓ Fitted' if self.is_fitted else '✗ Not fitted'}")
        print(f"Number of models: {len(self.models)}")
        print(f"Ensemble weights: {self.ensemble_weights}")
        
        if self.model_scores:
            print("\n" + "-" * 70)
            print("MODEL PERFORMANCE")
            print("-" * 70)
            
            for model_name, scores in self.model_scores.items():
                print(f"\n{model_name}:")
                print(f"  Accuracy: {scores['accuracy']:.4f}")
                print(f"  F1 (macro): {scores['f1_macro']:.4f}")
                print(f"  F1 (weighted): {scores['f1_weighted']:.4f}")
                print(f"  Training time: {scores['training_time']:.2f}s")
                
                if 'cv_mean' in scores:
                    print(f"  CV accuracy: {scores['cv_mean']:.4f} "
                         f"(±{scores['cv_std']:.4f})")
        
        print("\n" + "=" * 70)


def load_and_preprocess_data(preprocessor_path: Path, train_file: Path):
    """
    Load raw data and apply preprocessing pipeline
    
    Args:
        preprocessor_path: Path to preprocessor.pkl
        train_file: Path to train.csv
        
    Returns:
        Tuple of (X_transformed, y, feature_names)
    """
    logger.info("\n Loading raw data...")
    df = pd.read_csv(train_file)
    logger.info(f"✓ Loaded {len(df)} samples")
    
    # Prepare data
    X = df.drop(['Id', 'Cover_Type'], axis=1)
    y = df['Cover_Type']
    
    # Convert labels from 1-7 to 0-6 for sklearn compatibility
    y = y - 1
    
    logger.info(f"  Features: {X.shape[1]}")
    logger.info(f"  Classes: {y.nunique()} (labels: {y.min()}-{y.max()})")
    
    # Load and apply preprocessor
    if preprocessor_path.exists():
        logger.info("\n Applying preprocessing pipeline...")
        
        with open(preprocessor_path, 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        pipeline = preprocessor_data['pipeline']
        
        # Apply transformations step by step
        X_transformed = pipeline.elevation_processor.transform(X)
        X_transformed = pipeline.aspect_transformer.transform(X_transformed)
        X_transformed = pipeline.soil_consolidator.transform(X_transformed)
        
        # Apply scaling if available
        if pipeline.scaler is not None:
            # Get numerical columns
            numerical_cols = X_transformed.select_dtypes(include=[np.number]).columns.tolist()
            
            logger.info(f"  Found {len(numerical_cols)} numerical columns")
            logger.info(f"  Scaler expects {pipeline.scaler.n_features_in_} features")
            
            # Ensure we have the right number of features
            if len(numerical_cols) != pipeline.scaler.n_features_in_:
                logger.warning(f"  Feature mismatch! Adjusting...")
                
                # Use only the features that match the scaler
                if len(numerical_cols) > pipeline.scaler.n_features_in_:
                    numerical_cols = numerical_cols[:pipeline.scaler.n_features_in_]
                else:
                    logger.error(f"  Too few features: {len(numerical_cols)} < {pipeline.scaler.n_features_in_}")
                    logger.error("  Regenerate preprocessor with: python src/generate_preprocessor.py")
                    raise ValueError("Feature count mismatch - regenerate preprocessor")
            
            # Scale numerical features
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                numerical_data = X_transformed[numerical_cols].values
                scaled_data = pipeline.scaler.transform(numerical_data)
                X_transformed[numerical_cols] = scaled_data
            
            logger.info(f"✓ Scaling applied to {len(numerical_cols)} features")
        
        feature_names = X_transformed.columns.tolist()
        logger.info(f"✓ Preprocessing complete: {X.shape[1]} → {X_transformed.shape[1]} features")
        
        return X_transformed, y, feature_names
    else:
        logger.warning("  Preprocessor not found, using raw features")
        logger.warning("   Run: python src/generate_preprocessor.py")
        return X, y, X.columns.tolist()


def main():
    """Main function to generate ensemble_models.pkl"""
    
    print("=" * 70)
    print("FOREST COVER ENSEMBLE MODEL GENERATOR")
    print("=" * 70)
    
    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "src" / "data"
    raw_dir = data_dir / "raw"
    artifacts_dir = data_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = raw_dir / "train.csv"
    preprocessor_path = artifacts_dir / "preprocessor.pkl"
    
    # Check if training data exists
    if not train_file.exists():
        logger.error(f" Training data not found: {train_file}")
        logger.info(f"   Current directory: {Path.cwd()}")
        logger.info(f"   Looking for: {train_file.absolute()}")
        logger.info("   Please ensure train.csv is in: src/data/raw/")
        return
    
    # Load and preprocess data
    try:
        X, y, feature_names = load_and_preprocess_data(preprocessor_path, train_file)
    except Exception as e:
        logger.error(f" Data loading failed: {e}")
        return
    
    # Initialize trainer
    logger.info("\n Initializing ensemble trainer...")
    trainer = EnsembleModelTrainer(random_state=42)
    
    # Train ensemble
    logger.info("\n Starting ensemble training...")
    trainer.fit(X, y, perform_cv=True)
    
    # Evaluate ensemble
    logger.info("\n Evaluating ensemble performance...")
    y_pred = trainer.predict(X)
    
    # Note: y and y_pred are both 0-6 now
    ensemble_accuracy = accuracy_score(y, y_pred)
    ensemble_f1 = f1_score(y, y_pred, average='weighted')
    
    logger.info(f"✓ Ensemble accuracy: {ensemble_accuracy:.4f}")
    logger.info(f"✓ Ensemble F1 (weighted): {ensemble_f1:.4f}")
    
    # Show class distribution
    logger.info(f"\n Class Distribution (0-6, add 1 for original labels):")
    for cls in sorted(y.unique()):
        count = (y == cls).sum()
        pred_count = (y_pred == cls).sum()
        logger.info(f"  Class {cls} (original {cls+1}): {count} actual, {pred_count} predicted")
    
    # Save ensemble
    output_path = artifacts_dir / "ensemble_models.pkl"
    trainer.save(output_path)
    
    # Show feature importance
    importance_df = trainer.get_feature_importance(feature_names)
    
    if importance_df is not None:
        logger.info("\n Top 10 Most Important Features:")
        avg_importance = importance_df.mean(axis=1).sort_values(ascending=False)
        
        for i, (feature_name, importance) in enumerate(avg_importance.head(10).items(), 1):
            logger.info(f"  {i:2d}. {feature_name:40s} {importance:.4f}")
    
    # Print summary
    print("\n")
    trainer.summary()
    
    print("\n" + "=" * 70)
    print(" ENSEMBLE GENERATION COMPLETE!")
    print("=" * 70)
    print(f"\n Saved to: {output_path}")
    print(f"   Ready for deployment!")
    print("\nNext steps:")
    print("  1. Test predictions: python src/predict.py")
    print("  2. Start API: uvicorn api.app:app --reload")
    print("  3. Generate submission: python src/generate_submission.py")


if __name__ == "__main__":
    main()