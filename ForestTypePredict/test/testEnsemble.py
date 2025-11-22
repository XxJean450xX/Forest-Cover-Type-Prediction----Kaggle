"""
Test Ensemble Models
Diagnostic script to verify ensemble functionality
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from generateEnsembleModels import EnsembleModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_ensemble():
    """Test ensemble models comprehensively"""
    
    print("=" * 70)
    print("ENSEMBLE MODEL DIAGNOSTIC TEST")
    print("=" * 70)
    
    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "src" / "data"
    raw_dir = data_dir / "raw"
    artifacts_dir = data_dir / "artifacts"
    
    train_file = raw_dir / "train.csv"
    preprocessor_path = artifacts_dir / "preprocessor.pkl"
    ensemble_path = artifacts_dir / "ensemble_models.pkl"
    
    # Check files
    if not ensemble_path.exists():
        logger.error(f" Ensemble not found: {ensemble_path}")
        logger.info("   Run: python src/generate_ensemble_models.py")
        return
    
    if not preprocessor_path.exists():
        logger.error(f" Preprocessor not found: {preprocessor_path}")
        logger.info("   Run: python src/generate_preprocessor.py")
        return
    
    if not train_file.exists():
        logger.error(f" Training data not found: {train_file}")
        return
    
    # Load data
    logger.info("\n Loading training data...")
    df = pd.read_csv(train_file)
    X = df.drop(['Id', 'Cover_Type'], axis=1)
    y = df['Cover_Type'] - 1  # Convert to 0-6
    
    logger.info(f"✓ Loaded {len(df)} samples")
    logger.info(f"  Features: {X.shape[1]}")
    logger.info(f"  Classes: {y.nunique()} (range: {y.min()}-{y.max()})")
    
    # Load preprocessor
    logger.info("\n Loading preprocessor...")
    with open(preprocessor_path, 'rb') as f:
        preprocessor_data = pickle.load(f)
    
    pipeline = preprocessor_data['pipeline']
    
    # Preprocess
    logger.info(" Preprocessing data...")
    X_transformed = pipeline.elevation_processor.transform(X)
    X_transformed = pipeline.aspect_transformer.transform(X_transformed)
    X_transformed = pipeline.soil_consolidator.transform(X_transformed)
    
    if pipeline.scaler is not None:
        numerical_cols = X_transformed.select_dtypes(include=[np.number]).columns.tolist()
        numerical_data = X_transformed[numerical_cols].values
        
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            scaled_data = pipeline.scaler.transform(numerical_data)
            X_transformed[numerical_cols] = scaled_data
    
    logger.info(f"✓ Preprocessing complete: {X.shape[1]} → {X_transformed.shape[1]} features")
    
    # Load ensemble
    logger.info("\n Loading ensemble...")
    with open(ensemble_path, 'rb') as f:
        ensemble_data = pickle.load(f)
    
    trainer = EnsembleModelTrainer(random_state=ensemble_data['random_state'])
    trainer.models = ensemble_data['models']
    trainer.model_scores = ensemble_data['model_scores']
    trainer.ensemble_weights = ensemble_data['ensemble_weights']
    trainer.is_fitted = ensemble_data['is_fitted']
    
    logger.info(f"✓ Loaded {len(trainer.models)} models: {list(trainer.models.keys())}")
    
    # Test 1: Individual model predictions
    print("\n" + "=" * 70)
    print("TEST 1: INDIVIDUAL MODEL PREDICTIONS")
    print("=" * 70)
    
    X_array = X_transformed.values if isinstance(X_transformed, pd.DataFrame) else X_transformed
    
    for model_name, model in trainer.models.items():
        y_pred_individual = model.predict(X_array)
        acc = accuracy_score(y, y_pred_individual)
        f1 = f1_score(y, y_pred_individual, average='weighted')
        
        print(f"\n{model_name}:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 (weighted): {f1:.4f}")
        print(f"  Predictions range: {y_pred_individual.min()}-{y_pred_individual.max()}")
        print(f"  Unique classes predicted: {np.unique(y_pred_individual)}")
    
    # Test 2: Ensemble probability predictions
    print("\n" + "=" * 70)
    print("TEST 2: ENSEMBLE PROBABILITY PREDICTIONS")
    print("=" * 70)
    
    try:
        proba = trainer.predict_proba(X_array)
        print(f"\n✓ Probability shape: {proba.shape}")
        print(f"  Expected: ({len(X)}, 7)")
        print(f"  Probability sum (should be ~1.0): {proba[0].sum():.6f}")
        print(f"  Min probability: {proba.min():.6f}")
        print(f"  Max probability: {proba.max():.6f}")
        print(f"\nSample probabilities (first observation):")
        for i, p in enumerate(proba[0]):
            print(f"  Class {i}: {p:.4f}")
    except Exception as e:
        print(f"\n Error in predict_proba: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Ensemble class predictions
    print("\n" + "=" * 70)
    print("TEST 3: ENSEMBLE CLASS PREDICTIONS")
    print("=" * 70)
    
    try:
        y_pred = trainer.predict(X_array)
        
        print(f"\n✓ Predictions shape: {y_pred.shape}")
        print(f"  Expected: ({len(X)},)")
        print(f"  Predictions range: {y_pred.min()}-{y_pred.max()}")
        print(f"  Expected range: 0-6")
        print(f"  Unique classes predicted: {sorted(np.unique(y_pred))}")
        
        # Calculate metrics
        acc = accuracy_score(y, y_pred)
        f1_macro = f1_score(y, y_pred, average='macro')
        f1_weighted = f1_score(y, y_pred, average='weighted')
        
        print(f"\n Ensemble Performance:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 (macro): {f1_macro:.4f}")
        print(f"  F1 (weighted): {f1_weighted:.4f}")
        
    except Exception as e:
        print(f"\n Error in predict: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Confusion matrix
    print("\n" + "=" * 70)
    print("TEST 4: CONFUSION MATRIX")
    print("=" * 70)
    
    try:
        cm = confusion_matrix(y, y_pred)
        print("\nConfusion Matrix (rows=true, cols=pred):")
        print("     ", "  ".join([f"C{i}" for i in range(7)]))
        for i, row in enumerate(cm):
            print(f"C{i}:", "  ".join([f"{val:4d}" for val in row]))
        
        # Per-class accuracy
        print("\n Per-Class Performance:")
        for cls in range(7):
            true_count = (y == cls).sum()
            pred_count = (y_pred == cls).sum()
            correct = cm[cls, cls]
            cls_accuracy = correct / true_count if true_count > 0 else 0
            
            print(f"  Class {cls} (orig {cls+1}): "
                  f"true={true_count:5d}, pred={pred_count:5d}, "
                  f"correct={correct:5d}, acc={cls_accuracy:.4f}")
        
    except Exception as e:
        print(f"\n Error in confusion matrix: {e}")
    
    # Test 5: Sample predictions
    print("\n" + "=" * 70)
    print("TEST 5: SAMPLE PREDICTIONS (First 10)")
    print("=" * 70)
    
    try:
        print("\n  ID | True | Pred | Confidence")
        print("  " + "-" * 35)
        for i in range(min(10, len(y))):
            true_cls = y.iloc[i] if isinstance(y, pd.Series) else y[i]
            pred_cls = y_pred[i]
            confidence = proba[i, pred_cls]
            match = "✓" if true_cls == pred_cls else "✗"
            
            print(f"  {i:3d} | {true_cls:4d} | {pred_cls:4d} | {confidence:.4f} {match}")
    except Exception as e:
        print(f"\n Error in sample predictions: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    
    if acc > 0.0:
        print("\n Ensemble is working correctly!")
        print(f"   Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        
        if acc < 0.5:
            print("\n  WARNING: Low accuracy detected")
            print("   - Check if features are scaled correctly")
            print("   - Verify label encoding (should be 0-6)")
            print("   - Review model hyperparameters")
        elif acc > 0.99:
            print("\n  WARNING: Suspiciously high accuracy")
            print("   - This is training set - expect overfitting")
            print("   - Check CV scores for realistic performance")
    else:
        print("\n Ensemble predictions are failing!")
        print("   - Check error messages above")
        print("   - Verify data types (DataFrame vs numpy)")
        print("   - Ensure label encoding matches (0-6)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_ensemble()
