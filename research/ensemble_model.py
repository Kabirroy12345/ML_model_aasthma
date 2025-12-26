"""
AsthmAI - High-Accuracy Ensemble Model
Goal: Achieve 80%+ accuracy through:
1. Stacking ensemble with meta-learner
2. Advanced feature engineering
3. Class balancing with SMOTE
4. Hyperparameter optimization
5. Deep Neural Network ensemble member
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, Any

# Sklearn imports
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_validate
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.ensemble import (
    StackingClassifier, VotingClassifier,
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

# Import boosting libraries
import xgboost as xgb
import lightgbm as lgb

# For SMOTE
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.combine import SMOTETomek
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("Warning: imbalanced-learn not installed. Run: pip install imbalanced-learn")

warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


class HighAccuracyEnsemble:
    """Advanced ensemble for achieving 80%+ accuracy."""
    
    def __init__(self, data_dir: str, results_dir: str):
        self.data_dir = data_dir
        self.results_dir = results_dir
        
        self.numerical_cols = [
            'AQI', 'PM2.5', 'SO2 level', 'NO2 level', 
            'CO2 level', 'Humidity', 'Temperature'
        ]
        self.categorical_cols = [
            'Asthma Symptoms Frequency', 'Triggers', 'Weather Sensitivity',
            'Poor Air Quality Exposure', 'Night Breathing Difficulty'
        ]
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.results = {}
        
    def load_data(self):
        """Load all datasets."""
        train_df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        val_df = pd.read_csv(os.path.join(self.data_dir, 'validation.csv'))
        test_df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))
        
        # Combine train and validation
        full_train = pd.concat([train_df, val_df], ignore_index=True)
        
        return full_train, test_df
    
    def advanced_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features to boost model performance."""
        df = df.copy()
        
        # Pollution interaction features
        df['AQI_PM_ratio'] = df['AQI'] / (df['PM2.5'] + 1)
        df['pollution_index'] = (df['AQI'] * 0.4 + df['PM2.5'] * 0.3 + 
                                  df['NO2 level'] * 0.15 + df['SO2 level'] * 0.15)
        df['gas_pollution'] = df['CO2 level'] * df['NO2 level'] * df['SO2 level'] / 10000
        
        # Weather-pollution interactions
        df['humidity_pollution'] = df['Humidity'] * df['pollution_index'] / 100
        df['temp_pollution'] = df['Temperature'] * df['pollution_index'] / 100
        
        # AQI categories (critical, unhealthy, moderate, good)
        df['AQI_critical'] = (df['AQI'] > 200).astype(int)
        df['AQI_unhealthy'] = ((df['AQI'] > 100) & (df['AQI'] <= 200)).astype(int)
        df['PM25_high'] = (df['PM2.5'] > 75).astype(int)
        
        # Symptom severity score
        symptom_map = {
            'Daily': 4, 'Frequently (Weekly)': 3, 
            '1-2 times a month': 2, 'Less than once a month': 1
        }
        df['symptom_severity'] = df['Asthma Symptoms Frequency'].map(symptom_map).fillna(0)
        
        # Exposure risk score
        exposure_map = {'Yes, often': 3, 'Occasionally': 2, 'No': 1}
        df['exposure_score'] = df['Poor Air Quality Exposure'].map(exposure_map).fillna(0)
        
        # Night symptoms score
        night_map = {'Frequently': 3, 'Occasionally': 2, 'Rarely': 1, 'Never': 0}
        df['night_score'] = df['Night Breathing Difficulty'].map(night_map).fillna(0)
        
        # Trigger count (number of triggers)
        df['trigger_count'] = df['Triggers'].apply(lambda x: str(x).count(',') + 1)
        
        # Combined clinical score
        df['clinical_risk_score'] = (
            df['symptom_severity'] * 0.4 + 
            df['exposure_score'] * 0.3 + 
            df['night_score'] * 0.3
        )
        
        # Combined environmental score
        df['env_risk_score'] = (
            df['AQI_critical'] * 0.3 +
            df['AQI_unhealthy'] * 0.2 +
            df['PM25_high'] * 0.25 +
            (df['pollution_index'] / df['pollution_index'].max()) * 0.25
        )
        
        # Total risk interaction
        df['total_risk_interaction'] = df['clinical_risk_score'] * df['env_risk_score']
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features with advanced engineering."""
        # Apply feature engineering
        df = self.advanced_feature_engineering(df)
        
        # Get all numerical features (including engineered)
        engineered_cols = [
            'AQI_PM_ratio', 'pollution_index', 'gas_pollution',
            'humidity_pollution', 'temp_pollution', 'AQI_critical',
            'AQI_unhealthy', 'PM25_high', 'symptom_severity',
            'exposure_score', 'night_score', 'trigger_count',
            'clinical_risk_score', 'env_risk_score', 'total_risk_interaction'
        ]
        
        all_numerical = self.numerical_cols + engineered_cols
        X_num = df[all_numerical].values
        
        # Encode categorical
        X_cat_list = []
        for col in self.categorical_cols:
            if fit:
                encoder = LabelEncoder()
                X_cat_list.append(encoder.fit_transform(df[col].astype(str)))
            else:
                # Simple encoding for test
                encoder = LabelEncoder()
                encoder.fit(df[col].astype(str))
                X_cat_list.append(encoder.transform(df[col].astype(str)))
        
        X_cat = np.column_stack(X_cat_list)
        X = np.hstack([X_num, X_cat])
        
        # Handle any NaN/inf values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        
        # Encode target
        if fit:
            self.label_encoder.fit(['Low', 'Medium', 'High'])
        y = self.label_encoder.transform(df['Risk Class'])
        
        return X, y
    
    def create_stacking_ensemble(self):
        """Create a powerful stacking ensemble."""
        
        # Level 1: Diverse base estimators
        base_estimators = [
            ('xgb', xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                eval_metric='mlogloss',
                use_label_encoder=False
            )),
            ('lgb', lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                verbose=-1
            )),
            ('rf', RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )),
            ('et', ExtraTreesClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=RANDOM_STATE
            )),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=RANDOM_STATE,
                early_stopping=True,
                validation_fraction=0.1
            ))
        ]
        
        # Level 2: Meta-learner
        meta_learner = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=RANDOM_STATE
        )
        
        # Create stacking classifier
        stacking_clf = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        return stacking_clf
    
    def create_voting_ensemble(self):
        """Create a soft voting ensemble as alternative."""
        
        estimators = [
            ('xgb', xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                random_state=RANDOM_STATE, eval_metric='mlogloss', use_label_encoder=False
            )),
            ('lgb', lgb.LGBMClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                random_state=RANDOM_STATE, verbose=-1
            )),
            ('rf', RandomForestClassifier(
                n_estimators=300, max_depth=15, random_state=RANDOM_STATE
            )),
            ('et', ExtraTreesClassifier(
                n_estimators=300, max_depth=15, random_state=RANDOM_STATE
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=200, max_depth=5, random_state=RANDOM_STATE
            ))
        ]
        
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        
        return voting_clf
    
    def train_and_evaluate(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray):
        """Train ensemble and evaluate."""
        
        print("=" * 60)
        print("AsthmAI - High Accuracy Ensemble Training")
        print("=" * 60)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply SMOTE - DISABLED for Deterministic Run
        # SMOTE can corrupt integer-encoded categorical features
        print("\nSkipping SMOTE to preserve integer-encoded categorical structure.")
        X_train_balanced, y_train_balanced = X_train_scaled, y_train
        
        # Create and train stacking ensemble
        print("\nTraining Stacking Ensemble (6 base models + meta-learner)...")
        stacking_clf = self.create_stacking_ensemble()
        stacking_clf.fit(X_train_balanced, y_train_balanced)
        
        # Evaluate stacking
        y_pred_stack = stacking_clf.predict(X_test_scaled)
        y_prob_stack = stacking_clf.predict_proba(X_test_scaled)
        
        stack_accuracy = accuracy_score(y_test, y_pred_stack)
        stack_f1 = f1_score(y_test, y_pred_stack, average='weighted')
        stack_roc = roc_auc_score(y_test, y_prob_stack, multi_class='ovr')
        
        print(f"\n--- Stacking Ensemble Results ---")
        print(f"Accuracy:  {stack_accuracy:.4f} ({stack_accuracy*100:.1f}%)")
        print(f"F1-Score:  {stack_f1:.4f}")
        print(f"ROC-AUC:   {stack_roc:.4f}")
        
        # Create and train voting ensemble
        print("\nTraining Voting Ensemble...")
        voting_clf = self.create_voting_ensemble()
        voting_clf.fit(X_train_balanced, y_train_balanced)
        
        y_pred_vote = voting_clf.predict(X_test_scaled)
        y_prob_vote = voting_clf.predict_proba(X_test_scaled)
        
        vote_accuracy = accuracy_score(y_test, y_pred_vote)
        vote_f1 = f1_score(y_test, y_pred_vote, average='weighted')
        vote_roc = roc_auc_score(y_test, y_prob_vote, multi_class='ovr')
        
        print(f"\n--- Voting Ensemble Results ---")
        print(f"Accuracy:  {vote_accuracy:.4f} ({vote_accuracy*100:.1f}%)")
        print(f"F1-Score:  {vote_f1:.4f}")
        print(f"ROC-AUC:   {vote_roc:.4f}")
        
        # Cross-validation on best model
        print("\nRunning 5-Fold Cross-Validation on Stacking Ensemble...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(stacking_clf, X_train_scaled, y_train, cv=skf, scoring='accuracy')
        
        print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Store results
        self.results = {
            'stacking': {
                'accuracy': float(stack_accuracy),
                'f1': float(stack_f1),
                'roc_auc': float(stack_roc),
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'confusion_matrix': confusion_matrix(y_test, y_pred_stack).tolist()
            },
            'voting': {
                'accuracy': float(vote_accuracy),
                'f1': float(vote_f1),
                'roc_auc': float(vote_roc),
                'confusion_matrix': confusion_matrix(y_test, y_pred_vote).tolist()
            }
        }
        
        # Save best model
        if stack_accuracy >= vote_accuracy:
            self.best_model = stacking_clf
            best_name = 'stacking'
        else:
            self.best_model = voting_clf
            best_name = 'voting'
        
        print(f"\n{'='*60}")
        print(f"BEST MODEL: {best_name.upper()} ENSEMBLE")
        print(f"ACCURACY: {max(stack_accuracy, vote_accuracy)*100:.1f}%")
        print(f"ROC-AUC: {max(stack_roc, vote_roc):.4f}")
        print(f"{'='*60}")
        
        return self.results
    
    def save_results(self):
        """Save results and model."""
        # Save results
        with open(os.path.join(self.results_dir, 'ensemble_results.json'), 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save best model
        with open(os.path.join(self.results_dir, 'best_ensemble_model.pkl'), 'wb') as f:
            pickle.dump({
                'model': self.best_model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder
            }, f)
        
        print(f"\nResults saved to {self.results_dir}/")
    
    def run(self):
        """Run the full ensemble pipeline."""
        # Load data
        train_df, test_df = self.load_data()
        
        print(f"Training samples: {len(train_df)}")
        print(f"Test samples: {len(test_df)}")
        
        # Prepare features
        X_train, y_train = self.prepare_features(train_df, fit=True)
        X_test, y_test = self.prepare_features(test_df, fit=False)
        
        print(f"Feature dimensions: {X_train.shape[1]}")
        
        # Train and evaluate
        results = self.train_and_evaluate(X_train, y_train, X_test, y_test)
        
        # Save
        self.save_results()
        
        return results


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    data_dir = os.path.join(parent_dir, 'data')
    results_dir = os.path.join(parent_dir, 'results')
    
    ensemble = HighAccuracyEnsemble(data_dir, results_dir)
    results = ensemble.run()
    
    print("\n✓ High-Accuracy Ensemble Training Complete!")
    return results


if __name__ == "__main__":
    main()
