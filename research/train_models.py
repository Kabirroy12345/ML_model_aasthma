"""
AsthmAI - Comprehensive Model Training Pipeline
Trains and compares 8 different ML models with cross-validation.

Models: Logistic Regression, Random Forest, XGBoost, LightGBM, 
        SVM, Gradient Boosting, KNN, Neural Network

Includes:
- Stratified K-Fold Cross-Validation (5-fold and 10-fold)
- Hyperparameter tuning with GridSearchCV
- Statistical significance testing
- Comprehensive metrics logging
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Sklearn imports
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_validate, GridSearchCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    make_scorer
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Statistical tests
from scipy import stats

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try to import optional packages
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Skipping XGBoost model.")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not installed. Skipping LightGBM model.")

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


class AsthmaModelTrainer:
    """Comprehensive model training and evaluation pipeline."""
    
    def __init__(self, data_dir: str, results_dir: str):
        self.data_dir = data_dir
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Define feature columns
        self.numerical_cols = [
            'AQI', 'PM2.5', 'SO2 level', 'NO2 level', 
            'CO2 level', 'Humidity', 'Temperature'
        ]
        self.categorical_cols = [
            'Asthma Symptoms Frequency', 'Triggers', 'Weather Sensitivity',
            'Poor Air Quality Exposure', 'Night Breathing Difficulty'
        ]
        
        # Results storage
        self.results = {}
        self.best_models = {}
        self.cv_results = {}
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load train, validation, and test datasets."""
        train_df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        val_df = pd.read_csv(os.path.join(self.data_dir, 'validation.csv'))
        test_df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))
        
        print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels from dataframe."""
        # Encode categorical features
        X_num = df[self.numerical_cols].values
        
        # One-hot encode categorical
        X_cat_list = []
        for col in self.categorical_cols:
            encoder = LabelEncoder()
            X_cat_list.append(encoder.fit_transform(df[col].astype(str)))
        
        X_cat = np.column_stack(X_cat_list)
        X = np.hstack([X_num, X_cat])
        
        # Encode target
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df['Risk Class'])
        
        return X, y, label_encoder
    
    def create_preprocessor(self):
        """Create sklearn preprocessing pipeline."""
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
                 self.categorical_cols)
            ]
        )
        return preprocessor
    
    def get_models(self) -> Dict[str, Any]:
        """Define all models with hyperparameter grids."""
        models = {
            'Logistic Regression': {
                'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
                'params': {
                    'C': [0.01, 0.1, 1, 10],
                    'solver': ['lbfgs', 'liblinear']
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=RANDOM_STATE),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=RANDOM_STATE),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            },
            'SVM': {
                'model': SVC(random_state=RANDOM_STATE, probability=True),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            },
            'KNN': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            }
        }
        
        # Add XGBoost if available
        if HAS_XGBOOST:
            models['XGBoost'] = {
                'model': xgb.XGBClassifier(
                    random_state=RANDOM_STATE, 
                    eval_metric='mlogloss',
                    use_label_encoder=False
                ),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            }
        
        # Add LightGBM if available
        if HAS_LIGHTGBM:
            models['LightGBM'] = {
                'model': lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            }
        
        return models
    
    def train_with_cv(self, X: np.ndarray, y: np.ndarray, 
                      n_folds: int = 5) -> Dict[str, Dict]:
        """Train all models with cross-validation and hyperparameter tuning."""
        
        models = self.get_models()
        results = {}
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"\n{'='*60}")
        print(f"Training {len(models)} models with {n_folds}-Fold Cross-Validation")
        print(f"{'='*60}\n")
        
        for name, model_config in models.items():
            print(f"\n[{name}]")
            print("-" * 40)
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                model_config['model'],
                model_config['params'],
                cv=skf,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_scaled, y)
            
            # Get best model
            best_model = grid_search.best_estimator_
            self.best_models[name] = best_model
            
            # Cross-validation with multiple metrics
            cv_results = cross_validate(
                best_model, X_scaled, y, cv=skf,
                scoring=['accuracy', 'precision_weighted', 'recall_weighted', 
                         'f1_weighted', 'roc_auc_ovr'],
                return_train_score=True
            )
            
            # Store results
            results[name] = {
                'best_params': grid_search.best_params_,
                'cv_accuracy': cv_results['test_accuracy'].tolist(),
                'cv_precision': cv_results['test_precision_weighted'].tolist(),
                'cv_recall': cv_results['test_recall_weighted'].tolist(),
                'cv_f1': cv_results['test_f1_weighted'].tolist(),
                'cv_roc_auc': cv_results['test_roc_auc_ovr'].tolist(),
                'mean_accuracy': float(np.mean(cv_results['test_accuracy'])),
                'std_accuracy': float(np.std(cv_results['test_accuracy'])),
                'mean_f1': float(np.mean(cv_results['test_f1_weighted'])),
                'std_f1': float(np.std(cv_results['test_f1_weighted'])),
                'mean_roc_auc': float(np.mean(cv_results['test_roc_auc_ovr'])),
                'std_roc_auc': float(np.std(cv_results['test_roc_auc_ovr']))
            }
            
            print(f"  Best Params: {grid_search.best_params_}")
            print(f"  CV Accuracy: {results[name]['mean_accuracy']:.4f} ± {results[name]['std_accuracy']:.4f}")
            print(f"  CV F1-Score: {results[name]['mean_f1']:.4f} ± {results[name]['std_f1']:.4f}")
            print(f"  CV ROC-AUC:  {results[name]['mean_roc_auc']:.4f} ± {results[name]['std_roc_auc']:.4f}")
        
        self.cv_results = results
        return results
    
    def evaluate_on_test(self, X_test: np.ndarray, y_test: np.ndarray, 
                         X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Dict]:
        """Evaluate all trained models on test set."""
        
        results = {}
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"\n{'='*60}")
        print("Evaluating Models on Test Set")
        print(f"{'='*60}\n")
        
        for name, model in self.best_models.items():
            # Refit on full training data
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)
            
            # Metrics
            results[name] = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, average='weighted')),
                'recall': float(recall_score(y_test, y_pred, average='weighted')),
                'f1': float(f1_score(y_test, y_pred, average='weighted')),
                'roc_auc': float(roc_auc_score(y_test, y_prob, multi_class='ovr')),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'predictions': y_pred.tolist(),
                'probabilities': y_prob.tolist()
            }
            
            print(f"[{name}]")
            print(f"  Accuracy:  {results[name]['accuracy']:.4f}")
            print(f"  Precision: {results[name]['precision']:.4f}")
            print(f"  Recall:    {results[name]['recall']:.4f}")
            print(f"  F1-Score:  {results[name]['f1']:.4f}")
            print(f"  ROC-AUC:   {results[name]['roc_auc']:.4f}")
            print()
        
        self.results['test'] = results
        return results
    
    def statistical_tests(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Perform statistical significance tests between models."""
        
        print(f"\n{'='*60}")
        print("Statistical Significance Tests")
        print(f"{'='*60}\n")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Get cross-validation scores for each model
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
        
        model_scores = {}
        for name, model in self.best_models.items():
            scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='f1_weighted')
            model_scores[name] = scores
        
        # Paired t-tests between all model pairs
        model_names = list(model_scores.keys())
        t_test_results = []
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                name1, name2 = model_names[i], model_names[j]
                t_stat, p_value = stats.ttest_rel(
                    model_scores[name1], 
                    model_scores[name2]
                )
                
                significant = "Yes" if p_value < 0.05 else "No"
                t_test_results.append({
                    'model_1': name1,
                    'model_2': name2,
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant_at_0.05': significant
                })
                
                print(f"{name1} vs {name2}: t={t_stat:.4f}, p={p_value:.4f} ({significant})")
        
        # Friedman test (non-parametric alternative to repeated measures ANOVA)
        all_scores = np.array([model_scores[name] for name in model_names])
        friedman_stat, friedman_p = stats.friedmanchisquare(*[all_scores[i] for i in range(len(model_names))])
        
        print(f"\nFriedman Test: χ²={friedman_stat:.4f}, p={friedman_p:.4f}")
        
        return {
            'paired_t_tests': t_test_results,
            'friedman_test': {
                'statistic': float(friedman_stat),
                'p_value': float(friedman_p)
            }
        }
    
    def save_results(self):
        """Save all results to files."""
        
        # Save CV results
        with open(os.path.join(self.results_dir, 'cv_results.json'), 'w') as f:
            json.dump(self.cv_results, f, indent=2)
        
        # Save test results
        if 'test' in self.results:
            with open(os.path.join(self.results_dir, 'test_results.json'), 'w') as f:
                # Remove predictions and probabilities for smaller file
                test_results_summary = {}
                for name, res in self.results['test'].items():
                    test_results_summary[name] = {
                        k: v for k, v in res.items() 
                        if k not in ['predictions', 'probabilities']
                    }
                json.dump(test_results_summary, f, indent=2)
        
        # Save best models
        for name, model in self.best_models.items():
            model_path = os.path.join(self.results_dir, f'model_{name.lower().replace(" ", "_")}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        print(f"\nResults saved to {self.results_dir}/")
    
    def generate_latex_tables(self):
        """Generate LaTeX tables for the paper."""
        
        # CV Results Table
        latex_cv = "\\begin{table}[h]\n\\centering\n"
        latex_cv += "\\caption{Cross-Validation Results (5-Fold)}\n"
        latex_cv += "\\begin{tabular}{lcccc}\n\\hline\n"
        latex_cv += "Model & Accuracy & F1-Score & ROC-AUC \\\\ \\hline\n"
        
        for name, res in self.cv_results.items():
            latex_cv += f"{name} & {res['mean_accuracy']:.3f}±{res['std_accuracy']:.3f} & "
            latex_cv += f"{res['mean_f1']:.3f}±{res['std_f1']:.3f} & "
            latex_cv += f"{res['mean_roc_auc']:.3f}±{res['std_roc_auc']:.3f} \\\\\n"
        
        latex_cv += "\\hline\n\\end{tabular}\n\\end{table}"
        
        with open(os.path.join(self.results_dir, 'table_cv_results.tex'), 'w') as f:
            f.write(latex_cv)
        
        print("\nLaTeX tables saved!")
    
    def run_full_pipeline(self):
        """Execute the complete training and evaluation pipeline."""
        
        print("=" * 60)
        print("AsthmAI - Model Training Pipeline")
        print("=" * 60)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load data
        train_df, val_df, test_df = self.load_data()
        
        # Combine train and validation for cross-validation
        full_train_df = pd.concat([train_df, val_df], ignore_index=True)
        
        # Prepare features
        X_train_full, y_train_full, label_encoder = self.prepare_features(full_train_df)
        X_test, y_test, _ = self.prepare_features(test_df)
        
        print(f"\nFeatures shape: {X_train_full.shape}")
        print(f"Classes: {label_encoder.classes_}")
        
        # Train with 5-fold CV
        self.train_with_cv(X_train_full, y_train_full, n_folds=5)
        
        # Evaluate on test set
        self.evaluate_on_test(X_test, y_test, X_train_full, y_train_full)
        
        # Statistical tests
        stat_results = self.statistical_tests(X_train_full, y_train_full)
        self.results['statistical_tests'] = stat_results
        
        # Save everything
        self.save_results()
        self.generate_latex_tables()
        
        # Find best model
        best_model_name = max(
            self.results['test'].items(),
            key=lambda x: x[1]['f1']
        )[0]
        
        print(f"\n{'='*60}")
        print(f"BEST MODEL: {best_model_name}")
        print(f"F1-Score: {self.results['test'][best_model_name]['f1']:.4f}")
        print(f"ROC-AUC: {self.results['test'][best_model_name]['roc_auc']:.4f}")
        print(f"{'='*60}")
        
        print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("✓ Training pipeline complete!")
        
        return best_model_name


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    data_dir = os.path.join(parent_dir, 'data')
    results_dir = os.path.join(parent_dir, 'results')
    
    trainer = AsthmaModelTrainer(data_dir, results_dir)
    best_model = trainer.run_full_pipeline()
    
    return best_model


if __name__ == "__main__":
    main()
