"""
AsthmAI - Model Explainability Module
Provides SHAP and LIME explanations for model predictions.

Features:
- SHAP global feature importance
- SHAP summary plots
- SHAP waterfall plots for individual predictions
- LIME local explanations
- Permutation importance
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')

# Try to import SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Warning: SHAP not installed. Run: pip install shap")

# Try to import LIME
try:
    import lime
    import lime.lime_tabular
    HAS_LIME = True
except ImportError:
    HAS_LIME = False
    print("Warning: LIME not installed. Run: pip install lime")


class ModelExplainer:
    """Provides interpretability for asthma risk prediction models."""
    
    def __init__(self, data_dir: str, results_dir: str, figures_dir: str):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.figures_dir = figures_dir
        os.makedirs(figures_dir, exist_ok=True)
        
        # Feature names
        self.numerical_cols = [
            'AQI', 'PM2.5', 'SO2 level', 'NO2 level', 
            'CO2 level', 'Humidity', 'Temperature'
        ]
        self.categorical_cols = [
            'Asthma Symptoms Frequency', 'Triggers', 'Weather Sensitivity',
            'Poor Air Quality Exposure', 'Night Breathing Difficulty'
        ]
        self.feature_names = self.numerical_cols + self.categorical_cols
        
        # Storage
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.explainer = None
        self.shap_values = None
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Load test data."""
        test_df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))
        train_df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        
        # Prepare features
        X_train, y_train = self._prepare_features(train_df)
        X_test, y_test = self._prepare_features(test_df)
        
        # Fit scaler on training data
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_test, test_df
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features from dataframe."""
        X_num = df[self.numerical_cols].values
        
        X_cat_list = []
        for col in self.categorical_cols:
            encoder = LabelEncoder()
            X_cat_list.append(encoder.fit_transform(df[col].astype(str)))
        
        X_cat = np.column_stack(X_cat_list)
        X = np.hstack([X_num, X_cat])
        
        y = self.label_encoder.fit_transform(df['Risk Class'])
        return X, y
    
    def load_best_model(self, model_name: str = 'random_forest'):
        """Load the best trained model."""
        model_path = os.path.join(self.results_dir, f'model_{model_name}.pkl')
        
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.models[model_name] = pickle.load(f)
            print(f"Loaded model: {model_name}")
            return self.models[model_name]
        else:
            # Try to find any model
            for fname in os.listdir(self.results_dir):
                if fname.startswith('model_') and fname.endswith('.pkl'):
                    with open(os.path.join(self.results_dir, fname), 'rb') as f:
                        model = pickle.load(f)
                    model_name = fname.replace('model_', '').replace('.pkl', '')
                    self.models[model_name] = model
                    print(f"Loaded model: {model_name}")
                    return model
            
            raise FileNotFoundError("No trained models found!")
    
    def compute_shap_values(self, model, X_train: np.ndarray, X_test: np.ndarray):
        """Compute SHAP values for model explanations."""
        if not HAS_SHAP:
            print("SHAP not available. Skipping SHAP analysis.")
            return None
        
        print("\nComputing SHAP values...")
        
        # Use appropriate explainer based on model type
        model_type = type(model).__name__
        
        if 'Forest' in model_type or 'Boost' in model_type or 'XGB' in model_type or 'LGB' in model_type:
            # Tree-based models
            self.explainer = shap.TreeExplainer(model)
        else:
            # Other models (use background samples)
            background = shap.sample(X_train, min(100, len(X_train)))
            self.explainer = shap.KernelExplainer(model.predict_proba, background)
        
        # Compute SHAP values
        self.shap_values = self.explainer.shap_values(X_test[:100])  # Limit for speed
        
        print("SHAP values computed!")
        return self.shap_values
    
    def plot_shap_summary(self, X_test: np.ndarray, save: bool = True):
        """Generate SHAP summary plot."""
        if not HAS_SHAP or self.shap_values is None:
            print("SHAP values not available.")
            return
        
        print("\nGenerating SHAP summary plot...")
        
        # Handle multi-class SHAP values
        if isinstance(self.shap_values, list):
            # Multi-class: use class 2 (High risk) for visualization
            shap_vals = self.shap_values[2] if len(self.shap_values) > 2 else self.shap_values[1]
        else:
            shap_vals = self.shap_values
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_vals, 
            X_test[:100],
            feature_names=self.feature_names,
            show=False,
            plot_size=(12, 8)
        )
        plt.tight_layout()
        
        if save:
            plt.savefig(
                os.path.join(self.figures_dir, 'shap_summary.png'), 
                dpi=300, 
                bbox_inches='tight'
            )
            plt.savefig(
                os.path.join(self.figures_dir, 'shap_summary.pdf'), 
                bbox_inches='tight'
            )
            print(f"Saved: {self.figures_dir}/shap_summary.png")
        
        plt.close()
    
    def plot_shap_bar(self, X_test: np.ndarray, save: bool = True):
        """Generate SHAP feature importance bar plot."""
        if not HAS_SHAP or self.shap_values is None:
            return
        
        print("\nGenerating SHAP bar plot...")
        
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[2] if len(self.shap_values) > 2 else self.shap_values[1]
        else:
            shap_vals = self.shap_values
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_vals, 
            X_test[:100],
            feature_names=self.feature_names,
            plot_type="bar",
            show=False
        )
        plt.tight_layout()
        
        if save:
            plt.savefig(
                os.path.join(self.figures_dir, 'shap_importance.png'), 
                dpi=300, 
                bbox_inches='tight'
            )
            print(f"Saved: {self.figures_dir}/shap_importance.png")
        
        plt.close()
    
    def plot_shap_waterfall(self, model, X_sample: np.ndarray, sample_idx: int = 0, save: bool = True):
        """Generate SHAP waterfall plot for a single prediction."""
        if not HAS_SHAP or self.explainer is None:
            return
        
        print(f"\nGenerating SHAP waterfall for sample {sample_idx}...")
        
        # Get SHAP explanation for single sample
        shap_values_single = self.explainer.shap_values(X_sample[sample_idx:sample_idx+1])
        
        if isinstance(shap_values_single, list):
            sv = shap_values_single[2][0] if len(shap_values_single) > 2 else shap_values_single[1][0]
            base_value = self.explainer.expected_value[2] if len(shap_values_single) > 2 else self.explainer.expected_value[1]
        else:
            sv = shap_values_single[0]
            base_value = self.explainer.expected_value
        
        # Create Explanation object
        explanation = shap.Explanation(
            values=sv,
            base_values=base_value,
            data=X_sample[sample_idx],
            feature_names=self.feature_names
        )
        
        plt.figure(figsize=(12, 6))
        shap.waterfall_plot(explanation, show=False)
        plt.tight_layout()
        
        if save:
            plt.savefig(
                os.path.join(self.figures_dir, f'shap_waterfall_{sample_idx}.png'),
                dpi=300,
                bbox_inches='tight'
            )
            print(f"Saved: {self.figures_dir}/shap_waterfall_{sample_idx}.png")
        
        plt.close()
    
    def compute_lime_explanation(self, model, X_train: np.ndarray, 
                                  X_test: np.ndarray, sample_idx: int = 0):
        """Generate LIME explanation for a single prediction."""
        if not HAS_LIME:
            print("LIME not available.")
            return None
        
        print(f"\nGenerating LIME explanation for sample {sample_idx}...")
        
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=self.feature_names,
            class_names=self.label_encoder.classes_,
            mode='classification'
        )
        
        # Generate explanation
        exp = explainer.explain_instance(
            X_test[sample_idx],
            model.predict_proba,
            num_features=len(self.feature_names)
        )
        
        # Save as HTML
        html_path = os.path.join(self.figures_dir, f'lime_explanation_{sample_idx}.html')
        exp.save_to_file(html_path)
        print(f"Saved: {html_path}")
        
        # Save as figure
        fig = exp.as_pyplot_figure()
        fig.set_size_inches(12, 8)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.figures_dir, f'lime_explanation_{sample_idx}.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
        print(f"Saved: {self.figures_dir}/lime_explanation_{sample_idx}.png")
        
        return exp
    
    def compute_permutation_importance(self, model, X_test: np.ndarray, y_test: np.ndarray):
        """Compute and plot permutation feature importance."""
        print("\nComputing permutation importance...")
        
        result = permutation_importance(
            model, X_test, y_test,
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Sort by importance
        sorted_idx = result.importances_mean.argsort()[::-1]
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(
            range(len(sorted_idx)),
            result.importances_mean[sorted_idx],
            xerr=result.importances_std[sorted_idx],
            color='steelblue',
            edgecolor='black'
        )
        plt.yticks(
            range(len(sorted_idx)),
            [self.feature_names[i] for i in sorted_idx]
        )
        plt.xlabel('Mean Decrease in Accuracy')
        plt.ylabel('Feature')
        plt.title('Permutation Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        plt.savefig(
            os.path.join(self.figures_dir, 'permutation_importance.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
        print(f"Saved: {self.figures_dir}/permutation_importance.png")
        
        # Save importance values
        importance_dict = {
            self.feature_names[i]: {
                'mean': float(result.importances_mean[i]),
                'std': float(result.importances_std[i])
            }
            for i in range(len(self.feature_names))
        }
        
        with open(os.path.join(self.results_dir, 'feature_importance.json'), 'w') as f:
            json.dump(importance_dict, f, indent=2)
        
        return result
    
    def run_full_explainability_pipeline(self):
        """Execute the complete explainability pipeline."""
        print("=" * 60)
        print("AsthmAI - Model Explainability Pipeline")
        print("=" * 60)
        
        # Load data
        X_train, X_test, y_test, test_df = self.load_data()
        
        # Load best model
        model = self.load_best_model()
        
        # Fit model on training data
        train_df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        X_train_full, y_train = self._prepare_features(train_df)
        X_train_scaled = self.scaler.transform(X_train_full)
        model.fit(X_train_scaled, y_train)
        
        # Permutation importance
        self.compute_permutation_importance(model, X_test, y_test)
        
        # SHAP analysis
        if HAS_SHAP:
            self.compute_shap_values(model, X_train, X_test)
            self.plot_shap_summary(X_test)
            self.plot_shap_bar(X_test)
            self.plot_shap_waterfall(model, X_test, sample_idx=0)
            self.plot_shap_waterfall(model, X_test, sample_idx=5)
        
        # LIME explanations
        if HAS_LIME:
            self.compute_lime_explanation(model, X_train, X_test, sample_idx=0)
            self.compute_lime_explanation(model, X_train, X_test, sample_idx=5)
        
        print("\n" + "=" * 60)
        print("✓ Explainability analysis complete!")
        print(f"Figures saved to: {self.figures_dir}")
        print("=" * 60)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    data_dir = os.path.join(parent_dir, 'data')
    results_dir = os.path.join(parent_dir, 'results')
    figures_dir = os.path.join(parent_dir, 'figures')
    
    explainer = ModelExplainer(data_dir, results_dir, figures_dir)
    explainer.run_full_explainability_pipeline()


if __name__ == "__main__":
    main()
