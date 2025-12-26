"""
AsthmAI - Publication Figure Generator
Generates all figures required for the conference paper.

Figures:
1. ROC curves comparison (all models)
2. Confusion matrices (heatmaps)
3. Learning curves
4. Model performance comparison bar chart
5. Feature correlation heatmap
6. Class distribution
7. Training history (for neural network)
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score
)
from sklearn.model_selection import learning_curve, StratifiedKFold

warnings.filterwarnings('ignore')

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


class FigureGenerator:
    """Generate publication-quality figures for the paper."""
    
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
        
        # Class names
        self.class_names = ['Low', 'Medium', 'High']
        self.class_colors = ['#2ecc71', '#f39c12', '#e74c3c']
        
    def load_data(self):
        """Load all datasets."""
        self.train_df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        self.val_df = pd.read_csv(os.path.join(self.data_dir, 'validation.csv'))
        self.test_df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))
        self.full_df = pd.read_csv(os.path.join(self.data_dir, 'dataset.csv'))
        
        print(f"Loaded: Train={len(self.train_df)}, Val={len(self.val_df)}, Test={len(self.test_df)}")
    
    def load_results(self):
        """Load training results."""
        with open(os.path.join(self.results_dir, 'cv_results.json'), 'r') as f:
            self.cv_results = json.load(f)
        
        with open(os.path.join(self.results_dir, 'test_results.json'), 'r') as f:
            self.test_results = json.load(f)
        
        print(f"Loaded results for {len(self.cv_results)} models")
    
    def load_models(self):
        """Load all trained models."""
        self.models = {}
        for fname in os.listdir(self.results_dir):
            if fname.startswith('model_') and fname.endswith('.pkl'):
                model_name = fname.replace('model_', '').replace('.pkl', '').replace('_', ' ').title()
                with open(os.path.join(self.results_dir, fname), 'rb') as f:
                    self.models[model_name] = pickle.load(f)
        print(f"Loaded {len(self.models)} models")
    
    def _prepare_features(self, df: pd.DataFrame):
        """Prepare features and labels."""
        X_num = df[self.numerical_cols].values
        
        X_cat_list = []
        for col in self.categorical_cols:
            encoder = LabelEncoder()
            X_cat_list.append(encoder.fit_transform(df[col].astype(str)))
        
        X_cat = np.column_stack(X_cat_list)
        X = np.hstack([X_num, X_cat])
        
        label_encoder = LabelEncoder()
        label_encoder.fit(self.class_names)
        y = label_encoder.transform(df['Risk Class'])
        
        return X, y
    
    def plot_class_distribution(self):
        """Plot class distribution bar chart."""
        print("\nGenerating class distribution plot...")
        
        class_counts = self.full_df['Risk Class'].value_counts()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(
            self.class_names,
            [class_counts.get(c, 0) for c in self.class_names],
            color=self.class_colors,
            edgecolor='black',
            linewidth=1.5
        )
        
        # Add value labels
        for bar, count in zip(bars, [class_counts.get(c, 0) for c in self.class_names]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 20,
                f'{count}',
                ha='center',
                va='bottom',
                fontsize=14,
                fontweight='bold'
            )
        
        ax.set_xlabel('Risk Class')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Dataset Class Distribution')
        ax.set_ylim(0, max(class_counts.values) * 1.15)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'class_distribution.png'))
        plt.savefig(os.path.join(self.figures_dir, 'class_distribution.pdf'))
        plt.close()
        print("Saved: class_distribution.png")
    
    def plot_feature_correlation(self):
        """Plot feature correlation heatmap."""
        print("\nGenerating correlation heatmap...")
        
        corr_df = self.full_df[self.numerical_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_df, dtype=bool))
        
        sns.heatmap(
            corr_df,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={'shrink': 0.8},
            ax=ax
        )
        
        ax.set_title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'correlation_heatmap.png'))
        plt.savefig(os.path.join(self.figures_dir, 'correlation_heatmap.pdf'))
        plt.close()
        print("Saved: correlation_heatmap.png")
    
    def plot_model_comparison(self):
        """Plot model performance comparison bar chart."""
        print("\nGenerating model comparison plot...")
        
        models = list(self.test_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        x = np.arange(len(models))
        width = 0.15
        
        fig, ax = plt.subplots(figsize=(14, 8))
        colors = plt.cm.Set2(np.linspace(0, 1, len(metrics)))
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = [self.test_results[m][metric] for m in models]
            bars = ax.bar(x + i * width, values, width, label=label, color=colors[i], edgecolor='black')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison on Test Set')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'model_comparison.png'))
        plt.savefig(os.path.join(self.figures_dir, 'model_comparison.pdf'))
        plt.close()
        print("Saved: model_comparison.png")
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models."""
        print("\nGenerating confusion matrices...")
        
        n_models = len(self.test_results)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.array(axes).flatten()
        
        for idx, (model_name, results) in enumerate(self.test_results.items()):
            cm = np.array(results['confusion_matrix'])
            
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                ax=axes[idx],
                cbar=False
            )
            axes[idx].set_title(f'{model_name}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        # Hide empty subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Confusion Matrices - All Models', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'confusion_matrices.png'))
        plt.savefig(os.path.join(self.figures_dir, 'confusion_matrices.pdf'))
        plt.close()
        print("Saved: confusion_matrices.png")
    
    def plot_roc_curves(self):
        """Plot ROC curves for all models."""
        print("\nGenerating ROC curves...")
        
        # Prepare data
        X_train, y_train = self._prepare_features(self.train_df)
        X_test, y_test = self._prepare_features(self.test_df)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Binarize labels for multi-class ROC
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
        n_classes = 3
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.models)))
        
        for (model_name, model), color in zip(self.models.items(), colors):
            # Fit and predict
            model.fit(X_train_scaled, y_train)
            y_prob = model.predict_proba(X_test_scaled)
            
            # Compute micro-average ROC curve
            fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=color, lw=2, 
                   label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # Random classifier line
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves - Multi-class (Micro-averaged)')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'roc_curves.png'))
        plt.savefig(os.path.join(self.figures_dir, 'roc_curves.pdf'))
        plt.close()
        print("Saved: roc_curves.png")
    
    def plot_cv_scores_boxplot(self):
        """Plot cross-validation scores as boxplots."""
        print("\nGenerating CV scores boxplot...")
        
        # Prepare data for boxplot
        data = []
        labels = []
        
        for model_name, results in self.cv_results.items():
            data.extend(results['cv_f1'])
            labels.extend([model_name] * len(results['cv_f1']))
        
        df_box = pd.DataFrame({'Model': labels, 'F1-Score': data})
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.boxplot(
            data=df_box,
            x='Model',
            y='F1-Score',
            palette='Set2',
            ax=ax
        )
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title('Cross-Validation F1-Score Distribution')
        ax.set_ylim(0.5, 1.0)
        ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='90% threshold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'cv_boxplot.png'))
        plt.savefig(os.path.join(self.figures_dir, 'cv_boxplot.pdf'))
        plt.close()
        print("Saved: cv_boxplot.png")
    
    def plot_learning_curve(self, model_name: str = 'Random Forest'):
        """Plot learning curve for best model."""
        print(f"\nGenerating learning curve for {model_name}...")
        
        # Find the model
        model_key = model_name.lower().replace(' ', '_')
        model = None
        for name, m in self.models.items():
            if model_key in name.lower().replace(' ', '_'):
                model = m
                break
        
        if model is None:
            print(f"Model {model_name} not found. Skipping learning curve.")
            return
        
        # Prepare data
        full_train_df = pd.concat([self.train_df, self.val_df])
        X, y = self._prepare_features(full_train_df)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Compute learning curve
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_scaled, y,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(train_sizes, train_mean, 'o-', color='#2196F3', label='Training Score')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.2, color='#2196F3')
        
        ax.plot(train_sizes, val_mean, 'o-', color='#4CAF50', label='Validation Score')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.2, color='#4CAF50')
        
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('F1-Score')
        ax.set_title(f'Learning Curve - {model_name}')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.5, 1.05)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'learning_curve.png'))
        plt.savefig(os.path.join(self.figures_dir, 'learning_curve.pdf'))
        plt.close()
        print("Saved: learning_curve.png")
    
    def plot_feature_distributions(self):
        """Plot distribution of numerical features by risk class."""
        print("\nGenerating feature distributions...")
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for idx, col in enumerate(self.numerical_cols):
            for cls, color in zip(self.class_names, self.class_colors):
                data = self.full_df[self.full_df['Risk Class'] == cls][col]
                axes[idx].hist(data, bins=20, alpha=0.5, label=cls, color=color, edgecolor='black')
            
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
            axes[idx].legend()
        
        # Hide last empty subplot
        axes[-1].set_visible(False)
        
        plt.suptitle('Feature Distributions by Risk Class', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'feature_distributions.png'))
        plt.close()
        print("Saved: feature_distributions.png")
    
    def generate_all_figures(self):
        """Generate all figures for the paper."""
        print("=" * 60)
        print("AsthmAI - Publication Figure Generator")
        print("=" * 60)
        
        # Load everything
        self.load_data()
        self.load_results()
        self.load_models()
        
        # Generate figures
        self.plot_class_distribution()
        self.plot_feature_correlation()
        self.plot_model_comparison()
        self.plot_confusion_matrices()
        self.plot_roc_curves()
        self.plot_cv_scores_boxplot()
        self.plot_learning_curve()
        self.plot_feature_distributions()
        
        print("\n" + "=" * 60)
        print("✓ All figures generated!")
        print(f"Output directory: {self.figures_dir}")
        print("=" * 60)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    data_dir = os.path.join(parent_dir, 'data')
    results_dir = os.path.join(parent_dir, 'results')
    figures_dir = os.path.join(parent_dir, 'figures')
    
    generator = FigureGenerator(data_dir, results_dir, figures_dir)
    generator.generate_all_figures()


if __name__ == "__main__":
    main()
