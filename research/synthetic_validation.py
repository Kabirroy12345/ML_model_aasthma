"""
AsthmAI - Synthetic Data Validation Framework
Goal: Validate that synthetic data matches original distribution and doesn't introduce bias.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def validate_synthetic_data():
    print("=" * 60)
    print("AsthmAI - Synthetic Data Validation")
    print("=" * 60)
    
    # Load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
    
    original_df = pd.read_csv(os.path.join(data_dir, 'dataset_original.csv'))
    synthetic_df = pd.read_csv(os.path.join(data_dir, 'dataset.csv'))
    
    print(f"Original samples: {len(original_df)}")
    print(f"Synthetic samples: {len(synthetic_df)}")
    
    numerical_cols = [
        'AQI', 'PM2.5', 'SO2 level', 'NO2 level', 
        'CO2 level', 'Humidity', 'Temperature'
    ]
    
    # 1. Statistical Tests (KS-Test)
    print("\n--- Kolmogorov-Smirnov Test (Distribution Similarity) ---")
    print("H0: Distributions are identical. p > 0.05 fails to reject H0 (Good)")
    
    ks_results = []
    for col in numerical_cols:
        stat, p_value = stats.ks_2samp(original_df[col], synthetic_df[col])
        result = "Pass" if p_value > 0.05 else "Fail"
        print(f"{col}: p-value = {p_value:.4f} ({result})")
        ks_results.append({'Feature': col, 'p-value': p_value, 'Result': result})
        
    # 2. KDE Plots (Visual Comparison)
    print("\nGenerating Distribution Comparison Plots...")
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_cols):
        sns.kdeplot(data=original_df, x=col, label='Original', ax=axes[i], fill=True, alpha=0.3)
        sns.kdeplot(data=synthetic_df, x=col, label='Synthetic', ax=axes[i], fill=True, alpha=0.3)
        axes[i].set_title(f'{col} Distribution')
        axes[i].legend()
        
    # Remove empty subplots
    for i in range(len(numerical_cols), 9):
        fig.delaxes(axes[i])
        
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'synthetic_validation_kde.png'), dpi=300)
    print("Saved: synthetic_validation_kde.png")
    
    # 3. PCA/t-SNE Visualization (Manifold overlap)
    print("\nGenerating PCA/t-SNE Manifold Comparison...")
    
    # Combine for projection
    orig_subset = original_df[numerical_cols].copy()
    orig_subset['Source'] = 'Original'
    
    # Sample synthetic to match size for fair comparison implies dense points
    # But here we want to see coverage, so take a sample of 500
    syn_subset = synthetic_df[numerical_cols].sample(n=500, random_state=42).copy()
    syn_subset['Source'] = 'Synthetic'
    
    combined = pd.concat([orig_subset, syn_subset])
    X_combined = combined[numerical_cols].values
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_combined)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=combined['Source'], alpha=0.6)
    plt.title('PCA Projection: Original vs Synthetic Data')
    plt.savefig(os.path.join(figures_dir, 'synthetic_validation_pca.png'), dpi=300)
    print("Saved: synthetic_validation_pca.png")
    
    print("\nValidation complete!")

if __name__ == "__main__":
    validate_synthetic_data()
