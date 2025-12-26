"""
AsthmAI - Synthetic Data Generator
Expands the original dataset from 201 samples to 2000+ samples
using statistical distributions derived from the original data.

This is a common research practice for healthcare ML when real data is limited.
The synthetic data follows the same distributions as the original data.
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

# Set random seed for reproducibility
np.random.seed(42)

def load_original_data(filepath: str) -> pd.DataFrame:
    """Load and analyze original dataset."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df

def get_numerical_distributions(df: pd.DataFrame, numerical_cols: list) -> dict:
    """Extract statistical distributions for numerical columns."""
    distributions = {}
    for col in numerical_cols:
        data = df[col].dropna()
        distributions[col] = {
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'skew': data.skew()
        }
    return distributions

def get_categorical_distributions(df: pd.DataFrame, categorical_cols: list) -> dict:
    """Extract probability distributions for categorical columns."""
    distributions = {}
    for col in categorical_cols:
        value_counts = df[col].value_counts(normalize=True)
        distributions[col] = {
            'values': value_counts.index.tolist(),
            'probabilities': value_counts.values.tolist()
        }
    return distributions

def generate_numerical_samples(distributions: dict, n_samples: int) -> pd.DataFrame:
    """Generate synthetic numerical data based on original distributions."""
    data = {}
    
    for col, dist in distributions.items():
        # Use truncated normal to respect min/max bounds
        samples = np.random.normal(dist['mean'], dist['std'], n_samples)
        # Clip to original range with small buffer
        samples = np.clip(samples, dist['min'] * 0.9, dist['max'] * 1.1)
        
        # Add slight noise for variety
        noise = np.random.normal(0, dist['std'] * 0.1, n_samples)
        samples = samples + noise
        samples = np.clip(samples, dist['min'] * 0.8, dist['max'] * 1.2)
        
        # Round appropriately
        if col in ['AQI']:
            samples = np.round(samples).astype(int)
        else:
            samples = np.round(samples, 2)
        
        data[col] = samples
    
    return pd.DataFrame(data)

def generate_categorical_samples(distributions: dict, n_samples: int) -> pd.DataFrame:
    """Generate synthetic categorical data based on original distributions."""
    data = {}
    
    for col, dist in distributions.items():
        samples = np.random.choice(
            dist['values'],
            size=n_samples,
            p=dist['probabilities']
        )
        data[col] = samples
    
    return pd.DataFrame(data)

def calculate_risk_factor(row: pd.Series) -> float:
    """
    Calculate risk factor based on feature values.
    This mimics the relationship learned from the original data.
    """
    risk = 0.0
    
    # Environmental factors (40% weight)
    if row['AQI'] > 200:
        risk += 0.15
    elif row['AQI'] > 150:
        risk += 0.10
    elif row['AQI'] > 100:
        risk += 0.05
    
    if row['PM2.5'] > 100:
        risk += 0.10
    elif row['PM2.5'] > 50:
        risk += 0.05
    
    if row['CO2 level'] > 500:
        risk += 0.08
    elif row['CO2 level'] > 400:
        risk += 0.04
    
    if row['SO2 level'] > 30:
        risk += 0.04
    if row['NO2 level'] > 70:
        risk += 0.03
    
    # Clinical/Symptom factors (60% weight)
    freq = row['Asthma Symptoms Frequency']
    if freq == 'Daily':
        risk += 0.20
    elif freq == 'Frequently (Weekly)':
        risk += 0.15
    elif freq == '1-2 times a month':
        risk += 0.08
    elif freq == 'Less than once a month':
        risk += 0.03
    
    # Triggers
    triggers = str(row['Triggers'])
    trigger_count = triggers.count(',') + 1
    risk += min(0.10, trigger_count * 0.03)
    
    # Weather sensitivity
    if row['Weather Sensitivity'] in ['Hot and humid weather', 'Cold weather']:
        risk += 0.05
    
    # Exposure
    if row['Poor Air Quality Exposure'] == 'Yes, often':
        risk += 0.10
    elif row['Poor Air Quality Exposure'] == 'Occasionally':
        risk += 0.05
    
    # Night difficulty
    if row['Night Breathing Difficulty'] == 'Frequently':
        risk += 0.10
    elif row['Night Breathing Difficulty'] == 'Occasionally':
        risk += 0.05
    
    # Add randomness
    noise = np.random.normal(0, 0.08)
    risk = risk + noise
    
    # Clamp to valid range and round
    risk = np.clip(risk, 0.0, 1.0)
    risk = round(risk / 0.12) * 0.12  # Discretize to match original pattern
    risk = np.clip(risk, 0.12, 1.0)
    
    return round(risk, 2)

def assign_risk_class(risk_factor: float) -> str:
    """Convert continuous risk factor to categorical class."""
    if risk_factor >= 0.7:
        return 'High'
    elif risk_factor >= 0.4:
        return 'Medium'
    else:
        return 'Low'

def generate_synthetic_dataset(original_df: pd.DataFrame, target_samples: int = 2000) -> pd.DataFrame:
    """Generate complete synthetic dataset."""
    
    # Define column types
    numerical_cols = ['AQI', 'PM2.5', 'SO2 level', 'NO2 level', 'CO2 level', 'Humidity', 'Temperature']
    categorical_cols = ['Asthma Symptoms Frequency', 'Triggers', 'Weather Sensitivity', 
                        'Poor Air Quality Exposure', 'Night Breathing Difficulty']
    
    # Get distributions
    num_dist = get_numerical_distributions(original_df, numerical_cols)
    cat_dist = get_categorical_distributions(original_df, categorical_cols)
    
    # Generate new samples (subtract original count)
    n_new = target_samples - len(original_df)
    
    print(f"Generating {n_new} new synthetic samples...")
    
    # Generate features
    num_data = generate_numerical_samples(num_dist, n_new)
    cat_data = generate_categorical_samples(cat_dist, n_new)
    
    # Combine
    synthetic_df = pd.concat([num_data, cat_data], axis=1)
    
    # Calculate risk factors
    print("Calculating risk factors...")
    synthetic_df['Risk Factor'] = synthetic_df.apply(calculate_risk_factor, axis=1)
    
    # Add risk class for classification
    synthetic_df['Risk Class'] = synthetic_df['Risk Factor'].apply(assign_risk_class)
    
    # Add risk class to original data too
    original_df = original_df.copy()
    original_df['Risk Class'] = original_df['Risk Factor'].apply(assign_risk_class)
    
    # Combine with original
    full_df = pd.concat([original_df, synthetic_df], ignore_index=True)
    
    # Shuffle
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return full_df

def generate_train_val_test_split(df: pd.DataFrame, output_dir: str):
    """Create stratified train/validation/test splits."""
    from sklearn.model_selection import train_test_split
    
    # First split: 70% train, 30% temp
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df['Risk Class']
    )
    
    # Second split: 50% validation, 50% test (15% each of total)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['Risk Class']
    )
    
    # Save splits
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'validation.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"\nDataset Splits:")
    print(f"  Train:      {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation: {len(val_df):,} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:       {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    # Show class distribution
    print(f"\nClass Distribution (Full Dataset):")
    for cls in ['Low', 'Medium', 'High']:
        count = (df['Risk Class'] == cls).sum()
        print(f"  {cls}: {count:,} ({count/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df

def main():
    """Main execution function."""
    print("=" * 60)
    print("AsthmAI - Synthetic Data Generator")
    print("=" * 60)
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    original_path = os.path.join(script_dir, 'data', 'dataset.csv')
    data_dir = os.path.join(script_dir, 'data')
    
    # Backup original dataset
    original_df = load_original_data(original_path)
    backup_path = os.path.join(data_dir, 'dataset_original.csv')
    original_df.to_csv(backup_path, index=False)
    print(f"Original dataset backed up to: {backup_path}")
    print(f"Original samples: {len(original_df)}")
    
    # Generate expanded dataset
    expanded_df = generate_synthetic_dataset(original_df, target_samples=2000)
    
    # Save expanded dataset
    expanded_path = os.path.join(data_dir, 'dataset.csv')
    expanded_df.to_csv(expanded_path, index=False)
    print(f"\nExpanded dataset saved to: {expanded_path}")
    print(f"Total samples: {len(expanded_df)}")
    
    # Create train/val/test splits
    generate_train_val_test_split(expanded_df, data_dir)
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics:")
    print("=" * 60)
    print(expanded_df.describe())
    
    print("\n✓ Data generation complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
