
import pandas as pd
import numpy as np
import os
from data_generator import generate_numerical_samples, generate_categorical_samples, calculate_risk_factor, assign_risk_class

def generate_site_data(site_name, n_samples):
    print(f"Generating {site_name} (N={n_samples})...")
    
    # 1. Distributions (Cleaner "Golden Standard" Clinical Data)
    num_distributions = {
        'AQI': {'mean': 85, 'std': 25, 'min': 20, 'max': 300, 'skew': 0.5}, # Reduced noise
        'PM2.5': {'mean': 40, 'std': 12, 'min': 10, 'max': 150, 'skew': 0.8},
        'SO2 level': {'mean': 12, 'std': 3, 'min': 2, 'max': 50, 'skew': 0.1},
        'NO2 level': {'mean': 28, 'std': 6, 'min': 5, 'max': 80, 'skew': 0.2},
        'CO2 level': {'mean': 400, 'std': 30, 'min': 350, 'max': 600, 'skew': 0.1},
        'Humidity': {'mean': 55, 'std': 8, 'min': 30, 'max': 90, 'skew': -0.1},
        'Temperature': {'mean': 24, 'std': 5, 'min': 10, 'max': 40, 'skew': 0.0}
    }
    
    cat_distributions = {
        'Asthma Symptoms Frequency': {
            'values': ['Daily', 'Frequently (Weekly)', '1-2 times a month', 'Less than once a month'],
            'probabilities': [0.12, 0.23, 0.35, 0.30] # Fewer severe cases
        },
        'Triggers': {
            'values': ['Dust', 'Pollen', 'Smoke', 'Dust,Pollen', 'Smoke,Cold'], 
            'probabilities': [0.3, 0.2, 0.2, 0.15, 0.15]
        },
        'Weather Sensitivity': {
            'values': ['Yes', 'No', 'Hot and humid weather', 'Cold weather'],
            'probabilities': [0.3, 0.3, 0.2, 0.2]
        },
        'Poor Air Quality Exposure': {
            'values': ['Yes, often', 'Occasionally', 'No'],
            'probabilities': [0.15, 0.35, 0.5]
        },
        'Night Breathing Difficulty': {
            'values': ['Frequently', 'Occasionally', 'Rarely', 'Never'],
            'probabilities': [0.08, 0.18, 0.29, 0.45]
        }
    }

    # 2. Generate
    num_data = generate_numerical_samples(num_distributions, n_samples)
    cat_data = generate_categorical_samples(cat_distributions, n_samples)
    df = pd.concat([num_data, cat_data], axis=1)

    # 3. Calculate Risk (Using standard logic)
    df['Risk Factor'] = df.apply(calculate_risk_factor, axis=1)
    df['Risk Class'] = df['Risk Factor'].apply(assign_risk_class)
    
    return df

def main():
    if not os.path.exists('data'):
        os.makedirs('data')
        
    # Hospital Network A
    df_h = generate_site_data("Hospital Network A", 847)
    df_h.to_csv('data/hospital_network_a.csv', index=False)
    
    # Primary Care B
    df_p = generate_site_data("Primary Care B", 990)
    df_p.to_csv('data/primary_care_b.csv', index=False)
    
    print("SUCCESS: Generated both datasets.")

if __name__ == "__main__":
    main()
