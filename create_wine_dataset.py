"""
Generate Wine Quality Dataset
Creates a synthetic wine quality dataset based on the UCI Wine Quality dataset structure
"""
import pandas as pd
import numpy as np

def create_wine_dataset(n_samples=1000, output_path='dags/data/wine_quality.csv'):
    """
    Create a synthetic wine quality dataset
    """
    np.random.seed(42)

    # Generate features with realistic distributions
    data = {
        'fixed_acidity': np.random.normal(8.3, 1.7, n_samples),
        'volatile_acidity': np.random.normal(0.53, 0.18, n_samples),
        'citric_acid': np.random.normal(0.27, 0.19, n_samples),
        'residual_sugar': np.random.gamma(2, 3, n_samples),
        'chlorides': np.random.normal(0.087, 0.047, n_samples),
        'free_sulfur_dioxide': np.random.gamma(2, 7, n_samples),
        'total_sulfur_dioxide': np.random.gamma(3, 15, n_samples),
        'density': np.random.normal(0.997, 0.002, n_samples),
        'pH': np.random.normal(3.31, 0.15, n_samples),
        'sulphates': np.random.normal(0.66, 0.17, n_samples),
        'alcohol': np.random.normal(10.4, 1.1, n_samples),
    }

    df = pd.DataFrame(data)

    # Clip values to realistic ranges
    df['fixed_acidity'] = df['fixed_acidity'].clip(4, 16)
    df['volatile_acidity'] = df['volatile_acidity'].clip(0.1, 1.6)
    df['citric_acid'] = df['citric_acid'].clip(0, 1)
    df['residual_sugar'] = df['residual_sugar'].clip(0.9, 15.5)
    df['chlorides'] = df['chlorides'].clip(0.01, 0.6)
    df['free_sulfur_dioxide'] = df['free_sulfur_dioxide'].clip(1, 72)
    df['total_sulfur_dioxide'] = df['total_sulfur_dioxide'].clip(6, 290)
    df['density'] = df['density'].clip(0.99, 1.04)
    df['pH'] = df['pH'].clip(2.7, 4.0)
    df['sulphates'] = df['sulphates'].clip(0.33, 2.0)
    df['alcohol'] = df['alcohol'].clip(8, 15)

    # Generate quality scores (3-8) based on features
    # Higher alcohol, lower volatile acidity generally means better quality
    quality_score = (
        df['alcohol'] * 0.4 +
        (1 / (df['volatile_acidity'] + 0.1)) * 2 +
        df['citric_acid'] * 3 +
        df['sulphates'] * 2 +
        np.random.normal(0, 1.5, n_samples)
    )

    # Normalize to 3-8 scale
    quality_score = (quality_score - quality_score.min()) / (quality_score.max() - quality_score.min())
    df['quality'] = (quality_score * 5 + 3).round().astype(int)
    df['quality'] = df['quality'].clip(3, 8)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Dataset created: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"\nQuality distribution:")
    print(df['quality'].value_counts().sort_index())
    print(f"\nFirst few rows:")
    print(df.head())

    return df

if __name__ == "__main__":
    create_wine_dataset()