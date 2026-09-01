"""
AsthmAI - Fix #2: Enhanced Synthetic Data Validation
Adds bootstrap CIs, Cohen's d effect sizes, chi-squared tests,
and Wasserstein distance to strengthen synthetic-to-real transfer claims.
"""
import pandas as pd
import numpy as np
import warnings
from scipy import stats
from scipy.stats import ks_2samp, wasserstein_distance, chi2_contingency
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')
RANDOM_STATE = 42


def cohens_d(a, b):
    """Cohen's d effect size between two distributions."""
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    if pooled_std == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std


def bootstrap_accuracy(model_fn, X, y, n_bootstrap=500, test_size=0.2, seed=42):
    """Bootstrap confidence interval for model accuracy."""
    rng  = np.random.RandomState(seed)
    accs = []
    n    = len(X)
    n_test = int(n * test_size)

    for _ in range(n_bootstrap):
        idx       = rng.permutation(n)
        train_idx = idx[n_test:]
        test_idx  = idx[:n_test]

        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        sc  = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)

        m = model_fn()
        m.fit(Xtr, ytr)
        accs.append(accuracy_score(yte, m.predict(Xte)))

    accs = np.array(accs)
    return {
        'mean':  accs.mean(),
        'std':   accs.std(),
        'ci_lo': np.percentile(accs, 2.5),
        'ci_hi': np.percentile(accs, 97.5),
    }


def run_synthetic_validation():
    print("=" * 60)
    print("AsthmAI - Enhanced Synthetic Data Validation")
    print("=" * 60)

    original  = pd.read_csv('data/dataset_original.csv')
    synthetic = pd.read_csv('data/dataset.csv')
    print(f"Original:  {len(original)} samples")
    print(f"Synthetic: {len(synthetic)} samples")

    numerical_cols = ['AQI', 'PM2.5', 'SO2 level', 'NO2 level',
                      'CO2 level', 'Humidity', 'Temperature']
    categorical_cols = ['Asthma Symptoms Frequency', 'Poor Air Quality Exposure',
                        'Night Breathing Difficulty', 'Weather Sensitivity']

    lines = [
        "AsthmAI Enhanced Synthetic Data Validation\n",
        "=" * 50 + "\n\n",
        f"Original samples:  {len(original)}\n",
        f"Synthetic samples: {len(synthetic)}\n\n",
    ]

    # ── 1. KS Tests ───────────────────────────────────────────────
    print("\n--- Kolmogorov-Smirnov Tests ---")
    print("H0: Distributions identical. p > 0.05 = PASS")
    lines.append("1. Kolmogorov-Smirnov Tests (Numerical Features):\n")
    lines.append(f"   {'Feature':20s} {'KS stat':>8s} {'p-value':>10s} {'Result':>8s}\n")

    for col in numerical_cols:
        if col in original.columns and col in synthetic.columns:
            stat, p = ks_2samp(original[col].dropna(), synthetic[col].dropna())
            result  = "PASS" if p > 0.05 else "FAIL"
            print(f"  {col:20s}  stat={stat:.4f}  p={p:.4f}  {result}")
            lines.append(f"   {col:20s} {stat:>8.4f} {p:>10.4f} {result:>8s}\n")

    # ── 2. Cohen's d Effect Sizes ─────────────────────────────────
    print("\n--- Cohen's d Effect Sizes ---")
    print("(|d| < 0.2 = negligible, 0.2-0.5 = small, 0.5-0.8 = medium, >0.8 = large)")
    lines.append("\n2. Cohen's d Effect Sizes (Numerical Features):\n")
    lines.append(f"   {'Feature':20s} {'Cohen d':>10s} {'Interpretation':>16s}\n")

    for col in numerical_cols:
        if col in original.columns and col in synthetic.columns:
            d      = cohens_d(original[col].dropna(), synthetic[col].dropna())
            abs_d  = abs(d)
            if abs_d < 0.2:    interp = "Negligible"
            elif abs_d < 0.5:  interp = "Small"
            elif abs_d < 0.8:  interp = "Medium"
            else:              interp = "Large"
            print(f"  {col:20s}  d={d:+.4f}  ({interp})")
            lines.append(f"   {col:20s} {d:>+10.4f} {interp:>16s}\n")

    # ── 3. Wasserstein Distance ───────────────────────────────────
    print("\n--- Wasserstein Distance (lower = more similar) ---")
    lines.append("\n3. Wasserstein Distances:\n")

    for col in numerical_cols:
        if col in original.columns and col in synthetic.columns:
            o_norm = (original[col] - original[col].mean()) / (original[col].std() + 1e-8)
            s_norm = (synthetic[col] - synthetic[col].mean()) / (synthetic[col].std() + 1e-8)
            wd     = wasserstein_distance(o_norm.dropna(), s_norm.dropna())
            print(f"  {col:20s}  W={wd:.4f}")
            lines.append(f"   {col:20s}  W={wd:.4f}\n")

    # ── 4. Chi-squared Tests for Categorical ─────────────────────
    print("\n--- Chi-Squared Tests (Categorical Features) ---")
    lines.append("\n4. Chi-Squared Tests (Categorical Features):\n")

    for col in categorical_cols:
        if col in original.columns and col in synthetic.columns:
            cats  = pd.Categorical(
                pd.concat([original[col], synthetic[col]]).astype(str)
            ).categories
            obs_o = original[col].astype(str).value_counts().reindex(cats, fill_value=0)
            obs_s = synthetic[col].astype(str).value_counts().reindex(cats, fill_value=0)
            ct    = np.array([obs_o.values, obs_s.values])
            try:
                chi2, p, dof, _ = chi2_contingency(ct)
                result = "PASS" if p > 0.05 else "FAIL"
                print(f"  {col:30s}  chi2={chi2:.2f}  p={p:.4f}  {result}")
                lines.append(f"   {col:30s}  chi2={chi2:.2f}  p={p:.4f}  {result}\n")
            except:
                pass

    # ── 5. Bootstrap CI for Accuracy ─────────────────────────────
    print("\n--- Bootstrap 95% CI for Model Accuracy (n=200 iterations) ---")
    lines.append("\n5. Bootstrap 95% Confidence Intervals:\n")

    le = LabelEncoder()
    numerical = ['AQI', 'PM2.5', 'SO2 level', 'NO2 level', 'CO2 level', 'Humidity', 'Temperature']

    # Map categoricals
    symp_map  = {'Daily': 4, 'Frequently (Weekly)': 3, '1-2 times a month': 2, 'Less than once a month': 1}
    exp_map   = {'Yes, often': 3, 'Occasionally': 2, 'No': 1}
    night_map = {'Frequently': 3, 'Occasionally': 2, 'Rarely': 1, 'Never': 0}

    syn = synthetic.copy()
    syn['symp']  = syn['Asthma Symptoms Frequency'].map(symp_map).fillna(0)
    syn['exp']   = syn['Poor Air Quality Exposure'].map(exp_map).fillna(0)
    syn['night'] = syn['Night Breathing Difficulty'].map(night_map).fillna(0)

    X = syn[numerical + ['symp', 'exp', 'night']].fillna(0).values
    y = le.fit_transform(syn['Risk Class'])

    def model_fn():
        base = [
            ('xgb',  XGBClassifier(n_estimators=100, random_state=RANDOM_STATE,
                                   eval_metric='mlogloss', verbosity=0)),
            ('lgbm', LGBMClassifier(n_estimators=100, random_state=RANDOM_STATE, verbose=-1)),
            ('rf',   RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
        ]
        return StackingClassifier(estimators=base,
                                   final_estimator=LogisticRegression(max_iter=300),
                                   cv=3)

    print("  Running 200 bootstrap iterations (may take ~2 min)...")
    ci = bootstrap_accuracy(model_fn, X, y, n_bootstrap=200)
    print(f"  Accuracy: {ci['mean']*100:.1f}% (95% CI: {ci['ci_lo']*100:.1f}%--{ci['ci_hi']*100:.1f}%)")
    lines.append(f"   Stacking Accuracy: {ci['mean']*100:.1f}% "
                 f"(95% CI: {ci['ci_lo']*100:.1f}%--{ci['ci_hi']*100:.1f}%, "
                 f"SD={ci['std']*100:.1f}%)\n")

    with open('results/synthetic_validation_v2.txt', 'w') as f:
        f.writelines(lines)

    print("\nSaved: results/synthetic_validation_v2.txt")
    return ci


if __name__ == "__main__":
    run_synthetic_validation()
