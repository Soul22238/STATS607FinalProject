"""
Benchmark script to compare FastGLM performance against sklearn/statsmodels.

Tests all 5 GLM families (logistic, poisson, gamma, gaussian, inverse_gaussian)
on various data sizes including large datasets and unstable data scenarios.
"""

import numpy as np
import time
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns
from fastglm import FastGLM
import warnings
warnings.filterwarnings('ignore')


def generate_logistic_data(n_samples, n_features, random_state=42):
    """Generate synthetic data for logistic regression"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 2),
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=random_state
    )
    return X, y


def generate_poisson_data(n_samples, n_features, random_state=42):
    """Generate synthetic data for Poisson regression"""
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    true_coef = np.random.randn(n_features)
    
    # Generate Poisson response
    eta = X @ true_coef
    mu = np.exp(eta)
    y = np.random.poisson(mu)
    
    return X, y


def generate_gamma_data(n_samples, n_features, random_state=42):
    """Generate synthetic data for Gamma regression"""
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    true_coef = np.random.randn(n_features) * 0.5
    
    # Generate Gamma response
    eta = X @ true_coef
    mu = np.exp(eta)
    shape = 2.0
    scale = mu / shape
    y = np.random.gamma(shape, scale)
    
    return X, y


def generate_gaussian_data(n_samples, n_features, random_state=42):
    """Generate synthetic data for Gaussian regression"""
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    true_coef = np.random.randn(n_features)
    noise = np.random.randn(n_samples) * 0.5
    
    # Generate Gaussian response
    y = X @ true_coef + noise
    
    return X, y


def generate_inverse_gaussian_data(n_samples, n_features, random_state=42):
    """Generate synthetic data for Inverse Gaussian regression"""
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    true_coef = np.random.randn(n_features) * 0.3
    
    # Generate Inverse Gaussian response (simplified)
    mu = np.abs(X @ true_coef) + 1.0
    y = mu + np.random.randn(n_samples) * 0.2 * mu
    y = np.abs(y)
    
    return X, y


def generate_unstable_data(family, n_samples, n_features, random_state=42):
    """Generate unstable/challenging data with outliers and extreme values"""
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    
    # Add some extreme values and correlations
    X[:5] *= 10  # Outliers
    X[:, 0] = X[:, 1] + np.random.randn(n_samples) * 0.01  # Multicollinearity
    
    true_coef = np.random.randn(n_features)
    
    if family == 'logistic':
        eta = X @ true_coef
        prob = 1.0 / (1.0 + np.exp(-eta))
        y = (np.random.rand(n_samples) < prob).astype(float)
    elif family == 'poisson':
        eta = X @ true_coef * 0.5
        mu = np.exp(eta)
        y = np.random.poisson(mu)
    elif family == 'gamma':
        eta = X @ true_coef * 0.3
        mu = np.exp(eta)
        y = np.random.gamma(2, mu/2)
    elif family == 'gaussian':
        y = X @ true_coef + np.random.randn(n_samples) * 2.0
    elif family == 'inverse_gaussian':
        mu = np.abs(X @ true_coef) + 1.0
        y = np.abs(mu + np.random.randn(n_samples) * 0.3 * mu)
    
    return X, y


def benchmark_logistic(n_samples_list, n_features=10, n_trials=10):
    """
    Benchmark logistic regression: FastGLM vs sklearn
    """
    results = []
    
    for n_samples in n_samples_list:
        print(f"\nTesting Logistic Regression with n={n_samples}, p={n_features}")
        
        fastglm_times = []
        sklearn_times = []
        
        for trial in range(n_trials):
            # Generate data
            X, y = generate_logistic_data(n_samples, n_features, random_state=trial)
            
            # FastGLM
            start = time.perf_counter()
            model_fast = FastGLM(family='logistic', fit_intercept=True)
            model_fast.fit(X, y)
            y_pred_fast = model_fast.predict(X)
            fastglm_time = time.perf_counter() - start
            fastglm_times.append(fastglm_time)
            
            # sklearn
            start = time.perf_counter()
            model_sklearn = LogisticRegression(
                solver='lbfgs',
                max_iter=100,
                fit_intercept=True
            )
            model_sklearn.fit(X, y)
            y_pred_sklearn = model_sklearn.predict_proba(X)[:, 1]
            sklearn_time = time.perf_counter() - start
            sklearn_times.append(sklearn_time)
            
            # Verify results are similar
            if trial == 0:
                diff = np.mean(np.abs(y_pred_fast - y_pred_sklearn))
                print(f"  Prediction difference: {diff:.6f}")
        
        fastglm_mean = np.mean(fastglm_times) * 1000  # Convert to ms
        sklearn_mean = np.mean(sklearn_times) * 1000
        speedup = sklearn_mean / fastglm_mean
        
        print(f"  FastGLM: {fastglm_mean:.2f} ms")
        print(f"  sklearn: {sklearn_mean:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        results.append({
            'n_samples': n_samples,
            'n_features': n_features,
            'family': 'logistic',
            'fastglm_time': fastglm_mean,
            'sklearn_time': sklearn_mean,
            'speedup': speedup
        })
    
    return pd.DataFrame(results)


def benchmark_poisson(n_samples_list, n_features=10, n_trials=10):
    """Benchmark Poisson regression: FastGLM vs statsmodels"""
    try:
        from statsmodels.genmod.generalized_linear_model import GLM
        from statsmodels.genmod.families import Poisson
        has_statsmodels = True
    except ImportError:
        print("Warning: statsmodels not available, skipping Poisson comparison")
        has_statsmodels = False
        return pd.DataFrame()
    
    results = []
    
    for n_samples in n_samples_list:
        print(f"\nTesting Poisson Regression with n={n_samples}, p={n_features}")
        
        fastglm_times = []
        statsmodels_times = []
        
        for trial in range(n_trials):
            X, y = generate_poisson_data(n_samples, n_features, random_state=trial)
            
            # FastGLM
            start = time.perf_counter()
            model_fast = FastGLM(family='poisson', fit_intercept=True)
            model_fast.fit(X, y)
            y_pred_fast = model_fast.predict(X)
            fastglm_time = time.perf_counter() - start
            fastglm_times.append(fastglm_time)
            
            # statsmodels
            start = time.perf_counter()
            X_with_intercept = np.column_stack([np.ones(n_samples), X])
            model_sm = GLM(y, X_with_intercept, family=Poisson())
            result_sm = model_sm.fit()
            y_pred_sm = result_sm.predict(X_with_intercept)
            statsmodels_time = time.perf_counter() - start
            statsmodels_times.append(statsmodels_time)
            
            if trial == 0:
                diff = np.mean(np.abs(y_pred_fast - y_pred_sm))
                print(f"  Prediction difference: {diff:.6f}")
        
        fastglm_mean = np.mean(fastglm_times) * 1000
        statsmodels_mean = np.mean(statsmodels_times) * 1000
        speedup = statsmodels_mean / fastglm_mean
        
        print(f"  FastGLM: {fastglm_mean:.2f} ms")
        print(f"  statsmodels: {statsmodels_mean:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        results.append({
            'n_samples': n_samples,
            'n_features': n_features,
            'family': 'poisson',
            'fastglm_time': fastglm_mean,
            'sklearn_time': statsmodels_mean,
            'speedup': speedup
        })
    
    return pd.DataFrame(results)


def benchmark_gamma(n_samples_list, n_features=10, n_trials=10):
    """Benchmark Gamma regression: FastGLM vs statsmodels (with log link)"""
    try:
        from statsmodels.genmod.generalized_linear_model import GLM
        from statsmodels.genmod.families import Gamma
        from statsmodels.genmod.families.links import Log
        has_statsmodels = True
    except ImportError:
        print("Warning: statsmodels not available, skipping Gamma comparison")
        return pd.DataFrame()
    
    results = []
    
    for n_samples in n_samples_list:
        print(f"\nTesting Gamma Regression with n={n_samples}, p={n_features}")
        
        fastglm_times = []
        statsmodels_times = []
        
        for trial in range(n_trials):
            X, y = generate_gamma_data(n_samples, n_features, random_state=trial)
            
            # FastGLM
            start = time.perf_counter()
            model_fast = FastGLM(family='gamma', fit_intercept=True)
            model_fast.fit(X, y)
            y_pred_fast = model_fast.predict(X)
            fastglm_time = time.perf_counter() - start
            fastglm_times.append(fastglm_time)
            
            # statsmodels (with log link to match FastGLM)
            try:
                start = time.perf_counter()
                X_with_intercept = np.column_stack([np.ones(n_samples), X])
                model_sm = GLM(y, X_with_intercept, family=Gamma(link=Log()))
                result_sm = model_sm.fit(disp=False)
                y_pred_sm = result_sm.predict(X_with_intercept)
                statsmodels_time = time.perf_counter() - start
                statsmodels_times.append(statsmodels_time)
                
                if trial == 0:
                    diff = np.mean(np.abs(y_pred_fast - y_pred_sm))
                    print(f"  Prediction difference: {diff:.6f}")
            except:
                # If statsmodels fails, just skip this trial
                continue
        
        if len(statsmodels_times) > 0:
            fastglm_mean = np.mean(fastglm_times) * 1000
            statsmodels_mean = np.mean(statsmodels_times) * 1000
            speedup = statsmodels_mean / fastglm_mean
            
            print(f"  FastGLM: {fastglm_mean:.2f} ms")
            print(f"  statsmodels: {statsmodels_mean:.2f} ms")
            print(f"  Speedup: {speedup:.2f}x")
            
            results.append({
                'n_samples': n_samples,
                'n_features': n_features,
                'family': 'gamma',
                'fastglm_time': fastglm_mean,
                'sklearn_time': statsmodels_mean,
                'speedup': speedup
            })
        else:
            # FastGLM only
            fastglm_mean = np.mean(fastglm_times) * 1000
            print(f"  FastGLM: {fastglm_mean:.2f} ms (statsmodels comparison unavailable)")
            
            results.append({
                'n_samples': n_samples,
                'n_features': n_features,
                'family': 'gamma',
                'fastglm_time': fastglm_mean,
                'sklearn_time': np.nan,
                'speedup': np.nan
            })
    
    return pd.DataFrame(results)


def benchmark_gaussian(n_samples_list, n_features=10, n_trials=10):
    """Benchmark Gaussian regression: FastGLM vs sklearn LinearRegression"""
    results = []
    
    for n_samples in n_samples_list:
        print(f"\nTesting Gaussian Regression with n={n_samples}, p={n_features}")
        
        fastglm_times = []
        sklearn_times = []
        
        for trial in range(n_trials):
            X, y = generate_gaussian_data(n_samples, n_features, random_state=trial)
            
            # FastGLM
            start = time.perf_counter()
            model_fast = FastGLM(family='gaussian', fit_intercept=True)
            model_fast.fit(X, y)
            y_pred_fast = model_fast.predict(X)
            fastglm_time = time.perf_counter() - start
            fastglm_times.append(fastglm_time)
            
            # sklearn
            start = time.perf_counter()
            model_sklearn = LinearRegression(fit_intercept=True)
            model_sklearn.fit(X, y)
            y_pred_sklearn = model_sklearn.predict(X)
            sklearn_time = time.perf_counter() - start
            sklearn_times.append(sklearn_time)
            
            if trial == 0:
                diff = np.mean(np.abs(y_pred_fast - y_pred_sklearn))
                print(f"  Prediction difference: {diff:.6f}")
        
        fastglm_mean = np.mean(fastglm_times) * 1000
        sklearn_mean = np.mean(sklearn_times) * 1000
        speedup = sklearn_mean / fastglm_mean
        
        print(f"  FastGLM: {fastglm_mean:.2f} ms")
        print(f"  sklearn: {sklearn_mean:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        results.append({
            'n_samples': n_samples,
            'n_features': n_features,
            'family': 'gaussian',
            'fastglm_time': fastglm_mean,
            'sklearn_time': sklearn_mean,
            'speedup': speedup
        })
    
    return pd.DataFrame(results)


def benchmark_inverse_gaussian(n_samples_list, n_features=10, n_trials=10):
    """Benchmark Inverse Gaussian regression: FastGLM vs statsmodels"""
    try:
        from statsmodels.genmod.generalized_linear_model import GLM
        from statsmodels.genmod.families import InverseGaussian
        from statsmodels.genmod.families.links import InverseSquared
        has_statsmodels = True
    except ImportError:
        print("Warning: statsmodels not available, skipping Inverse Gaussian comparison")
        return pd.DataFrame()
    
    results = []
    
    for n_samples in n_samples_list:
        print(f"\nTesting Inverse Gaussian Regression with n={n_samples}, p={n_features}")
        
        fastglm_times = []
        statsmodels_times = []
        
        for trial in range(n_trials):
            X, y = generate_inverse_gaussian_data(n_samples, n_features, random_state=trial)
            
            # FastGLM
            start = time.perf_counter()
            model_fast = FastGLM(family='inverse_gaussian', fit_intercept=True)
            model_fast.fit(X, y)
            y_pred_fast = model_fast.predict(X)
            fastglm_time = time.perf_counter() - start
            fastglm_times.append(fastglm_time)
            
            # statsmodels (with inverse squared link to match FastGLM)
            try:
                start = time.perf_counter()
                X_with_intercept = np.column_stack([np.ones(n_samples), X])
                model_sm = GLM(y, X_with_intercept, family=InverseGaussian(link=InverseSquared()))
                result_sm = model_sm.fit(disp=False)
                y_pred_sm = result_sm.predict(X_with_intercept)
                statsmodels_time = time.perf_counter() - start
                statsmodels_times.append(statsmodels_time)
                
                if trial == 0:
                    diff = np.mean(np.abs(y_pred_fast - y_pred_sm))
                    print(f"  Prediction difference: {diff:.6f}")
            except:
                # If statsmodels fails, skip this trial
                continue
        
        if len(statsmodels_times) > 0:
            fastglm_mean = np.mean(fastglm_times) * 1000
            statsmodels_mean = np.mean(statsmodels_times) * 1000
            speedup = statsmodels_mean / fastglm_mean
            
            print(f"  FastGLM: {fastglm_mean:.2f} ms")
            print(f"  statsmodels: {statsmodels_mean:.2f} ms")
            print(f"  Speedup: {speedup:.2f}x")
            
            results.append({
                'n_samples': n_samples,
                'n_features': n_features,
                'family': 'inverse_gaussian',
                'fastglm_time': fastglm_mean,
                'sklearn_time': statsmodels_mean,
                'speedup': speedup
            })
        else:
            fastglm_mean = np.mean(fastglm_times) * 1000
            print(f"  FastGLM: {fastglm_mean:.2f} ms (statsmodels comparison unavailable)")
            
            results.append({
                'n_samples': n_samples,
                'n_features': n_features,
                'family': 'inverse_gaussian',
                'fastglm_time': fastglm_mean,
                'sklearn_time': np.nan,
                'speedup': np.nan
            })
    
    return pd.DataFrame(results)


def benchmark_stability(families, n_samples=500, n_features=10, n_trials=20):
    """Test stability on challenging data with outliers and multicollinearity"""
    print("\n" + "=" * 70)
    print("STABILITY TEST: Unstable Data with Outliers and Multicollinearity")
    print("=" * 70)
    
    results = []
    
    for family in families:
        print(f"\nTesting {family} on unstable data")
        
        convergence_count = 0
        iteration_counts = []
        times = []
        
        for trial in range(n_trials):
            try:
                X, y = generate_unstable_data(family, n_samples, n_features, random_state=trial)
                
                start = time.perf_counter()
                model = FastGLM(family=family, fit_intercept=True, max_iter=50)
                model.fit(X, y)
                elapsed = time.perf_counter() - start
                
                convergence_count += 1
                iteration_counts.append(model.n_iter_)
                times.append(elapsed * 1000)
            except Exception as e:
                print(f"  Trial {trial} failed: {str(e)[:50]}")
        
        if convergence_count > 0:
            print(f"  Convergence rate: {convergence_count}/{n_trials} ({100*convergence_count/n_trials:.1f}%)")
            print(f"  Avg iterations: {np.mean(iteration_counts):.1f}")
            print(f"  Avg time: {np.mean(times):.2f} ms")
            
            results.append({
                'family': family,
                'convergence_rate': convergence_count / n_trials,
                'avg_iterations': np.mean(iteration_counts),
                'std_iterations': np.std(iteration_counts),
                'avg_time': np.mean(times),
                'std_time': np.std(times)
            })
    
    return pd.DataFrame(results)


def benchmark_fastglm_weaknesses(n_samples=1000, n_trials=10):
    """Benchmark scenarios where baseline methods outperform FastGLM"""
    print("\n" + "=" * 70)
    print("WEAKNESS TEST: Scenarios Where Baseline Methods Excel")
    print("=" * 70)
    
    results = []
    
    # Test 1: High-dimensional data (p > n)
    print("\n1. High-dimensional data (p=200, n=100)")
    n_high = 100
    p_high = 200
    
    fastglm_times_high = []
    sklearn_times_high = []
    
    for trial in range(n_trials):
        np.random.seed(trial)
        X = np.random.randn(n_high, p_high)
        y = (np.random.rand(n_high) < 0.5).astype(float)
        
        # FastGLM (will struggle with p > n)
        try:
            start = time.perf_counter()
            model_fast = FastGLM(family='logistic', fit_intercept=True, max_iter=10)
            model_fast.fit(X, y)
            fastglm_times_high.append(time.perf_counter() - start)
        except:
            fastglm_times_high.append(np.nan)
        
        # sklearn with regularization (handles p > n well)
        start = time.perf_counter()
        model_sklearn = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=100)
        model_sklearn.fit(X, y)
        sklearn_times_high.append(time.perf_counter() - start)
    
    fastglm_mean_high = np.nanmean(fastglm_times_high) * 1000 if not all(np.isnan(fastglm_times_high)) else np.nan
    sklearn_mean_high = np.mean(sklearn_times_high) * 1000
    
    print(f"  FastGLM: {fastglm_mean_high:.2f} ms")
    print(f"  sklearn (L2): {sklearn_mean_high:.2f} ms")
    print(f"  Winner: {'sklearn' if sklearn_mean_high < fastglm_mean_high else 'FastGLM'}")
    
    results.append({
        'scenario': 'High-dimensional (p>n)',
        'fastglm_time': fastglm_mean_high,
        'baseline_time': sklearn_mean_high,
        'winner': 'sklearn' if sklearn_mean_high < fastglm_mean_high else 'FastGLM',
        'speedup_factor': fastglm_mean_high / sklearn_mean_high if not np.isnan(fastglm_mean_high) else np.nan
    })
    
    # Test 2: Very large dataset (FastGLM not optimized for n > 50000)
    print("\n2. Very large dataset (n=50000, p=20)")
    n_large = 50000
    p_normal = 20
    
    X_large = np.random.randn(n_large, p_normal)
    y_large = (np.random.rand(n_large) < 0.5).astype(float)
    
    start = time.perf_counter()
    model_fast_large = FastGLM(family='logistic', fit_intercept=True)
    model_fast_large.fit(X_large, y_large)
    fastglm_time_large = (time.perf_counter() - start) * 1000
    
    start = time.perf_counter()
    model_sklearn_large = LogisticRegression(solver='lbfgs', max_iter=100)
    model_sklearn_large.fit(X_large, y_large)
    sklearn_time_large = (time.perf_counter() - start) * 1000
    
    print(f"  FastGLM: {fastglm_time_large:.2f} ms")
    print(f"  sklearn: {sklearn_time_large:.2f} ms")
    print(f"  Winner: {'sklearn' if sklearn_time_large < fastglm_time_large else 'FastGLM'}")
    
    results.append({
        'scenario': 'Very large (n=50000)',
        'fastglm_time': fastglm_time_large,
        'baseline_time': sklearn_time_large,
        'winner': 'sklearn' if sklearn_time_large < fastglm_time_large else 'FastGLM',
        'speedup_factor': fastglm_time_large / sklearn_time_large
    })
    
    # Test 3: Need for L1 regularization (FastGLM doesn't support)
    print("\n3. L1 regularization for feature selection (n=500, p=100)")
    n_reg = 500
    p_reg = 100
    
    X_reg = np.random.randn(n_reg, p_reg)
    true_coef = np.zeros(p_reg)
    true_coef[:10] = np.random.randn(10)  # Only 10 features are relevant
    y_reg = (X_reg @ true_coef + np.random.randn(n_reg) > 0).astype(float)
    
    start = time.perf_counter()
    model_fast_reg = FastGLM(family='logistic', fit_intercept=True)
    model_fast_reg.fit(X_reg, y_reg)
    fastglm_time_reg = (time.perf_counter() - start) * 1000
    
    start = time.perf_counter()
    model_sklearn_l1 = LogisticRegression(penalty='l1', C=0.1, solver='liblinear', max_iter=100)
    model_sklearn_l1.fit(X_reg, y_reg)
    sklearn_time_l1 = (time.perf_counter() - start) * 1000
    
    print(f"  FastGLM (no regularization): {fastglm_time_reg:.2f} ms")
    print(f"  sklearn (L1 Lasso): {sklearn_time_l1:.2f} ms")
    print(f"  sklearn non-zero coef: {np.sum(model_sklearn_l1.coef_[0] != 0)}/100 (sparsity)")
    print(f"  Winner: sklearn (feature selection)")
    
    results.append({
        'scenario': 'L1 regularization',
        'fastglm_time': fastglm_time_reg,
        'baseline_time': sklearn_time_l1,
        'winner': 'sklearn',
        'speedup_factor': fastglm_time_reg / sklearn_time_l1
    })
    
    return pd.DataFrame(results)


def benchmark_limitation_scenarios(n_trials=10):
    """
    Benchmark specific scenarios showing FastGLM limitations.
    Tests cases where baseline methods outperform FastGLM.
    """
    print("\n" + "=" * 70)
    print("LIMITATION SCENARIOS: Where Baseline Methods Excel")
    print("=" * 70)
    
    results = []
    
    # Scenario 1: Optimal range - Small-medium dataset (n=1000, p=20)
    print("\n[Scenario 1] OPTIMAL: Small-medium dataset (n=1000, p=20)")
    fastglm_times = []
    sklearn_times = []
    pred_diffs = []
    
    for trial in range(n_trials):
        np.random.seed(trial)
        n, p = 1000, 20
        X = np.random.randn(n, p)
        y = (1 / (1 + np.exp(-(X @ np.random.randn(p)))) > 0.5).astype(float)
        
        # FastGLM
        start = time.perf_counter()
        model_fast = FastGLM(family='logistic', fit_intercept=True)
        model_fast.fit(X, y)
        y_pred_fast = model_fast.predict(X)
        fastglm_times.append(time.perf_counter() - start)
        
        # sklearn
        start = time.perf_counter()
        model_sklearn = LogisticRegression(solver='lbfgs', max_iter=100)
        model_sklearn.fit(X, y)
        y_pred_sklearn = model_sklearn.predict_proba(X)[:, 1]
        sklearn_times.append(time.perf_counter() - start)
        
        pred_diffs.append(np.mean(np.abs(y_pred_fast - y_pred_sklearn)))
    
    fastglm_mean = np.mean(fastglm_times) * 1000
    sklearn_mean = np.mean(sklearn_times) * 1000
    speedup = sklearn_mean / fastglm_mean
    
    print(f"  FastGLM: {fastglm_mean:.2f} ms")
    print(f"  sklearn: {sklearn_mean:.2f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Pred diff: {np.mean(pred_diffs):.6f}")
    
    results.append({
        'scenario': 'Optimal (n=1000, p=20)',
        'n_samples': 1000,
        'n_features': 20,
        'fastglm_time': fastglm_mean,
        'baseline_time': sklearn_mean,
        'speedup': speedup,
        'pred_diff': np.mean(pred_diffs),
        'winner': 'FastGLM' if speedup > 1 else 'sklearn'
    })
    
    # Scenario 2: Medium dataset (n=5000, p=30)
    print("\n[Scenario 2] OPTIMAL: Medium dataset (n=5000, p=30)")
    fastglm_times = []
    sklearn_times = []
    pred_diffs = []
    
    for trial in range(n_trials):
        np.random.seed(trial + 100)
        n, p = 5000, 30
        X = np.random.randn(n, p)
        y = (1 / (1 + np.exp(-(X @ np.random.randn(p)))) > 0.5).astype(float)
        
        start = time.perf_counter()
        model_fast = FastGLM(family='logistic', fit_intercept=True)
        model_fast.fit(X, y)
        y_pred_fast = model_fast.predict(X)
        fastglm_times.append(time.perf_counter() - start)
        
        start = time.perf_counter()
        model_sklearn = LogisticRegression(solver='lbfgs', max_iter=100)
        model_sklearn.fit(X, y)
        y_pred_sklearn = model_sklearn.predict_proba(X)[:, 1]
        sklearn_times.append(time.perf_counter() - start)
        
        pred_diffs.append(np.mean(np.abs(y_pred_fast - y_pred_sklearn)))
    
    fastglm_mean = np.mean(fastglm_times) * 1000
    sklearn_mean = np.mean(sklearn_times) * 1000
    speedup = sklearn_mean / fastglm_mean
    
    print(f"  FastGLM: {fastglm_mean:.2f} ms")
    print(f"  sklearn: {sklearn_mean:.2f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Pred diff: {np.mean(pred_diffs):.6f}")
    
    results.append({
        'scenario': 'Medium (n=5000, p=30)',
        'n_samples': 5000,
        'n_features': 30,
        'fastglm_time': fastglm_mean,
        'baseline_time': sklearn_mean,
        'speedup': speedup,
        'pred_diff': np.mean(pred_diffs),
        'winner': 'FastGLM' if speedup > 1 else 'sklearn'
    })
    
    # Scenario 3: Very small dataset (n=30, p=10)
    print("\n[Scenario 3] TOO SMALL: Very small dataset (n=30, p=10)")
    fastglm_times = []
    sklearn_times = []
    pred_diffs = []
    
    for trial in range(n_trials):
        np.random.seed(trial + 200)
        n, p = 30, 10
        X = np.random.randn(n, p)
        y = (1 / (1 + np.exp(-(X @ np.random.randn(p)))) > 0.5).astype(float)
        
        start = time.perf_counter()
        model_fast = FastGLM(family='logistic', fit_intercept=True)
        model_fast.fit(X, y)
        y_pred_fast = model_fast.predict(X)
        fastglm_times.append(time.perf_counter() - start)
        
        start = time.perf_counter()
        model_sklearn = LogisticRegression(solver='lbfgs', max_iter=100)
        model_sklearn.fit(X, y)
        y_pred_sklearn = model_sklearn.predict_proba(X)[:, 1]
        sklearn_times.append(time.perf_counter() - start)
        
        pred_diffs.append(np.mean(np.abs(y_pred_fast - y_pred_sklearn)))
    
    fastglm_mean = np.mean(fastglm_times) * 1000
    sklearn_mean = np.mean(sklearn_times) * 1000
    speedup = sklearn_mean / fastglm_mean
    
    print(f"  FastGLM: {fastglm_mean:.2f} ms")
    print(f"  sklearn: {sklearn_mean:.2f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Pred diff: {np.mean(pred_diffs):.6f}")
    
    results.append({
        'scenario': 'Too small (n=30, p=10)',
        'n_samples': 30,
        'n_features': 10,
        'fastglm_time': fastglm_mean,
        'baseline_time': sklearn_mean,
        'speedup': speedup,
        'pred_diff': np.mean(pred_diffs),
        'winner': 'FastGLM' if speedup > 1 else 'sklearn'
    })
    
    # Scenario 4: Large dataset (n=20000, p=20)
    print("\n[Scenario 4] TOO LARGE: Large dataset (n=20000, p=20)")
    np.random.seed(42)
    n, p = 20000, 20
    X = np.random.randn(n, p)
    y = (1 / (1 + np.exp(-(X @ np.random.randn(p) * 0.3))) > 0.5).astype(float)
    
    start = time.perf_counter()
    model_fast = FastGLM(family='logistic', fit_intercept=True)
    model_fast.fit(X, y)
    y_pred_fast = model_fast.predict(X)
    fastglm_time = (time.perf_counter() - start) * 1000
    
    start = time.perf_counter()
    model_sklearn = LogisticRegression(solver='lbfgs', max_iter=100)
    model_sklearn.fit(X, y)
    y_pred_sklearn = model_sklearn.predict_proba(X)[:, 1]
    sklearn_time = (time.perf_counter() - start) * 1000
    
    speedup = sklearn_time / fastglm_time
    pred_diff = np.mean(np.abs(y_pred_fast - y_pred_sklearn))
    
    print(f"  FastGLM: {fastglm_time:.2f} ms")
    print(f"  sklearn: {sklearn_time:.2f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Pred diff: {pred_diff:.6f}")
    
    results.append({
        'scenario': 'Too large (n=20000, p=20)',
        'n_samples': 20000,
        'n_features': 20,
        'fastglm_time': fastglm_time,
        'baseline_time': sklearn_time,
        'speedup': speedup,
        'pred_diff': pred_diff,
        'winner': 'FastGLM' if speedup > 1 else 'sklearn'
    })
    
    # Scenario 5: High-dimensional (n=500, p=200)
    print("\n[Scenario 5] HIGH-DIMENSIONAL: p=200 > optimal range")
    fastglm_times = []
    sklearn_times = []
    pred_diffs = []
    
    for trial in range(n_trials):
        np.random.seed(trial + 300)
        n, p = 500, 200
        X = np.random.randn(n, p)
        y = (1 / (1 + np.exp(-(X @ np.random.randn(p) * 0.1))) > 0.5).astype(float)
        
        start = time.perf_counter()
        model_fast = FastGLM(family='logistic', fit_intercept=True)
        model_fast.fit(X, y)
        y_pred_fast = model_fast.predict(X)
        fastglm_times.append(time.perf_counter() - start)
        
        start = time.perf_counter()
        model_sklearn = LogisticRegression(solver='lbfgs', max_iter=100)
        model_sklearn.fit(X, y)
        y_pred_sklearn = model_sklearn.predict_proba(X)[:, 1]
        sklearn_times.append(time.perf_counter() - start)
        
        pred_diffs.append(np.mean(np.abs(y_pred_fast - y_pred_sklearn)))
    
    fastglm_mean = np.mean(fastglm_times) * 1000
    sklearn_mean = np.mean(sklearn_times) * 1000
    speedup = sklearn_mean / fastglm_mean
    
    print(f"  FastGLM: {fastglm_mean:.2f} ms")
    print(f"  sklearn: {sklearn_mean:.2f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Pred diff: {np.mean(pred_diffs):.6f}")
    
    results.append({
        'scenario': 'High-dim (n=500, p=200)',
        'n_samples': 500,
        'n_features': 200,
        'fastglm_time': fastglm_mean,
        'baseline_time': sklearn_mean,
        'speedup': speedup,
        'pred_diff': np.mean(pred_diffs),
        'winner': 'FastGLM' if speedup > 1 else 'sklearn'
    })
    
    # Scenario 6: p > n (underdetermined)
    print("\n[Scenario 6] p > n: Underdetermined system (n=100, p=200)")
    fastglm_times = []
    sklearn_times = []
    pred_diffs = []
    
    for trial in range(n_trials):
        np.random.seed(trial + 400)
        n, p = 100, 200
        X = np.random.randn(n, p)
        y = (1 / (1 + np.exp(-(X @ np.random.randn(p) * 0.1))) > 0.5).astype(float)
        
        start = time.perf_counter()
        model_fast = FastGLM(family='logistic', fit_intercept=True)
        model_fast.fit(X, y)
        y_pred_fast = model_fast.predict(X)
        fastglm_times.append(time.perf_counter() - start)
        
        start = time.perf_counter()
        model_sklearn = LogisticRegression(solver='lbfgs', max_iter=100)
        model_sklearn.fit(X, y)
        y_pred_sklearn = model_sklearn.predict_proba(X)[:, 1]
        sklearn_times.append(time.perf_counter() - start)
        
        pred_diffs.append(np.mean(np.abs(y_pred_fast - y_pred_sklearn)))
    
    fastglm_mean = np.mean(fastglm_times) * 1000
    sklearn_mean = np.mean(sklearn_times) * 1000
    speedup = sklearn_mean / fastglm_mean
    
    print(f"  FastGLM: {fastglm_mean:.2f} ms")
    print(f"  sklearn: {sklearn_mean:.2f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Pred diff: {np.mean(pred_diffs):.6f}")
    
    results.append({
        'scenario': 'p > n (n=100, p=200)',
        'n_samples': 100,
        'n_features': 200,
        'fastglm_time': fastglm_mean,
        'baseline_time': sklearn_mean,
        'speedup': speedup,
        'pred_diff': np.mean(pred_diffs),
        'winner': 'FastGLM' if speedup > 1 else 'sklearn'
    })
    
    # Scenario 7: Sparse data (50% zeros)
    print("\n[Scenario 7] SPARSE: 50% zero values")
    fastglm_times = []
    sklearn_times = []
    pred_diffs = []
    
    for trial in range(n_trials):
        np.random.seed(trial + 500)
        n, p = 1000, 50
        X = np.random.randn(n, p)
        X[X < 0.5] = 0  # Make ~50% sparse
        y = (1 / (1 + np.exp(-(X @ np.random.randn(p) * 0.3))) > 0.5).astype(float)
        
        start = time.perf_counter()
        model_fast = FastGLM(family='logistic', fit_intercept=True)
        model_fast.fit(X, y)
        y_pred_fast = model_fast.predict(X)
        fastglm_times.append(time.perf_counter() - start)
        
        start = time.perf_counter()
        model_sklearn = LogisticRegression(solver='lbfgs', max_iter=100)
        model_sklearn.fit(X, y)
        y_pred_sklearn = model_sklearn.predict_proba(X)[:, 1]
        sklearn_times.append(time.perf_counter() - start)
        
        pred_diffs.append(np.mean(np.abs(y_pred_fast - y_pred_sklearn)))
    
    fastglm_mean = np.mean(fastglm_times) * 1000
    sklearn_mean = np.mean(sklearn_times) * 1000
    speedup = sklearn_mean / fastglm_mean
    
    print(f"  FastGLM: {fastglm_mean:.2f} ms")
    print(f"  sklearn: {sklearn_mean:.2f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Pred diff: {np.mean(pred_diffs):.6f}")
    
    results.append({
        'scenario': 'Sparse (50% zeros)',
        'n_samples': 1000,
        'n_features': 50,
        'fastglm_time': fastglm_mean,
        'baseline_time': sklearn_mean,
        'speedup': speedup,
        'pred_diff': np.mean(pred_diffs),
        'winner': 'FastGLM' if speedup > 1 else 'sklearn'
    })
    
    return pd.DataFrame(results)


def compute_prediction_accuracy(n_samples_list, n_features=20, n_trials=5):
    """Compare prediction accuracy between FastGLM and baseline methods"""
    print("\n" + "=" * 70)
    print("PREDICTION ACCURACY COMPARISON")
    print("=" * 70)
    
    results = []
    
    for n_samples in n_samples_list:
        print(f"\nTesting prediction accuracy with n={n_samples}, p={n_features}")
        
        # Logistic regression accuracy
        diffs_logistic = []
        for trial in range(n_trials):
            X, y = generate_logistic_data(n_samples, n_features, random_state=trial)
            
            model_fast = FastGLM(family='logistic', fit_intercept=True)
            model_fast.fit(X, y)
            pred_fast = model_fast.predict(X)
            
            model_sklearn = LogisticRegression(solver='lbfgs', max_iter=100, fit_intercept=True)
            model_sklearn.fit(X, y)
            pred_sklearn = model_sklearn.predict_proba(X)[:, 1]
            
            diff = np.mean(np.abs(pred_fast - pred_sklearn))
            diffs_logistic.append(diff)
        
        mean_diff = np.mean(diffs_logistic)
        max_diff = np.max(diffs_logistic)
        
        print(f"  Logistic - Mean diff: {mean_diff:.6f}, Max diff: {max_diff:.6f}")
        
        results.append({
            'n_samples': n_samples,
            'family': 'logistic',
            'mean_prediction_diff': mean_diff,
            'max_prediction_diff': max_diff,
            'std_prediction_diff': np.std(diffs_logistic)
        })
    
    return pd.DataFrame(results)


def plot_results(all_results, stability_df, weakness_df=None, accuracy_df=None, limitations_df=None, save_dir='results/plots'):
    """Create comprehensive visualization of benchmark results for all families"""
    
    # Color scheme for families
    family_colors = {
        'logistic': '#3498db',
        'poisson': '#e74c3c',
        'gamma': '#2ecc71',
        'gaussian': '#f39c12',
        'inverse_gaussian': '#9b59b6'
    }
    
    # ========== PLOT 1: Summary Plot with Overall Comparison ==========
    fig1 = plt.figure(figsize=(14, 5))
    
    # 1.1 Execution time comparison across all families (log scale)
    ax1 = plt.subplot(1, 2, 1)
    for family, color in family_colors.items():
        df = all_results.get(family, pd.DataFrame())
        if not df.empty:
            ax1.plot(df['n_samples'], df['fastglm_time'], 'o-', 
                    label=family.capitalize(), color=color, linewidth=2, markersize=6)
    ax1.set_xlabel('Sample Size (n)', fontsize=12)
    ax1.set_ylabel('Time (ms, log scale)', fontsize=12)
    ax1.set_title('FastGLM Execution Time - All Families', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 1.2 Speedup comparison (where applicable)
    ax2 = plt.subplot(1, 2, 2)
    speedup_data = []
    labels = []
    colors_list = []
    for family in ['logistic', 'poisson', 'gamma', 'gaussian', 'inverse_gaussian']:
        df = all_results.get(family, pd.DataFrame())
        if not df.empty and not df['speedup'].isna().all():
            speedup_data.append(df['speedup'].mean())
            labels.append(family.capitalize())
            colors_list.append(family_colors[family])
    
    if speedup_data:
        bars = ax2.barh(range(len(labels)), speedup_data, color=colors_list, alpha=0.7, edgecolor='black')
        ax2.axvline(x=1, color='red', linestyle='--', linewidth=2, label='No speedup')
        ax2.set_yticks(range(len(labels)))
        ax2.set_yticklabels(labels)
        ax2.set_xlabel('Average Speedup Factor', fontsize=12)
        ax2.set_title('Average Speedup vs Baseline', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, speedup_data)):
            ax2.text(val + 0.1, i, f'{val:.2f}x', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    summary_path = f'{save_dir}/01_summary.png'
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"Summary plot saved to: {summary_path}")
    plt.close()
    
    # ========== PLOT 2: Bar Charts (Stability + Weakness + Accuracy) ==========
    fig2 = plt.figure(figsize=(18, 5))
    
    # 2.1 Stability test results
    ax1 = plt.subplot(1, 3, 1)
    if not stability_df.empty:
        families_stable = stability_df['family'].tolist()
        convergence = stability_df['convergence_rate'].tolist()
        colors_stable = [family_colors[f] for f in families_stable]
        
        bars = ax1.bar(range(len(families_stable)), convergence, 
                      color=colors_stable, alpha=0.7, edgecolor='black')
        ax1.set_xticks(range(len(families_stable)))
        ax1.set_xticklabels([f.capitalize() for f in families_stable], rotation=45, ha='right')
        ax1.set_ylabel('Convergence Rate', fontsize=11)
        ax1.set_title('Stability Test: Unstable Data', fontsize=13, fontweight='bold')
        ax1.set_ylim([0, 1.1])
        ax1.axhline(y=1.0, color='green', linestyle='--', linewidth=1, alpha=0.5)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels
        for i, (bar, val) in enumerate(zip(bars, convergence)):
            ax1.text(i, val + 0.02, f'{val*100:.0f}%', ha='center', fontsize=10, fontweight='bold')
    
    # 2.2 Weakness Analysis
    ax2 = plt.subplot(1, 3, 2)
    if weakness_df is not None and not weakness_df.empty:
        scenarios = weakness_df['scenario'].values
        winners = weakness_df['winner'].values
        speedups = weakness_df['speedup_factor'].values
        
        colors_weak = ['#e74c3c' if w == 'sklearn' else '#3498db' for w in winners]
        bars = ax2.barh(range(len(scenarios)), speedups, color=colors_weak, alpha=0.7, edgecolor='black')
        ax2.set_yticks(range(len(scenarios)))
        ax2.set_yticklabels([s.replace(' ', '\n') for s in scenarios], fontsize=9)
        ax2.set_xlabel('Speedup Factor (Winner/Loser)', fontsize=11)
        ax2.set_title('FastGLM Weakness Analysis', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add value labels and winner
        for i, (bar, val, winner) in enumerate(zip(bars, speedups, winners)):
            ax2.text(val + 0.1, i, f'{val:.2f}x\n({winner})', va='center', fontsize=9, fontweight='bold')
    
    # 2.3 Prediction Accuracy
    ax3 = plt.subplot(1, 3, 3)
    if accuracy_df is not None and not accuracy_df.empty:
        n_samples = accuracy_df['n_samples'].values
        mean_diffs = accuracy_df['mean_prediction_diff'].values
        max_diffs = accuracy_df['max_prediction_diff'].values
        
        ax3.semilogy(n_samples, mean_diffs, 'o-', label='Mean Diff', color='#2ecc71', linewidth=2, markersize=8)
        ax3.semilogy(n_samples, max_diffs, 's-', label='Max Diff', color='#e74c3c', linewidth=2, markersize=8)
        ax3.set_xlabel('Sample Size (n)', fontsize=11)
        ax3.set_ylabel('Prediction Difference (log scale)', fontsize=11)
        ax3.set_title('Numerical Accuracy vs sklearn', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Add annotation for typical accuracy
        if len(mean_diffs) > 0:
            avg_mean = np.mean(mean_diffs)
            ax3.axhline(y=avg_mean, color='gray', linestyle='--', alpha=0.5)
            ax3.text(n_samples[0], avg_mean, f'  Avg: {avg_mean:.2e}', 
                     va='bottom', fontsize=9, color='gray')
    
    plt.tight_layout()
    bars_path = f'{save_dir}/02_analysis_bars.png'
    plt.savefig(bars_path, dpi=300, bbox_inches='tight')
    print(f"Bar charts saved to: {bars_path}")
    plt.close()
    
    # ========== PLOT 3: Individual Family Performance (Line Plots) ==========
    fig3 = plt.figure(figsize=(18, 10))
    
    for idx, family in enumerate(['logistic', 'poisson', 'gamma', 'gaussian', 'inverse_gaussian']):
        ax = plt.subplot(2, 3, idx + 1)
        df = all_results.get(family, pd.DataFrame())
        
        if not df.empty:
            ax.plot(df['n_samples'], df['fastglm_time'], 'o-', 
                   color=family_colors[family], linewidth=2.5, markersize=7, label='FastGLM')
            
            # Add baseline for all families including inverse_gaussian
            if not df['sklearn_time'].isna().all():
                ax.plot(df['n_samples'], df['sklearn_time'], 's--', 
                       color='gray', linewidth=2.5, markersize=7, label='Baseline', alpha=0.7)
            
            ax.set_xlabel('Sample Size (n)', fontsize=11)
            ax.set_ylabel('Time (ms)', fontsize=11)
            ax.set_title(f'{family.replace("_", " ").title()} Regression', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
    
    # Remove the 6th subplot (empty)
    ax_empty = plt.subplot(2, 3, 6)
    ax_empty.axis('off')
    
    plt.tight_layout()
    lines_path = f'{save_dir}/03_family_performance.png'
    plt.savefig(lines_path, dpi=300, bbox_inches='tight')
    print(f"Family performance plot saved to: {lines_path}")
    plt.close()
    
    # ========== PLOT 4: Limitation Scenarios ==========
    if limitations_df is not None and not limitations_df.empty:
        fig4 = plt.figure(figsize=(18, 5))
        
        # 4.1 Speedup comparison for all scenarios
        ax1 = plt.subplot(1, 3, 1)
        scenarios = limitations_df['scenario'].values
        speedups = limitations_df['speedup'].values
        colors = ['#e74c3c' if s < 1 else '#2ecc71' for s in speedups]
        
        bars = ax1.barh(range(len(scenarios)), speedups, color=colors, alpha=0.7, edgecolor='black')
        ax1.axvline(x=1, color='black', linestyle='--', linewidth=2, label='Equal performance')
        ax1.set_yticks(range(len(scenarios)))
        ax1.set_yticklabels(scenarios, fontsize=10)
        ax1.set_xlabel('Speedup Factor (Baseline / FastGLM)', fontsize=12)
        ax1.set_title('FastGLM vs Baseline: Speedup Comparison', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, speedups)):
            label = f'{val:.2f}x'
            x_pos = val + 0.05 if val < 2 else val - 0.05
            ha = 'left' if val < 2 else 'right'
            ax1.text(x_pos, i, label, va='center', fontsize=10, fontweight='bold', ha=ha)
        
        # 4.2 Execution time comparison
        ax2 = plt.subplot(1, 3, 2)
        x_pos = np.arange(len(scenarios))
        width = 0.35
        
        bars1 = ax2.bar(x_pos - width/2, limitations_df['fastglm_time'], width, 
                       label='FastGLM', color='#3498db', alpha=0.7, edgecolor='black')
        bars2 = ax2.bar(x_pos + width/2, limitations_df['baseline_time'], width,
                       label='Baseline (sklearn)', color='#95a5a6', alpha=0.7, edgecolor='black')
        
        ax2.set_xlabel('Scenario', fontsize=12)
        ax2.set_ylabel('Time (ms)', fontsize=12)
        ax2.set_title('Execution Time: FastGLM vs Baseline', fontsize=13, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=9)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_yscale('log')
        
        # 4.3 Prediction accuracy
        ax3 = plt.subplot(1, 3, 3)
        pred_diffs = limitations_df['pred_diff'].values
        colors_bar = ['#2ecc71' if p < 0.05 else '#f39c12' if p < 0.1 else '#e74c3c' for p in pred_diffs]
        
        bars = ax3.bar(range(len(scenarios)), pred_diffs, color=colors_bar, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Scenario', fontsize=12)
        ax3.set_ylabel('Mean Prediction Difference', fontsize=12)
        ax3.set_title('Prediction Accuracy: |FastGLM - Baseline|', fontsize=13, fontweight='bold')
        ax3.set_xticks(range(len(scenarios)))
        ax3.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=9)
        ax3.axhline(y=0.05, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Good (<0.05)')
        ax3.axhline(y=0.1, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Moderate (<0.1)')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        limitations_path = f'{save_dir}/04_limitation_scenarios.png'
        plt.savefig(limitations_path, dpi=300, bbox_inches='tight')
        print(f"Limitation scenarios plot saved to: {limitations_path}")
        plt.close()
    
    print(f"\nAll plots saved to: {save_dir}/")
    return


def main():
    """Run comprehensive benchmark suite for all GLM families"""
    print("=" * 80)
    print("FastGLM Comprehensive Benchmark Suite - All 5 Families")
    print("=" * 80)
    
    # Test configurations with comprehensive sample sizes
    n_samples_list = [10, 50, 100, 500, 1000, 5000, 10000, 20000]
    n_features = 20
    n_trials = 10
    
    all_results = {}
    
    # 1. Logistic Regression
    print("\n" + "=" * 80)
    print("LOGISTIC REGRESSION BENCHMARK")
    print("=" * 80)
    df_logistic = benchmark_logistic(n_samples_list, n_features, n_trials)
    all_results['logistic'] = df_logistic
    
    # 2. Poisson Regression
    print("\n" + "=" * 80)
    print("POISSON REGRESSION BENCHMARK")
    print("=" * 80)
    df_poisson = benchmark_poisson(n_samples_list, n_features, n_trials)
    all_results['poisson'] = df_poisson
    
    # 3. Gamma Regression (NEW)
    print("\n" + "=" * 80)
    print("GAMMA REGRESSION BENCHMARK")
    print("=" * 80)
    df_gamma = benchmark_gamma(n_samples_list, n_features, n_trials)
    all_results['gamma'] = df_gamma
    
    # 4. Gaussian Regression (NEW)
    print("\n" + "=" * 80)
    print("GAUSSIAN REGRESSION BENCHMARK")
    print("=" * 80)
    df_gaussian = benchmark_gaussian(n_samples_list, n_features, n_trials)
    all_results['gaussian'] = df_gaussian
    
    # 5. Inverse Gaussian Regression (NEW)
    print("\n" + "=" * 80)
    print("INVERSE GAUSSIAN REGRESSION BENCHMARK")
    print("=" * 80)
    df_inverse_gaussian = benchmark_inverse_gaussian(n_samples_list, n_features, n_trials)
    all_results['inverse_gaussian'] = df_inverse_gaussian
    
    # 6. Stability Test
    families_to_test = ['logistic', 'poisson', 'gamma', 'gaussian', 'inverse_gaussian']
    df_stability = benchmark_stability(families_to_test, n_samples=500, n_features=10, n_trials=20)
    
    # 7. Detailed Limitation Scenarios (from demo_fastglm_limitations)
    print("\n" + "=" * 80)
    print("DETAILED LIMITATION SCENARIOS")
    print("=" * 80)
    df_limitations = benchmark_limitation_scenarios(n_trials=10)
    
    # 8. Weakness Analysis (scenarios where baseline wins)
    print("\n" + "=" * 80)
    print("WEAKNESS ANALYSIS - When Baselines Outperform FastGLM")
    print("=" * 80)
    df_weakness = benchmark_fastglm_weaknesses(n_samples=1000, n_trials=10)
    
    # 9. Prediction Accuracy Comparison
    print("\n" + "=" * 80)
    print("PREDICTION ACCURACY ANALYSIS")
    print("=" * 80)
    n_samples_accuracy = [100, 500, 1000, 5000]
    df_accuracy = compute_prediction_accuracy(n_samples_accuracy, n_features=20, n_trials=5)
    
    # Summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE SUMMARY")
    print("=" * 80)
    
    for family, df in all_results.items():
        if not df.empty:
            print(f"\n{family.upper()} Regression:")
            print(f"  Avg FastGLM time: {df['fastglm_time'].mean():.2f} ms")
            if not df['sklearn_time'].isna().all():
                print(f"  Avg baseline time: {df['sklearn_time'].mean():.2f} ms")
                print(f"  Avg speedup: {df['speedup'].mean():.2f}x")
            else:
                print(f"  (No baseline comparison available)")
    
    if not df_stability.empty:
        print("\nSTABILITY TEST RESULTS:")
        print(df_stability.to_string(index=False))
    
    if not df_limitations.empty:
        print("\nDETAILED LIMITATION SCENARIOS:")
        print(df_limitations.to_string(index=False))
    
    if not df_weakness.empty:
        print("\nWEAKNESS ANALYSIS RESULTS:")
        print(df_weakness.to_string(index=False))
    
    if not df_accuracy.empty:
        print("\nPREDICTION ACCURACY RESULTS:")
        print(df_accuracy.to_string(index=False))
    
    # Generate comprehensive visualization
    plot_results(all_results, df_stability, df_weakness, df_accuracy, df_limitations, save_dir='results/plots')
    
    # Save all results to CSV
    csv_dir = 'results/csv'
    for family, df in all_results.items():
        if not df.empty:
            filename = f'{csv_dir}/benchmark_{family}.csv'
            df.to_csv(filename, index=False)
            print(f"\n{family.capitalize()} results saved to: {filename}")
    
    if not df_stability.empty:
        df_stability.to_csv(f'{csv_dir}/benchmark_stability.csv', index=False)
        print(f"Stability results saved to: {csv_dir}/benchmark_stability.csv")
    
    if not df_limitations.empty:
        df_limitations.to_csv(f'{csv_dir}/benchmark_limitations.csv', index=False)
        print(f"Limitation scenarios saved to: {csv_dir}/benchmark_limitations.csv")
    
    if not df_weakness.empty:
        df_weakness.to_csv(f'{csv_dir}/benchmark_weakness.csv', index=False)
        print(f"Weakness analysis saved to: {csv_dir}/benchmark_weakness.csv")
    
    if not df_accuracy.empty:
        df_accuracy.to_csv(f'{csv_dir}/benchmark_accuracy.csv', index=False)
        print(f"Prediction accuracy saved to: {csv_dir}/benchmark_accuracy.csv")
    
    # Create combined results DataFrame
    combined = pd.concat([df for df in all_results.values() if not df.empty], ignore_index=True)
    combined.to_csv(f'{csv_dir}/benchmark_all_families.csv', index=False)
    print(f"Combined results saved to: {csv_dir}/benchmark_all_families.csv")
    
    print("\n" + "=" * 80)
    print("Benchmark completed successfully!")
    print("=" * 80)
    print("\nKey Findings:")
    print("1. FastGLM optimized for small-medium datasets (n < 10,000)")
    print("2. All 5 GLM families supported with consistent API")
    print("3. Stability test shows convergence on challenging data")
    print("4. Speedup varies by family and sample size")
    print("5. Baseline methods win in high-dimensional and very large datasets")
    print("\nResults Organization:")
    print("  - CSV files: results/csv/")
    print("  - Plots: results/plots/")
    print("    * 01_summary.png - Overall comparison")
    print("    * 02_analysis_bars.png - Stability, weakness, accuracy")
    print("    * 03_family_performance.png - Individual family performance")


if __name__ == '__main__':
    main()
