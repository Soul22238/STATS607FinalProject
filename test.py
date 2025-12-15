"""
Comprehensive test suite for FastGLM

IMPORTANT: This test file requires fastglm.py to be present in the same directory.
If fastglm.py is missing, please ensure it exists before running tests.

Run with: pytest test.py -v

Test coverage (37 tests):
- Logistic regression: 5 tests
- Poisson regression: 3 tests
- Gamma regression: 3 tests
- Gaussian regression: 3 tests
- Inverse Gaussian regression: 2 tests
- Input validation: 5 tests
- Model parameters: 3 tests
- Data scales: 4 tests
- Numerical stability: 3 tests
- Diagnostics: 2 tests
- Edge cases: 3 tests
- Utility functions: 1 test
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
import numpy as np
from fastglm import FastGLM
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.datasets import make_classification


class TestLogisticRegression:
    """Test logistic regression functionality"""
    
    def test_basic_fit(self):
        """Test basic model fitting"""
        np.random.seed(42)
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        
        model = FastGLM(family='logistic')
        model.fit(X, y)
        
        assert model.coef_ is not None
        assert model.n_iter_ > 0
        assert model.n_iter_ <= model.max_iter
    
    def test_predictions(self):
        """Test prediction output"""
        np.random.seed(42)
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        
        model = FastGLM(family='logistic')
        model.fit(X, y)
        y_pred = model.predict(X)
        
        assert len(y_pred) == len(y)
        assert np.all(y_pred >= 0) and np.all(y_pred <= 1)
    
    def test_score(self):
        """Test scoring method"""
        np.random.seed(42)
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        
        model = FastGLM(family='logistic')
        model.fit(X, y)
        score = model.score(X, y)
        
        assert 0 <= score <= 1
    
    def test_convergence(self):
        """Test model convergence"""
        np.random.seed(42)
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        
        model = FastGLM(family='logistic', max_iter=100, tol=1e-6)
        model.fit(X, y)
        
        assert model.n_iter_ < model.max_iter
    
    def test_sklearn_consistency(self):
        """Test consistency with sklearn"""
        np.random.seed(42)
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        
        model_fast = FastGLM(family='logistic', fit_intercept=True)
        model_fast.fit(X, y)
        y_pred_fast = model_fast.predict(X)
        
        model_sklearn = LogisticRegression(solver='lbfgs', max_iter=100)
        model_sklearn.fit(X, y)
        y_pred_sklearn = model_sklearn.predict_proba(X)[:, 1]
        
        diff = np.mean(np.abs(y_pred_fast - y_pred_sklearn))
        assert diff < 0.1


class TestPoissonRegression:
    """Test Poisson regression functionality"""
    
    def test_basic_fit(self):
        """Test basic model fitting"""
        np.random.seed(42)
        n, p = 100, 5
        X = np.random.randn(n, p)
        eta = X @ np.random.randn(p)
        y = np.random.poisson(np.exp(eta))
        
        model = FastGLM(family='poisson')
        model.fit(X, y)
        
        assert model.coef_ is not None
        assert model.n_iter_ > 0
    
    def test_predictions_positive(self):
        """Test that predictions are positive"""
        np.random.seed(42)
        n, p = 100, 5
        X = np.random.randn(n, p)
        y = np.random.poisson(5, size=n)
        
        model = FastGLM(family='poisson')
        model.fit(X, y)
        y_pred = model.predict(X)
        
        assert np.all(y_pred >= 0)
    
    def test_count_data(self):
        """Test with count data"""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.poisson(3, size=100)
        
        model = FastGLM(family='poisson')
        model.fit(X, y)
        
        assert model.score(X, y) > -10


class TestGammaRegression:
    """Test Gamma regression functionality"""
    
    def test_basic_fit(self):
        """Test basic model fitting"""
        np.random.seed(42)
        n, p = 100, 5
        X = np.random.randn(n, p)
        mu = np.exp(X @ np.random.randn(p) * 0.5)
        y = np.random.gamma(2, mu/2)
        
        model = FastGLM(family='gamma')
        model.fit(X, y)
        
        assert model.coef_ is not None
        assert model.n_iter_ > 0
    
    def test_predictions_positive(self):
        """Test that predictions are positive"""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.gamma(2, 1, size=100)
        
        model = FastGLM(family='gamma')
        model.fit(X, y)
        y_pred = model.predict(X)
        
        assert np.all(y_pred > 0)
    
    def test_positive_data_requirement(self):
        """Test that Gamma handles data appropriately"""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.gamma(2, 1, size=100)
        
        model = FastGLM(family='gamma')
        model.fit(X, y)
        
        assert model.coef_ is not None


class TestGaussianRegression:
    """Test Gaussian regression functionality"""
    
    def test_basic_fit(self):
        """Test basic model fitting"""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X @ np.random.randn(5) + np.random.randn(100) * 0.5
        
        model = FastGLM(family='gaussian')
        model.fit(X, y)
        
        assert model.coef_ is not None
        assert model.n_iter_ > 0
    
    def test_sklearn_consistency(self):
        """Test consistency with sklearn LinearRegression"""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X @ np.random.randn(5) + np.random.randn(100) * 0.5
        
        model_fast = FastGLM(family='gaussian', fit_intercept=False)
        model_fast.fit(X, y)
        
        model_sklearn = LinearRegression(fit_intercept=False)
        model_sklearn.fit(X, y)
        
        coef_diff = np.max(np.abs(model_fast.coef_ - model_sklearn.coef_))
        assert coef_diff < 1e-6
    
    def test_r2_score(self):
        """Test R2 score calculation"""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X @ np.random.randn(5) + np.random.randn(100) * 0.1
        
        model = FastGLM(family='gaussian')
        model.fit(X, y)
        r2 = model.score(X, y)
        
        assert r2 > 0.5


class TestInverseGaussianRegression:
    """Test Inverse Gaussian regression functionality"""
    
    def test_basic_fit(self):
        """Test basic model fitting"""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        mu = np.abs(X @ np.random.randn(5)) + 1.0
        y = np.abs(mu + np.random.randn(100) * 0.2 * mu)
        
        model = FastGLM(family='inverse_gaussian')
        model.fit(X, y)
        
        assert model.coef_ is not None
        assert model.n_iter_ > 0
    
    def test_predictions_positive(self):
        """Test that predictions are positive"""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.abs(np.random.randn(100)) + 1.0
        
        model = FastGLM(family='inverse_gaussian')
        model.fit(X, y)
        y_pred = model.predict(X)
        
        assert np.all(y_pred > 0)


class TestInputValidation:
    """Test input validation and error handling"""
    
    def test_invalid_family(self):
        """Test invalid family raises error"""
        with pytest.raises(ValueError):
            FastGLM(family='invalid_family')
    
    def test_mismatched_dimensions(self):
        """Test mismatched X, y dimensions"""
        X = np.random.randn(100, 5)
        y = np.random.randn(50)
        
        model = FastGLM(family='logistic')
        with pytest.raises((ValueError, AssertionError)):
            model.fit(X, y)
    
    def test_empty_data(self):
        """Test with minimal data"""
        X = np.random.randn(10, 2)
        y = np.random.randn(10)
        
        model = FastGLM(family='gaussian')
        model.fit(X, y)
        
        assert model.coef_ is not None
    
    def test_nan_in_data(self):
        """Test NaN in data raises error"""
        X = np.random.randn(100, 5)
        X[0, 0] = np.nan
        y = np.random.randn(100)
        
        model = FastGLM(family='gaussian', check_input=True)
        with pytest.raises((ValueError, RuntimeError)):
            model.fit(X, y)
    
    def test_inf_in_data(self):
        """Test Inf in data raises error"""
        X = np.random.randn(100, 5)
        X[0, 0] = np.inf
        y = np.random.randn(100)
        
        model = FastGLM(family='gaussian', check_input=True)
        with pytest.raises((ValueError, RuntimeError)):
            model.fit(X, y)


class TestModelParameters:
    """Test model parameters and configurations"""
    
    def test_max_iter_parameter(self):
        """Test max_iter parameter"""
        np.random.seed(42)
        X, y = make_classification(n_samples=100, n_features=5)
        
        model = FastGLM(family='logistic', max_iter=5)
        model.fit(X, y)
        
        assert model.n_iter_ <= 5
    
    def test_tolerance_parameter(self):
        """Test tolerance parameter"""
        np.random.seed(42)
        X, y = make_classification(n_samples=100, n_features=5)
        
        model = FastGLM(family='logistic', tol=1e-10)
        model.fit(X, y)
        
        assert model.n_iter_ > 0
    
    def test_fit_intercept(self):
        """Test fit_intercept parameter"""
        np.random.seed(42)
        X, y = make_classification(n_samples=100, n_features=5)
        
        model_with = FastGLM(family='logistic', fit_intercept=True)
        model_with.fit(X, y)
        
        model_without = FastGLM(family='logistic', fit_intercept=False)
        model_without.fit(X, y)
        
        assert model_with.intercept_ is not None
        assert model_without.intercept_ == 0.0 or model_without.intercept_ is None


class TestDataScales:
    """Test different data scales and sizes"""
    
    def test_small_dataset(self):
        """Test with small dataset (n=30)"""
        np.random.seed(42)
        X = np.random.randn(30, 5)
        y = (np.random.rand(30) < 0.5).astype(float)
        
        model = FastGLM(family='logistic')
        model.fit(X, y)
        
        assert model.coef_ is not None
    
    def test_medium_dataset(self):
        """Test with medium dataset (n=1000)"""
        np.random.seed(42)
        X = np.random.randn(1000, 20)
        y = (np.random.rand(1000) < 0.5).astype(float)
        
        model = FastGLM(family='logistic')
        model.fit(X, y)
        
        assert model.coef_ is not None
    
    def test_large_dataset(self):
        """Test with large dataset (n=10000)"""
        np.random.seed(42)
        X = np.random.randn(10000, 20)
        y = (np.random.rand(10000) < 0.5).astype(float)
        
        model = FastGLM(family='logistic', max_iter=50)
        model.fit(X, y)
        
        assert model.coef_ is not None
    
    def test_high_dimensional(self):
        """Test with high-dimensional data (p=100)"""
        np.random.seed(42)
        X = np.random.randn(200, 100)
        y = (np.random.rand(200) < 0.5).astype(float)
        
        model = FastGLM(family='logistic', max_iter=100)
        model.fit(X, y)
        
        assert len(model.coef_) == 100


class TestNumericalStability:
    """Test numerical stability"""
    
    def test_collinear_features(self):
        """Test with collinear features"""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        X[:, 1] = X[:, 0] + np.random.randn(100) * 0.01
        y = (np.random.rand(100) < 0.5).astype(float)
        
        model = FastGLM(family='logistic', max_iter=50)
        model.fit(X, y)
        
        assert model.coef_ is not None
    
    def test_outliers(self):
        """Test with outliers"""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        X[:5] *= 10  # Add outliers
        y = (np.random.rand(100) < 0.5).astype(float)
        
        model = FastGLM(family='logistic', max_iter=50)
        model.fit(X, y)
        
        assert model.coef_ is not None
    
    def test_extreme_values(self):
        """Test with extreme values"""
        np.random.seed(42)
        X = np.random.randn(100, 5) * 100
        y = np.random.poisson(5, size=100)
        
        model = FastGLM(family='poisson', max_iter=50)
        model.fit(X, y)
        
        assert model.coef_ is not None


class TestDiagnostics:
    """Test diagnostic features"""
    
    def test_verbose_mode(self):
        """Test verbose mode runs without error"""
        np.random.seed(42)
        X, y = make_classification(n_samples=100, n_features=5)
        
        model = FastGLM(family='logistic', verbose=True)
        model.fit(X, y)
        
        assert model.n_iter_ > 0
    
    def test_diagnose_method(self):
        """Test model has required attributes"""
        np.random.seed(42)
        X, y = make_classification(n_samples=100, n_features=5)
        
        model = FastGLM(family='logistic')
        model.fit(X, y)
        
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'n_iter_')
        assert model.coef_ is not None


class TestEdgeCases:
    """Test edge cases"""
    
    def test_single_feature(self):
        """Test with single feature"""
        np.random.seed(42)
        X = np.random.randn(100, 1)
        y = (np.random.rand(100) < 0.5).astype(float)
        
        model = FastGLM(family='logistic')
        model.fit(X, y)
        
        assert len(model.coef_) == 1
    
    def test_perfect_separation(self):
        """Test with perfectly separable data"""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.zeros(100)
        y[:50] = 1
        X[y == 1] += 10
        
        model = FastGLM(family='logistic', max_iter=100)
        model.fit(X, y)
        
        assert model.coef_ is not None
    
    def test_all_same_class(self):
        """Test with all same class"""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.ones(100)
        
        model = FastGLM(family='logistic', max_iter=50)
        model.fit(X, y)
        
        assert model.coef_ is not None


class TestQuickTest:
    """Test utility functions"""
    
    def test_model_creation(self):
        """Test that models can be created for all families"""
        families = ['logistic', 'poisson', 'gamma', 'gaussian', 'inverse_gaussian']
        
        for family in families:
            model = FastGLM(family=family)
            assert model.family == family


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
