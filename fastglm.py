"""
FastGLM: Ultra-fast GLM Solver for Small Datasets

This module implements optimized Generalized Linear Models (GLM) specifically
designed for small datasets (n < 2000, p < 50) using IRLS with Cholesky decomposition.

Supported families:
- Logistic Regression (Binomial with logit link) - Binary classification
- Poisson Regression (Poisson with log link) - Count data
- Gamma Regression (Gamma with log link) - Positive continuous, right-skewed data
- Gaussian Regression (Normal with identity link) - Standard linear regression
- Inverse Gaussian (Inverse Gaussian with inverse squared link) - Waiting times, survival data

Key optimizations:
- Direct Cholesky decomposition instead of generic solvers
- Dense matrix operations optimized for small data
- Minimal overhead in initialization and iteration
- No unnecessary abstraction layers

"""

import numpy as np
from scipy import linalg
from typing import Literal, Tuple, Dict, Optional
import warnings
import time
import warnings


class FastGLM:
    """
    Fast GLM solver using IRLS (Iteratively Reweighted Least Squares) with Cholesky decomposition.
    
    Parameters
    ----------
    family : {'logistic', 'poisson', 'gamma', 'gaussian', 'inverse_gaussian'}
        The distribution family to use:
        - 'logistic': Binomial with logit link (binary classification)
        - 'poisson': Poisson with log link (count data)
        - 'gamma': Gamma with log link (positive continuous, right-skewed)
        - 'gaussian': Normal with identity link (standard linear regression)
        - 'inverse_gaussian': Inverse Gaussian with inverse squared link (waiting times)
    fit_intercept : bool, default=True
        Whether to fit an intercept term
    max_iter : int, default=25
        Maximum number of IRLS iterations
    tol : float, default=1e-6
        Convergence tolerance for coefficient changes
    """
    
    def __init__(
        self,
        family: Literal['logistic', 'poisson', 'gamma', 'gaussian', 'inverse_gaussian'] = 'logistic',
        fit_intercept: bool = True,
        max_iter: int = 25,
        tol: float = 1e-6,
        verbose: bool = False,
        check_input: bool = True
    ):
        self.family = family
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.check_input = check_input
        
        # Validate family
        valid_families = ['logistic', 'poisson', 'gamma', 'gaussian', 'inverse_gaussian']
        if self.family not in valid_families:
            raise ValueError(f"family must be one of {valid_families}, got '{self.family}'")
        
        # Model parameters (set after fitting)
        self.coef_ = None
        self.intercept_ = 0.0
        self.n_iter_ = 0
        self.fit_time_ = 0.0
        
        # Diagnostics
        self.warnings_ = []
        self.data_diagnostics_ = {}
        
    def _link(self, mu: np.ndarray) -> np.ndarray:
        """Apply link function: eta = g(mu)"""
        if self.family == 'logistic':
            # logit link: log(mu / (1 - mu))
            mu = np.clip(mu, 1e-7, 1 - 1e-7)
            return np.log(mu / (1 - mu))
        elif self.family == 'poisson':
            # log link: log(mu)
            mu = np.clip(mu, 1e-7, None)
            return np.log(mu)
        elif self.family == 'gamma':
            # log link: log(mu)
            mu = np.clip(mu, 1e-7, None)
            return np.log(mu)
        elif self.family == 'gaussian':
            # identity link: mu
            return mu
        elif self.family == 'inverse_gaussian':
            # inverse squared link: 1/mu^2
            mu = np.clip(mu, 1e-7, None)
            return 1.0 / (mu ** 2)
        
    def _inverse_link(self, eta: np.ndarray) -> np.ndarray:
        """Apply inverse link function: mu = g^(-1)(eta)"""
        if self.family == 'logistic':
            # inverse logit: 1 / (1 + exp(-eta))
            return 1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500)))
        elif self.family == 'poisson':
            # inverse log: exp(eta)
            return np.exp(np.clip(eta, -500, 500))
        elif self.family == 'gamma':
            # inverse log: exp(eta)
            return np.exp(np.clip(eta, -500, 500))
        elif self.family == 'gaussian':
            # inverse identity: eta
            return eta
        elif self.family == 'inverse_gaussian':
            # inverse of inverse squared: 1/sqrt(eta)
            eta = np.clip(eta, 1e-7, None)
            return 1.0 / np.sqrt(eta)
        
    def _variance(self, mu: np.ndarray) -> np.ndarray:
        """Variance function V(mu)"""
        if self.family == 'logistic':
            # V(mu) = mu * (1 - mu)
            mu = np.clip(mu, 1e-7, 1 - 1e-7)
            return mu * (1 - mu)
        elif self.family == 'poisson':
            # V(mu) = mu
            return np.clip(mu, 1e-7, None)
        elif self.family == 'gamma':
            # V(mu) = mu^2
            mu = np.clip(mu, 1e-7, None)
            return mu ** 2
        elif self.family == 'gaussian':
            # V(mu) = 1 (constant variance)
            return np.ones_like(mu)
        elif self.family == 'inverse_gaussian':
            # V(mu) = mu^3
            mu = np.clip(mu, 1e-7, None)
            return mu ** 3
        
    def _derivative_link(self, mu: np.ndarray) -> np.ndarray:
        """Derivative of link function: g'(mu)"""
        if self.family == 'logistic':
            # d/dmu[log(mu/(1-mu))] = 1/(mu(1-mu))
            mu = np.clip(mu, 1e-7, 1 - 1e-7)
            return 1.0 / (mu * (1 - mu))
        elif self.family == 'poisson':
            # d/dmu[log(mu)] = 1/mu
            mu = np.clip(mu, 1e-7, None)
            return 1.0 / mu
        elif self.family == 'gamma':
            # d/dmu[log(mu)] = 1/mu
            mu = np.clip(mu, 1e-7, None)
            return 1.0 / mu
        elif self.family == 'gaussian':
            # d/dmu[mu] = 1
            return np.ones_like(mu)
        elif self.family == 'inverse_gaussian':
            # d/dmu[1/mu^2] = -2/mu^3
            mu = np.clip(mu, 1e-7, None)
            return -2.0 / (mu ** 3)
        
    def _initialize_mu(self, y: np.ndarray) -> np.ndarray:
        """Initialize starting values for mu"""
        if self.family == 'logistic':
            # Initialize with adjusted proportions
            return np.clip((y + 0.5) / 2, 0.1, 0.9)
        elif self.family == 'poisson':
            # Initialize with adjusted counts
            return np.clip(y + 0.1, 0.1, None)
        elif self.family == 'gamma':
            # Initialize with adjusted positive values
            return np.clip(y + 0.1, 0.1, None)
        elif self.family == 'gaussian':
            # Initialize with the values themselves
            return y.copy()
        elif self.family == 'inverse_gaussian':
            # Initialize with adjusted positive values
            return np.clip(y + 0.1, 0.1, None)
    
    def _check_data_characteristics(self, X: np.ndarray, y: np.ndarray) -> None:
        """Check data characteristics and issue warnings for suboptimal scenarios"""
        n_samples, n_features = X.shape
        self.warnings_ = []
        self.data_diagnostics_ = {
            'n_samples': n_samples,
            'n_features': n_features,
            'sparsity': None,
            'condition_number': None,
            'recommended_method': 'FastGLM'
        }
        
        # Check 1: Very small dataset (n < 50)
        if n_samples < 50:
            msg = (f"⚠️  VERY SMALL DATASET (n={n_samples}): FastGLM may not be faster than sklearn "
                   f"due to initialization overhead. Consider sklearn for n < 50.")
            self.warnings_.append(msg)
            self.data_diagnostics_['recommended_method'] = 'sklearn (marginal benefit)'
            if self.verbose:
                warnings.warn(msg)
        
        # Check 2: Large dataset (n > 50,000)
        if n_samples > 50000:
            estimated_time = n_samples * n_features * n_features / 1e8  # rough estimate
            msg = (f"⚠️  LARGE DATASET (n={n_samples}): FastGLM uses O(np²) dense operations. "
                   f"sklearn's iterative methods (LBFGS) are more memory-efficient and likely faster. "
                   f"Estimated FastGLM overhead: ~{estimated_time:.2f}s vs sklearn's O(np) per iteration.")
            self.warnings_.append(msg)
            self.data_diagnostics_['recommended_method'] = 'sklearn (LBFGS more efficient)'
            if self.verbose:
                warnings.warn(msg)
        
        # Check 3: High-dimensional data (p > 100)
        if n_features > 100:
            cholesky_ops = n_features ** 3 / 1e9  # billion operations
            msg = (f"⚠️  HIGH-DIMENSIONAL DATA (p={n_features}): Cholesky decomposition is O(p³) ≈ "
                   f"{cholesky_ops:.2f}B operations. sklearn's LBFGS is O(p) per iteration and will be faster. "
                   f"FastGLM is optimal for p < 50.")
            self.warnings_.append(msg)
            self.data_diagnostics_['recommended_method'] = 'sklearn (lower complexity)'
            if self.verbose:
                warnings.warn(msg)
        
        # Check 4: p > n (high-dimensional, underdetermined)
        if n_features > n_samples:
            msg = (f"⚠️  HIGH-DIMENSIONAL (p={n_features} > n={n_samples}): X'WX will be singular. "
                   f"FastGLM adds ridge regularization (λ=1e-8) for stability, but sklearn with "
                   f"proper L2 regularization (penalty='l2') is recommended.")
            self.warnings_.append(msg)
            self.data_diagnostics_['recommended_method'] = 'sklearn (needs regularization)'
            if self.verbose:
                warnings.warn(msg)
        
        # Check 5: Sparse matrix detection
        if hasattr(X, 'sparse') or (hasattr(X, 'nnz') and hasattr(X, 'shape')):
            msg = (f"⚠️  SPARSE MATRIX DETECTED: FastGLM only supports dense matrices. "
                   f"sklearn can utilize sparse matrix operations for {X.nnz / (X.shape[0] * X.shape[1]):.1%} "
                   f"memory savings and faster computation.")
            self.warnings_.append(msg)
            self.data_diagnostics_['recommended_method'] = 'sklearn (sparse support)'
            if self.verbose:
                warnings.warn(msg)
        else:
            # Check sparsity in dense matrix
            sparsity = np.sum(X == 0) / X.size
            self.data_diagnostics_['sparsity'] = sparsity
            if sparsity > 0.5:
                msg = (f"⚠️  HIGHLY SPARSE DATA ({sparsity:.1%} zeros): Consider using scipy.sparse "
                       f"with sklearn for better memory efficiency and speed.")
                self.warnings_.append(msg)
                if self.verbose:
                    warnings.warn(msg)
        
        # Check 6: Condition number (multicollinearity)
        try:
            if n_features <= 100:  # Only compute for reasonable sizes
                cond = np.linalg.cond(X)
                self.data_diagnostics_['condition_number'] = cond
                if cond > 1e10:
                    msg = (f"⚠️  ILL-CONDITIONED DATA (cond={cond:.2e}): Severe multicollinearity detected. "
                           f"Ridge regularization recommended. Use sklearn with penalty='l2', C=<regularization>.")
                    self.warnings_.append(msg)
                    if self.verbose:
                        warnings.warn(msg)
        except:
            pass
    
    def diagnose(self) -> Dict:
        """
        Get diagnostic information about the fitted model and data characteristics.
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'n_samples': Number of samples
            - 'n_features': Number of features
            - 'n_iterations': Number of IRLS iterations
            - 'fit_time': Fitting time in seconds
            - 'sparsity': Proportion of zero values (if available)
            - 'condition_number': Matrix condition number (if available)
            - 'recommended_method': Recommended method ('FastGLM' or alternative)
            - 'warnings': List of warning messages
            - 'performance_estimate': Estimated performance vs baselines
        """
        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        diagnostics = self.data_diagnostics_.copy()
        diagnostics['n_iterations'] = self.n_iter_
        diagnostics['fit_time'] = self.fit_time_
        diagnostics['warnings'] = self.warnings_
        
        # Estimate performance vs baseline
        n = diagnostics['n_samples']
        p = diagnostics['n_features']
        
        if n < 50:
            diagnostics['performance_estimate'] = 'sklearn likely faster (0.8-1.0x)'
        elif n > 50000:
            diagnostics['performance_estimate'] = 'sklearn likely faster (0.5-0.9x)'
        elif p > 100:
            diagnostics['performance_estimate'] = 'sklearn likely faster (0.3-0.7x)'
        elif 50 <= n <= 10000 and p <= 50:
            if self.family in ['poisson', 'gamma']:
                diagnostics['performance_estimate'] = 'FastGLM much faster (5-15x)'
            elif self.family == 'gaussian':
                diagnostics['performance_estimate'] = 'FastGLM faster (2-5x)'
            else:
                diagnostics['performance_estimate'] = 'FastGLM faster (1.5-2x)'
        else:
            diagnostics['performance_estimate'] = 'FastGLM comparable (1-2x)'
        
        return diagnostics
    
    def print_diagnostics(self) -> None:
        """Print formatted diagnostic information"""
        diag = self.diagnose()
        
        print("=" * 70)
        print("FastGLM Diagnostic Report")
        print("=" * 70)
        print(f"Family: {self.family}")
        print(f"Data shape: n={diag['n_samples']}, p={diag['n_features']}")
        print(f"Iterations: {diag['n_iterations']} / {self.max_iter}")
        print(f"Fit time: {diag['fit_time']:.4f}s")
        
        if diag['sparsity'] is not None:
            print(f"Sparsity: {diag['sparsity']:.1%}")
        
        if diag['condition_number'] is not None:
            print(f"Condition number: {diag['condition_number']:.2e}")
        
        print(f"\nRecommended method: {diag['recommended_method']}")
        print(f"Performance estimate: {diag['performance_estimate']}")
        
        if diag['warnings']:
            print(f"\n⚠️  Warnings ({len(diag['warnings'])}):")
            for i, warning in enumerate(diag['warnings'], 1):
                print(f"  {i}. {warning}")
        else:
            print("\n✓ No warnings - data characteristics are optimal for FastGLM")
        
        print("=" * 70)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FastGLM':
        """
        Fit the GLM model using IRLS with Cholesky decomposition.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : FastGLM
            Fitted estimator
        """
        start_time = time.time()
        
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        
        n_samples, n_features = X.shape
        
        # Check data characteristics and issue warnings
        if self.check_input:
            self._check_data_characteristics(X, y)
        
        # Add intercept column if needed
        if self.fit_intercept:
            X = np.column_stack([np.ones(n_samples), X])
        
        # Initialize mu and eta
        mu = self._initialize_mu(y)
        eta = self._link(mu)
        
        # IRLS iterations
        for iteration in range(self.max_iter):
            # Compute working weights and response
            g_prime = self._derivative_link(mu)
            V = self._variance(mu)
            
            # Weight matrix W = 1 / (g'(mu)^2 * V(mu))
            # For numerical stability, clip weights
            W = 1.0 / (g_prime**2 * V + 1e-8)
            W = np.clip(W, 1e-10, 1e10)
            W = np.nan_to_num(W, nan=1.0, posinf=1e10, neginf=1e-10)
            
            # Working response: z = eta + (y - mu) * g'(mu)
            z = eta + (y - mu) * g_prime
            z = np.nan_to_num(z, nan=eta, posinf=500, neginf=-500)
            
            # Weighted least squares: solve (X^T W X) beta = X^T W z
            # Using Cholesky decomposition for speed
            sqrt_W = np.sqrt(W)
            X_weighted = X * sqrt_W[:, np.newaxis]
            z_weighted = z * sqrt_W
            
            # Form normal equations: X^T W X and X^T W z
            XtWX = X_weighted.T @ X_weighted
            XtWz = X_weighted.T @ z_weighted
            
            # Add small ridge for numerical stability
            XtWX += np.eye(XtWX.shape[0]) * 1e-8
            
            # Solve using Cholesky decomposition (fast for small matrices)
            try:
                L = linalg.cholesky(XtWX, lower=True)
                beta_new = linalg.cho_solve((L, True), XtWz)
            except linalg.LinAlgError:
                # Fallback to lstsq if Cholesky fails
                warnings.warn("Cholesky decomposition failed, using lstsq")
                beta_new = linalg.lstsq(XtWX, XtWz)[0]
            
            # Update eta and mu
            eta = X @ beta_new
            mu = self._inverse_link(eta)
            
            # Check convergence
            if iteration > 0:
                delta = np.max(np.abs(beta_new - beta_old))
                if delta < self.tol:
                    break
            
            beta_old = beta_new.copy()
        
        self.n_iter_ = iteration + 1
        self.fit_time_ = time.time() - start_time
        
        # Store coefficients
        if self.fit_intercept:
            self.intercept_ = beta_new[0]
            self.coef_ = beta_new[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta_new
        
        # Print summary if verbose
        if self.verbose:
            print(f"\nFastGLM fit completed in {self.fit_time_:.4f}s ({self.n_iter_} iterations)")
            if self.warnings_:
                print(f"⚠️  {len(self.warnings_)} warning(s) issued. Call .diagnose() for details.")
            
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the fitted model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples
            
        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted mean response
        """
        X = np.asarray(X, dtype=np.float64)
        eta = X @ self.coef_ + self.intercept_
        return self._inverse_link(eta)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (only for logistic regression).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples
            
        Returns
        -------
        proba : array of shape (n_samples, 2)
            Probability of each class
        """
        if self.family != 'logistic':
            raise ValueError("predict_proba only available for logistic regression")
        
        prob_1 = self.predict(X)
        return np.column_stack([1 - prob_1, prob_1])
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the coefficient of determination R^2 (for Poisson)
        or accuracy (for logistic).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True values
            
        Returns
        -------
        score : float
            Score value
        """
        y = np.asarray(y).ravel()
        
        if self.family == 'logistic':
            # Return accuracy for classification
            y_pred = (self.predict(X) > 0.5).astype(int)
            return np.mean(y_pred == y)
        else:
            # Return R^2 for regression (all other families)
            y_pred = self.predict(X)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot)


