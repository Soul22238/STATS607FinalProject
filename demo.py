#!/usr/bin/env python3
"""
FastGLM Complete Demo - Run All

This script runs the complete FastGLM workflow:
1. Verify environment and dependencies
2. Run quick feature demonstration
3. Run full benchmark suite
4. Run all unit tests

Usage: python demo.py
"""

import sys
import subprocess
import os

def print_header(title):
    """Print section header"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def check_environment():
    """Check if virtual environment is activated and packages installed"""
    print_header("Step 1: Environment Check")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check if in virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if in_venv:
        print("Virtual environment: Active")
    else:
        print("WARNING: No virtual environment detected")
        print("Recommended: source .venv/bin/activate")
    
    # Check required packages
    required = ['numpy', 'scipy', 'sklearn', 'statsmodels', 'matplotlib', 'pandas', 'pytest']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"Package {package}: OK")
        except ImportError:
            missing.append(package)
            print(f"Package {package}: MISSING")
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\nAll dependencies satisfied")
    return True

def run_quick_demo():
    """Run quick demonstration of all GLM families"""
    print_header("Step 2: Quick Feature Demo")
    
    from fastglm import FastGLM
    import numpy as np
    import time
    
    families = ['logistic', 'poisson', 'gamma', 'gaussian', 'inverse_gaussian']
    
    for i, family in enumerate(families, 1):
        print(f"\n[{i}/5] Testing {family.upper()} regression...")
        
        np.random.seed(42)
        X = np.random.randn(200, 5)
        
        if family == 'logistic':
            y = (np.random.rand(200) < 0.5).astype(float)
        elif family == 'poisson':
            y = np.random.poisson(5, size=200)
        elif family == 'gamma':
            y = np.random.gamma(2, 1, size=200)
        elif family == 'gaussian':
            y = X @ np.random.randn(5) + np.random.randn(200) * 0.5
        elif family == 'inverse_gaussian':
            y = np.abs(np.random.randn(200)) + 1.0
        
        start = time.perf_counter()
        model = FastGLM(family=family)
        model.fit(X, y)
        elapsed = (time.perf_counter() - start) * 1000
        
        score = model.score(X, y)
        print(f"  Iterations: {model.n_iter_}, Score: {score:.3f}, Time: {elapsed:.2f} ms")
    
    print("\nQuick demo completed successfully")

def run_benchmarks():
    """Run full benchmark suite"""
    print_header("Step 3: Full Benchmark Suite")
    print("This will take 5-10 minutes...")
    print("Benchmarking all 5 GLM families across multiple data sizes")
    print()
    
    try:
        result = subprocess.run(
            [sys.executable, 'benchmark.py'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print("\nBenchmarks completed successfully")
            print("Results saved to:")
            print("  - results/csv/")
            print("  - results/plots/")
            return True
        else:
            print(f"\nBenchmark failed with code {result.returncode}")
            return False
    except Exception as e:
        print(f"\nError running benchmarks: {e}")
        return False

def run_tests():
    """Run unit tests with pytest"""
    print_header("Step 4: Unit Tests")
    print("Running 37 unit tests...")
    print()
    
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', 'test.py', '-v', '--tb=short'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print("\nAll tests passed")
            return True
        else:
            print(f"\nSome tests failed (code {result.returncode})")
            return False
    except Exception as e:
        print(f"\nError running tests: {e}")
        return False

def main():
    """Run complete demo workflow"""
    print("=" * 70)
    print("FastGLM Complete Demo - Run All")
    print("=" * 70)
    print("\nThis will:")
    print("1. Check environment and dependencies")
    print("2. Run quick feature demonstration")
    print("3. Run full benchmark suite (5-10 minutes)")
    print("4. Run all unit tests")
    print()
    
    # Step 1: Check environment
    if not check_environment():
        print("\nEnvironment check failed. Please install dependencies first.")
        sys.exit(1)
    
    # Step 2: Quick demo
    try:
        run_quick_demo()
    except Exception as e:
        print(f"\nQuick demo failed: {e}")
        sys.exit(1)
    
    # Step 3: Ask user if they want to continue with benchmarks
    print("\n" + "-" * 70)
    response = input("Continue with full benchmarks? (y/n, default=y): ").strip().lower()
    if response not in ['', 'y', 'yes']:
        print("Skipping benchmarks and tests")
        print("\nTo run later:")
        print("  Benchmarks: python benchmark.py")
        print("  Tests: pytest test.py -v")
        return
    
    # Step 4: Run benchmarks
    benchmark_success = run_benchmarks()
    
    # Step 5: Run tests
    test_success = run_tests()
    
    # Summary
    print_header("Demo Complete - Summary")
    print(f"Environment check: OK")
    print(f"Quick demo: OK")
    print(f"Benchmarks: {'OK' if benchmark_success else 'FAILED'}")
    print(f"Tests: {'OK' if test_success else 'FAILED'}")
    print()
    
    if benchmark_success and test_success:
        print("All steps completed successfully!")
        print("\nCheck results:")
        print("  - results/csv/ for benchmark data")
        print("  - results/plots/ for visualizations")
    else:
        print("Some steps failed. Check output above for details.")
    
    print("=" * 70)

if __name__ == '__main__':
    main()
