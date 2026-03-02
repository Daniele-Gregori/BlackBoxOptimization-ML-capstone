# Bayesian Optimization for 8 Black Box Functions
"""
I'll implement Bayesian Optimization using Gaussian Processes with scikit-learn. You'll fill in the data dictionaries.

```python
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# YOUR DATA - Fill in the dictionaries below
# ============================================================

# Function 1: 17 samples, 2 input dims, 1 output
X1 = {}  # {0: [x1, x2], 1: [x1, x2], ...}
y1 = {}  # {0: y, 1: y, ...}

# Function 2: 17 samples, 2 input dims, 1 output
X2 = {}
y2 = {}

# Function 3: 22 samples, 3 input dims, 1 output
X3 = {}
y3 = {}

# Function 4: 37 samples, 4 input dims, 1 output
X4 = {}
y4 = {}

# Function 5: 27 samples, 4 input dims, 1 output
X5 = {}
y5 = {}

# Function 6: 27 samples, 5 input dims, 1 output
X6 = {}
y6 = {}

# Function 7: 37 samples, 6 input dims, 1 output
X7 = {}
y7 = {}

# Function 8: 47 samples, 8 input dims, 1 output
X8 = {}
y8 = {}

# ============================================================
# BAYESIAN OPTIMIZATION CLASS
# ============================================================

class BayesianOptimizer:
    def __init__(self, X_dict, y_dict, bounds, maximize=True):
        """
        X_dict: dictionary of input samples
        y_dict: dictionary of output values
        bounds: list of (min, max) tuples for each dimension
        maximize: True to maximize, False to minimize
        """
        self.X = np.array(list(X_dict.values()))
        self.y = np.array(list(y_dict.values())).reshape(-1, 1)
        self.bounds = np.array(bounds)
        self.maximize = maximize
        self.dim = self.X.shape[1]
        
        # Normalize data for better GP performance
        self.X_mean = self.X.mean(axis=0)
        self.X_std = self.X.std(axis=0) + 1e-8
        self.y_mean = self.y.mean()
        self.y_std = self.y.std() + 1e-8
        
        self.X_norm = (self.X - self.X_mean) / self.X_std
        self.y_norm = (self.y - self.y_mean) / self.y_std
        
        # Initialize GP
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(self.dim), nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=False,
            n_restarts_optimizer=10
        )
        self.gp.fit(self.X_norm, self.y_norm.ravel())
    
    def _normalize_X(self, X):
        return (X - self.X_mean) / self.X_std
    
    def _denormalize_y(self, y_norm):
        return y_norm * self.y_std + self.y_mean
    
    def predict(self, X):
        """Return mean and std of prediction"""
        X = np.atleast_2d(X)
        X_norm = self._normalize_X(X)
        mu_norm, sigma_norm = self.gp.predict(X_norm, return_std=True)
        mu = self._denormalize_y(mu_norm)
        sigma = sigma_norm * self.y_std
        return mu, sigma
    
    def expected_improvement(self, X, xi=0.01):
        """Expected Improvement acquisition function"""
        X = np.atleast_2d(X)
        mu, sigma = self.predict(X)
        
        if self.maximize:
            best = self.y.max()
            imp = mu - best - xi
        else:
            best = self.y.min()
            imp = best - mu - xi
        
        Z = np.where(sigma > 1e-8, imp / sigma, 0)
        ei = np.where(sigma > 1e-8, 
                      imp * norm.cdf(Z) + sigma * norm.pdf(Z), 
                      0)
        return ei
    
    def upper_confidence_bound(self, X, kappa=2.0):
        """UCB acquisition function"""
        mu, sigma = self.predict(X)
        if self.maximize:
            return mu + kappa * sigma
        else:
            return -(mu - kappa * sigma)
    
    def probability_of_improvement(self, X, xi=0.01):
        """Probability of Improvement acquisition function"""
        X = np.atleast_2d(X)
        mu, sigma = self.predict(X)
        
        if self.maximize:
            best = self.y.max()
            Z = (mu - best - xi) / (sigma + 1e-8)
        else:
            best = self.y.min()
            Z = (best - mu - xi) / (sigma + 1e-8)
        
        return norm.cdf(Z)
    
    def suggest_next(self, acquisition='EI', n_restarts=25, **acq_params):
        """
        Suggest next point to evaluate
        acquisition: 'EI', 'UCB', or 'PI'
        """
        acq_funcs = {
            'EI': self.expected_improvement,
            'UCB': self.upper_confidence_bound,
            'PI': self.probability_of_improvement
        }
        acq_func = acq_funcs[acquisition]
        
        def objective(x):
            return -acq_func(x.reshape(1, -1), **acq_params).ravel()[0]
        
        best_x = None
        best_acq = float('inf')
        
        # Multi-start optimization
        for _ in range(n_restarts):
            x0 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            
            result = minimize(
                objective,
                x0,
                bounds=self.bounds,
                method='L-BFGS-B'
            )
            
            if result.fun < best_acq:
                best_acq = result.fun
                best_x = result.x
        
        return best_x, -best_acq
    
    def update(self, X_new, y_new):
        """Add new observation and refit GP"""
        X_new = np.atleast_2d(X_new)
        y_new = np.atleast_1d(y_new).reshape(-1, 1)
        
        self.X = np.vstack([self.X, X_new])
        self.y = np.vstack([self.y, y_new])
        
        # Renormalize
        self.X_mean = self.X.mean(axis=0)
        self.X_std = self.X.std(axis=0) + 1e-8
        self.y_mean = self.y.mean()
        self.y_std = self.y.std() + 1e-8
        
        self.X_norm = (self.X - self.X_mean) / self.X_std
        self.y_norm = (self.y - self.y_mean) / self.y_std
        
        self.gp.fit(self.X_norm, self.y_norm.ravel())


# ============================================================
# SETUP OPTIMIZERS FOR ALL 8 FUNCTIONS
# ============================================================

def create_optimizer(X_dict, y_dict, bounds, maximize=True):
    """Helper to create optimizer with error checking"""
    if not X_dict or not y_dict:
        print("Warning: Empty data dictionary")
        return None
    return BayesianOptimizer(X_dict, y_dict, bounds, maximize)

# Define bounds for each function (ADJUST THESE BASED ON YOUR DATA)
# Format: [(min1, max1), (min2, max2), ...]

bounds1 = [(0, 1), (0, 1)]  # 2D
bounds2 = [(0, 1), (0, 1)]  # 2D
bounds3 = [(0, 1), (0, 1), (0, 1)]  # 3D
bounds4 = [(0, 1), (0, 1), (0, 1), (0, 1)]  # 4D
bounds5 = [(0, 1), (0, 1), (0, 1), (0, 1)]  # 4D
bounds6 = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]  # 5D
bounds7 = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]  # 6D
bounds8 = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]  # 8D

# ============================================================
# MAIN EXECUTION
# ============================================================

def run_bayesian_optimization():
    """Run BO on all 8 functions"""
    
    datasets = [
        (X1, y1, bounds1, "Function 1 (2D)"),
        (X2, y2, bounds2, "Function 2 (2D)"),
        (X3, y3, bounds3, "Function 3 (3D)"),
        (X4, y4, bounds4, "Function 4 (4D)"),
        (X5, y5, bounds5, "Function 5 (4D)"),
        (X6, y6, bounds6, "Function 6 (5D)"),
        (X7, y7, bounds7, "Function 7 (6D)"),
        (X8, y8, bounds8, "Function 8 (8D)"),
    ]
    
    optimizers = []
    results = []
    
    for X, y, bounds, name in datasets:
        print(f"\n{'='*50}")
        print(f"Optimizing {name}")
        print(f"{'='*50}")
        
        if not X or not y:
            print(f"Skipping {name} - no data provided")
            optimizers.append(None)
            results.append(None)
            continue
        
        # Auto-detect bounds from data
        X_arr = np.array(list(X.values()))
        auto_bounds = [(X_arr[:, i].min() - 0.1 * abs(X_arr[:, i].min()), 
                        X_arr[:, i].max() + 0.1 * abs(X_arr[:, i].max())) 
                       for i in range(X_arr.shape[1])]
        
        print(f"Auto-detected bounds: {auto_bounds}")
        
        # Create optimizer
        opt = BayesianOptimizer(X, y, auto_bounds, maximize=True)
        optimizers.append(opt)
        
        # Get current best
        y_arr = np.array(list(y.values()))
        current_best_idx = np.argmax(y_arr)
        current_best_y = y_arr[current_best_idx]
        current_best_x = X_arr[current_best_idx]
        
        print(f"Current best: y = {current_best_y:.6f} at x = {current_best_x}")
        
        # Suggest next point using different acquisition functions
        print("\nSuggested next points:")
        
        for acq in ['EI', 'UCB', 'PI']:
            next_x, acq_value = opt.suggest_next(acquisition=acq)
            pred_mu, pred_std = opt.predict(next_x)
            print(f"  {acq}: x = {next_x}, predicted y = {pred_mu[0]:.4f} \[PlusMinus] {pred_std[0]:.4f}")
        
        results.append({
            'optimizer': opt,
            'current_best_x': current_best_x,
            'current_best_y': current_best_y,
            'suggested_EI': opt.suggest_next(acquisition='EI')[0],
            'suggested_UCB': opt.suggest_next(acquisition='UCB')[0],
            'suggested_PI': opt.suggest_next(acquisition='PI')[0]
        })
    
    return optimizers, results


# ============================================================
# BATCH SUGGESTION (for parallel evaluation)
# ============================================================

def suggest_batch(optimizer, batch_size=5, acquisition='EI'):
    """Suggest multiple points using kriging believer strategy"""
    if optimizer is None:
        return None
    
    suggestions = []
    temp_opt = BayesianOptimizer(
        {i: x for i, x in enumerate(optimizer.X)},
        {i: y[0] for i, y in enumerate(optimizer.y)},
        optimizer.bounds,
        optimizer.maximize
    )
    
    for _ in range(batch_size):
        next_x, _ = temp_opt.suggest_next(acquisition=acquisition)
        suggestions.append(next_x)
        
        # Update with predicted mean (kriging believer)
        pred_y, _ = temp_opt.predict(next_x)
        temp_opt.update(next_x, pred_y)
    
    return np.array(suggestions)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    # Uncomment after filling in data
    # optimizers, results = run_bayesian_optimization()
    
    # Example of getting batch suggestions:
    # batch = suggest_batch(optimizers[0], batch_size=5)
    
    print("Fill in the X and y dictionaries, then run run_bayesian_optimization()")

"""
```


## How to Use:

1. **Fill in your data** in the dictionaries at the top (X1, y1, X2, y2, etc.)

2. **Run optimization:**
```python
optimizers, results = run_bayesian_optimization()
```

3. **Get next suggested point for any function:**
```python
next_point, acq_value = optimizers[0].suggest_next(acquisition='EI')
```

4. **After evaluating, update the model:**
```python
optimizers[0].update(new_x, new_y)
```

5. **For batch suggestions:**
```python
batch_points = suggest_batch(optimizers[0], batch_size=5)
```
"""