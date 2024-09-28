## Quantile Regression PDLP
A Python package for performing quantile regression using the PDLP solver from Google's OR-Tools, with an interface and summaries similar to `statsmodels`.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Quantile Regression as a Linear Programming Problem](#quantile-regression-as-a-linear-programming-problem)
    - [Mathematical Formulation](#mathematical-formulation)
    - [LP Formulation](#lp-formulation)
- [Features](#features)
- [Documentation](#documentation)
    - [Class: `QuantileRegression`](#class-quantileregression)
- [Examples](#examples)
    - [Example 1: Basic Usage](#example-1-basic-usage)
    - [Example 2: Multiple Quantiles](#example-2-multiple-quantiles)
    - [Example 3: Weighted Quantile Regression with L1 Regularization](#example-3-weighted-quantile-regression-with-l1-regularization)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Installation
### Prerequisites
- Python 3.6 or higher.
- `ortools` package.

### Install from Source
Clone the repository:
```bash
git clone https://github.com/joshvern/quantile_regression_pdlp.git
cd quantile_regression_pdlp
pip install -U pip setuptools wheel
pip install .
```

## Usage
```python
from quantile_regression_pdlp import QuantileRegression
import numpy as np


#Generate sample data
np.random.seed(42)
X = np.random.rand(100, 2)
y = X @ np.array([1.5, -2.0]) + np.random.randn(100) * 0.5


#Initialize and fit the model
model = QuantileRegression(tau=0.5, n_bootstrap=500, random_state=42)
model.fit(X, y)


#Print the summary
print(model.summary())


#Make predictions
X_new = np.array([[0.1, 0.2], [0.5, 0.8]])
y_pred = model.predict(X_new)
print('Predictions:', y_pred)
```

## Quantile Regression as a Linear Programming Problem

Quantile regression aims to estimate the conditional quantiles of a response variable given certain predictor variables. Unlike ordinary least squares regression, which minimizes the sum of squared residuals, quantile regression minimizes a weighted sum of absolute residuals and can incorporate regularization.

### Mathematical Formulation

For a quantile $\tau \in (0, 1)$, the weighted quantile regression problem can be formulated as:

$$
\min_{\beta} \sum_{i=1}^{n} w_i \rho_{\tau}(y_i - x_i^{\top} \beta)
$$

where:

- $y_i$ is the response variable.
- $x_i$ is the vector of predictor variables.
- $w_i$ is the weight assigned to the (i)th observation.
- $\beta$ is the vector of coefficients.
- $\rho_{\tau}(u)$ is the quantile loss function defined as:

$$
\rho_{\tau}(u) = u(\tau - \mathbb{I}(u < 0))
$$

This function is piecewise linear and convex, allowing the quantile regression problem to be expressed as a linear programming (LP) problem. Additionally, L1 regularization can be incorporated to promote sparsity in the coefficients.

### LP Formulation

By introducing auxiliary variables and incorporating regularization, the problem can be rewritten:

#### **Variables**

- Introduce non-negative slack variables $r_i^{+}$ and $r_i^{-}$ for each observation:

$$
y_i - x_i^{\top} \beta = r_i^{+} - r_i^{-}
$$

- For L1 regularization, introduce auxiliary variables $z_j$ for each coefficient $\beta_j$ (excluding the intercept):

$$
z_j \geq \beta_j \\
z_j \geq -\beta_j
$$

#### **Objective Function**

The objective function incorporates both the weighted quantile loss and the L1 regularization term:

$$
\min_{\beta, r^{+}, r^{-}, z} \sum_{i=1}^{n} (\tau w_i r_i^{+} + (1 - \tau) w_i r_i^{-}) + \lambda \sum_{j=1}^{p} z_j
$$

where:

- $\lambda$ is the regularization strength parameter.
- $p$ is the number of predictor variables (excluding the intercept).

#### **Constraints**

Subject to the constraints:

1. **Residual Constraints**:

$$
y_i - x_i^{\top} \beta = r_i^{+} - r_i^{-}, \quad \forall i = 1, \dots, n
$$

2. **Non-negativity Constraints**:

$$
r_i^{+} \geq 0, \quad r_i^{-} \geq 0, \quad \forall i = 1, \dots, n
$$

3. **L1 Regularization Constraints**:

$$
z_j \geq \beta_j, \quad \forall j = 1, \dots, p \\
z_j \geq -\beta_j, \quad \forall j = 1, \dots, p
$$

This LP formulation can be efficiently solved using the PDLP solver provided by Google's OR-Tools.

## Features
- **Custom Quantiles**: Supports estimation for any quantile $\tau \in (0, 1)$.
- **Weighted Quantile Regression**: Assigns different weights to observations, allowing differential influence on regression estimates.
- **L1 Regularization (Lasso)**: Promotes sparsity in the model by penalizing the absolute values of the coefficients.
- **Statistical Summaries**: Provides standard errors, t-values, and p-values computed via bootstrapping.
- **Bootstrap Estimation**: Standard errors are estimated using bootstrap resampling.
- **Simple API**: Designed to be similar to `statsmodels` for ease of use.
- **Efficient Solver**: Utilizes the PDLP solver for efficient computation, suitable for large datasets.

## Documentation
### Class: `QuantileRegression`
*Initialization*
```python
QuantileRegression(tau=0.5, n_bootstrap=1000, random_state=None, regularization='none', alpha=0.0)
```
- **Parameters**:
    - `tau` (float, default=0.5): The quantile to estimate, must be between 0 and 1.
    - `n_bootstrap` (int, default=1000): Number of bootstrap samples for estimating standard errors.
    - `random_state` (int, default=None): Seed for the random number generator.
    - `regularization` (str, default='none'): Type of regularization to apply. Options are `'l1'` for Lasso regularization or `'none'` for no regularization.
    - `alpha` (float, default=0.0): Regularization strength. Must be a non-negative float. Higher values imply stronger regularization.

*Attributes*
- `coef_` (ndarray): Estimated coefficients for the regression model.
- `intercept_` (float): Estimated intercept term.
- `stderr_` (ndarray): Standard errors of the coefficients.
- `tvalues_` (ndarray): T-statistics of the coefficients.
- `pvalues_` (ndarray): P-values of the coefficients.

*Methods*
- `fit(X, y, weights=None)`: Fit the quantile regression model to the data.
    - **Parameters**:
        - `X`: array-like of shape (n_samples, n_features). Training data.
        - `y`: array-like of shape (n_samples,). Target values.
        - `weights` (ndarray, optional): Weights for each observation, shape (n_samples,). Default is `None`, which assigns equal weight to all observations.
    - **Returns**: `self`

- `predict(X)`: Predict using the quantile regression model.
    - **Parameters**:
        - `X`: array-like of shape (n_samples, n_features). Samples.
    - **Returns**: `y_pred` (ndarray of shape (n_samples,))

- `summary()`: Return a summary of the regression results.
    - **Returns**: `summary_df` (pandas DataFrame)

## Examples
### Example 1: Basic Usage
```python
from quantile_regression_pdlp import QuantileRegression
import numpy as np

#Data
np.random.seed(0)
X = np.random.rand(50, 1)
y = np.random.rand(50) * 2 + np.random.randn(50) * 0.5

#Model
model = QuantileRegression(tau=0.5, n_bootstrap=500, random_state=0)
model.fit(X, y)
print(model.summary())
```
**Output**
```
           Coefficient  Std. Error   t-value     P>|t|
Intercept     1.000226    0.236704  4.225639  0.000103
X1           -0.054309    0.545591 -0.099542  0.921114
```

### Example 2: Multiple Quantiles
```python
quantiles = [0.25, 0.5, 0.75]
models = {}

for tau in quantiles:
    model = QuantileRegression(tau=tau, n_bootstrap=500, random_state=42)
    model.fit(X, y)
    models[tau] = model

# Accessing summaries
for tau, model in models.items():
    print(f"\nQuantile: {tau}")
    print(model.summary())
```
**Output**
```
Quantile: 0.25
           Coefficient  Std. Error   t-value     P>|t|
Intercept     0.989807    0.241534  4.097998  0.000156
X1           -0.827227    0.350021 -2.363365  0.022121

Quantile: 0.5
           Coefficient  Std. Error   t-value     P>|t|
Intercept     1.000226    0.233869  4.276863  0.000087
X1           -0.054309    0.531167 -0.102245  0.918979

Quantile: 0.75
           Coefficient  Std. Error   t-value     P>|t|
Intercept     1.612909    0.400987  4.022350  0.000199
X1           -0.163155    0.691681 -0.235882  0.814507
```

### Example 3: Weighted Quantile Regression with L1 Regularization
```python
from quantile_regression_pdlp import QuantileRegression
import numpy as np

#Generate sample data
np.random.seed(42)
X = np.random.rand(200, 3)  # 3 predictor variables
y = 1.0 + 2.5 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(200) * 0.3

#Assign weights: higher weights to observations with y > median
weights = np.where(y > np.median(y), 1.5, 1.0)

#Initialize and fit the model with L1 regularization
model = QuantileRegression(
    tau=0.5,
    n_bootstrap=500,
    random_state=42,
    regularization='l1',
    alpha=0.1
)
model.fit(X, y, weights=weights)

#Print the summary
print(model.summary())

#Make predictions
X_new = np.array([
    [0.2, 0.3, 0.5],
    [0.6, 0.8, 0.1],
    [0.4, 0.5, 0.9]
])
y_pred = model.predict(X_new)
print('Predictions:', y_pred)
```
**Output**
```
           Coefficient  Std. Error    t-value     P>|t|
Intercept     1.055083    0.093393  11.297225  0.000000
X1            2.494034    0.100485  24.819935  0.000000
X2           -1.560294    0.106501 -14.650449  0.000000
X3            0.428887    0.105701   4.057566  0.000071
Predictions: [1.300245   1.34615695 1.65854778]
```

## Dependencies
- `ortools` (Google's OR-Tools)
- `numpy`
- `pandas`
- `scipy`
- `tqdm`

Install dependencies using:
```bash
pip install numpy pandas scipy tqdm ortools
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request on GitHub.
1. Fork the repository on GitHub.
2. Create a new branch for your feature or bugfix.
3. Commit your changes with clear messages.
4. Submit a pull request.

## License
This project is licensed under the MIT License.
