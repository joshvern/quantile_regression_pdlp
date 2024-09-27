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
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
## Installation
### Prerequisites
- Python 3.6 or higher.
- `ortools` package.
### Install via pip
```bash
pip install quantile_regression_pdlp
```
### Install from Source
Clone the repository:
```bash
git clone https://github.com/{yourusername}/quantile_regression_pdlp.git
cd quantile_regression_pdlp
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

Quantile regression aims to estimate the conditional quantiles of a response variable given certain predictor variables. Unlike ordinary least squares regression, which minimizes the sum of squared residuals, quantile regression minimizes a weighted sum of absolute residuals.

### Mathematical Formulation

For a quantile $\tau \in (0, 1)$, the quantile regression problem can be formulated as:

$$
\min_{\beta} \sum_{i=1}^{n} \rho_{\tau}(y_i - x_i^{\top} \beta)
$$

where:

- $y_i$ is the response variable.
- $x_i$ is the vector of predictor variables.
- $\beta$ is the vector of coefficients.
- $\rho_{\tau}(u)$ is the quantile loss function defined as:

$$
\rho_{\tau}(u) = u(\tau - \mathbb{I}(u < 0))
$$

This function is piecewise linear and convex, allowing the quantile regression problem to be expressed as a linear programming (LP) problem.

### LP Formulation

By introducing auxiliary variables, the problem can be rewritten:

- **Variables**: Introduce non-negative slack variables $r_i^{+}$ and $r_i^{-}$ such that:

$$
y_i - x_i^{\top} \beta = r_i^{+} - r_i^{-}
$$

- **Objective Function**:

$$
\min_{\beta, r^{+}, r^{-}} \sum_{i=1}^{n} (\tau r_i^{+} + (1 - \tau) r_i^{-})
$$

- **Constraints**:

$$
r_i^{+} >= 0, \quad r_i^{-} >= 0
$$

This LP formulation can be efficiently solved using the PDLP solver provided by Google's OR-Tools.

## Features
- **Custom Quantiles**: Supports estimation for any quantile $\tau \in (0, 1)$.
- **Statistical Summaries**: Provides standard errors, t-values, and p-values computed via bootstrapping.
- **Bootstrap Estimation**: Standard errors are estimated using bootstrap resampling.
- **Simple API**: Designed to be similar to `statsmodels` for ease of use.
- **Efficient Solver**: Utilizes the PDLP solver for efficient computation, suitable for large datasets.
## Documentation
### Class: `QuantileRegression`
*Initialization*
```python
QuantileRegression(tau=0.5, n_bootstrap=1000, random_state=None)
```
- Parameters:
    - `tau` (float, default=0.5): The quantile to estimate, must be between 0 and 1.
    - `n_bootstrap` (int, default=1000): Number of bootstrap samples for estimating standard errors.
    - `random_state` (int, default=None): Seed for the random number generator.
*Attributes*
- `coef_` (ndarray): Estimated coefficients for the regression model.
- `intercept_` (float): Estimated intercept term.
- `stderr_` (ndarray): Standard errors of the coefficients.
- `tvalues_` (ndarray): T-statistics of the coefficients.
- `pvalues_` (ndarray): P-values of the coefficients.
*Methods*
- `fit(X, y)`: Fit the quantile regression model to the data.
    - Parameters:
        - `X`: array-like of shape (n_samples, n_features). Training data.
        - `y`: array-like of shape (n_samples,). Target values.
    - Returns: `self`
- `predict(X)`: Predict using the quantile regression model.
    - Parameters:
        - `X`: array-like of shape (n_samples, n_features). Samples.
    - Returns: `y_pred` (ndarray of shape (n_samples,))
- `summary()`: Return a summary of the regression results.
    - Returns: `summary_df` (pandas DataFrame)
## Examples
### Example 1: Basic Usage
```python
from quantile_regression_pdlp import QuantileRegression
import numpy as np

#Data
np.random.seed(0)
X = np.random.rand(50, 1)
y = 2 * X.squeeze() + np.random.randn(50) * 0.5

#Model
model = QuantileRegression(tau=0.5, n_bootstrap=500, random_state=0)
model.fit(X, y)
print(model.summary())
```
Output
```
           Coefficient  Std. Error   t-value     P>|t|
Intercept     0.053282    0.274354  0.194209  0.846815
X1            1.718165    0.372070  4.617858  0.000028
```
### Example 2: Multiple Quantiles
```python
quantiles = [0.25, 0.5, 0.75]
models = {}

for tau in quantiles:
  model = QuantileRegression(tau=tau, n_bootstrap=500, random_state=42)
  model.fit(X, y)
  models[tau] = model

#Accessing summaries
for tau, model in models.items():
  print(f"\nQuantile: {tau}")
  print(model.summary())
```
Output
```
Quantile: 0.25
           Coefficient  Std. Error   t-value         P>|t|
Intercept    -0.435959    0.197562 -2.206691  3.205236e-02
X1            2.030023    0.270743  7.497967  1.127290e-09

Quantile: 0.5
           Coefficient  Std. Error   t-value     P>|t|
Intercept     0.053282    0.274341  0.194218  0.846808
X1            1.718165    0.380929  4.510460  0.000040

Quantile: 0.75
           Coefficient  Std. Error   t-value     P>|t|
Intercept     0.364761    0.282678  1.290378  0.202976
X1            1.641364    0.477440  3.437846  0.001205
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
