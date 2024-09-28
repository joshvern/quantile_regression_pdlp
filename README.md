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
    - [Example 2: Estimating Multiple Quantiles Simultaneously](#example-2-estimating-multiple-quantiles-simultaneously)
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

Quantile regression aims to estimate the conditional quantiles of a response variable given certain predictor variables. Unlike ordinary least squares regression, which minimizes the sum of squared residuals, quantile regression minimizes a weighted sum of absolute residuals and can incorporate regularization. Estimating multiple quantiles simultaneously allows for a more comprehensive understanding of the conditional distribution and ensures quantile estimates do not cross.

### Mathematical Formulation

For multiple quantiles $\tau_1 < \tau_2 < \dots < \tau_K$, the weighted quantile regression problem can be formulated as:

$$
\min_{\beta^{(k)}} \sum_{k=1}^{K} \sum_{i=1}^{n} w_i \rho_{\tau_k}(y_i - x_i^{\top} \beta^{(k)})
$$

where:
- $y_i$ is the response variable.
- $x_i$ is the vector of predictor variables.
- $w_i$ is the weight assigned to the #th observation.
- $\beta^{(k)}$ is the vector of coefficients for quantile $\tau_k$.
- $\rho_{\tau_k}(u)$ is the quantile loss function defined as:

$$
\rho_{\tau_k}(u) = u(\tau_k - \mathbb{I}(u < 0))
$$

To prevent quantile crossing, we impose additional constraints ensuring that the predicted quantile levels are ordered appropriately:

$$
x_i^{\top} \beta^{(1)} \leq x_i^{\top} \beta^{(2)} \leq \dots \leq x_i^{\top} \beta^{(K)}, \quad \forall i = 1, \dots, n
$$

### LP Formulation

By introducing auxiliary variables and incorporating regularization, the problem can be rewritten:

#### Variables

- For each quantile $\tau_k$, introduce:
  - Coefficients $\beta^{(k)}$.
  - Non-negative slack variables $r_{i}^{+(k)}$ and $r_{i}^{-(k)}$ for each observation:

$$
y_i - x_i^{\top} \beta^{(k)} = r_{i}^{+(k)} - r_{i}^{-(k)}, \quad \forall i, \forall k
$$

- For L1 regularization, introduce auxiliary variables $z_j^{(k)}$ for each coefficient $\beta_j^{(k)}$ (excluding the intercept):

$$
z_j^{(k)} \geq \beta_j^{(k)} \\
z_j^{(k)} \geq -\beta_j^{(k)}, \quad \forall j, \forall k
$$

#### Objective Function

The objective function incorporates both the weighted quantile loss for all quantiles and the L1 regularization term:

$$
\min_{\beta^{(k)}, r^{+(k)}, r^{-(k)}, z^{(k)}} \sum_{k=1}^{K} \left( \sum_{i=1}^{n} \left( \tau_k w_i r_{i}^{+(k)} + (1 - \tau_k) w_i r_{i}^{-(k)} \right) + \lambda \sum_{j=1}^{p} z_j^{(k)} \right)
$$

where:
- $\lambda$ is the regularization strength parameter.
- $p$ is the number of predictor variables (excluding the intercept).

#### Constraints

Subject to the constraints:

1. **Residual Constraints**:

$$
y_i - x_i^{\top} \beta^{(k)} = r_{i}^{+(k)} - r_{i}^{-(k)}, \quad \forall i = 1, \dots, n; \quad \forall k = 1, \dots, K
$$

2. **Non-negativity Constraints**:

$$
r_{i}^{+(k)} \geq 0, \quad r_{i}^{-(k)} \geq 0, \quad \forall i = 1, \dots, n; \quad \forall k = 1, \dots, K
$$

3. **L1 Regularization Constraints**:

$$
z_j^{(k)} \geq \beta_j^{(k)}, \quad z_j^{(k)} \geq -\beta_j^{(k)}, \quad \forall j = 1, \dots, p; \quad \forall k = 1, \dots, K
$$

4. **Non-Crossing Constraints**:

$$
x_i^{\top} \beta^{(k)} \leq x_i^{\top} \beta^{(k+1)}, \quad \forall i = 1, \dots, n; \quad \forall k = 1, \dots, K-1
$$

This LP formulation can be efficiently solved using the PDLP solver provided by Google's OR-Tools.

## Features
- **Custom Quantiles**: Supports estimation for any quantile $\tau \in (0, 1)$.
- **Multiple Quantiles**: Estimates multiple quantiles simultaneously without crossing.
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
    - `tau` (float or list of floats, default=0.5): The quantile(s) to estimate, each must be between 0 and 1. Can be a single float or a list of floats.
    - `n_bootstrap` (int, default=1000): Number of bootstrap samples for estimating standard errors.
    - `random_state` (int, default=None): Seed for the random number generator.
    - `regularization` (str, default='none'): Type of regularization to apply. Options are `'l1'` for Lasso regularization or `'none'` for no regularization.
    - `alpha` (float, default=0.0): Regularization strength. Must be a non-negative float. Higher values imply stronger regularization.

*Attributes*
- `coef_` (dict): Estimated coefficients for each quantile. Keys are quantile values, and values are arrays of coefficients.
- `intercept_` (dict): Estimated intercept term for each quantile. Keys are quantile values, and values are floats.
- `stderr_` (dict): Standard errors of the coefficients for each quantile. Keys are quantile values, and values are arrays of standard errors.
- `tvalues_` (dict): T-statistics of the coefficients for each quantile. Keys are quantile values, and values are arrays of t-values.
- `pvalues_` (dict): P-values of the coefficients for each quantile. Keys are quantile values, and values are arrays of p-values.

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
    - **Returns**: `y_pred` (dict of ndarrays)

- `summary()`: Return a summary of the regression results.
    - **Returns**: `summary_dict` (dict of pandas DataFrames)

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
print(model.summary()[0.5])
```
**Output**
```
           Coefficient  Std. Error   t-value     P>|t|
Intercept     1.000226    0.236704  4.225639  0.000103
X1           -0.054309    0.545591 -0.099542  0.921114
```

### Example 2: Estimating Multiple Quantiles Simultaneously
```python
from quantile_regression_pdlp import QuantileRegression
import numpy as np

# Data
np.random.seed(0)
X = np.random.rand(50, 1)
y = np.random.rand(50) * 2 + np.random.randn(50) * 0.5

# Model
quantiles = [0.25, 0.5, 0.75]
model = QuantileRegression(tau=quantiles, n_bootstrap=500, random_state=0)
model.fit(X, y)

# Accessing summaries
summaries = model.summary()
for tau, summary_df in summaries.items():
    print(f"\nQuantile: {tau}")
    print(summary_df)

# Making predictions
X_new = np.array([[0.1], [0.5], [0.9]])
predictions = model.predict(X_new)
for tau, y_pred in predictions.items():
    print(f"\nPredictions for quantile {tau}: {y_pred}")
```
**Output**
```
Quantile: 0.25
           Coefficient  Std. Error   t-value     P>|t|
Intercept     0.989803    0.232086  4.264809  0.000091
X1           -0.827212    0.359381 -2.301768  0.025639

Quantile: 0.5
           Coefficient  Std. Error   t-value     P>|t|
Intercept     1.000221    0.231796  4.315086  0.000077
X1           -0.054280    0.531496 -0.102128  0.919072

Quantile: 0.75
           Coefficient  Std. Error   t-value     P>|t|
Intercept     1.612909    0.383085  4.210315  0.000109
X1           -0.163154    0.670333 -0.243393  0.808717

Predictions for quantile 0.25: [0.90708228 0.57619763 0.24531299]

Predictions for quantile 0.5: [0.99479299 0.97308079 0.95136859]

Predictions for quantile 0.75: [1.59659361 1.53133192 1.46607023]
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

#Initialize and fit the model with multiple quantiles and L1 regularization
model = QuantileRegression(
    tau=[0.25, 0.5, 0.75],  # Multiple quantiles
    n_bootstrap=500,
    random_state=42,
    regularization='l1',
    alpha=0.1
)
model.fit(X, y, weights=weights)

#Print the summaries for each quantile
summaries = model.summary()
for tau, summary_df in summaries.items():
    print(f"\nQuantile: {tau}")
    print(summary_df)

#Make predictions for each quantile
X_new = np.array([
    [0.2, 0.3, 0.5],
    [0.6, 0.8, 0.1],
    [0.4, 0.5, 0.9]
])
y_pred = model.predict(X_new)
for tau, preds in y_pred.items():
    print(f"\nPredictions for quantile {tau}: {preds}")
```
**Output**
```
Quantile: 0.25
           Coefficient  Std. Error    t-value         P>|t|
Intercept     0.810970    0.082416   9.839943  0.000000e+00
X1            2.545751    0.099900  25.483014  0.000000e+00
X2           -1.594534    0.086994 -18.329156  0.000000e+00
X3            0.512027    0.087006   5.884950  1.687805e-08

Quantile: 0.5
           Coefficient  Std. Error    t-value     P>|t|
Intercept     1.055079    0.093192  11.321600  0.000000
X1            2.494038    0.100447  24.829305  0.000000
X2           -1.560300    0.105679 -14.764491  0.000000
X3            0.428890    0.105631   4.060266  0.000071

Quantile: 0.75
           Coefficient  Std. Error    t-value     P>|t|
Intercept     1.224177    0.099520  12.300762  0.000000
X1            2.515053    0.117779  21.354093  0.000000
X2           -1.531661    0.120866 -12.672435  0.000000
X3            0.439465    0.128081   3.431139  0.000732

Predictions for quantile 0.25: [1.09777369 1.11399647 1.49282796]

Predictions for quantile 0.5: [1.30024154 1.34615084 1.65854521]

Predictions for quantile 0.75: [1.48742136 1.55182608 1.8598857 ]
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
