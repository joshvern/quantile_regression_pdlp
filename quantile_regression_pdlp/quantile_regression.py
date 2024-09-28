# quantile_regression_pdlp/quantile_regression.py

from ortools.linear_solver import pywraplp
import numpy as np
import pandas as pd
from scipy.stats import t
from tqdm import tqdm


class QuantileRegression:
    """
    Quantile Regression using PDLP solver from Google's OR-Tools, with statistical summaries.

    Parameters
    ----------
    tau : float, default=0.5
        The quantile to estimate, which must be between 0 and 1.

    n_bootstrap : int, default=1000
        Number of bootstrap samples to use for estimating standard errors.

    random_state : int, default=None
        Seed for the random number generator.

    regularization : str, default='none'
        Type of regularization to apply. Options are 'l1' for Lasso regularization or 'none' for no regularization.

    alpha : float, default=0.0
        Regularization strength. Must be a non-negative float. Higher values imply stronger regularization.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients for the linear regression problem.

    intercept_ : float
        Independent term in the linear model.

    stderr_ : ndarray of shape (n_features + 1,)
        Standard errors of the coefficients (including intercept).

    tvalues_ : ndarray of shape (n_features + 1,)
        T-statistics for the coefficients.

    pvalues_ : ndarray of shape (n_features + 1,)
        P-values for the coefficients.

    Methods
    -------
    fit(X, y, weights=None)
        Fit the quantile regression model.

    predict(X)
        Predict using the quantile regression model.

    summary()
        Return a summary of the regression results.
    """

    def __init__(self, tau=0.5, n_bootstrap=1000, random_state=None, regularization='none', alpha=0.0):
        if not 0 < tau < 1:
            raise ValueError("The quantile tau must be between 0 and 1.")
        self.tau = tau
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.regularization = regularization  # Added regularization parameter
        self.alpha = alpha  # Added alpha parameter for regularization strength
        self.coef_ = None
        self.intercept_ = None
        self.stderr_ = None
        self.tvalues_ = None
        self.pvalues_ = None
        self._is_fitted = False

    def fit(self, X, y, weights=None):
        """
        Fit the quantile regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        weights : array-like of shape (n_samples,), optional
            Weights for each observation. Default is None, which assigns equal weight to all observations.

        Returns
        -------
        self : object
            Returns self.
        """
        X = np.asarray(X)
        y = np.asarray(y).flatten()

        if weights is None:
            weights = np.ones_like(y)
        else:
            weights = np.asarray(weights)
            if weights.shape[0] != y.shape[0]:
                raise ValueError("Weights array must have the same length as the number of observations.")

        n_samples, n_features = X.shape

        # Add intercept term by appending a column of ones to X
        X_augmented = np.hstack((np.ones((n_samples, 1)), X))

        # Solve the linear programming problem
        beta_values = self._solve_lp(X_augmented, y, weights)  # Modified to include weights

        self.intercept_ = beta_values[0]
        self.coef_ = beta_values[1:]

        # Estimate standard errors via bootstrapping
        self._compute_standard_errors(X_augmented, y, weights)  # Modified to include weights

        self._is_fitted = True
        return self

    def predict(self, X):
        """
        Predict using the quantile regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        if not self._is_fitted:
            raise Exception("Model is not fitted yet. Please call 'fit' before 'predict'.")

        X = np.asarray(X)
        return np.dot(X, self.coef_) + self.intercept_

    def summary(self):
        """
        Return a summary of the regression results.

        Returns
        -------
        summary_df : pandas DataFrame
            Summary table with coefficients, standard errors, t-values, and p-values.
        """
        if not self._is_fitted:
            raise Exception("Model is not fitted yet. Please call 'fit' before 'summary'.")

        coef = np.concatenate(([self.intercept_], self.coef_))
        index = ['Intercept'] + [f'X{i}' for i in range(1, len(self.coef_) + 1)]
        summary_df = pd.DataFrame({
            'Coefficient': coef,
            'Std. Error': self.stderr_,
            't-value': self.tvalues_,
            'P>|t|': self.pvalues_,
        }, index=index)
        return summary_df

    def _solve_lp(self, X, y, weights):
        """
        Solve the quantile regression problem formulated as a linear program.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features + 1)
            Augmented feature matrix with intercept.

        y : ndarray of shape (n_samples,)
            Target values.

        weights : ndarray of shape (n_samples,)
            Weights for each observation.

        Returns
        -------
        beta_values : ndarray of shape (n_features + 1,)
            Estimated coefficients (including intercept).
        """
        n_samples, n_features = X.shape

        # Create the solver instance
        solver = pywraplp.Solver.CreateSolver('PDLP')
        if not solver:
            raise Exception("PDLP solver is not available.")

        infinity = solver.infinity()

        # Define variables: coefficients and residuals
        beta = [solver.NumVar(-infinity, infinity, f'beta_{j}') for j in range(n_features)]
        r_pos = [solver.NumVar(0, infinity, f'r_pos_{i}') for i in range(n_samples)]
        r_neg = [solver.NumVar(0, infinity, f'r_neg_{i}') for i in range(n_samples)]

        # If L1 regularization is specified, introduce auxiliary variables
        if self.regularization == 'l1' and self.alpha > 0:
            z = [solver.NumVar(0, infinity, f'z_{j}') for j in range(1, n_features)]
            for j in range(1, n_features):
                # z_j >= beta_j
                solver.Add(beta[j] <= z[j-1])
                # z_j >= -beta_j
                solver.Add(-beta[j] <= z[j-1])

        # Add constraints: y_i = x_i^T beta + r_pos_i - r_neg_i
        for i in range(n_samples):
            constraint_expr = sum(X[i, j] * beta[j] for j in range(n_features)) + r_pos[i] - r_neg[i]
            solver.Add(constraint_expr == y[i])

        # Define the objective function
        objective = solver.Objective()
        for i in range(n_samples):
            objective.SetCoefficient(r_pos[i], self.tau * weights[i])
            objective.SetCoefficient(r_neg[i], (1 - self.tau) * weights[i])

        # Add L1 regularization to the objective if specified
        if self.regularization == 'l1' and self.alpha > 0:
            for j in range(n_features - 1):
                objective.SetCoefficient(z[j], self.alpha)

        objective.SetMinimization()

        # Solve the LP problem
        status = solver.Solve()

        if status != pywraplp.Solver.OPTIMAL:
            raise Exception('Solver did not find an optimal solution.')

        # Extract the coefficients
        beta_values = np.array([beta[j].solution_value() for j in range(n_features)])

        return beta_values

    def _compute_standard_errors(self, X, y, weights):
        """
        Compute standard errors via bootstrapping.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features + 1)
            Augmented feature matrix with intercept.

        y : ndarray of shape (n_samples,)
            Target values.

        weights : ndarray of shape (n_samples,)
            Weights for each observation.
        """
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        n_features = X.shape[1]
        beta_bootstrap = np.zeros((self.n_bootstrap, n_features))

        for i in tqdm(range(self.n_bootstrap), desc='Bootstrapping'):
            sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]
            weights_sample = weights[sample_indices]

            try:
                beta_sample = self._solve_lp(X_sample, y_sample, weights_sample)
                beta_bootstrap[i, :] = beta_sample
            except Exception:
                beta_bootstrap[i, :] = np.nan

        # Remove any iterations where the solver failed
        valid_bootstrap = beta_bootstrap[~np.isnan(beta_bootstrap).any(axis=1)]

        # Compute standard errors
        stderr = np.std(valid_bootstrap, axis=0, ddof=1)
        self.stderr_ = stderr

        # Compute t-values and p-values
        coef_full = np.concatenate(([self.intercept_], self.coef_))
        stderr_full = self.stderr_
        self.tvalues_ = coef_full / stderr_full

        df = len(y) - (n_features - 1)  # Degrees of freedom
        self.pvalues_ = 2 * (1 - t.cdf(np.abs(self.tvalues_), df=df))
