# quantile_regression_pdlp/quantile_regression.py

from ortools.linear_solver import pywraplp
import numpy as np
import pandas as pd
from scipy.stats import t
from tqdm import tqdm
from sklearn.base import BaseEstimator, RegressorMixin


class QuantileRegression(BaseEstimator, RegressorMixin):
    """
    Quantile Regression using PDLP solver from Google's OR-Tools, with statistical summaries.

    Parameters
    ----------
    tau : float or list of floats, default=0.5
        The quantile(s) to estimate, each must be between 0 and 1.
        Can be a single float for one quantile or a list of floats for multiple quantiles.

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
    coef_ : dict
        Estimated coefficients for each quantile. Keys are quantile values, and values are arrays of coefficients.

    intercept_ : dict
        Estimated intercept term for each quantile. Keys are quantile values, and values are floats.

    stderr_ : dict
        Standard errors of the coefficients for each quantile. Keys are quantile values, and values are arrays of standard errors.

    tvalues_ : dict
        T-statistics of the coefficients for each quantile. Keys are quantile values, and values are arrays of t-values.

    pvalues_ : dict
        P-values of the coefficients for each quantile. Keys are quantile values, and values are arrays of p-values.

    feature_names_ : list
        List of feature names. If input X is a pandas DataFrame, the column names are used; otherwise, generic names are assigned.

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
        self.tau = tau
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.regularization = regularization
        self.alpha = alpha

        # Attributes initialized during fitting
        self.coef_ = None
        self.intercept_ = None
        self.stderr_ = None
        self.tvalues_ = None
        self.pvalues_ = None
        self.feature_names_ = None
        self._is_fitted = None

    def fit(self, X, y, weights=None):
        """
        Fit the quantile regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Can be a NumPy array or a pandas DataFrame.

        y : array-like of shape (n_samples,)
            Target values. Can be a NumPy array or a pandas Series.

        weights : array-like of shape (n_samples,), optional
            Weights for each observation. Default is None, which assigns equal weight to all observations.

        Returns
        -------
        self : object
            Returns self.
        """
        # Handle pandas DataFrames and Series
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X = X.values
        else:
            self.feature_names_ = [f'X{i}' for i in range(1, X.shape[1] + 1)]
            X = np.asarray(X)

        if isinstance(y, pd.Series):
            y = y.values
        else:
            y = np.asarray(y).flatten()

        if weights is None:
            weights = np.ones_like(y)
        else:
            weights = np.asarray(weights)
            if weights.shape[0] != y.shape[0]:
                raise ValueError("Weights array must have the same length as the number of observations.")

        # Validate tau in __init__
        self._validate_tau()

        n_samples, n_features = X.shape

        # Add intercept term by appending a column of ones to X
        X_augmented = np.hstack((np.ones((n_samples, 1)), X))

        # Initialize storage for multiple quantiles
        self.coef_ = {q: np.zeros(n_features) for q in self.tau}
        self.intercept_ = {q: 0.0 for q in self.tau}
        self.stderr_ = {q: np.zeros(n_features + 1) for q in self.tau}
        self.tvalues_ = {q: np.zeros(n_features + 1) for q in self.tau}
        self.pvalues_ = {q: np.zeros(n_features + 1) for q in self.tau}
        self._is_fitted = {q: False for q in self.tau}

        # Solve LP for all quantiles simultaneously with non-crossing constraints
        coefficients = self._solve_multiple_lp(X_augmented, y, weights)

        # Extract the coefficients
        for q in self.tau:
            self.intercept_[q] = coefficients[q][0]
            self.coef_[q] = coefficients[q][1:]

        # Compute standard errors via bootstrapping
        self._compute_standard_errors(X_augmented, y, weights)

        # Mark all quantiles as fitted
        for q in self.tau:
            self._is_fitted[q] = True

        return self

    def _validate_tau(self):
        """
        Validate the tau parameter to ensure all quantiles are between 0 and 1 and properly sorted.

        Raises
        ------
        ValueError, TypeError
        """
        if isinstance(self.tau, float):
            if not 0 < self.tau < 1:
                raise ValueError("Each quantile tau must be between 0 and 1.")
            self.tau = [self.tau]
        elif isinstance(self.tau, list):
            if not all(isinstance(q, float) and 0 < q < 1 for q in self.tau):
                raise ValueError("All quantiles tau must be floats between 0 and 1.")
            self.tau = sorted(self.tau)  # Sort to enforce ordering
        else:
            raise TypeError("tau must be a float or a list of floats.")

    def _solve_multiple_lp(self, X, y, weights, return_coefficients=True):
        """
        Solve multiple quantile regression problems as a single LP with non-crossing constraints.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features + 1)
            Augmented feature matrix with intercept.

        y : ndarray of shape (n_samples,)
            Target values.

        weights : ndarray of shape (n_samples,)
            Weights for each observation.

        return_coefficients : bool, default=True
            Whether to return the estimated coefficients.

        Returns
        -------
        coefficients : dict (only if return_coefficients is True)
            Estimated coefficients for each quantile.
        """
        n_samples, n_features = X.shape
        n_quantiles = len(self.tau)

        # Create the solver instance
        solver = pywraplp.Solver.CreateSolver('PDLP')
        if not solver:
            raise Exception("PDLP solver is not available.")

        infinity = solver.infinity()

        # Define variables for each quantile
        beta = {q: [solver.NumVar(-infinity, infinity, f'beta_{j}_q{q}') for j in range(n_features)] for q in self.tau}
        r_pos = {q: [solver.NumVar(0, infinity, f'r_pos_{i}_q{q}') for i in range(n_samples)] for q in self.tau}
        r_neg = {q: [solver.NumVar(0, infinity, f'r_neg_{i}_q{q}') for i in range(n_samples)] for q in self.tau}

        # If L1 regularization is specified, introduce auxiliary variables for each quantile and feature
        if self.regularization == 'l1' and self.alpha > 0:
            z = {q: [solver.NumVar(0, infinity, f'z_{j}_q{q}') for j in range(1, n_features)] for q in self.tau}
            for q in self.tau:
                for j in range(1, n_features):
                    # z_j_q >= beta_j_q
                    solver.Add(beta[q][j] <= z[q][j - 1])
                    # z_j_q >= -beta_j_q
                    solver.Add(-beta[q][j] <= z[q][j - 1])

        # Add constraints and objective
        objective = solver.Objective()
        for q in self.tau:
            for i in range(n_samples):
                # Residual constraints: y_i = x_i^T beta_q + r_pos_q_i - r_neg_q_i
                constraint_expr = sum(X[i, j] * beta[q][j] for j in range(n_features)) + r_pos[q][i] - r_neg[q][i]
                solver.Add(constraint_expr == y[i])

                # Objective coefficients
                objective.SetCoefficient(r_pos[q][i], q * weights[i])
                objective.SetCoefficient(r_neg[q][i], (1 - q) * weights[i])

        # Add L1 regularization to the objective if specified
        if self.regularization == 'l1' and self.alpha > 0:
            for q in self.tau:
                for j in range(n_features - 1):
                    objective.SetCoefficient(z[q][j], self.alpha)

        objective.SetMinimization()

        # Add non-crossing constraints
        for i in range(n_samples):
            for k in range(n_quantiles - 1):
                q_lower = self.tau[k]
                q_upper = self.tau[k + 1]
                # Predicted values for quantile q_lower <= predicted values for quantile q_upper
                pred_lower = sum(X[i, j] * beta[q_lower][j] for j in range(n_features))
                pred_upper = sum(X[i, j] * beta[q_upper][j] for j in range(n_features))
                solver.Add(pred_lower <= pred_upper)

        # Solve the LP problem
        status = solver.Solve()

        if status != pywraplp.Solver.OPTIMAL:
            raise Exception('Solver did not find an optimal solution.')

        if return_coefficients:
            # Extract the coefficients
            coefficients = {}
            for q in self.tau:
                intercept = beta[q][0].solution_value()
                coef = np.array([beta[q][j].solution_value() for j in range(1, n_features)])
                coefficients[q] = np.concatenate(([intercept], coef))
            return coefficients

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
        n_quantiles = len(self.tau)
        beta_bootstrap = {q: np.zeros((self.n_bootstrap, n_features)) for q in self.tau}

        for i in tqdm(range(self.n_bootstrap), desc='Bootstrapping'):
            sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]
            weights_sample = weights[sample_indices]

            try:
                # Solve for multiple quantiles
                beta_sample = self._solve_multiple_lp(X_sample, y_sample, weights_sample, return_coefficients=True)
                for q in self.tau:
                    beta_bootstrap[q][i, :] = beta_sample[q]
            except Exception:
                for q in self.tau:
                    beta_bootstrap[q][i, :] = np.nan

        # Compute standard errors, t-values, and p-values for each quantile
        for q in self.tau:
            # Remove any iterations where the solver failed
            valid_bootstrap = beta_bootstrap[q][~np.isnan(beta_bootstrap[q]).any(axis=1)]

            if valid_bootstrap.size == 0:
                raise Exception(f"All bootstrap iterations failed for quantile {q}.")

            # Compute standard errors
            stderr = np.std(valid_bootstrap, axis=0, ddof=1)
            self.stderr_[q] = stderr

            # Compute t-values and p-values
            coef_full = np.concatenate(([self.intercept_[q]], self.coef_[q]))
            stderr_full = self.stderr_[q]
            tvalues_full = coef_full / stderr_full
            self.tvalues_[q] = tvalues_full

            df = len(y) - (X.shape[1] - 1)  # Degrees of freedom
            pvalues_full = 2 * (1 - t.cdf(np.abs(tvalues_full), df=df))
            self.pvalues_[q] = pvalues_full

    def predict(self, X):
        """
        Predict using the quantile regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples. Can be a NumPy array or a pandas DataFrame.

        Returns
        -------
        y_pred : dict of ndarrays
            Predicted values for each quantile. Keys are quantile values, and values are arrays of predictions.
        """
        if not all(self._is_fitted.values()):
            raise Exception("Model is not fitted yet. Please call 'fit' before 'predict'.")

        # Handle pandas DataFrames
        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = np.asarray(X)

        n_samples = X.shape[0]

        # Add intercept term
        X_augmented = np.hstack((np.ones((n_samples, 1)), X))

        y_pred = {}
        for q in self.tau:
            y_pred[q] = X_augmented @ np.concatenate(([self.intercept_[q]], self.coef_[q]))
        return y_pred

    def summary(self):
        """
        Return a summary of the regression results.

        Returns
        -------
        summary_dict : dict of pandas DataFrames
            Summary tables for each quantile with coefficients, standard errors, t-values, and p-values.
        """
        if not all(self._is_fitted.values()):
            raise Exception("Model is not fitted yet. Please call 'fit' before 'summary'.")

        summary_dict = {}
        for q in self.tau:
            coef = np.concatenate(([self.intercept_[q]], self.coef_[q]))
            index = ['Intercept'] + self.feature_names_
            summary_df = pd.DataFrame({
                'Coefficient': coef,
                'Std. Error': self.stderr_[q],
                't-value': self.tvalues_[q],
                'P>|t|': self.pvalues_[q],
            }, index=index)
            summary_dict[q] = summary_df

        return summary_dict

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            'tau': self.tau,
            'n_bootstrap': self.n_bootstrap,
            'random_state': self.random_state,
            'regularization': self.regularization,
            'alpha': self.alpha,
        }

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : object
            Returns self.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
