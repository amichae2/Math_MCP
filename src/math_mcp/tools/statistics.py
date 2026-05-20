"""Statistics and probability tools using SciPy and NumPy."""

import json
from typing import Any

import numpy as np
from scipy import optimize, special, stats
import sympy

from ..utils.errors import tool_error_handler
from ..utils.latex_utils import safe_latex

_DEFAULT_PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]
_DISTRIBUTION_ALIASES: dict[str, str] = {
    "normal": "norm",
    "gaussian": "norm",
    "exponential": "expon",
    "chi2": "chi2",
    "uniform_discrete": "randint",
    "binomial": "binom",
    "poisson": "poisson",
    "geometric": "geom",
    "nbinom": "nbinom",
    "bernoulli": "bernoulli",
    "lognormal": "lognorm",
    "weibull": "weibull_min",
    "pareto": "pareto",
    "cauchy": "cauchy",
    "gamma": "gamma",
    "beta": "beta",
    "uniform": "uniform",
    "f": "f",
    "t": "t",
    "hypergeometric": "hypergeom",
}
_RANDOM_ALIASES: dict[str, str] = {
    "uniform": "uniform",
    "normal": "normal",
    "exponential": "exponential",
    "poisson": "poisson",
    "binomial": "binomial",
    "beta": "beta",
    "gamma": "gamma",
    "lognormal": "lognormal",
    "multivariate_normal": "multivariate_normal",
    "dirichlet": "dirichlet",
    "wishart": "wishart",
}


def _clean_numeric_data(data: list[float], *, name: str = "data") -> tuple[np.ndarray, str | None]:
    if not data:
        raise ValueError(f"{name} must not be empty")
    array = np.asarray(data, dtype=float)
    finite_mask = np.isfinite(array)
    warning = None
    if not np.all(finite_mask):
        array = array[finite_mask]
        warning = f"non-finite values were removed from {name}"
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one finite number")
    return array, warning


def _mode_values(data: np.ndarray) -> float | list[float]:
    values, counts = np.unique(data, return_counts=True)
    max_count = counts.max()
    modes = values[counts == max_count]
    if modes.size == 1:
        return float(modes[0])
    return modes.astype(float).tolist()


def _get_distribution(name: str) -> Any:
    attr_name = _DISTRIBUTION_ALIASES.get(name.lower(), name.lower())
    distribution = getattr(stats, attr_name, None)
    if distribution is None:
        raise ValueError(f"unsupported distribution: {name}")
    return distribution


def _prepare_distribution_params(params: dict[str, float] | None) -> dict[str, float]:
    values = dict(params or {})
    if "scale" in values and values["scale"] <= 0:
        raise ValueError("scale must be positive")
    if "p" in values and not 0 <= values["p"] <= 1:
        raise ValueError("p must be between 0 and 1")
    return values


def _make_distribution(distribution: Any, params: dict[str, float]) -> Any:
    shape_keys = [key for key in params if key not in {"loc", "scale"}]
    shape_args = [params[key] for key in shape_keys]
    kwargs = {key: params[key] for key in ("loc", "scale") if key in params}
    return distribution(*shape_args, **kwargs)


def _cohens_d(data1: np.ndarray, data2: np.ndarray | None = None, *, popmean: float | None = None) -> float:
    if data2 is None:
        if popmean is None:
            raise ValueError("popmean is required for one-sample Cohen's d")
        return float((np.mean(data1) - popmean) / np.std(data1, ddof=1))
    pooled = np.sqrt(
        (((data1.size - 1) * np.var(data1, ddof=1)) + ((data2.size - 1) * np.var(data2, ddof=1)))
        / (data1.size + data2.size - 2)
    )
    return float((np.mean(data1) - np.mean(data2)) / pooled)


def _confidence_interval_mean_diff(
    data1: np.ndarray,
    data2: np.ndarray | None = None,
    *,
    alpha: float,
    paired: bool = False,
    popmean: float | None = None,
) -> list[float] | None:
    if data2 is None:
        if popmean is None or data1.size < 2:
            return None
        mean_diff = np.mean(data1) - popmean
        se = stats.sem(data1)
        critical = stats.t.ppf(1 - alpha / 2, data1.size - 1)
        margin = critical * se
        return [float(mean_diff - margin), float(mean_diff + margin)]
    differences = data1 - data2 if paired else None
    if paired and differences is not None:
        return _confidence_interval_mean_diff(differences, alpha=alpha, popmean=0.0)
    if data1.size < 2 or data2.size < 2:
        return None
    mean_diff = np.mean(data1) - np.mean(data2)
    var1 = np.var(data1, ddof=1)
    var2 = np.var(data2, ddof=1)
    se = np.sqrt(var1 / data1.size + var2 / data2.size)
    df_num = (var1 / data1.size + var2 / data2.size) ** 2
    df_den = ((var1 / data1.size) ** 2) / (data1.size - 1) + ((var2 / data2.size) ** 2) / (data2.size - 1)
    df = df_num / df_den
    critical = stats.t.ppf(1 - alpha / 2, df)
    margin = critical * se
    return [float(mean_diff - margin), float(mean_diff + margin)]


def _design_matrix(
    x: list[float] | list[list[float]],
    *,
    model: str,
    degree: int,
    fit_intercept: bool,
) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    if not x:
        raise ValueError("x must not be empty")
    if isinstance(x[0], list):
        predictors = np.asarray(x, dtype=float)
        if predictors.ndim != 2:
            raise ValueError("x must be a list of rows or a single predictor list")
        if model == "polynomial":
            raise ValueError("polynomial regression expects a single predictor list")
        feature_names = [f"x{i + 1}" for i in range(predictors.shape[1])]
        base_predictors = predictors
    else:
        base = np.asarray(x, dtype=float).reshape(-1, 1)
        if model == "polynomial":
            if degree < 1:
                raise ValueError("degree must be positive")
            predictors = np.column_stack([base[:, 0] ** power for power in range(1, degree + 1)])
            feature_names = [f"x^{power}" if power > 1 else "x1" for power in range(1, degree + 1)]
        else:
            predictors = base
            feature_names = ["x1"]
        base_predictors = base
    if fit_intercept:
        design = np.column_stack([np.ones(predictors.shape[0]), predictors])
        names = ["intercept", *feature_names]
    else:
        design = predictors
        names = feature_names
    return design, names, base_predictors


def _ols_fit(design: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float, np.ndarray, np.ndarray]:
    beta = np.linalg.pinv(design.T @ design) @ design.T @ y
    predictions = design @ beta
    residuals = y - predictions
    n_obs, n_params = design.shape
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 if ss_tot == 0 else 1 - ss_res / ss_tot
    adjusted_r_squared = 1.0 if n_obs <= n_params else 1 - (1 - r_squared) * (n_obs - 1) / (n_obs - n_params)
    sigma2 = ss_res / max(n_obs - n_params, 1)
    covariance = sigma2 * np.linalg.pinv(design.T @ design)
    std_errors = np.sqrt(np.diag(covariance))
    return beta, predictions, r_squared, adjusted_r_squared, sigma2, covariance, std_errors


def _rescale_standardized_coefficients(
    beta: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    fit_intercept: bool,
) -> np.ndarray:
    if not fit_intercept:
        return beta / scales
    slopes = beta[1:] / scales
    intercept = beta[0] - float(np.sum((beta[1:] * means) / scales))
    return np.concatenate([[intercept], slopes])


def _bootstrap_statistic(data: np.ndarray, statistic: str) -> Any:
    operations: dict[str, Any] = {
        "mean": np.mean,
        "median": np.median,
        "std": lambda values: np.std(values, ddof=1),
        "var": lambda values: np.var(values, ddof=1),
    }
    if statistic == "custom_function":
        raise ValueError("custom_function is not supported through this API; use mean, median, std, or var")
    if statistic not in operations:
        raise ValueError("statistic must be one of: mean, median, std, var, custom_function")
    return operations[statistic](data)


@tool_error_handler("describe")
async def describe(
    data: list[float],
    percentiles: list[float] | None = None,
) -> dict[str, Any]:
    """Compute comprehensive descriptive statistics for a dataset."""
    cleaned, warning = _clean_numeric_data(data)
    percentile_values = percentiles or _DEFAULT_PERCENTILES
    percentile_results = np.percentile(cleaned, percentile_values)
    q1 = float(np.percentile(cleaned, 25))
    q3 = float(np.percentile(cleaned, 75))
    result = {
        "result": f"n={cleaned.size}, mean={float(np.mean(cleaned)):.6g}, std={float(np.std(cleaned, ddof=1 if cleaned.size > 1 else 0)):.6g}",
        "n": int(cleaned.size),
        "mean": float(np.mean(cleaned)),
        "std": float(np.std(cleaned, ddof=1 if cleaned.size > 1 else 0)),
        "variance": float(np.var(cleaned, ddof=1 if cleaned.size > 1 else 0)),
        "min": float(np.min(cleaned)),
        "max": float(np.max(cleaned)),
        "range": float(np.max(cleaned) - np.min(cleaned)),
        "median": float(np.median(cleaned)),
        "mode": _mode_values(cleaned),
        "skewness": float(stats.skew(cleaned, bias=False)) if cleaned.size > 2 else 0.0,
        "kurtosis": float(stats.kurtosis(cleaned, fisher=True, bias=False)) if cleaned.size > 3 else 0.0,
        "percentiles": {str(int(level) if float(level).is_integer() else level): float(value) for level, value in zip(percentile_values, percentile_results)},
        "iqr": float(q3 - q1),
        "sem": float(stats.sem(cleaned)) if cleaned.size > 1 else 0.0,
        "latex": None,
    }
    if warning is not None:
        result["warning"] = warning
    return result


@tool_error_handler("distribution")
async def distribution(
    name: str,
    operation: str,
    params: dict[str, float] | None = None,
    x: float | list[float] | None = None,
    q: float | list[float] | None = None,
    size: int = 1,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Evaluate, fit, and sample probability distributions."""
    distribution_object = _get_distribution(name)
    parameter_values = _prepare_distribution_params(params)
    op = operation.lower()
    if op not in {"pdf", "cdf", "sf", "ppf", "rvs", "stats", "fit"}:
        raise ValueError("operation must be one of: pdf, cdf, sf, ppf, rvs, stats, fit")

    if op == "fit":
        if not isinstance(x, list):
            raise ValueError("fit requires x to be a data array")
        fit_data, warning = _clean_numeric_data(x, name="x")
        fitted = distribution_object.fit(fit_data)
        arg_names = []
        shapes = getattr(distribution_object, "shapes", None)
        if shapes:
            arg_names.extend([part.strip() for part in shapes.split(",")])
        arg_names.extend(["loc", "scale"])
        fitted_params = {name: float(value) for name, value in zip(arg_names, fitted)}
        frozen = distribution_object(*fitted[:-2], loc=fitted[-2], scale=fitted[-1])
        log_likelihood = float(np.sum(frozen.logpdf(fit_data) if hasattr(frozen, "logpdf") else frozen.logpmf(fit_data)))
        result = {"params": fitted_params, "log_likelihood": log_likelihood, "latex": None}
        if warning is not None:
            result["warning"] = warning
        return result

    frozen = _make_distribution(distribution_object, parameter_values)
    if op in {"pdf", "cdf", "sf"}:
        if x is None:
            raise ValueError(f"{op} requires x")
        values = np.asarray(x if isinstance(x, list) else [x], dtype=float)
        evaluator_name = "pmf" if op == "pdf" and hasattr(frozen, "pmf") and not hasattr(frozen, "pdf") else op
        evaluator = getattr(frozen, evaluator_name)
        evaluated = evaluator(values)
        result_value: float | list[float] = float(evaluated[0]) if values.size == 1 else np.asarray(evaluated, dtype=float).tolist()
        return {
            "result": result_value,
            "latex": f"{op} = {result_value}",
        }
    if op == "ppf":
        if q is None:
            raise ValueError("ppf requires q")
        quantiles = np.asarray(q if isinstance(q, list) else [q], dtype=float)
        if np.any((quantiles < 0) | (quantiles > 1)):
            raise ValueError("q values must be between 0 and 1")
        values = frozen.ppf(quantiles)
        result_value = float(values[0]) if quantiles.size == 1 else np.asarray(values, dtype=float).tolist()
        return {
            "result": result_value,
            "latex": f"F^{{-1}} = {result_value}",
        }
    if op == "rvs":
        samples = frozen.rvs(size=size, random_state=random_state)
        sample_array = np.asarray(samples, dtype=float)
        response = {
            "result": f"Generated {int(sample_array.size)} samples",
            "sample_stats": {
                "mean": float(np.mean(sample_array)),
                "std": float(np.std(sample_array, ddof=1 if sample_array.size > 1 else 0)),
            },
            "latex": None,
        }
        if sample_array.size <= 500:
            response["samples"] = sample_array.tolist()
        return response
    if op == "stats":
        mean_value, variance_value, skew_value, kurtosis_value = frozen.stats(moments="mvsk")
        return {
            "mean": float(mean_value),
            "variance": float(variance_value),
            "skewness": float(skew_value),
            "kurtosis": float(kurtosis_value),
            "latex": None,
        }
    raise ValueError("unsupported distribution operation")


@tool_error_handler("hypothesis_test")
async def hypothesis_test(
    test: str,
    data1: list[float],
    data2: list[float] | None = None,
    params: dict | None = None,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Perform a named statistical hypothesis test."""
    params = dict(params or {})
    cleaned1, warning1 = _clean_numeric_data(data1, name="data1")
    cleaned2, warning2 = _clean_numeric_data(data2, name="data2") if data2 is not None else (None, None)
    warnings_list = [warning for warning in [warning1, warning2] if warning is not None]
    test_name = test.lower()
    statistic: float
    p_value: float
    confidence_interval: list[float] | None = None
    effect_size: float | None = None
    degrees_of_freedom: float | None = None

    if test_name == "ttest_1samp":
        popmean = float(params.get("popmean", 0.0))
        result = stats.ttest_1samp(cleaned1, popmean, alternative=alternative)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        confidence_interval = _confidence_interval_mean_diff(cleaned1, alpha=alpha, popmean=popmean)
        effect_size = _cohens_d(cleaned1, popmean=popmean)
        degrees_of_freedom = float(cleaned1.size - 1)
    elif test_name == "ttest_ind":
        if cleaned2 is None:
            raise ValueError("ttest_ind requires data2")
        equal_var = bool(params.get("equal_var", True))
        result = stats.ttest_ind(cleaned1, cleaned2, alternative=alternative, equal_var=equal_var)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        confidence_interval = _confidence_interval_mean_diff(cleaned1, cleaned2, alpha=alpha)
        effect_size = _cohens_d(cleaned1, cleaned2)
        if equal_var:
            degrees_of_freedom = float(cleaned1.size + cleaned2.size - 2)
        else:
            var1 = np.var(cleaned1, ddof=1)
            var2 = np.var(cleaned2, ddof=1)
            numerator = (var1 / cleaned1.size + var2 / cleaned2.size) ** 2
            denominator = ((var1 / cleaned1.size) ** 2) / (cleaned1.size - 1) + ((var2 / cleaned2.size) ** 2) / (cleaned2.size - 1)
            degrees_of_freedom = float(numerator / denominator)
    elif test_name == "ttest_paired":
        if cleaned2 is None or cleaned1.size != cleaned2.size:
            raise ValueError("ttest_paired requires data2 of the same length")
        result = stats.ttest_rel(cleaned1, cleaned2, alternative=alternative)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        confidence_interval = _confidence_interval_mean_diff(cleaned1, cleaned2, alpha=alpha, paired=True)
        effect_size = _cohens_d(cleaned1 - cleaned2, popmean=0.0)
        degrees_of_freedom = float(cleaned1.size - 1)
    elif test_name == "wilcoxon":
        if cleaned2 is None:
            raise ValueError("wilcoxon requires paired data2")
        result = stats.wilcoxon(cleaned1, cleaned2, alternative=alternative)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        effect_size = float(abs(stats.norm.ppf(max(p_value / 2, 1e-16))) / np.sqrt(cleaned1.size))
        degrees_of_freedom = float(cleaned1.size - 1)
    elif test_name == "mannwhitney":
        if cleaned2 is None:
            raise ValueError("mannwhitney requires data2")
        result = stats.mannwhitneyu(cleaned1, cleaned2, alternative=alternative)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        effect_size = float(statistic / (cleaned1.size * cleaned2.size))
    elif test_name == "ks":
        if cleaned2 is not None:
            result = stats.ks_2samp(cleaned1, cleaned2, alternative=alternative)
        else:
            distribution_name = params.get("distribution", "norm")
            result = stats.kstest(cleaned1, distribution_name, alternative=alternative)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    elif test_name == "chi2_contingency":
        observed = np.asarray(params.get("observed", data1), dtype=float)
        chi2, p_value, dof, _ = stats.chi2_contingency(observed)
        statistic = float(chi2)
        degrees_of_freedom = float(dof)
        total = np.sum(observed)
        effect_size = float(np.sqrt(chi2 / total)) if total > 0 else None
    elif test_name == "f_oneway":
        groups = [cleaned1]
        if cleaned2 is not None:
            groups.append(cleaned2)
        for group in params.get("groups", []):
            groups.append(_clean_numeric_data(group, name="group")[0])
        result = stats.f_oneway(*groups)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        grand_mean = np.mean(np.concatenate(groups))
        ss_between = float(sum(group.size * (np.mean(group) - grand_mean) ** 2 for group in groups))
        ss_total = float(sum(np.sum((group - grand_mean) ** 2) for group in groups))
        effect_size = float(ss_between / ss_total) if ss_total > 0 else None
        degrees_of_freedom = float(sum(group.size for group in groups) - len(groups))
    elif test_name == "kruskal":
        groups = [cleaned1]
        if cleaned2 is not None:
            groups.append(cleaned2)
        for group in params.get("groups", []):
            groups.append(_clean_numeric_data(group, name="group")[0])
        result = stats.kruskal(*groups)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    elif test_name == "shapiro":
        result = stats.shapiro(cleaned1)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    elif test_name == "bartlett":
        if cleaned2 is None:
            raise ValueError("bartlett requires data2")
        result = stats.bartlett(cleaned1, cleaned2)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    elif test_name == "levene":
        if cleaned2 is None:
            raise ValueError("levene requires data2")
        result = stats.levene(cleaned1, cleaned2, center=params.get("center", "median"))
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    elif test_name == "pearsonr":
        if cleaned2 is None:
            raise ValueError("pearsonr requires data2")
        result = stats.pearsonr(cleaned1, cleaned2)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        effect_size = statistic
    elif test_name == "spearmanr":
        if cleaned2 is None:
            raise ValueError("spearmanr requires data2")
        result = stats.spearmanr(cleaned1, cleaned2, alternative=alternative)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        effect_size = statistic
    else:
        raise ValueError("unsupported test name")

    reject_h0 = bool(p_value < alpha)
    response = {
        "result": f"{'Reject' if reject_h0 else 'Fail to reject'} H0 (p={p_value:.6g} {'<' if reject_h0 else '>='} {alpha})",
        "test": test_name,
        "statistic": statistic,
        "p_value": p_value,
        "alpha": alpha,
        "reject_h0": reject_h0,
        "confidence_interval": confidence_interval,
        "effect_size": effect_size,
        "degrees_of_freedom": degrees_of_freedom,
        "latex": f"stat={statistic:.6g}, p={p_value:.6g}",
    }
    if warnings_list:
        response["warning"] = "; ".join(warnings_list)
    if test_name in {"ttest_1samp", "ttest_ind", "ttest_paired", "f_oneway"} and cleaned1.size <= 5000:
        shapiro_p = float(stats.shapiro(cleaned1).pvalue) if cleaned1.size >= 3 else 1.0
        if shapiro_p < 0.05:
            response["assumption_warning"] = "data may not be normally distributed; consider a nonparametric alternative"
    return response


@tool_error_handler("regression")
async def regression(
    x: list[float] | list[list[float]],
    y: list[float],
    model: str = "linear",
    degree: int = 1,
    fit_intercept: bool = True,
    standardize: bool = False,
    include_stats: bool = True,
) -> dict[str, Any]:
    """Perform regression analysis for linear, polynomial, or logistic models."""
    y_array, warning = _clean_numeric_data(y, name="y")
    model_name = model.lower()
    if model_name not in {"linear", "polynomial", "logistic"}:
        raise ValueError("model must be one of: linear, polynomial, logistic")
    design, names, predictors = _design_matrix(x, model=model_name, degree=degree, fit_intercept=fit_intercept)
    if design.shape[0] != y_array.size:
        raise ValueError("x and y must contain the same number of observations")

    means = np.zeros(design.shape[1] - (1 if fit_intercept else 0))
    scales = np.ones_like(means)
    fitted_design = design.copy()
    if standardize and design.shape[1] > (1 if fit_intercept else 0):
        feature_slice = slice(1, None) if fit_intercept else slice(None)
        means = np.mean(design[:, feature_slice], axis=0)
        scales = np.std(design[:, feature_slice], axis=0, ddof=0)
        scales[scales == 0] = 1.0
        fitted_design[:, feature_slice] = (design[:, feature_slice] - means) / scales

    warning_messages: list[str] = []
    if warning is not None:
        warning_messages.append(warning)
    if predictors is not None and predictors.shape[1] > 1:
        corr = np.corrcoef(predictors, rowvar=False)
        off_diag = corr[~np.eye(corr.shape[0], dtype=bool)] if corr.ndim == 2 else np.array([])
        if off_diag.size and np.any(np.abs(off_diag) > 0.9):
            warning_messages.append("predictors show strong multicollinearity")

    if model_name in {"linear", "polynomial"}:
        beta, predictions, r_squared, adjusted_r_squared, sigma2, covariance, std_errors = _ols_fit(fitted_design, y_array)
        if standardize and scales.size:
            beta = _rescale_standardized_coefficients(beta, means, scales, fit_intercept)
            transformed_design = design
            predictions = transformed_design @ beta
            residuals = y_array - predictions
            sigma2 = float(np.sum(residuals**2) / max(y_array.size - design.shape[1], 1))
            covariance = sigma2 * np.linalg.pinv(design.T @ design)
            std_errors = np.sqrt(np.diag(covariance))
            r_squared = 1.0 if np.allclose(y_array, predictions) else 1 - float(np.sum((y_array - predictions) ** 2)) / float(np.sum((y_array - np.mean(y_array)) ** 2))
            adjusted_r_squared = 1.0 if y_array.size <= design.shape[1] else 1 - (1 - r_squared) * (y_array.size - 1) / (y_array.size - design.shape[1])
        dof = max(y_array.size - design.shape[1], 1)
        t_stats = np.divide(beta, std_errors, out=np.zeros_like(beta), where=std_errors > 0)
        p_values = (2 * stats.t.sf(np.abs(t_stats), dof)).tolist() if include_stats else None
        f_statistic = None
        f_p_value = None
        if include_stats and design.shape[1] > 1 and y_array.size > design.shape[1]:
            ss_res = float(np.sum((y_array - predictions) ** 2))
            ss_tot = float(np.sum((y_array - np.mean(y_array)) ** 2))
            ss_reg = ss_tot - ss_res
            f_statistic = float((ss_reg / (design.shape[1] - 1)) / (ss_res / (y_array.size - design.shape[1]))) if ss_res > 0 else float("inf")
            f_p_value = float(stats.f.sf(f_statistic, design.shape[1] - 1, y_array.size - design.shape[1])) if np.isfinite(f_statistic) else 0.0
        latex_symbol_map = {name: sympy.Symbol(name) for name in names if name != "intercept"}
        latex_expr = sympy.Float(beta[0]) if fit_intercept else sympy.Integer(0)
        coefficient_slice = beta[1:] if fit_intercept else beta
        name_slice = names[1:] if fit_intercept else names
        for coefficient, name in zip(coefficient_slice, name_slice):
            latex_expr += sympy.Float(coefficient) * latex_symbol_map[name]
        response = {
            "result": f"R^2={r_squared:.6g}",
            "coefficients": beta.tolist(),
            "coefficient_names": names,
            "std_errors": std_errors.tolist() if include_stats else None,
            "p_values": p_values,
            "r_squared": float(r_squared),
            "adjusted_r_squared": float(adjusted_r_squared),
            "f_statistic": f_statistic,
            "f_p_value": f_p_value,
            "residual_std": float(np.sqrt(sigma2)),
            "predictions": None,
            "latex": safe_latex(latex_expr),
        }
    else:
        if not np.all(np.isin(y_array, [0.0, 1.0])):
            raise ValueError("logistic regression requires binary y values in {0, 1}")

        def neg_log_likelihood(beta: np.ndarray) -> float:
            linear_term = fitted_design @ beta
            probs = special.expit(linear_term)
            probs = np.clip(probs, 1e-12, 1 - 1e-12)
            return float(-np.sum(y_array * np.log(probs) + (1 - y_array) * np.log(1 - probs)))

        result = optimize.minimize(neg_log_likelihood, np.zeros(fitted_design.shape[1]), method="BFGS")
        if not result.success:
            raise ValueError("logistic regression failed; check for perfect separation or singular predictors")
        beta = result.x
        hessian_inv = np.asarray(result.hess_inv, dtype=float)
        std_errors = np.sqrt(np.diag(hessian_inv))
        if standardize and scales.size:
            beta = _rescale_standardized_coefficients(beta, means, scales, fit_intercept)
        linear_term = design @ beta
        predictions = special.expit(linear_term)
        ll_model = -neg_log_likelihood(result.x)
        baseline_prob = np.mean(y_array)
        ll_null = float(np.sum(y_array * np.log(np.clip(baseline_prob, 1e-12, 1 - 1e-12)) + (1 - y_array) * np.log(np.clip(1 - baseline_prob, 1e-12, 1 - 1e-12))))
        r_squared = 1 - ll_model / ll_null if ll_null != 0 else 1.0
        z_scores = np.divide(beta, std_errors, out=np.zeros_like(beta), where=std_errors > 0)
        p_values = (2 * stats.norm.sf(np.abs(z_scores))).tolist() if include_stats else None
        response = {
            "result": f"logistic model fitted, pseudo-R^2={r_squared:.6g}",
            "coefficients": beta.tolist(),
            "coefficient_names": names,
            "std_errors": std_errors.tolist() if include_stats else None,
            "p_values": p_values,
            "r_squared": float(r_squared),
            "adjusted_r_squared": None,
            "f_statistic": None,
            "f_p_value": None,
            "residual_std": None,
            "predictions": None,
            "latex": None,
        }
    if warning_messages:
        response["warning"] = "; ".join(warning_messages)
    return response


@tool_error_handler("bootstrap")
async def bootstrap(
    data: list[float],
    statistic: str = "mean",
    n_resamples: int = 10000,
    confidence_level: float = 0.95,
    method: str = "percentile",
    random_state: int | None = None,
) -> dict[str, Any]:
    """Estimate bootstrap confidence intervals and uncertainty."""
    cleaned, warning = _clean_numeric_data(data)
    if cleaned.size < 2:
        raise ValueError("bootstrap requires at least two observations")
    rng = np.random.default_rng(random_state)
    estimate = float(_bootstrap_statistic(cleaned, statistic))
    if method == "bca" and hasattr(stats, "bootstrap") and statistic != "custom_function":
        stat_map = {
            "mean": np.mean,
            "median": np.median,
            "std": lambda values, axis=-1: np.std(values, axis=axis, ddof=1),
            "var": lambda values, axis=-1: np.var(values, axis=axis, ddof=1),
        }
        result = stats.bootstrap(
            (cleaned,),
            stat_map[statistic],
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            method="BCa",
            random_state=rng,
        )
        standard_error = float(result.standard_error)
        bias = 0.0
        ci_lower = float(result.confidence_interval.low)
        ci_upper = float(result.confidence_interval.high)
    else:
        batch_size = 10000 if n_resamples > 100000 else n_resamples
        bootstrap_values: list[np.ndarray] = []
        remaining = n_resamples
        while remaining > 0:
            current_batch = min(batch_size, remaining)
            resamples = rng.choice(cleaned, size=(current_batch, cleaned.size), replace=True)
            if statistic == "mean":
                bootstrap_values.append(np.mean(resamples, axis=1))
            elif statistic == "median":
                bootstrap_values.append(np.median(resamples, axis=1))
            elif statistic == "std":
                bootstrap_values.append(np.std(resamples, axis=1, ddof=1))
            elif statistic == "var":
                bootstrap_values.append(np.var(resamples, axis=1, ddof=1))
            else:
                raise ValueError("custom_function is not supported through this API; use mean, median, std, or var")
            remaining -= current_batch
        distribution_values = np.concatenate(bootstrap_values)
        standard_error = float(np.std(distribution_values, ddof=1))
        bias = float(np.mean(distribution_values) - estimate)
        alpha_tail = (1 - confidence_level) / 2
        if method == "basic":
            lower = float(np.quantile(distribution_values, 1 - alpha_tail))
            upper = float(np.quantile(distribution_values, alpha_tail))
            ci_lower = float(2 * estimate - lower)
            ci_upper = float(2 * estimate - upper)
        else:
            ci_lower = float(np.quantile(distribution_values, alpha_tail))
            ci_upper = float(np.quantile(distribution_values, 1 - alpha_tail))
    response = {
        "result": f"{statistic} = {estimate:.6g} [{ci_lower:.6g}, {ci_upper:.6g}] ({confidence_level:.0%} CI)",
        "estimate": estimate,
        "standard_error": standard_error,
        "bias": bias,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "method": method,
        "n_resamples": n_resamples,
        "latex": None,
    }
    warnings_list = []
    if warning is not None:
        warnings_list.append(warning)
    if n_resamples < 1000:
        warnings_list.append("n_resamples is small; confidence intervals may be unstable")
    if cleaned.size < 5:
        warnings_list.append("sample size is very small; bootstrap results may be unreliable")
    if np.allclose(cleaned, cleaned[0]):
        warnings_list.append("data are constant; bootstrap variation is limited")
    if warnings_list:
        response["warning"] = "; ".join(warnings_list)
    return response


@tool_error_handler("random_sample")
async def random_sample(
    distribution: str = "uniform",
    params: dict[str, float] | None = None,
    size: int = 10,
    dimensions: int = 1,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Generate random samples from NumPy distributions."""
    if size < 1:
        raise ValueError("size must be positive")
    rng = np.random.default_rng(random_state)
    params = dict(params or {})
    name = distribution.lower()
    if name not in _RANDOM_ALIASES:
        raise ValueError(f"unsupported random distribution: {distribution}")

    if name == "uniform":
        samples = rng.uniform(params.get("low", 0.0), params.get("high", 1.0), size=size)
    elif name == "normal":
        samples = rng.normal(params.get("loc", 0.0), params.get("scale", 1.0), size=size)
    elif name == "exponential":
        samples = rng.exponential(params.get("scale", 1.0), size=size)
    elif name == "poisson":
        samples = rng.poisson(params.get("lam", params.get("mu", 1.0)), size=size)
    elif name == "binomial":
        samples = rng.binomial(int(params.get("n", 1)), params.get("p", 0.5), size=size)
    elif name == "beta":
        samples = rng.beta(params.get("a", 1.0), params.get("b", 1.0), size=size)
    elif name == "gamma":
        samples = rng.gamma(params.get("shape", params.get("k", 1.0)), params.get("scale", 1.0), size=size)
    elif name == "lognormal":
        samples = rng.lognormal(params.get("mean", 0.0), params.get("sigma", 1.0), size=size)
    elif name == "multivariate_normal":
        mean = np.asarray(params.get("mean", [0.0] * dimensions), dtype=float)
        cov = np.asarray(params.get("cov", np.eye(dimensions).tolist()), dtype=float)
        samples = rng.multivariate_normal(mean, cov, size=size)
    elif name == "dirichlet":
        alpha = np.asarray(params.get("alpha", [1.0] * dimensions), dtype=float)
        samples = rng.dirichlet(alpha, size=size)
    else:
        df = float(params.get("df", max(dimensions, 1)))
        scale = np.asarray(params.get("scale", np.eye(dimensions).tolist()), dtype=float)
        samples = stats.wishart(df=df, scale=scale).rvs(size=size, random_state=rng)

    sample_array = np.asarray(samples, dtype=float)
    flattened = sample_array.reshape(-1)
    response = {
        "result": f"Generated {size} {distribution} samples",
        "samples": flattened[:20].tolist(),
        "sample_stats": {
            "mean": float(np.mean(flattened)),
            "std": float(np.std(flattened, ddof=1 if flattened.size > 1 else 0)),
        },
        "latex": None,
    }
    warnings_list = []
    if size > 1_000_000:
        warnings_list.append("large sample request may consume significant memory")
    if warnings_list:
        response["warning"] = "; ".join(warnings_list)
    return response


def register(server: Any) -> None:
    """Register all statistics tools."""
    server.tool("describe")(describe)
    server.tool("distribution")(distribution)
    server.tool("hypothesis_test")(hypothesis_test)
    server.tool("regression")(regression)
    server.tool("bootstrap")(bootstrap)
    server.tool("random_sample")(random_sample)