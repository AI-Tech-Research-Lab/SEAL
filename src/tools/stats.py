import scipy.stats as stats
import numpy as np


def get_correlation(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[float, float, float]:
    """
    Calculate correlation metrics between prediction and target arrays.

    This function computes the Root Mean Square Error (RMSE), Spearman's rank correlation
    coefficient (rho), and Kendall's tau between the prediction and target arrays.

    ### Args:
        `prediction (np.ndarray)`: Array of predicted values.
        `target (np.ndarray)`: Array of true target values.

    ### Returns:
        `tuple[float, float, float]`: A tuple containing:
            - rmse (float): Root Mean Square Error.
            - rho (float): Spearman's rank correlation coefficient.
            - tau (float): Kendall's tau.

    Note:
        The input arrays should have the same shape and be non-empty.
    """
    rmse = np.sqrt(((prediction - target) ** 2).mean())
    rho, _ = stats.spearmanr(prediction, target)
    tau, _ = stats.kendalltau(prediction, target)

    return rmse, rho, tau
