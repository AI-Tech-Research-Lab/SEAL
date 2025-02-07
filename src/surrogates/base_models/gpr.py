from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    RationalQuadratic,
    ExpSineSquared,
    DotProduct,
)
import numpy as np


class GPR:
    """Gaussian Process Regression model for surrogate prediction"""

    def __init__(
        self,
        kernel="rbf",
        alpha=1,
        normalize_y=False,
        n_restarts_optimizer=10,
    ):
        """
        Initialize the Gaussian Process Regression model.

        Args:
            kernel (str): The kernel type to use. Options: 'rbf', 'matern', 'rational_quadratic', 'exp_sine_squared', 'dot_product'.
            alpha (float): Value added to the diagonal of the kernel matrix during fitting.
            normalize_y (bool): Whether to normalize the target values.
            n_restarts_optimizer (int): The number of restarts of the optimizer for finding the kernel's parameters.
            random_state (int): Random state for reproducibility.
        """
        self.kernel_type = kernel
        self.kernel = self._get_kernel(kernel)
        self.model = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=alpha,
            normalize_y=normalize_y,
            n_restarts_optimizer=n_restarts_optimizer,
        )
        self.name = "gpr"

    def _get_kernel(self, kernel_type):
        if kernel_type == "rbf":
            return RBF()
        elif kernel_type == "matern":
            return Matern()
        elif kernel_type == "rational_quadratic":
            return RationalQuadratic()
        elif kernel_type == "exp_sine_squared":
            return ExpSineSquared()
        elif kernel_type == "dot_product":
            return DotProduct()
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

    def fit(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Fit the Gaussian Process Regression model to the input data.

        Args:
            inputs (np.ndarray): Input features for training.
            targets (np.ndarray): Target values for training.
        """
        self.model.fit(inputs, targets.ravel())

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained Gaussian Process Regression model.

        Args:
            test_data (np.ndarray): Input features for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        return self.model.predict(test_data).reshape(-1, 1)
