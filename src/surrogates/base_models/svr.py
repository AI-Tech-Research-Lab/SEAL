from sklearn.svm import SVR
import numpy as np


class SVRModel:
    """Support Vector Regression model for surrogate prediction"""

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1e-6,
        epsilon: float = 1e-6,
        gamma: str = "auto",
        **kwargs,
    ) -> None:
        """
        Initialize the SVR model with default parameters.

        Args:
            kernel (str): Kernel type to be used in the algorithm. Default is 'rbf'.
            C (float): Regularization parameter. Default is 1.0.
            epsilon (float): Epsilon in the epsilon-SVR model. Default is 0.1.
            gamma (str): Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. Default is 'scale'.
            **kwargs: Additional keyword arguments to pass to SVR.

        Returns:
            None
        """
        self.name = "svr"
        self.model = SVR(
            kernel=kernel,
            C=C,
            epsilon=epsilon,
            gamma=gamma,
            **kwargs,
        )

    def fit(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Fit the SVR model to the input data.

        Args:
            inputs (np.ndarray): Input features for training.
            targets (np.ndarray): Target values for training.
        """
        self.model.fit(inputs, targets.ravel())

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained SVR model.

        Args:
            test_data (np.ndarray): Input features for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        return self.model.predict(test_data).reshape(-1, 1)
