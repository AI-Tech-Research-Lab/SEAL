from sklearn.ensemble import RandomForestRegressor
import numpy as np


class RandomForestModel:
    """Random Forest Regressor model for surrogate prediction"""

    def __init__(
        self,
        n_estimators=100,
        max_depth=5,
        criterion="poisson",
        **kwargs,
    ):
        """
        Initialize the Random Forest model.

        Args:
            n_estimators (int): The number of trees in the forest.
            max_depth (int): The maximum depth of the tree.
            random_state (int): Random state for reproducibility.
            **kwargs: Additional keyword arguments to pass to RandomForestRegressor.
        """
        self.name = "random_forest"
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            criterion=criterion,
            **kwargs,
        )

    def fit(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Fit the Random Forest model to the input data.

        Args:
            inputs (np.ndarray): Input features for training.
            targets (np.ndarray): Target values for training.
        """
        self.model.fit(inputs, targets.ravel())

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained Random Forest model.

        Args:
            test_data (np.ndarray): Input features for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        return self.model.predict(test_data).reshape(-1, 1)
