from sklearn.neighbors import KNeighborsRegressor
import numpy as np


class Neighbors:
    """
    KNeighborsRegressor

    Attributes:
        radius (float): Range of parameter space to use for radius_neighbors queries
        weights (str): Weight function used in prediction ('uniform' or 'distance')
        name (str): Name identifier for the model
        model (KNeighborsRegressor): The underlying sklearn model
    """

    def __init__(self, n_neighbors=5, weights="distance"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.name = "neighbors"
        self.model = None

    def fit(self, train_data: np.ndarray, train_label: np.ndarray) -> None:
        """Fit the RBF model to training data.

        Args:
            train_data (np.ndarray): Training input samples of shape (n_samples, n_features)
            train_label (np.ndarray): Target values of shape (n_samples,)
        """
        self.model = KNeighborsRegressor(
            n_neighbors=5,
            weights=self.weights,
            algorithm="auto",
            p=2,  # Euclidean distance
            metric="minkowski",
        )
        self.model.fit(train_data, train_label)

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        """Predict using the RBF model.

        Args:
            test_data (np.ndarray): Test samples of shape (n_samples, n_features)

        Returns:
            np.ndarray: Predicted values of shape (n_samples,)

        Raises:
            AssertionError: If model hasn't been fitted yet
        """
        assert (
            self.model is not None
        ), "RBF model does not exist, call fit to obtain rbf model first"
        return self.model.predict(test_data).reshape(-1, 1)
