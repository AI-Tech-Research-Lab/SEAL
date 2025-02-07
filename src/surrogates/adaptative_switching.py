from surrogates.base_models import get_base_predictor
from tools.stats import get_correlation

import numpy as np


class AdaptiveSwitching:
    """
    This class implements an adaptive switching that works as an ensemble of
    smaller surrogate models.

    It performs cross validation to select the best model to use in the final
    prediction.
    """

    def __init__(
        self,
        model_pool=["random_forest", "carts", "neighbors", "mlp", "svr", "gpr"],
        n_fold=10,
    ):
        self.name = "adaptive switching"

        # Provide all the models on which to perform the selection
        self.model_pool = model_pool
        self.n_fold = n_fold

        self.model = None

    def fit(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Selects a best predictor with cross-validation based on a cross validation
        to select the best model according to the Kendall's tau metric.


        ### Args:
            `inputs (np.ndarray)`: Input data for training the surrogate model.
            `targets (np.ndarray)`: Target values for training the surrogate model.

        """

        # Sanity check for the predict
        test_msg = "# of training samples have to be > # of dimensions"
        assert len(inputs) > len(inputs[0]), test_msg

        # Select the best predictor
        _best_model = self._n_fold_validation(inputs, targets, n=self.n_fold)

        # Store the best model to perform predictions
        self.model = get_base_predictor(_best_model, inputs, targets)

    def _n_fold_validation(
        self,
        train_data: np.ndarray,
        train_target: np.ndarray,
        n: int,
    ) -> str:
        """
        Performs a cross validation to select the best model according to the
        Kendall's tau metric.

        ### Args:
            `train_data (np.ndarray)`: Input data for training the surrogate model.
            `train_target (np.ndarray)`: Target values for training the surrogate model.
            `n (int)`: Number of folds for the cross validation.

        ### Returns:
            `str`: Name of the best model.
        """

        n_samples = len(train_data)
        perm = np.random.permutation(n_samples)

        # Keep track of the metrics for each model
        models_rmse = np.full((n, len(self.model_pool)), np.nan)
        models_rho = np.full((n, len(self.model_pool)), np.nan)
        models_tau = np.full((n, len(self.model_pool)), np.nan)

        for i, tst_split in enumerate(np.array_split(perm, n)):
            trn_split = np.setdiff1d(perm, tst_split, assume_unique=True)

            # Loop over all considered surrogate model in pool
            for j, model in enumerate(self.model_pool):
                acc_predictor = get_base_predictor(
                    model,
                    train_data[trn_split],
                    train_target[trn_split],
                )

                # Compute the correlation metrics
                rmse, rho, tau = get_correlation(
                    acc_predictor.predict(train_data[tst_split]),
                    train_target[tst_split],
                )

                # Store the metrics
                models_rmse[i, j] = rmse
                models_rho[i, j] = rho
                models_tau[i, j] = tau

        # Select the model with the highest stable mean Kendall's tau
        _tau_metric = np.mean(models_tau, axis=0) - np.std(models_tau, axis=0)
        winner_idx = int(np.argmax(_tau_metric))
        best_model = self.model_pool[winner_idx]

        print(
            f"\t * Best model: {best_model} | "
            f"tau = {np.mean(models_tau, axis=0)[winner_idx].max()}"
        )

        return best_model

    def predict(self, test_data):
        """
        Uses the previously selected best model to perform predictions.
        """
        return self.model.predict(test_data)
