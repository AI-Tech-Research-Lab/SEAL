from continual_learning.continual_problems import DataContinualTrainer

from torch import nn


class NaiveBaselineTrainer(DataContinualTrainer):
    """
    Represent the base problem where the model is never expanded.
    Used as a baseline to compare the effectiveness of the expansions within the NAS.
    """

    def get_task_model(
        self,
        task_id: str,
        base_model: nn.Module,
        training: bool = True,
    ) -> nn.Module:
        """
        All the tasks are evaluated with the same model.
        NOTE: the base model is RE-TRAINED with each new task it sees.
        """
        return base_model
