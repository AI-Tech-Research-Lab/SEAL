# When to expand

This study seeks to understand the best moment to expand the neural networks. At which tasks of the continual learning session is better to expand the model.

We do this by studying the `capacity` of the model to acquire the new tasks and we control the behaviour by using a `capacity_tau` parameter.

The experiments are run for different values of `capacity_tau` and we compare the results in terms of `average_accuracy` and `average_forgetting`.
