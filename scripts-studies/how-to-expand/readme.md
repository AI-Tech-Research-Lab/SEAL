# How to expand

This ablation study seeks to understand what is the **best way to expand the neural networks**. In other words, these experiments try to understand some possible hyperparameters for the **growing policy**.

The following is a list of the experiments:

### 1. Initialisation from Random vs OFA

Here we want to understand the best initialisation for the expanded weights. As the networks grows, new weights are introduced to the model, these weights will introduce priors from their distribution.

The initial hypothesis is that OFA weights will work better than random weights. To prove this we have randomly sample architecture and evaluated them under three different settings:

1. Expansion from random.
2. Expansion from OFA.
3. Expansion from random + distillation loss.
4. Expansion from OFA + distillation loss.

Here we are interested in both the `average_accuracy` and the `average_forgetting` of the methods.

### 2. Avoiding forgetting

Expanding the network and adding new weights interactions will, in most cases, mean an increase in the forgetting metrics. To avoid this, while retaining plasticity, we want to check strategies to retain old knowledge:

1. Not freezing + no distillation.
2. Freezing old weights + no distillation.
3. Freezing old weights + distillation.
4. Not freezing + distillation.

Here the main metric we want to see is the `average_forgetting`.
