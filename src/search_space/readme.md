This folder contains the definition of the SEARCH SPACE to use in the Neural Architecture Search.

The search space on this project will always come from a `OFA` (Once for All) supernet. Here there is an explanation for each of the given files:

### 1. `base_space.py`

Defines the basic ofa search space depending on the family of OFA models we are pasing. It defines how the architectures are encoded and decoded, it defines the sampling strategy and some extra utilities.
