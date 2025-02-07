The evaluation of our models is done on a Dataset-wise primitive. We want to check results for a specific dataset under an specific continual setting.

The code is organised as follows:

### 1. Base classes

The `ContinualDataset` class is an abstract class that defines the interface for a continual dataset. It forces the implementation of the logic to **LOAD** and **SPLIT** the dataset into **CONTINUAL** subsets.

From this base class, a few extra continual problems are defined:

- `DataContinualDataset` sets up the problem of `Data Continual Learning`. Mimics the idea of training the network on splits of the dataset one at the time. Each split is seen as a different task.

- `ClassContinualDataset` sets up the problem of `Class Incremental Learning`. The idea is that the network is trained to recognise a set of classes, then another set of classes and so on. Each set of classes is seen as a different task.

We are interested in the problem were no memory of previous tasks is allowed. In this way we measure both the **"stability" of the model** ( persisting old knowledge) and the **"plasticity"** ( ability to learn new knowledge).
