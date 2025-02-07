# NAS executions

There are 2 types of NAS problem we can run. `SingleObjectiveNAS` and `MultiObjectiveNAS`, depending on whether we want to run the search using one or two objectives.

Additionally, as a baseline, I also added some *fixed* architecture searchs. The idea is to compare how the full continual learning problem without expansion direction compares with the growing one.

### 1. Single objective NAS

To run the `single_objective_nas` you can use the following command. Please note that you have to be positioned in the `root` of the repository.

```bash
python scripts-nas/00_single_objective_nas.py --dataset ${DATASET_NAME} --random_seed ${RANDOM_SEED} [Optional: --fixed ${true | false} ] 
```

The results will be logged in the `nas-results` directory.

### 2. Multi objective NAS

To run the `multi_objective_nas` you can use the following command. Please note that you have to be positioned in the `root` of the repository.

```bash
python scripts-nas/01_multi_objective_nas.py --dataset ${DATASET_NAME} --random_seed ${RANDOM_SEED} [Optional: --fixed ${true | false} ] 
```
