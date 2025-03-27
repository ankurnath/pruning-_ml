# H\_DeepPruner

## Setup

To set up the environment, use the provided environment.yml file. This will create a Conda environment named `H_DeepPruner`:

```bash
conda env create -f environment.yml
conda activate H_DeepPruner
```

## Data Preparation

### Downloading Datasets

To download datasets, run the following command, replacing `Facebook` and `Wiki` with the desired dataset names:

```bash
python data_processing/data_download.py --datasets Facebook Wiki
```

### Splitting Datasets

To split a dataset into training and test sets, specify the dataset name and the desired training ratio (e.g., 30% training data):

```bash
python data_processing/train_test_split.py --dataset Facebook --ratio 0.3
```

## Running Experiments

### MaxCover

#### Training

- Train the GNN:
  ```bash
  python gnnpruner_train.py --problem MaxCover
  ```
- Train the MCTS:
  ```bash
  python main.py --problem MaxCover
  ```

#### Testing

- Run the trained model on a specific dataset:
  ```bash
  python main.py --problem MaxCover --dataset Facebook
  ```

### MaxCut

#### Training

- Train the GNN:
  ```bash
  python gnnpruner_train.py --problem MaxCut
  ```
- Train the MCTS:
  ```bash
  python main.py --problem MaxCut
  ```

#### Testing

- Run the trained model on a specific dataset:
  ```bash
  python main.py --problem MaxCut --dataset Facebook
  ```

### Influence Maximization (IM)

#### Training

- Train the GNN:
  ```bash
  python gnnpruner_train.py --problem IM
  ```
- Train the MCTS:
  ```bash
  python main.py --problem IM
  ```

#### Testing

- Run the trained model on a specific dataset:
  ```bash
  python main.py --problem IM --dataset Facebook
  ```

