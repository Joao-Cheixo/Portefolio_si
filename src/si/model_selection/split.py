from typing import Tuple

import numpy as np

from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    np.random.seed(random_state)
    # get dataset size
    n_samples = dataset.shape()[0]
    # get number of samples in the test set
    n_test = int(n_samples * test_size)
    # get the dataset permutations
    permutations = np.random.permutation(n_samples)
    # get samples in the test set
    test_idxs = permutations[:n_test]
    # get samples in the training set
    train_idxs = permutations[n_test:]
    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test


def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into stratified training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
   

    unique_labels, label_counts = np.unique(dataset.y, return_counts=True)
    train_indices = [] 
    test_indices = [] 

    np.random.seed(random_state) 
    for label, count in zip(unique_labels, label_counts): 

        num_test_samples = int(count * test_size)

        class_indices = np.where(dataset.y == label)[0]
        np.random.seed(random_state) 
        np.random.shuffle(class_indices) 
        test_indices.extend(class_indices[:num_test_samples]) 

        train_indices.extend(class_indices[num_test_samples:]) 

    train_dataset = Dataset(dataset.X[train_indices], dataset.y[train_indices], features=dataset.features, label=dataset.label)
    test_dataset = Dataset(dataset.X[test_indices], dataset.y[test_indices], features=dataset.features, label=dataset.label)

    return train_dataset, test_dataset #tuple



if __name__ == '__main__':
   
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    y = np.array([0, 1, 0, 1])
    features = np.array(['a', 'b', 'c'])
    label = 'target'
    dataset = Dataset(X, y, features, label)

    

    train, test = train_test_split(dataset, test_size=0.25, random_state=42)
    print("Train Set:")
    print(train.X)
    print(train.y)
    print("Test Set:")
    print(test.X)
    print(test.y)

    

    strat_train, strat_test = stratified_train_test_split(dataset, test_size=0.25, random_state=42)
    print("Stratified Train Set:")
    print(strat_train.X)
    print(strat_train.y)
    print("Stratified Test Set:")
    print(strat_test.X)
    print(strat_test.y)