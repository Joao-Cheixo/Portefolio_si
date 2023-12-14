from typing import Tuple, Sequence, Union

import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, X: np.ndarray, y: np.ndarray = None, features: Sequence[str] = None, label: str = None) -> None:
        """
        Dataset represents a tabular dataset for single output classification.

        Parameters
        ----------
        X: numpy.ndarray (n_samples, n_features)
            The feature matrix
        y: numpy.ndarray (n_samples, 1)
            The label vector
        features: list of str (n_features)
            The feature names
        label: str (1)
            The label name
        """
        if X is None:
            raise ValueError("X cannot be None")
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if features is not None and len(X[0]) != len(features):
            raise ValueError("Number of features must match the number of columns in X")
        if features is None:
            features = [f"feat_{str(i)}" for i in range(X.shape[1])]
        if y is not None and label is None:
            label = "y"
        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def shape(self) -> Tuple[int, int]:
        """
        Returns the shape of the dataset
        Returns
        -------
        tuple (n_samples, n_features)
        """
        return self.X.shape

    def has_label(self) -> bool:
        """
        Returns True if the dataset has a label
        Returns
        -------
        bool
        """
        return self.y is not None

    def get_classes(self) -> np.ndarray:
        """
        Returns the unique classes in the dataset
        Returns
        -------
        numpy.ndarray (n_classes)
        """
        if self.has_label():
            return np.unique(self.y)
        else:
            raise ValueError("Dataset does not have a label")

    def get_mean(self) -> np.ndarray:
        """
        Returns the mean of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmean(self.X, axis=0)

    def get_variance(self) -> np.ndarray:
        """
        Returns the variance of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanvar(self.X, axis=0)

    def get_median(self) -> np.ndarray:
        """
        Returns the median of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmedian(self.X, axis=0)

    def get_min(self) -> np.ndarray:
        """
        Returns the minimum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmin(self.X, axis=0)

    def get_max(self) -> np.ndarray:
        """
        Returns the maximum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmax(self.X, axis=0)

    def summary(self) -> pd.DataFrame:
        """
        Returns a summary of the dataset
        Returns
        -------
        pandas.DataFrame (n_features, 5)
        """
        data = {
            "mean": self.get_mean(),
            "median": self.get_median(),
            "min": self.get_min(),
            "max": self.get_max(),
            "var": self.get_variance()
        }
        return pd.DataFrame.from_dict(data, orient="index", columns=self.features)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, label: str = None):
        """
        Creates a Dataset object from a pandas DataFrame

        Parameters
        ----------
        df: pandas.DataFrame
            The DataFrame
        label: str
            The label name

        Returns
        -------
        Dataset
        """
        if label:
            X = df.drop(label, axis=1).to_numpy()
            y = df[label].to_numpy()
        else:
            X = df.to_numpy()
            y = None

        features = df.columns.tolist()
        return cls(X, y, features=features, label=label)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the dataset to a pandas DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        if self.y is None:
            return pd.DataFrame(self.X, columns=self.features)
        else:
            df = pd.DataFrame(self.X, columns=self.features)
            df[self.label] = self.y
            return df








    #1

    def dropna(self):
        """
        Remove samples containing at least one null value (NaN) and update the y vector accordingly.

        Returns
        -------
        self: Modified Dataset object
        """
        # Find indices of rows with NaN values in the feature matrix X
        nan_indices = np.isnan(self.X).any(axis=1)
        
        # Remove rows with NaN values from the feature matrix X
        self.X = self.X[~nan_indices]
        
        # Update the label vector y by removing corresponding entries
        if self.y is not None:
            self.y = self.y[~nan_indices]
        
        return self
    








    2#

    def fillna(self, values: list[float]) -> 'Dataset':
        """
        Replaces all null values with another value or the mean or median of the feature/variable
        
        Parameters:
        -----------
        values: list of medians or means to use as replacements for NaN values
        
        Return:
        Modified Dataset 
        """
        #verifica se todos os valores em values são do tipo int ou float. Se algum valor não for, cria um ValueError
        if not all(isinstance(value, (int, float)) for value in values): 
            raise ValueError("value could be only float")
        num_columns = self.X.shape[1]
        #verifica se o número de valores em values é pelo menos igual ao número de colunas em self.X, se não for cria um ValueError
        if len(values) < num_columns:
            raise ValueError("values have at least as many values as columns in X")
        #verifica se os valores em values correspondem à média ou à mediana das features atuais do conjunto de dados. 
        #Se não corresponderem, cria um ValueError.
        if not np.array_equal(values, self.get_mean()) and not np.array_equal(values, self.get_median()):
            raise ValueError("values are the array of means or medians of the variables")

        for cols in range(num_columns): 
            col_values = self.X[:, cols] #cria a variável col_values com todas as linhas e a coluna cols de X
            nan_v = np.isnan(col_values) #cria a nan_v onde verifica quais valores são NaN na coluna atual

            if np.any(nan_v): 
                replace_value = values[cols] #substitui esses valores por valores especificados em values para essa coluna
                col_values[nan_v] = replace_value
                self.X[:, cols] = col_values #atualiza a coluna no conjunto de dados com os valores substituídos.

        return self 
    





    3#

    
    def remove_by_index(self, index):
        """
        Remove a sample by its index and update the y vector accordingly.

        Parameters
        ----------
        index : int
            The index of the sample to remove.

        Returns
        -------
        self: Modified Dataset object
        """
        # Check if the provided index is valid
        if index < 0 or index >= len(self.X):
            raise ValueError("Invalid index provided for sample removal")
        
        # Remove the sample at the specified index from the feature matrix X
        self.X = np.delete(self.X, index, axis=0)
        
        # Update the label vector y by removing the corresponding entry
        if self.y is not None:
            self.y = np.delete(self.y, index, axis=0)
        
        return self








    @classmethod
    def from_random(cls,
                    n_samples: int,
                    n_features: int,
                    n_classes: int = 2,
                    features: Sequence[str] = None,
                    label: str = None):
        """
        Creates a Dataset object from random data

        Parameters
        ----------
        n_samples: int
            The number of samples
        n_features: int
            The number of features
        n_classes: int
            The number of classes
        features: list of str
            The feature names
        label: str
            The label name

        Returns
        -------
        Dataset
        """
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        return cls(X, y, features=features, label=label)


if __name__ == '__main__':
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([1, 2])
    features = np.array(['a', 'b', 'c'])
    label = 'y'
    dataset = Dataset(X, y, features, label)
    print(dataset.shape())
    print(dataset.has_label())
    print(dataset.get_classes())
    print(dataset.get_mean())
    print(dataset.get_variance())
    print(dataset.get_median())
    print(dataset.get_min())
    print(dataset.get_max())
    print(dataset.summary())