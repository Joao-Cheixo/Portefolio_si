
import numpy as np
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification
from typing import Callable

class SelectPercentile:
    """
    SelectPercentile selects the k best features according to a scoring function and a given percentile.

    Parameters
    ----------
    scorefunc: Callable
        The scoring function
    percentile: float
        The percentile of features to select

    """


    def __init__(self, score_func: Callable = f_classification, percentile: int = 50) -> None:
        """
        Initializes a SelectPercentile instance with the given score function and percentile.

        Args:
            scorefunc (Callable): The score function to use for feature selection. Defaults to f_classification.
            percentile (float): The percentile of features to select. Defaults to 0.5.
        """
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None


    def fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        Fits the SelectPercentile model to the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to
        
        Returns
        -------
        self : object
        """
        self.F, self.p = self.score_func(dataset)
        return self


    def transform(self, dataset: Dataset) -> Dataset:
        """
        Transforms the dataset by removing features with low variance

        Parameters
        ----------
        dataset: Dataset
            The dataset to transform
        
        Returns
        -------
        Dataset
        """
        n_feat = int(len(dataset.features) * self.percentile)
        idxs = np.argsort(self.F)[-n_feat:]
        percentile = dataset.X[:, idxs]
        percentile_name = [dataset.features[idx] for idx in idxs]
    
    
        return Dataset(percentile, dataset.y, percentile_name, dataset.label)
    


    
    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Fits the SelectPercentile model to the dataset and transforms it

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit and transform
        
        Returns
        -------
        Dataset
        """
        self.fit(dataset)
        return self.transform(dataset)








if __name__ == '__main__':
    dataset = Dataset(X=np.array([[0, 2, 4, 1, 5],
                                  [0, 1, 4, 0, 5 ],
                                  [0, 1, 4, 1, 5]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4", 'f5'],
                      label="y")

    selector = SelectPercentile()
    selector = selector.fit(dataset)
    dataset = selector.transform(dataset)
    print(dataset.features)
