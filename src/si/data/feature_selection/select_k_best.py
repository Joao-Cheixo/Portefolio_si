import numpy as np

from src.dataset import Dataset
from statistics.f_classification import f_classification


class SelectKBest:
    def __init__(self, score_func, k):
        """
        Initializes the select k best.
        :param score_func: Function that calculates the score for each feature.
        :param k: Number of features to select.
        """
        self.score_func = score_func
        self.k = k
        self.f_value = None
        self.p_value = None

    def fit(self, dataset: Dataset):
        """
        Calculates the score for each feature.
        :param dataset: Dataset object.
        :return: SelectKBest object.
        """
        self.f_value, self.p_value = self.score_func(dataset)

        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Selects the k best features.
        :param dataset: Dataset object.
        :return: Dataset object.
        """
        if self.f_value is None:
            raise Exception("You must fit the select k best before transform the dataset.")

        indexes = np.argsort(self.f_value)[-self.k:]  # get the indexes of the k best features, as the sorting is
        # in ascending order, we get the last k indexes
        best_features = dataset.x[:, indexes]
        best_features_names = [dataset.features_names[i] for i in indexes]

        return Dataset(best_features, dataset.y, best_features_names, dataset.label_name)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Calculates the score for each feature and selects the k best features.
        :param dataset: Dataset object.
        :return: Dataset object.
        """
        self.fit(dataset)
        return self.transform(dataset)