import numpy as np

from si.data.dataset import Dataset


class VarianceThreshold:
    def __init__(self, threshold: float = 0.0):
        """
        Initializes the variance threshold.
        :param threshold: Threshold for the variance.
        """
        self.threshold = threshold
        self.variance = None  # None because we don't have a dataset yet

    def fit(self, dataset: Dataset):
        """
        Calculates the variance of each feature.
        :param dataset: Dataset object.
        """

        # variance = dataset.get_variance()
        # self.variance = variance
        self.variance = np.var(dataset.x, axis=0)

        return self

    def transform(self, dataset: Dataset):
        """
        Selects the features that have a variance greater than the threshold.
        :param dataset: Dataset object.
        :return: Dataset object.
        """
        if self.variance is None:
            raise Exception("You must fit the variance threshold before transform the dataset.")

        selected_features = np.where(self.variance > self.threshold)[0] # [0] because np.where returns a tuple
        selected_features_names = [dataset.features_names[i] for i in selected_features]
        selected_features_data = dataset.x[:, selected_features]

        return Dataset(selected_features_data, dataset.y, selected_features_names, dataset.label_name)

    def fit_transform(self, dataset: Dataset):
        """
        Calculates the variance of each feature and selects the features that have a variance
        greater than the threshold.
        :param dataset: Dataset object.
        :return: Dataset object.
        """
        self.fit(dataset)
        return self.transform(dataset)