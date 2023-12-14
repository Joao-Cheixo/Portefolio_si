
import numpy as np
from typing import Callable
from si.metrics.rmse import rmse
from si.data.dataset import Dataset
from si.statistics.euclidean_distance import euclidean_distance
from si.model_selection.split import train_test_split


class KNNRegressor:
    """
    KNN Classifier
    The k-Nearst Neighbors classifier is a machine learning model that estimates the average value of the k most similar 
    examples instead of the most common class.

    Parameters
    ----------
    k: int
        The number of nearest neighbors to use
    distance: Callable
        The distance function to use

    Attributes
    ----------
    dataset: np.ndarray
        The training data
    """
    def __init__(self, k: int = 1, distance: Callable = euclidean_distance):
        #definição do construtor:
        """
        Initialize the KNN Regressor

        Parameters
        ----------
        k: int
            The number of nearest neighbors to use
        distance: Callable
            The distance function to use
        """
        # parameters
        self.k = k
        self.distance = distance

        # attributes
        self.dataset = None #dataset inicializado como None

    def fit(self, dataset: Dataset) -> 'KNNRegressor':
        #função fit recebe como parâmetro um objeto Dataset para ajustar o modelo
        """
        It fits the model to the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: KNNRegressor
            The fitted model
        """
        self.dataset = dataset #atribui o dataset fornecido ao atributo self.dataset da classe KNNRegressor
        return self #devolve o self após ajustar os dados
    
    def _get_closest_label(self, x: np.ndarray):
        """
        Calculates the mean of the class with the highest frequency.

        Parameters 
            x: Array of samples.

        Returns: 
            Indexes of the classes with the highest frequency
        """
        distances = self.distance(x, self.dataset.X)
        
        knn = np.argsort(distances)[:self.k] 
        
        knn_labels = self.dataset.y[knn]
        
        match_class_mean = np.mean(knn_labels) 

        return match_class_mean 
    
    
    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        It predicts the classes of the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the classes of (test dataset)

        Returns
        -------
        predictions: np.ndarray
            An array of predicted values for the testing dataset (Y_pred)
        """
        
        return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X) #devolve um array NumPy contendo os 
        #valores previstos para o dataset de teste (Y_pred)

    def score(self, dataset: Dataset) -> float:
        """
        Returns the accuracy of the model.
        :return: Accuracy of the model.
        """
        predictions = self.predict(dataset) 

        return rmse(dataset.y, predictions) 


if __name__ == '__main__':
    num_samples = 600
    num_features = 100

    X = np.random.rand(num_samples, num_features)
    y = np.random.rand(num_samples) 

    dataset_ = Dataset(X=X, y=y)

    #features and class name 
    dataset_.features = ["feature_" + str(i) for i in range(num_features)]
    dataset_.label = "target"

    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # regressor KNN
    knn_regressor = KNNRegressor(k=5)  

    # fit the model to the train dataset
    knn_regressor.fit(dataset_train)

    # evaluate the model on the test dataset
    score = knn_regressor.score(dataset_test)
    print(f'The rmse of the model is: {score}')