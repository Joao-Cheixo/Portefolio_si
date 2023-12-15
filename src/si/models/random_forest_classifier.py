import numpy as np


from typing import Literal
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.model_selection.split import train_test_split

class RandomForestClassifier:
    """
    Ensemble machine learning technique that combines multiple decision trees to improve prediction accuracy 
    and reduce overfitting
    """
   
    def __init__(self, n_estimators:int = 100, 
                 max_features:int = None,
                 min_sample_split:int = 2,
                 max_depth:int = 15,
                 mode: Literal['gini','entropy'] = 'gini',
                 seed:int = None):
        """
        Random Forest is an ensemble machine learning technique that combines multiple decision trees to improve prediction 
        accuracy and reduce overfitting.
    
        parameters:
        -----------
        n_estimators: 
            number of decision trees to use
        
        max_features:
            maximum number of features to use per tree
        
        min_sample_split:
            minimum samples allowed in a split

        max_depth:
            maximum depth of the trees

        mode:
            impurity calculation mode (gini or entropy)

        seed:
            random seed to use to assure reproducibility

        estimated parameters:
        ---------------------
        trees: list
            the trees of the random forest and respective features used for training (initialized as an empty list)
        """
        
        self.n_estimators = n_estimators 
        self.max_features = max_features 
        self.min_sample_split = min_sample_split 
        self.max_depth = max_depth 
        self.mode = mode 
        self.seed = seed 

        
        self.trees = [] 
        self.training = {} 

    def set_random_seed(self):
        """
        Set a random seed for NumPy

        Returns
        -------
        A random value
        """
        if self.seed is not None:

            np.random.seed(self.seed) 


    def fit(self, dataset:Dataset)->'RandomForestClassifier':
        """
        Train the decision trees of the random forest

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: RandomForest
            The fitted model
        """
        self.set_random_seed() 
            
        n_samples, n_features = dataset.shape() 
        if self.max_features is None: 
            self.max_features = int(np.sqrt(n_features)) 

       
        for x in range(self.n_estimators): 

            bootstrap_samples = np.random.choice(n_samples, n_samples, replace = True) 

            bootstrap_features = np.random.choice(n_features, self.max_features, replace=False) 

            random_dataset = Dataset(dataset.X[bootstrap_samples][:,bootstrap_features], dataset.y[bootstrap_samples])

            tree = DecisionTreeClassifier(min_sample_split=self.min_sample_split, max_depth=self.max_depth, mode = self.mode)

            tree.fit(random_dataset) 

            self.trees.append((bootstrap_features, tree)) 

        return self
    

    def predict(self, dataset:Dataset)-> np.ndarray:
        """
        Predicts the class labels for a dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset for which to make predictions.

        Returns
        -------
        np.ndarray
            An array of predicted class labels.
        """
        n_samples = dataset.shape()[0] 
        predictions = np.zeros((self.n_estimators, n_samples), dtype=object) 
        
        for tree_idx, (feature_idx, tree) in enumerate(self.trees): 

            data_samples = Dataset(dataset.X[:, feature_idx], dataset.y) 
            tree_preds = tree.predict(data_samples)  

            predictions[tree_idx, :] = tree_preds  

        return predictions 


    def score(self, dataset: Dataset) -> float:
        """
        Computes the accuracy between predicted and real labels

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the RandomForest on

        Returns
        -------
        random_forest: float
            The Mean Square Error of the model
        """
        #avaliar o modelo:
        predictions = self.predict(dataset) 

        return accuracy(dataset.y, predictions) 



if __name__ == '__main__':
    

    np.random.seed(42)
    X = np.random.rand(100, 3)
    theta_true = np.array([3, 1.5, -2])
    noise = 0.1 * np.random.randn(100)
    y = X.dot(theta_true) + noise

    dataset = Dataset(X, y, features=['Feature1', 'Feature2', 'Feature3'], label='Target')

    random_forest = RandomForestClassifier(n_estimators=100, max_features=None, min_sample_split=2, max_depth=15, mode='gini', seed=None)
    random_forest.fit(dataset)

    print("Trees:", len(random_forest.trees))

    predictions = random_forest.predict(dataset)
    print("Predictions for the first 5 samples:", predictions[:, :5])

    accuracy_score = random_forest.score(dataset)
    print("Accuracy:", accuracy_score)



    from si.io.csv_file import read_csv
    from sklearn.ensemble import RandomForestClassifier as RFC


    data = read_csv('/home/joao/Portefolio_si/datasets/iris/iris.csv', sep=',', features=True, label=True)
    train, test = train_test_split(data, test_size=0.33, random_state=42)
    model = RandomForestClassifier(min_sample_split=3, max_depth=3, mode='gini', n_estimators=5)

    model = RFC(n_estimators=5, max_depth=3, min_samples_split=3)
    model.fit(train.X, train.y)
    print(model.score(test.X, test.y))