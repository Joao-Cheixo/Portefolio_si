import numpy as np
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.sigmoid_function import sigmoid_function

import numpy as np 


class StackingClassifier:




    def __init__(self, models: list, final_model):
        """
        Initialize the StackingClassifier with a list of models and a final model.
        :param models: List of models to be stacked.
        :param final_model: Final model to make predictions.
        """
        # parameters
        self.models = models
        self.final_model = final_model
    






    def fit(self, dataset: Dataset) -> 'StackingClassifier':
        """
        Fit the models according to the training data.

        Parameters
        ----------
        dataset : Dataset
            The training data.

        Returns
        -------
        self : StackingClassifier
            The fitted model.
        """
        for model in self.models:
            model.fit(dataset) 
        
        predictions=[]
        for model in self.models:
            prever=model.predict(dataset)
            predictions.append(prever)
        
        predictions=np.array(predictions).T 
        self.final_model.fit(Dataset(dataset.X, predictions))
        return self






    def predict(self, dataset: Dataset) -> np.array:
        """
        Computes the prevision of all the models and returns the final model prediction.
        :param dataset: Dataset object to predict the labels of.
        :return: the final model prediction
        """
        predictions = []
        for model in self.models:
            prever=model.predict(dataset)
            predictions.append(prever)
        
        predictions=np.array(predictions).T
        y_pred_final=self.final_model.predict(Dataset(dataset.X, predictions))                
        return y_pred_final
    




    def score(self, dataset: Dataset) -> float:
        """
        Calculates the accuracy of the model.
        :return: Accuracy of the model.
        """
        y_pred = self.predict(dataset)
        score = accuracy(dataset.y, y_pred)

        return score
    





if __name__ == '__main__':
    from si.io.csv_file import read_csv
    from si.model_selection.split import stratified_train_test_split
    from si.models.knn_classifier import KNNClassifier
    from si.models.logistic_regression import LogisticRegression
    from si.models.decision_tree_classifier import DecisionTreeClassifier

    data = "/home/joao/Portefolio_si/datasets/breast_bin/breast-bin.csv"  
    breast=read_csv(data, sep=",",features=True,label=True)
    train_data, test_data = stratified_train_test_split(breast, test_size=0.20, random_state=42)


    knn = KNNClassifier(k=3)
    
    
    LG=LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)

    
    DT=DecisionTreeClassifier(min_sample_split=3, max_depth=3, mode='gini')

    # Final Model (Choose a different model as the final model)
    final_model = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)

    # Initialize StackingClassifier
    models = [knn, LG, DT]
    exercise = StackingClassifier(models, final_model)
    exercise.fit(train_data)
    print(exercise.score(test_data))