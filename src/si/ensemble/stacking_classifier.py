import numpy as np
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy

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
        Fit the models to the dataset.
        :param dataset: Dataset object to fit the models to.
        :return: self: StackingClassifier
        """
        
        self.F, self.p = self.score_func(dataset) #recorre à função score_func, utilizando o conjunto de dados fornecido
        #e calcula os valores de F e p para cada feature no conjunto de dados
        return self 







    def predict(self, dataset: Dataset) -> np.array:
        """
        Computes the prevision of all the models and returns the final model prediction.
        :param dataset: Dataset object to predict the labels of.
        :return: the final model prediction
        """
        # gets the model predictions
        predictions = []
        for model in self.models:
            predictions.append(model.predict(dataset))

        # gets the final model previsions
        y_pred = self.final_model.predict(Dataset(np.array(predictions).T, dataset.y))

        return y_pred
    




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

    data = ""  
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