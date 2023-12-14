import math 

def rmse(y_true: float, Y_pred: float) -> float:
    """
    Calculates the error following the RMSE formula

    Parameters
    ----------
    y_true: 
        real values of y
    Y_pred: 
        predicted values of y

    Returns: float
        float corresponding to the error between y_true and y_pred

    """
     
    if len(y_true) != len(Y_pred): 
        raise ValueError("Input lists must have the same length") 
    if not y_true or not Y_pred: 
        return 0.0  

    squared_diff = [(true - pred) ** 2 for true, pred in zip(y_true, Y_pred)] 

    mean_squared_diff = sum(squared_diff) / len(y_true) 
 
    rmse = math.sqrt(mean_squared_diff) 

    return rmse 


if __name__ == "__main__":
    

    y_true_1 = [3, -0.5, 2, 7]
    y_pred_1 = [3, -0.5, 2, 7]
    assert rmse(y_true_1, y_pred_1) == 0, "Erro no Caso de Teste 1"

    

    y_true_2 = [3, -0.5, 2, 7]
    y_pred_2 = [2.5, 0.0, 2, 8]
    assert math.isclose(rmse(y_true_2, y_pred_2), 0.612, rel_tol=1e-3), "Erro no Caso de Teste 2"

    
    y_true_3 = []
    y_pred_3 = []
    assert rmse(y_true_3, y_pred_3) == 0, "Erro no Caso de Teste 3"