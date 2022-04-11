from typing import Tuple
import numpy as np

class Model:
    ID_DICT = {"NAME": "Nicholas Zotalis", "BU_ID": "U81029802", "BU_EMAIL": "nzotalis@bu.edu"}

    def __init__(self):
        self.theta = None

    def preprocess(self, X: np.array, y: np.array) -> Tuple[np.array, np.array]:
        ###############################################
        ####      add preprocessing code here      ####
        ###############################################
        x_ones = np.ones((X.shape[0], 1))
        X = np.hstack((x_ones, X))
        return X, y

    def train(self, X_train: np.array, y_train: np.array):
        """
        Train model with training data
        """
        ###############################################
        ####   initialize and train your model     ####
        ###############################################

        #theta=(X^t X+λI)^-1 X^t y
        #Solve for regularized linear regression using normal equations
        L = 5
        self.theta = np.linalg.inv(((X_train.T @ X_train) + L * np.eye(X_train.shape[1]))) @ X_train.T @ y_train

    def predict(self, X_val: np.array) -> np.array:
        """
        Predict with model and given feature
        """
        ###############################################
        ####      add model prediction code here   ####
        ###############################################
        
        return np.dot(X_val, self.theta)
        

