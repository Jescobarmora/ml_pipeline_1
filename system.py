import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

import warnings
warnings.filterwarnings("ignore")


class ML_System_Classification:
    def __init__(self, path):
        self.path = path

    def load_data(self):
        dataset = pd.read_csv(self.path + "iris_dataset.csv", header=0, sep=";", decimal=",")
        prueba = pd.read_csv(self.path + "iris_prueba.csv", header=0, sep=";", decimal=",")
        return dataset, prueba
    
    def processing_data(self, dataset, prueba):
        covariables = [x for x in dataset.columns if x not in ["y"]]
        X = dataset[covariables]
        y = dataset["y"].values.ravel()
        X_nuevo = prueba[covariables]
        y_nuevo = prueba["y"].values.ravel()
        return X, y, X_nuevo, y_nuevo
    
    def train_and_evaluate_model(self, X_train, y_train, X_test, y_test):
        # Escalado de los datos
        scaler = preprocessing.StandardScaler()
        X_train_Z = scaler.fit_transform(X_train)
        X_test_Z = scaler.transform(X_test)
        
        # Configuración del modelo
        modelo = LogisticRegression(random_state=123)
        parametros = {'C': np.arange(0.1, 5.1, 0.1)}
        grilla = GridSearchCV(estimator=modelo, param_grid=parametros, cv=5, scoring=make_scorer(accuracy_score), n_jobs=-1)
        
        # Entrenamiento y predicción
        grilla.fit(X_train_Z, y_train)
        y_hat_test = grilla.predict(X_test_Z)
        y_hat_train = grilla.predict(X_train_Z)
        
        # Evaluación
        u_test = accuracy_score(y_test, y_hat_test)
        u_train = accuracy_score(y_train, y_hat_train)
        
        return u_test, u_train
    
    def finalize_model(self, X, y, u1, u2):
        if np.abs(u1 - u2) < 0.1:
            modelo_completo = LogisticRegression(random_state=123)
            parametros = {'C': np.arange(0.1, 5.1, 0.1)}
            grilla_completa = GridSearchCV(estimator=modelo_completo, param_grid=parametros, cv=5, scoring=make_scorer(accuracy_score), n_jobs=-1)
            
            scaler = preprocessing.StandardScaler()
            X_Z = scaler.fit_transform(X)
            
            grilla_completa.fit(X_Z, y)
        else:
            print("No cumple la condición np.abs(u1 - u2) < 0.1, continúa sin GridSearchCV.")
            scaler = preprocessing.StandardScaler()
            X_Z = scaler.fit_transform(X)
            
            modelo_completo = LogisticRegression(random_state=123)
            modelo_completo.fit(X_Z, y)
            grilla_completa = modelo_completo
            
        return grilla_completa, scaler
    
    def forecast(self, grilla_completa, scaler, X_nuevo):
        X_nuevo_Z = scaler.transform(X_nuevo)
        y_hat_nuevo = grilla_completa.predict(X_nuevo_Z)
        return y_hat_nuevo
    
    def evaluate(self, y_nuevo, y_hat_nuevo):
        accuracy = accuracy_score(y_nuevo, y_hat_nuevo)
        return accuracy
    
    def ML_Flow_regression(self):
        try:
            dataset, prueba = self.load_data()
            X, y, X_nuevo, y_nuevo = self.processing_data(dataset, prueba)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)
            
            # Entrenar y evaluar el primer modelo
            u1, u2 = self.train_and_evaluate_model(X_train, y_train, X_test, y_test)
            
            # Entrenar y evaluar el segundo modelo (invirtiendo los conjuntos de entrenamiento y prueba)
            u3, u4 = self.train_and_evaluate_model(X_test, y_test, X_train, y_train)
            
            # Decidir el modelo final
            grilla_completa, scaler_completo = self.finalize_model(X, y, u1, u3)
            
            # Predecir y evaluar en los nuevos datos
            y_hat_nuevo = self.forecast(grilla_completa, scaler_completo, X_nuevo)
            metric = self.evaluate(y_nuevo, y_hat_nuevo)
            
            return {'success': True, 'accuracy': metric * 100}
        except Exception as e:
            return {'success': False, 'message': str(e)}

# Ejecución
path = "/code/Python/Quiz_1/"
ml_system = ML_System_Classification(path)
resultado = ml_system.ML_Flow_regression()
print(resultado)
