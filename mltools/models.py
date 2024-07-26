import json

import joblib
import tensorflow as tf
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class ModelWrapper:
    def __init__(
        self, model_class=None, params=None, name=None, experiment: str = None
    ):
        self.model = model_class(**params) if model_class is not None else None
        self.params = params
        self.name = name
        self.experiment = experiment
        self.is_fitted = False

    def fit(self, X_train, y_train, sampling_strategy=None):
        if self.model is None:
            raise ValueError("Model is not set.")

        match sampling_strategy:
            case "oversampling":
                self.model = Pipeline(
                    [
                        ("sampling", RandomOverSampler(random_state=98)),
                        ("model", self.model),
                    ]
                )
                self.name = f"{self.name}_oversampling"
            case "undersampling":
                self.model = Pipeline(
                    [
                        ("sampling", RandomUnderSampler(random_state=98)),
                        ("model", self.model),
                    ]
                )
                self.name = f"{self.name}_undersampling"
            case "smote":
                self.model = Pipeline(
                    [
                        ("sampling", SMOTE(random_state=98)),
                        ("model", self.model),
                    ]
                )
                self.name = f"{self.name}_smote"

        self.model.fit(X_train, y_train)
        self.is_fitted = True
        print(f"{self.name} model fitted.")

    def predict(self, X_test):
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print(f"Accuracy for {self.name}: {accuracy}")
        print(f"Classification Report for {self.name}:\n{report}")
        print()
        print(f"Confusion Matrix for {self.name}:\n{confusion_matrix(y_test, y_pred)}")
        return accuracy, report

    def save(self, model_dir="models"):
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        model_path = f"{model_dir}/{self.name}_experiment_{self.experiment}.joblib"
        params_path = (
            f"{model_dir}/{self.name}_experiment_{self.experiment}_params.json"
        )
        joblib.dump(self.model, model_path)
        with open(params_path, "w") as f:
            json.dump(self.params, f)
        print(f"Model and parameters saved to {model_dir}.")

    def load(self, name, experiment, model_dir="models"):
        self.name = name
        self.experiment = experiment
        model_path = f"{model_dir}/{self.name}_experiment_{self.experiment}.joblib"
        params_path = (
            f"{model_dir}/{self.name}_experiment_{self.experiment}_params.json"
        )

        self.model = joblib.load(model_path)
        with open(params_path, "r") as f:
            self.params = json.load(f)

        # Optionally, you can also instantiate the model with parameters if needed
        self.is_fitted = True
        print(f"Model and parameters loaded from {model_dir}.")

    # @staticmethod
    # def get_model_class(model_name):
    #     if model_name == "random_forest":
    #         from sklearn.ensemble import RandomForestClassifier

    #         return RandomForestClassifier
    #     elif model_name == "logistic_regression":
    #         from sklearn.linear_model import LogisticRegression

    #         return LogisticRegression
    #     elif model_name == "xgboost":
    #         from xgboost import XGBClassifier

    #         return XGBClassifier
    #     # Add more model types as needed
    #     else:
    #         raise ValueError(f"Model {model_name} not recognized.")


class KerasModelWrapper:
    def __init__(self, model=None, name=None, experiment: str = None):
        self.model = model
        self.name = name
        self.experiment = experiment
        self.is_fitted = False

    def fit(self, X_train, y_train):
        if self.model is None:
            raise ValueError("Model is not set.")
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        print(f"{self.name} model fitted.")

    def predict(self, X_test):
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print(f"Accuracy for {self.name}: {accuracy}")
        print(f"Classification Report for {self.name}:\n{report}")
        print()
        print(f"Confusion Matrix for {self.name}:\n{confusion_matrix(y_test, y_pred)}")
        return accuracy, report

    def save(self, model_dir="models"):
        model_path = f"{model_dir}/{self.name}_experiment_{self.experiment}.h5"
        self.model.save(model_path)
        print(f"Model saved to {model_dir}.")

    def load(self, name, experiment, model_dir="models"):
        self.name = name
        self.experiment = experiment
        model_path = f"{model_dir}/{self.name}_experiment_{self.experiment}.h5"

        self.model = tf.keras.models.load_model(model_path)

        self.is_fitted = True
        print(f"Model and parameters loaded from {model_dir}.")
