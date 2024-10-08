import json

import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import class_weight


class ModelWrapper:
    """
    A wrapper class for machine learning models.

    Parameters:
    -----------
    model_class : class, optional
        The class of the machine learning model to be wrapped.
    params : dict, optional
        The parameters to be passed to the model class during initialization.
    name : str, optional
        The name of the model.

    Attributes:
    -----------
    model : object
        The machine learning model object.
    params : dict
        The parameters passed to the model during initialization.
    name : str
        The name of the model.
    is_fitted : bool
        Indicates whether the model has been fitted or not.

    Methods:
    --------
    fit(X_train, y_train, sampling_strategy=None)
        Fits the model to the training data.
    predict(X_test)
        Predicts the target variable for the given test data.
    get_results(X_test, y_test, output_dict=False)
        Returns the classification report and confusion matrix for the given
        test data.
    save_params(model_dir="params")
        Saves the model parameters to a JSON file.
    evaluate(X_test, y_test)
        Returns accuracy and classification report on a tuple.

    """

    def __init__(self, model_class=None, params=None, name=None):
        self.model = model_class(**params) if model_class is not None else None
        self.params = params
        self.name = name
        self.is_fitted = False

    def fit(self, X_train, y_train, sampling_strategy=None):
        """
        Fits the model to the training data.

        Parameters:
        -----------
        X_train : array-like
            The training input samples.
        y_train : array-like
            The target values.
        sampling_strategy : str, optional
            The sampling strategy to be applied to the training data.
            It can be one of "oversampling", "undersampling", or "smote".

        Raises:
        -------
        ValueError
            If the model is not set.

        """

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
        """
        Predicts the target variable for the given test data.

        Parameters:
        -----------
        X_test : array-like
            The test input samples.

        Raises:
        -------
        ValueError
            If the model is not fitted yet.

        Returns:
        --------
        array-like
            The predicted target values.

        """

        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self.model.predict(X_test)

    def get_results(self, X_test, y_test, output_dict=False):
        """
        Returns the classification report and confusion matrix given the test data.

        Parameters:
        -----------
        X_test : array-like
            The test input samples.
        y_test : array-like
            The true target values.
        output_dict : bool, optional
            If True, returns the classification report as a dictionary.

        Raises:
        -------
        ValueError
            If the model is not fitted yet.

        Returns:
        --------
        dict or tuple
            If output_dict is True, returns the classification report as a
            dictionary and the confusion matrix as a tuple.
            Otherwise, returns the classification report as a string and the
            confusion matrix as a numpy array.

        """

        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        y_pred = self.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=output_dict)
        cm = confusion_matrix(y_test, y_pred)

        return report, cm

    def save_params(self, model_dir="params"):
        """
        Saves the model parameters to a JSON file.

        Parameters:
        -----------
        model_dir : str, optional
            The directory to save the parameters file.

        Raises:
        -------
        ValueError
            If the model is not set.

        """

        if self.model is None:
            raise ValueError("Model is not set.")
        params_path = f"{model_dir}/{self.name}_params.json"
        with open(params_path, "w") as f:
            json.dump(self.params, f)
        print(f"Model and parameters saved to {model_dir}.")

    def evaluate(self, X_test, y_test):
        """
        Evaluates the model performance on the given test data.

        It prints the accuracy score, classification report, and confusion matrix.

        Parameters:
        -----------
        X_test : array-like
            The test input samples.
        y_test : array-like
            The true target values.

        Raises:
        -------
        ValueError
            If the model is not fitted yet.

        Returns:
        --------
        float, str
            The accuracy score and the classification report.

        """

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

    # def save(self, model_dir="models"):
    #     if not self.is_fitted:
    #         raise ValueError("Model is not fitted yet.")
    #     model_path = f"{model_dir}/{self.name}_experiment_{self.experiment}.joblib"
    #     params_path = (
    #         f"{model_dir}/{self.name}_experiment_{self.experiment}_params.json"
    #     )
    #     joblib.dump(self.model, model_path)
    #     with open(params_path, "w") as f:
    #         json.dump(self.params, f)
    #     print(f"Model and parameters saved to {model_dir}.")

    # def load(self, name, experiment, model_dir="models"):
    #     self.name = name
    #     self.experiment = experiment
    #     model_path = f"{model_dir}/{self.name}_experiment_{self.experiment}.joblib"
    #     params_path = (
    #         f"{model_dir}/{self.name}_experiment_{self.experiment}_params.json"
    #     )

    #     self.model = joblib.load(model_path)
    #     with open(params_path, "r") as f:
    #         self.pa


class NeuralNetworkWrapper:
    """
    A wrapper class for TensorFlow Keras neural network models.

    Parameters
    ----------
    model : tf.keras.models.Model
        The Keras model to be wrapped.
    name : str, optional
        The name of the model.

    Attributes
    ----------
    model : tf.keras.models.Model
        The Keras model object.
    name : str
        The name of the model.
    is_fitted : bool
        Indicates whether the model has been fitted.

    Methods
    -------
    fit(X_train, y_train, sampling_strategy=None)
        Fits the model to the training data with an optional sampling strategy.
    predict(X_test)
        Predicts the target variable for the given test data.
    get_results(X_test, y_test, output_dict=False)
        Returns the classification report and confusion matrix for the given test data.
    evaluate(X_test, y_test)
        Evaluates the model performance on the given test data.
    """

    def __init__(self, model, name=None):
        """
        Initializes the NeuralNetworkWrapper with the specified model and name.

        Parameters
        ----------
        model : tf.keras.models.Model
            The Keras model to be wrapped.
        name : str, optional
            The name of the model.

        Raises
        ------
        ValueError
            If the model is not set or if it is not a Keras model.
        """
        if model is None:
            raise ValueError("Model is not set.")
        if not isinstance(model, tf.keras.models.Model):
            raise ValueError("Model should be a Keras model.")

        self.model = model
        self.name = name
        self.is_fitted = False

    def fit(self, X_train, y_train, sampling_strategy: str = None):
        """
        Fits the Keras model to the training data with optional sampling strategy.

        Parameters
        ----------
        X_train : array-like
            The training input samples.
        y_train : array-like
            The target values.
        sampling_strategy : str, optional
            The sampling strategy to be applied to the training data.
            It can be one of "oversampling", "undersampling", or "smote".

        Raises
        ------
        ValueError
            If the model is not set or if the sampling strategy is invalid.
        """
        if self.model is None:
            raise ValueError("Model is not set.")
        if sampling_strategy not in [None, "oversampling", "undersampling", "smote"]:
            raise ValueError(
                "Sampling strategy should be one of 'oversampling', 'undersampling', or 'smote'."
            )

        sampler = None
        if sampling_strategy:
            if sampling_strategy == "oversampling":
                sampler = RandomOverSampler(random_state=98)
                self.name = f"{self.name}_oversampling"
            elif sampling_strategy == "undersampling":
                sampler = RandomUnderSampler(random_state=98)
                self.name = f"{self.name}_undersampling"
            elif sampling_strategy == "smote":
                sampler = SMOTE(random_state=98)
                self.name = f"{self.name}_smote"
        else:
            class_weights = class_weight.compute_class_weight(
                class_weight="balanced",
                classes=np.unique(y_train),
                y=y_train.values.flatten(),
            )
            class_weights_dict = dict(enumerate(class_weights))

        print(f"Type of y_train: {type(y_train)}")
        print(f"Shape of y_train: {y_train.shape}")

        if sampler is not None:
            print(f"Resampling the training data using {sampling_strategy}.")
            X_train, y_train = sampler.fit_resample(X_train, y_train)

        encoder = OneHotEncoder(sparse_output=False)
        y_train_encoded = encoder.fit_transform(y_train.values.reshape(-1, 1))

        self.model.fit(
            X_train.values,
            y_train_encoded,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            class_weight=(class_weights_dict if not sampling_strategy else None),
        )
        self.is_fitted = True
        print(f"{self.name} model fitted.")

    def predict(self, X_test):
        """
        Predicts the target variable for the given test data.

        Parameters
        ----------
        X_test : array-like
            The test input samples.

        Raises
        ------
        ValueError
            If the model is not fitted yet.

        Returns
        -------
        array-like
            The predicted target values.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        y_pred = self.model.predict(X_test.values)
        y_pred = np.argmax(y_pred, axis=1)

        return y_pred

    def get_results(self, X_test, y_test, output_dict=False):
        """
        Returns the classification report and confusion matrix given the test data.

        Parameters
        ----------
        X_test : array-like
            The test input samples.
        y_test : array-like
            The true target values.
        output_dict : bool, optional
            If True, returns the classification report as a dictionary.

        Raises
        ------
        ValueError
            If the model is not fitted yet.

        Returns
        -------
        dict or tuple
            If output_dict is True, returns the classification report as a
            dictionary and the confusion matrix as a tuple.
            Otherwise, returns the classification report as a string and the
            confusion matrix as a numpy array.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        y_pred = self.predict(X_test)
        report = classification_report(y_test.values, y_pred, output_dict=output_dict)
        cm = confusion_matrix(y_test.values, y_pred)

        return report, cm

    def evaluate(self, X_test, y_test):
        """
        Evaluates the model performance on the given test data.

        It prints the accuracy score, classification report, and confusion matrix.

        Parameters
        ----------
        X_test : array-like
            The test input samples.
        y_test : array-like
            The true target values.

        Raises
        ------
        ValueError
            If the model is not fitted yet.

        Returns
        -------
        float, str
            The accuracy score and the classification report.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test.values, y_pred)
        report = classification_report(y_test.values, y_pred)
        print(f"Accuracy for {self.name}: {accuracy}")
        print(f"Classification Report for {self.name}:\n{report}")
        print()
        print(
            f"Confusion Matrix for {self.name}:\n{confusion_matrix(y_test.values, y_pred)}"
        )
        return accuracy, report
