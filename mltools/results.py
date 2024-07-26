import pandas as pd

from mltools.models import ModelWrapper, NeuralNetworkWrapper


class ExperimentResults:
    """
    Class to store and manage experiment results, including classification
    reports and confusion matrices.

    Attributes
    ----------
    results_df : pd.DataFrame
        DataFrame to store the classification reports.
    confusion_matrices : dict
        Dictionary to store confusion matrices, with model names as keys.
    """

    def __init__(self):
        """
        Initializes an empty ExperimentResults object.
        """
        self.results_df = pd.DataFrame()
        self.confusion_matrices = {}

    def add_result(self, model_wrapper: ModelWrapper, X_test, y_test):
        """
        Adds results from a model evaluation to the ExperimentResults.

        Parameters
        ----------
        model_wrapper : ModelWrapper
            The model wrapper containing the fitted model and metadata.
        X_test : pd.DataFrame or np.ndarray
            The test features.
        y_test : pd.DataFrame or np.ndarray
            The test targets.

        Raises
        ------
        ValueError
            If the model is not fitted.
        """
        if not model_wrapper.is_fitted:
            raise ValueError(f"Model {model_wrapper.name} is not fitted yet.")

        report, cm = model_wrapper.get_results(X_test, y_test, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df["model"] = model_wrapper.name
        self.results_df = pd.concat([self.results_df, report_df], axis=0)

        # Store the confusion matrix separately because it is an np.ndarray
        self.confusion_matrices[model_wrapper.name] = cm

    def get_results(self):
        """
        Returns the classification reports as a DataFrame.

        Returns
        -------
        pd.DataFrame
            A copy of the results DataFrame.
        """
        return self.results_df.copy()

    def get_confusion_matrix(self, model_name):
        """
        Retrieves the confusion matrix for a specific model.

        Parameters
        ----------
        model_name : str
            The name of the model.

        Returns
        -------
        np.ndarray or None
            The confusion matrix for the specified model, or None if not found.
        """
        return self.confusion_matrices.get(model_name)

    def save_results(self, path):
        """
        Saves the results DataFrame to a CSV file.

        Parameters
        ----------
        path : str
            The path where the results should be saved.
        """
        self.results_df.to_csv(path, index=False)

    def load_results(self, path):
        """
        Loads results from a CSV file into the results DataFrame.

        Parameters
        ----------
        path : str
            The path to the CSV file to load.
        """
        self.results_df = pd.read_csv(path)


class ExperimentAutomation:
    """
    Automates the process of running experiments with multiple models and
    sampling strategies.

    Attributes
    ----------
    models_list : list of ModelWrapper
        List of model wrappers to be used in the experiments.
    results : ExperimentResults
        The object to store the results of the experiments.
    """

    def __init__(self, models_list):
        """
        Initializes the ExperimentAutomation with a list of models.

        Parameters
        ----------
        models_list : list of ModelWrapper
            The list of model wrappers to run experiments on.
        """
        self.models_list = models_list
        self.results = ExperimentResults()

    def run_experiment(self, X_train, X_test, y_train, y_test):
        """
        Runs experiments for each model in the list with different sampling
        strategies and stores the results.

        Parameters
        ----------
        X_train : pd.DataFrame or np.ndarray
            The training features.
        X_test : pd.DataFrame or np.ndarray
            The testing features.
        y_train : pd.DataFrame or np.ndarray
            The training targets.
        y_test : pd.DataFrame or np.ndarray
            The testing targets.
        """
        sampling_strategies = [None, "oversampling", "undersampling", "smote"]

        for model_wrapper in self.models_list:
            for strategy in sampling_strategies:
                model_wrapper_copy = self.clone_wrapper(model_wrapper)
                model_wrapper_copy.fit(X_train, y_train, sampling_strategy=strategy)
                self.results.add_result(model_wrapper_copy, X_test, y_test)
                print(
                    f"Results recorded for {model_wrapper_copy.name} "
                    f"with {strategy or 'no sampling'}."
                )

    def clone_wrapper(self, model_wrapper):
        """
        Creates a deep copy of a ModelWrapper object.

        Parameters
        ----------
        model_wrapper : ModelWrapper
            The ModelWrapper object to be copied.

        Returns
        -------
        ModelWrapper
            A deep copy of the input ModelWrapper object.
        """
        if isinstance(model_wrapper, ModelWrapper):
            model_class = model_wrapper.model.__class__
            params = model_wrapper.params.copy()
            name = model_wrapper.name
            return ModelWrapper(model_class=model_class, params=params, name=name)
        elif isinstance(model_wrapper, NeuralNetworkWrapper):
            model = model_wrapper.model
            name = model_wrapper.name
            return NeuralNetworkWrapper(model=model, name=name)
        else:
            raise ValueError("Invalid model wrapper type.")

    def save_results(self, path):
        """
        Saves the experiment results to a CSV file.

        Parameters
        ----------
        path : str
            The path to the CSV file where the results will be saved.
        """
        self.results.save_results(path)

    def load_results(self, path):
        """
        Loads experiment results from a CSV file.

        Parameters
        ----------
        path : str
            The path to the CSV file to load results from.
        """
        self.results.load_results(path)

    def get_results(self):
        """
        Retrieves the experiment results as a DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the experiment results.
        """
        return self.results.get_results()

    def get_confusion_matrix(self, model_name):
        """
        Retrieves the confusion matrix for a specific model.

        Parameters
        ----------
        model_name : str
            The name of the model for which to retrieve the confusion matrix.

        Returns
        -------
        np.ndarray or None
            The confusion matrix for the specified model, or None if not found.
        """
        return self.results.get_confusion_matrix(model_name)
