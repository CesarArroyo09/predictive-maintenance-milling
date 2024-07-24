import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class PreprocessingPipeline:
    def __init__(self):
        self._transformed_machine_failure = None
        self._X_train = None
        self._y_train = None
        self._X_test = None
        self._y_test = None
        # self._standard_scaler = StandardScaler()

    def fit(self, machine_failure):
        self._transformed_machine_failure = machine_failure.copy()
        self._transform_categorical()
        self._label_encoding_target()
        self._transform_numerical()

    def transform_split(self):
        return self._X_train, self._X_test, self._y_train, self._y_test

    def save(self, dir_path: str, suffix: str):
        if not os.path.isabs(dir_path):
            dir_path = os.path.join(os.getcwd(), dir_path)

        os.makedirs(dir_path, exist_ok=True)
        self._X_train.to_csv(
            os.path.join(dir_path, f"X_train_{suffix}.csv"), index=False
        )
        self._y_train.to_csv(
            os.path.join(dir_path, f"y_train_{suffix}.csv"), index=False
        )
        self._X_test.to_csv(os.path.join(dir_path, f"X_test_{suffix}.csv"), index=False)
        self._y_test.to_csv(os.path.join(dir_path, f"y_test_{suffix}.csv"), index=False)

    def _transform_categorical(self):
        self._transformed_machine_failure = self._transformed_machine_failure.drop(
            columns=["UDI", "Product ID"]
        )
        self._transformed_machine_failure = pd.get_dummies(
            self._transformed_machine_failure, drop_first=True, dtype=int
        )

    def _label_encoding_target(self):
        self._transformed_machine_failure["Flags quantity"] = (
            self._transformed_machine_failure["TWF"]
            + self._transformed_machine_failure["HDF"]
            + self._transformed_machine_failure["PWF"]
            + self._transformed_machine_failure["OSF"]
            + self._transformed_machine_failure["RNF"]
        )
        self._transformed_machine_failure["Failure type"] = 0

        one_failure_mask = self._transformed_machine_failure["Flags quantity"] == 1
        self._transformed_machine_failure.loc[
            one_failure_mask & (self._transformed_machine_failure["TWF"] == 1),
            "Failure type",
        ] = 1
        self._transformed_machine_failure.loc[
            one_failure_mask & (self._transformed_machine_failure["HDF"] == 1),
            "Failure type",
        ] = 2
        self._transformed_machine_failure.loc[
            one_failure_mask & (self._transformed_machine_failure["PWF"] == 1),
            "Failure type",
        ] = 3
        self._transformed_machine_failure.loc[
            one_failure_mask & (self._transformed_machine_failure["OSF"] == 1),
            "Failure type",
        ] = 4
        self._transformed_machine_failure.loc[
            one_failure_mask & (self._transformed_machine_failure["RNF"] == 1),
            "Failure type",
        ] = 5

        two_failure_mask = self._transformed_machine_failure["Flags quantity"] == 2
        two_failure_with_rnf_mask = two_failure_mask & (
            self._transformed_machine_failure["RNF"] == 1
        )
        self._transformed_machine_failure.loc[
            two_failure_with_rnf_mask & (self._transformed_machine_failure["TWF"] == 1),
            "Failure type",
        ] = 1
        self._transformed_machine_failure.loc[
            two_failure_with_rnf_mask & (self._transformed_machine_failure["HDF"] == 1),
            "Failure type",
        ] = 2
        self._transformed_machine_failure.loc[
            two_failure_with_rnf_mask & (self._transformed_machine_failure["PWF"] == 1),
            "Failure type",
        ] = 3
        self._transformed_machine_failure.loc[
            two_failure_with_rnf_mask & (self._transformed_machine_failure["OSF"] == 1),
            "Failure type",
        ] = 4

        two_failure_without_rnf_mask = (
            self._transformed_machine_failure["Flags quantity"] == 2
        ) & (self._transformed_machine_failure["RNF"] == 0)
        self._transformed_machine_failure.loc[
            two_failure_without_rnf_mask, "Failure type"
        ] = 6

        multi_failure_mask = self._transformed_machine_failure["Flags quantity"] > 2
        self._transformed_machine_failure.loc[multi_failure_mask, "Failure type"] = 6

        columns_to_drop = [
            "TWF",
            "HDF",
            "PWF",
            "OSF",
            "RNF",
            "Flags quantity",
            "Machine failure",
        ]
        self._transformed_machine_failure = self._transformed_machine_failure.drop(
            columns=columns_to_drop
        )

        self._transformed_machine_failure["Machine failure"] = (
            self._transformed_machine_failure["Failure type"] != 0
        ).astype(int)

    def _transform_numerical(self):
        X = self._transformed_machine_failure.drop(
            columns=["Failure type", "Machine failure"]
        )
        y = self._transformed_machine_failure[["Machine failure", "Failure type"]]
        X_train, X_test, self._y_train, self._y_test = train_test_split(
            X, y, test_size=0.2, random_state=95, stratify=y["Failure type"]
        )

        standard_scaler_columns = [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ]

        X_train_categorical = X_train.drop(columns=standard_scaler_columns)
        X_test_categorical = X_test.drop(columns=standard_scaler_columns)
        X_train_numerical = X_train[standard_scaler_columns].copy()
        X_test_numerical = X_test[standard_scaler_columns].copy()

        standard_scaler = StandardScaler()
        standard_scaler.fit(X_train_numerical)
        X_train_numerical = standard_scaler.transform(
            X_train_numerical[standard_scaler.feature_names_in_]
        )
        X_test_numerical = standard_scaler.transform(
            X_test_numerical[standard_scaler.feature_names_in_]
        )
        X_train_numerical = pd.DataFrame(
            X_train_numerical, columns=standard_scaler.feature_names_in_
        )
        X_test_numerical = pd.DataFrame(
            X_test_numerical, columns=standard_scaler.feature_names_in_
        )
        self._X_train = pd.concat(
            [X_train_categorical.reset_index(drop=True), X_train_numerical], axis=1
        )
        self._X_test = pd.concat(
            [X_test_categorical.reset_index(drop=True), X_test_numerical], axis=1
        )
