import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def get_training_data():
    file_path = "ENB2012_data.csv"

    file = pd.read_csv(file_path)
    column_number = len(file.columns)
    columns_used = [0, 1, 2, 3, 4, 5, 6]

    training = pd.read_csv(file_path, usecols=columns_used, skiprows=1, header=None, index_col=None)
    results = pd.read_csv(file_path, usecols=[column_number - 2, column_number - 1], skiprows=1,
                          header=None, index_col=None)

    print("Training Data:")
    print(training)
    print("----------------------")
    print("Expected Results:")
    print(results)

    return training, results


def get_lr_classifier(data, results):
    lr = LinearRegression()
    lr.fit(data, results)
    print(lr)
    return lr


def get_rf_classifier(data, results):
    rf = RandomForestRegressor()
    rf.fit(data, results)
    print(rf)
    return rf


def score_classifier(classifier, data, results):
    print(classifier.score(data, results))


if __name__ == "__main__":

    training_data, expected_output = get_training_data()
    linear_classifier = get_lr_classifier(training_data, expected_output)
    score_classifier(linear_classifier, training_data, expected_output)

    forest_classifier = get_rf_classifier(training_data, expected_output)
    score_classifier(forest_classifier, training_data, expected_output)
