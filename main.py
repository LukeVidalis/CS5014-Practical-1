import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
import numpy as np


def get_training_data():
    file_path = "ENB2012_data.csv"
    scaler = MinMaxScaler()

    file = pd.read_csv(file_path)
    column_number = len(file.columns)
    columns_used = [0, 1, 2, 3, 4, 5, 6, 7]

    training = pd.read_csv(file_path, usecols=columns_used, index_col=None)
    results = pd.read_csv(file_path, usecols=[column_number - 2, column_number - 1], skiprows=None, index_col=None)

    plot_graphs(training, results)
    pearson_plot(training, results)

    # After examining the values using Pearson's Correlation Coefficient we deduce that X6 and X8 increase the
    # computational demand more than the influence the results so we remove them.
    training = training.drop(columns=['X6', 'X8'])

    # Normalizing the values using MinMax Scaler which puts all the values between 0 and 1
    scaler.fit(training)
    training = scaler.transform(training)
    scaler.fit(results)
    results = scaler.transform(results)
    # print("Training Data:")
    # print(training)
    # print("----------------------")
    # print("Expected Results:")
    # print(results)

    # The data is being split firstly between a training and testing data. Since the testing data will not change
    # throughout running the program it is being saved first in a non randomized manner. The second split is done by
    # splitting the current training data into training and validation. This process is randomized with the shuffle
    # parameter and will return different results every time. The 0.11111111111 test size is done so that the testing
    # and validation sets have the same size.
    X_train, X_testing, y_train, y_testing = train_test_split(training, results, test_size=0.10, shuffle=False,
                                                              random_state=40)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.1111111111,
                                                                    shuffle=True)

    return X_train, y_train, X_testing, y_testing, X_validation, y_validation


# Creates and plots the relation between the feature and target values
# This method creates 16 graphs in total which are split into two plots: One for Y1 and one for Y2
def plot_graphs(training, results):

    x_column_names = training.columns
    y_column_names = results.columns

    x_features_num = training.shape[1]
    y_features_num = results.shape[1]

    x = training.values
    y = results.values

    for y_col in range(0, y_features_num):
        plt.figure(num=y_column_names[y_col]+" Values")
        for x_col in range(0, x_features_num):
            plt.subplot(x_features_num/2, 2, x_col+1).set_title(str(y_column_names[y_col]) + " values for " +
                                                                str(x_column_names[x_col]))
            plt.scatter(x[:, x_col], y[:, y_col])
            plt.tight_layout(pad=0.3, w_pad=0.5, h_pad=0.4)
        plt.show()
        plt.clf()


# Creates and displays a plot of the Pearson Correlation coefficient for each of the Y values in regards to the X values
# The library used to calculate pcc is SciPy
def pearson_plot(training, results):
    x_column_names = training.columns
    y_column_names = results.columns

    x_features_num = training.shape[1]
    y_features_num = results.shape[1]
    plt.figure(num="Pearson's Correlation coefficient for Y Values")

    for y_col in range(0, y_features_num):
        pp = []
        for x_col in range(0, x_features_num):
            value = pearsonr(training.values[:, x_col], results.values[:, y_col])[0]
            pp.append(value)

        plt.subplot(1, 2, y_col+1).set_title("PCC for " + str(y_column_names[y_col]))
        plt.bar(x_column_names, pp)

    plt.show()


def get_lr_classifier(data, results):
    lr = LinearRegression()
    lr.fit(data, results)
    return lr


def get_rf_classifier(data, results):
    rf = RandomForestRegressor()
    rf.fit(data, results)
    return rf


def score_classifier(classifier, data, results):
    print(classifier.score(data, results))


def plot_lr_results(model, X, y):
    theta = model.coef_
    y_predicted = model.predict(X)

    x_features_num = X.shape[1]
    g = np.empty((len(X), 2))

    for i in range(0, len(X)):
        gUnitY1 = 0
        gUnitY2 = 0
        for j in range(0, x_features_num):
            gUnitY1 += theta[0][j]*X[i][j]
            gUnitY2 += theta[1][j]*X[i][j]

        g[i][0] = gUnitY1
        g[i][1] = gUnitY2

    plt.subplot(1, 2, 1).set_title("Y1")
    plt.scatter(g[:, 0], y[:, 0], c='r')
    plt.plot(g[:, 0], y_predicted[:, 0], c='b')
    plt.subplot(1, 2, 2).set_title("Y2")
    plt.scatter(g[:, 1], y[:, 1], c='r')
    plt.plot(g[:, 1], y_predicted[:, 1], c='b')
    plt.show()
    plt.clf()


if __name__ == "__main__":

    X_train, y_train, X_testing, y_testing, X_validation, y_validation = get_training_data()
    linear_classifier = get_lr_classifier(X_train, y_train)

    print("Score Classifier Linear:")
    score_classifier(linear_classifier, X_testing, y_testing)

    error = cross_val_score(linear_classifier, X_validation, y_validation, cv=2, scoring="neg_mean_squared_error").mean()
    print("SME Validation Set:", -error)

    error = cross_val_score(linear_classifier, X_testing, y_testing, cv=2, scoring="neg_mean_squared_error").mean()
    print("SME Testing Set:", -error)

    plot_lr_results(linear_classifier, X_testing, y_testing)

    forest_classifier = get_rf_classifier(X_train, y_train)

    error = cross_val_score(forest_classifier, X_testing, y_testing, cv=2, scoring="neg_mean_squared_error").mean()
    print("SME Testing Set:", -error)

    error = cross_val_score(forest_classifier, X_validation, y_validation, cv=2,
                            scoring="neg_mean_squared_error").mean()
    print("SME Validation Set:", -error)

    print("Score Classifier Forest:")
    score_classifier(forest_classifier, X_testing, y_testing)
