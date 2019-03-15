import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def get_training_data():
    file_path = "ENB2012_data.csv"

    file = pd.read_csv(file_path)
    column_number = len(file.columns)
    columns_used = [0, 1, 2, 3, 4, 5, 6, 7]
    scaler = StandardScaler()
    test = pd.read_csv(file_path)
    training = pd.read_csv(file_path, usecols=columns_used, index_col=None)
    results = pd.read_csv(file_path, usecols=[column_number - 2, column_number - 1], skiprows=None, index_col=None)

    #plot_graphs(training, results)
    pearson_plot(training, results)
    scaler.fit(training)
    training = scaler.transform(training)
    scaler.fit(results)
    results = scaler.transform(results)
    # print("Training Data:")
    # print(training)
    # print("----------------------")
    # print("Expected Results:")
    # print(results)
    return training, results


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
    print(lr)
    return lr


def get_rf_classifier(data, results):
    rf = RandomForestRegressor()
    rf.fit(data, results)
    print(rf)
    return rf



def score_classifier(classifier, data, results):
    print(classifier.score(data, results)*100, "%")


if __name__ == "__main__":

    training_data, expected_output = get_training_data()
    linear_classifier = get_lr_classifier(training_data, expected_output)
    score_classifier(linear_classifier, training_data, expected_output)

    forest_classifier = get_rf_classifier(training_data, expected_output)
    score_classifier(forest_classifier, training_data, expected_output)
