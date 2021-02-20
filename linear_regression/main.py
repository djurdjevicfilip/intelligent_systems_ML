
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.stats import stats
from sklearn import metrics
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.decomposition import PCA

# Print Utils
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler

from linear_regression import LinearRegressionGradientDescent


def printDivider(text, lineLength=30):
    str = ""
    for i in range(lineLength):
        str += "-"

    str += " " + text + " "

    for i in range(lineLength):
        str += "-"

    print(str)


def printGrid(X, y):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(13, 15))

    i = 0
    for row in ax:
        for col in row:
            col.scatter(X[i], y, s=23, c='red', marker='o', alpha=0.7, edgecolors='black', linewidths=2,
                        label=labels[i])

            col.set_xlabel(labels[i])
            col.set_ylabel('Price')
            i += 1

    fig.suptitle('Price based on other attributes', fontsize=20)

def plotLinearReg(X, y):
    spots = 200
    estates = pd.DataFrame(data=np.linspace(0, max(X['Area']), num=spots))

    plt.scatter(X, y, s=23, c='red', marker='o', alpha=0.7,
                edgecolors='black', linewidths=2, label='houses')
    line, = plt.plot(estates[0], lr.predict(estates), lw=2, c='blue')
    line.set_label('Ordinary LR model')

    line, = plt.plot(estates[0], lrgd.predict(estates), lw=2, c='red')
    line.set_label('LRGD model')

    # Lokacija legende (gore levo)
    plt.legend(loc='upper left')
    plt.show()

def printData():
    # Write the first 5 rows
    printDivider("HEAD")
    print(data.head())

    # Write the last 5 rows
    printDivider("TAIL")
    print(data.tail())

    printDivider("INFO")
    print(data.info())

    printDivider("STATISTICS")
    print(data.describe())


def polyTransform(X):
    poly = PolynomialFeatures(degree=3, include_bias=False)
    poly.fit(X)
    X_poly = poly.transform(X)

    return pd.DataFrame(X_poly, columns=poly.get_feature_names(X.columns))

def msError(mse_history):
    plt.figure('MS Error')
    plt.plot(np.arange(0, len(mse_history), 1), mse_history)
    plt.xlabel('Iteration', fontsize=13)
    plt.ylabel('MS error value', fontsize=13)
    plt.xticks(np.arange(0, len(mse_history), 2))
    plt.title('Mean-square error function')
    plt.tight_layout()
    plt.legend(['MS Error'])
    plt.show()

def msHyperplane(mse_history, lrgd):
    spots = 1000
    c0, c1 = np.meshgrid(np.linspace(-500, 500, spots), np.linspace(0, 2, spots))
    c0 = c0.flatten()
    c1 = c1.flatten()
    mse_values = []
    for i in range(len(c0)):
        lrgd.set_coefficients(c0[i], c1[i])
        mse_values.append(lrgd.cost())
    fig = plt.figure('MSE hyperplane')
    ax = plt.subplot(1, 1, 1, projection='3d')
    surf = ax.plot_surface(c0.reshape(spots, spots), c1.reshape(spots, spots),
                           np.array(mse_values).reshape(spots, spots),
                           cmap=cm.coolwarm)
    min_mse_ind = mse_values.index(min(mse_values))
    ax.scatter(c0[min_mse_ind], c1[min_mse_ind], mse_values[min_mse_ind],
               c='r', s=250, marker='^')
    fig.colorbar(surf, shrink=0.5)
    ax.set_xlabel('c0')
    ax.set_ylabel('c1')
    ax.set_zlabel('mse')
    plt.title('MSE error')
    plt.tight_layout()
    plt.show()

# Read the dataset
data = pd.read_csv('datasets/house_prices_train.csv')

# Remove outliers
z_scores = stats.zscore(data)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3.5).all(axis=1)

data = data[filtered_entries]

printData()

X = []
y = data['Price']
labels = []
for col in data:
    X.append(data.loc[:, [col]])
    labels.append(col)

printGrid(X,y)
plt.show()
X = data.drop('Price', axis = 1)

feature_names = ['Area', 'Bedroom_no', 'Bath_no', 'Year_built']

X = X[feature_names]


ss = StandardScaler()

X[feature_names] = ss.fit_transform(X[feature_names])
y = y/1000

X_poly = polyTransform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, random_state=0)

lr = LinearRegression()
# Kreiranje i obucavanje modela
lrgd = LinearRegressionGradientDescent()
lrgd.fit(X_train, y_train)
res_coeff, mse_history = lrgd.perform_gradient_descent(0.008, 5000)
lr.fit(X_train, y_train)
plt.show()

printDivider('Linear Regression')

parameters = []
parameters.append(lr.intercept_)
parameters += lr.coef_

printDivider("Model parameters", 3)
for name, param in zip(X_poly.columns,parameters):
    print(name+": "+str(param))
printDivider("end", 3)

print(f'LR train score: {lr.score(X_test, y_test):.2f}')
print(f'LR test score: {lr.score(X_test, y_test):.2f}')

cv_res = cross_validate(lr, X, y, cv=10)
print(f'CV score: {cv_res["test_score"].mean():0.3f}')

lr.coef_ = lrgd.coeff[1:].reshape(1, -1)[0]
lr.intercept_ = lrgd.coeff[0]


plt.figure()



plt.scatter(X['Area'], y, s=23, c='red', marker='o', alpha=0.7, edgecolors='black', linewidths=2)



y_pred = lr.predict(X_test)
print("RMSE Test:"+str(metrics.mean_squared_error(y_test, y_pred)))
y_pred = lr.predict(X_train)

print("RMSE Train: "+str(metrics.mean_squared_error(y_train, y_pred)))

printDivider('Linear Regression w/ GradientDescent')

parameters = []
parameters.append(lr.intercept_)
parameters += lr.coef_
printDivider("Model parameters", 3)
for name, param in zip(X_poly.columns,parameters[0]):
    print(name+": "+str(param))

printDivider("end", 3)
print(f'LRGD train score: {lr.score(X_train, y_train):.2f}')
print(f'LRGD test score: {lr.score(X_test, y_test):.2f}')
cv_res = cross_validate(lr, X, y, cv=10)
print(f'CV score: {cv_res["test_score"].mean():0.3f}')


y_pred = lrgd.predict(X_test)
print("RMSE Test:"+str(metrics.mean_squared_error(y_test, y_pred)))
y_pred = lrgd.predict(X_train)

print("RMSE Train: "+str(metrics.mean_squared_error(y_train, y_pred)))