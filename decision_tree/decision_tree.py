import pandas as pd
from scipy.stats import stats
import numpy as np
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier


def printDivider(text, lineLength=30):
    str = ""
    for i in range(lineLength):
        str += "-"

    str += " " + text + " "

    for i in range(lineLength):
        str += "-"

    print(str)


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


def featurePlot(data, feature_name):
    bins = 20
    data['qcut_fare'] = pd.qcut(data[feature_name].rank(method='first'), bins)
    # Sortiranje DataFrame-a po koloni qcut_fare.
    data = data.sort_values(by='qcut_fare')
    # Konverzija vrednosti kolone qcut_fare (interval) u string.
    data['qcut_fare'] = data['qcut_fare'].astype(str)
    fig = plt.figure()
    # Prikaz zavisnosti prezivljavanja od cene karte.
    splot = sb.displot(data, x='qcut_fare', hue='type', hue_order=[0, 1],
               multiple='fill', bins=bins)

    splot.fig.savefig("decision_tree_photos/"+feature_name)

    plt.xticks(rotation=90)
    data.drop(columns=['qcut_fare'], inplace=True)


def printGrid(data):

    for feature in data:
        featurePlot(data, feature)

def removeOutliers(data, max=4):
    # Remove outliers
    z_scores = stats.zscore(data)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < max).all(axis=1)
    return data[filtered_entries]

def oneHotEncoding(data, feature_names):
    ohe = OneHotEncoder(dtype=int, sparse=False)
    # fit_transform zahteva promenu oblika
    embarked = ohe.fit_transform(data[feature_names].to_numpy().reshape(-1, 1))
    data.drop(columns=feature_names, inplace=True)
    return data.join(pd.DataFrame(data=embarked,
                                  columns=ohe.get_feature_names(feature_names[0])))

def combineFeatures(X, data):
    for i in range(len(data.columns)):
        column1 = data.columns[i]
        for j in range(i + 1, len(data.columns)):
            column2 = data.columns[j]
            print(column1 + "_" + column2 + "_ratio")
            if column1 != column2:
                X[column1 + "_" + column2 + "_ratio"] = data[column1] / data[column2]
    return X

def printTree(dtc_model, X):
    fig, axes = plt.subplots(1, 1, figsize=(8, 3), dpi=400)
    # Za isrtavanje stabla zadaje se model stabla.
    # Moguce je zadati maksimalnu dubinu stabla, imena atributa, imena labela,
    # velicinu fonta, farbanje cvorova stabla na osnovu klase pripadnosti itd.
    tree.plot_tree(decision_tree=dtc_model, max_depth=5,
                   feature_names=X.columns,
                   fontsize=3, filled=True)
    # Cuvanje stabla kao slike.
    fig.savefig('decision_tree_photos/tree.png')


# Read the dataset
data = pd.read_csv('datasets/cakes_train.csv')

# Remap
data=data.replace(to_replace="cupcake",value="1")
data=data.replace(to_replace="muffin",value="2")

data['type'] = data['type'].astype(int)

print(len(data))
#sb.displot(data[data['milk'].notna()], x='milk', col='eggs', hue='type', multiple='fill', bins=10)
#plt.show()

printData()

#removeOutliers(data)
X = data.drop('type', axis = 1)
plt.figure()
sb.heatmap(X.corr(), annot=True, fmt='.2f')
plt.show()

y = data['type']
X = combineFeatures(X, data.drop('type', axis = 1))

printGrid(data)



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, shuffle=True)
dtc_model = DecisionTreeClassifier(criterion='entropy' )

#plt.show()
dtc_model.fit(X_train, y_train)


print(f'Model training score: {dtc_model.score(X_train, y_train):0.3f}')
print(f'Model test score: {dtc_model.score(X_test, y_test):0.3f}')
print("Feature importances:\n{}".format(dtc_model.feature_importances_))
cv_res = cross_validate(dtc_model, X, y, cv=10)
print(f'CV score: {cv_res["test_score"].mean():0.3f}')

printTree(dtc_model, X)



y_pred = dtc_model.predict(X_test)
print("RMSE Test:"+str(metrics.mean_squared_error(y_test, y_pred)))
y_pred = dtc_model.predict(X_train)

print("RMSE Train: "+str(metrics.mean_squared_error(y_train, y_pred)))

# Correlations