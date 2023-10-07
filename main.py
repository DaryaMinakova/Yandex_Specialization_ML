import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


penguins_data = pd.read_csv("penguins.csv")
columns_name = penguins_data.columns.tolist()
penguins_data.dropna(subset=columns_name, inplace=True)
labels = penguins_data["species"]
features = penguins_data[["bill_length_mm", "bill_depth_mm"]]

train_features, test_features, train_labels, test_labels = train_test_split(
            features, labels, test_size=0.2, random_state=123)

best = -1
worth = 1
for i in range(1, 11):
    for weight in ["uniform", "distance"]:
        knn = KNeighborsClassifier(n_neighbors=i, weights=weight)
        knn.fit(train_features, train_labels)

        features_predicted = knn.predict(test_features)

        def accuracy(y_true, y_pred):
            return np.sum(y_true == y_pred) / len(y_true)

        ac = accuracy(test_labels, features_predicted)
        if ac > best:
            best = ac
        if ac < worth:
            worth = ac

print("Best accuracy: {:.6f}".format(best))
print("Worst accuracy: {:.6f}".format(worth))