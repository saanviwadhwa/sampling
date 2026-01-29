

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, NearMiss


data = pd.read_csv("C:/Users/saanv/Downloads/sampling/Creditcard_data.csv")

features = data.iloc[:, :-1]
target = data.iloc[:, -1]

print("Class distribution before sampling:")
print(target.value_counts())

sampling_methods = [
    ("Sampling1_Under", RandomUnderSampler(random_state=1)),
    ("Sampling2_Over", RandomOverSampler(random_state=1)),
    ("Sampling3_SMOTE", SMOTE(random_state=1)),
    ("Sampling4_ADASYN", ADASYN(random_state=1)),
    ("Sampling5_NearMiss", NearMiss())
]

model_list = [
    ("M1_SVM", SVC()),
    ("M2_Logistic", LogisticRegression(max_iter=1200)),
    ("M3_DecisionTree", DecisionTreeClassifier()),
    ("M4_KNN", KNeighborsClassifier()),
    ("M5_RandomForest", RandomForestClassifier())
]

accuracy_table = pd.DataFrame(
    index=[m[0] for m in model_list],
    columns=[s[0] for s in sampling_methods]
)

for samp_name, sampler in sampling_methods:

    X_bal, y_bal = sampler.fit_resample(features, target)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_bal, y_bal, test_size=0.3, random_state=10
    )

    for model_name, clf in model_list:
        clf.fit(X_tr, y_tr)
        predictions = clf.predict(X_te)
        score = accuracy_score(y_te, predictions)

        accuracy_table.loc[model_name, samp_name] = round(score * 100, 2)


print("\nAccuracy Matrix:\n")
print(accuracy_table)

accuracy_table.to_csv("final_sampling_results.csv")
print("\nSaved as final_sampling_results.csv")
