import numpy as np
import re
import pandas as pd
import json

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import zscore
from sympy.physics.quantum import L2

#data from how long to beat
# {
#   "lines": [
#     {"game_type": "game",    "comp_all": "180000",  "review_score_g": 90, "list_comp": 1},
#     {"game_type": "endless", "comp_all": 75000,   "review_score_g": 75, "list_comp": 0}
#   ]
# }

#genres = ['Fighting', 'Shooter','Isometric','First-person shooter','Platform', 'Strategy', 'Open World','Role-Playing', 'Horror', 'Point-and-Click', 'Third-Person','Hack and Slash', 'Sports', 'City-Building', 'Survival', 'First-Person', 'Management','Virtual Reality', 'Stealth', 'Racing/Driving', 'Turn-Based', 'Tactical', 'Sandbox']
genres = []

with open("data_new.json") as f:
    data = json.load(f)

df = pd.json_normalize(data, record_path="lines")
cols = ["comp_all","review_score_g"]  # adjust

df = df[~(df[cols] == 0).any(axis=1)] #remove a row when value is missing
df = df[df["game_type"] == "game"] #train only with game type, because sport/endless etc are not relevant
df = df[df["list_replay"] == 0] #train only with game type, because sport/endless etc are not relevant

df["comp_all"] = pd.to_numeric(df["comp_all"], errors="coerce")
df["list_comp"] = pd.to_numeric(df["list_comp"], errors="coerce").astype("Int64")
df["review_score_g"] = pd.to_numeric(df["review_score_g"], errors="coerce").astype("Int64")
df['review_score_g'].dropna()
df['comp_all'].fillna(df['comp_all'].median(), inplace=True)
z_scores = zscore(df["comp_all"])

# ensure numeric and handle NaNs
vals = pd.to_numeric(df['comp_all'], errors='coerce')
z_scores = zscore(vals, nan_policy='propagate') if hasattr(zscore, 'nan_policy') else (vals - vals.mean())/vals.std(ddof=0)

# ensure comp_all is numeric
vals = pd.to_numeric(df['comp_all'], errors='coerce')

# compute z-scores (handle constant column)
if vals.std(ddof=0) == 0 or vals.isna().all():
    df_filtered = df.copy()  # nothing to filter
else:
    z = (vals - vals.mean()) / vals.std(ddof=0)
    df_filtered = df.loc[z.abs() <= 3].copy()

# optional: reset index
df_filtered.reset_index(drop=True, inplace=True)
df = df_filtered
# for name, z in zip(df['custom_title'], z_scores):
#     # if (float(z) > 3 ):
#     print(name, np.nan if pd.isna(z) else float(z))


# features and target (iris-like: X numeric matrix, y class labels)
X = df[["comp_all", "review_score_g"]].to_numpy()
y = df["list_comp"].astype(int).to_numpy()

# optional train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


#Initialize the scaler
scaler = StandardScaler()

#Fit the scaler on the training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

#Transforming the test data using the same scaler
X_test_scaled = scaler.transform(X_test)

#Initialise the Support Vector Classifier
model = SVC(kernel='rbf', probability=True, random_state=42)

#Train the model using the training data
model.fit(X_train_scaled, y_train)

#Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test,y_pred)

print(f'Accuracy:', accuracy)

#Get detailed classification report
print("\nClassification Report:")
print(classification_report(y_test,y_pred))

#Define a aparameter grid for hyperparameter tuning
param_grid = {'C': [0.1,1,10], 'kernel': ['linear', 'rbf']}

#Initialize GridSearchCV with the SVC model and parameter grid
grid_search = GridSearchCV(SVC(), param_grid,cv=5)

#Fit the grid search to the training data
grid_search.fit(X_train_scaled, y_train)

#Get the best parameters
print("Best parameters:", grid_search.best_params_)

#Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", np.mean(cv_scores))


# The longer the game, the better its rating must be in order to complete it. Example: a game lasting about 50 hours and a rating of 90 always match, but a game lasting about 50 hours and a rating of 75 does not.
# test_object = [[185400, 90, 1]]
# test_scaled = scaler.transform(test_object)          # <- scale before predict
# predict = model.predict(test_scaled)
#
# print(f"\nAn old game ~50 hours and 90 rating:", predict)
#
# test_object = [[72000, 75, 1]]
# test_scaled = scaler.transform(test_object)          # <- scale before predict
# predict = model.predict(test_scaled)
#
# print(f"\nAn old game ~20 hours and 75 rating:", predict)
#
# test_object = [[90000, 78, 0]]
# test_scaled = scaler.transform(test_object)          # <- scale before predict
# predict = model.predict(test_scaled)
#
# print(f"\nA new game ~30 hours and 75 rating:", predict)
#
# test_object = [[100000, 88, 0]]
# test_scaled = scaler.transform(test_object)          # <- scale before predict
# predict = model.predict(test_scaled)
#
# print(f"\nA new game ~30 hours and 90 rating:", predict)
#
# test_object = [[108000, 65, 1]]
# test_scaled = scaler.transform(test_object)          # <- scale before predict
# predict = model.predict(test_scaled)
#
# print(f"\nAn old game ~30 hours and 65 rating:", predict)

# choose the 10 best games from my backlog list that I'll probably manage to finish
games = []
for item in data['lines']:
    if item['game_type'] == "game" and item['list_comp'] == 0 and item['list_replay'] == 0:
        test_object = [[item["comp_all"], item["review_score_g"]]]
        test_scaled = scaler.transform(test_object)  # <- scale before predict
        predict = model.predict(test_scaled)
        if predict == 1:
            games.append(item)

games.sort(key=lambda g: g["review_score_g"], reverse=True)
i = 0;
for item in games:
    if i < 100:
        print(f'You should try to play this game:', item['custom_title'], ' rating: ', item['review_score_g'], ' time to beat: ~', item['comp_all'] / 3600, 'hours')
    i = i + 1