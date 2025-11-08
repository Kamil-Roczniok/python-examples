import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#data from how long to beat
# {
#   "lines": [
#     {"game_type": "game",    "comp_all": "180000",  "review_score_g": 90, "list_comp": 1},
#     {"game_type": "endless", "comp_all": 75000,   "review_score_g": 75, "list_comp": 0}
#   ]
# }

with open("data.json") as f:
    data = json.load(f)

df = pd.json_normalize(data, record_path="lines")
cols = ["comp_all","review_score_g"]  # adjust

df["release_year"] = pd.to_datetime(df["release_world"], errors="coerce")
df["release_year"] = df["release_year"].dt.year.astype("Int64")

df = df[~(df[cols] == 0).any(axis=1)]
df = df[df["game_type"] == "game"]


df["comp_all"] = pd.to_numeric(df["comp_all"], errors="coerce")
df["list_comp"] = pd.to_numeric(df["list_comp"], errors="coerce").astype("Int64")
df["review_score_g"] = pd.to_numeric(df["review_score_g"], errors="coerce").astype("Int64")
df = df.dropna(subset=["comp_all", "list_comp", "review_score_g", "release_year"])

# features and target (iris-like: X numeric matrix, y class labels)
X = df[["comp_all", "review_score_g", "release_year"]].to_numpy()
y = df["list_comp"].astype(int).to_numpy()

# optional train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Initialize the scaler
scaler = StandardScaler()

#Fit the scaler on the training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

#Transforming the test data using the same scaler
X_test_scaled = scaler.transform(X_test)

#Initialise the Support Vector Classifier
model = SVC(kernel='linear',  probability=True, random_state=42)

#Train the model using the training data
model.fit(X_train_scaled, y_train)

#Make predidctions on the test set
y_pred = model.predict(X_test_scaled)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test,y_pred)

# The longer the game, the better its rating must be in order to complete it. Example: a game lasting about 50 hours and a rating of 90 always match, but a game lasting about 50 hours and a rating of 75 does not.
test_object = [[185400, 90, 2015]]
test_scaled = scaler.transform(test_object)          # <- scale before predict
predict = model.predict(test_scaled)

print(f"\nA game ~50 hours and 90 rating:", predict)

test_object = [[72000, 75, 2020]]
test_scaled = scaler.transform(test_object)          # <- scale before predict
predict = model.predict(test_scaled)

print(f"\nA game ~20 hours and 75 rating:", predict)

test_object = [[108000, 65, 2013]]
test_scaled = scaler.transform(test_object)          # <- scale before predict
predict = model.predict(test_scaled)

print(f"\nA game ~30 hours and 75 rating:", predict)

test_object = [[108000, 65, 1998]]
test_scaled = scaler.transform(test_object)          # <- scale before predict
predict = model.predict(test_scaled)

print(f"\nAn old game ~30 hours and 65 rating:", predict)