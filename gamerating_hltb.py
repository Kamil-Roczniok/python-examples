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
df["old_game"] = df["release_year"].dt.year < 2016
df["old_game"] = pd.to_numeric(df["old_game"], errors="coerce").astype("Int64") #games before 2015 yes/no because more acceptance for old games before witcher3,gta5, bloodborne etc

df = df[~(df[cols] == 0).any(axis=1)] #remove a row when value is missing
df = df[df["game_type"] == "game"] #train only with game type, because sport/endless etc are not relevant
df = df[df["list_replay"] == 0] #train only with game type, because sport/endless etc are not relevant

df["comp_all"] = pd.to_numeric(df["comp_all"], errors="coerce")
df["list_comp"] = pd.to_numeric(df["list_comp"], errors="coerce").astype("Int64")
df["review_score_g"] = pd.to_numeric(df["review_score_g"], errors="coerce").astype("Int64")
df = df.dropna(subset=["comp_all", "list_comp", "review_score_g", "old_game"])

# features and target (iris-like: X numeric matrix, y class labels)
X = df[["comp_all", "review_score_g", "old_game"]].to_numpy()
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
model = SVC(kernel='rbf', probability=True, random_state=42)

#Train the model using the training data
model.fit(X_train_scaled, y_train)

#Make predidctions on the test set
y_pred = model.predict(X_test_scaled)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test,y_pred)

# The longer the game, the better its rating must be in order to complete it. Example: a game lasting about 50 hours and a rating of 90 always match, but a game lasting about 50 hours and a rating of 75 does not.
test_object = [[185400, 90, 1]]
test_scaled = scaler.transform(test_object)          # <- scale before predict
predict = model.predict(test_scaled)

print(f"\nAn old game ~50 hours and 90 rating:", predict)

test_object = [[72000, 75, 1]]
test_scaled = scaler.transform(test_object)          # <- scale before predict
predict = model.predict(test_scaled)

print(f"\nAn old game ~20 hours and 75 rating:", predict)

test_object = [[90000, 78, 0]]
test_scaled = scaler.transform(test_object)          # <- scale before predict
predict = model.predict(test_scaled)

print(f"\nA new game ~30 hours and 75 rating:", predict)

test_object = [[100000, 88, 0]]
test_scaled = scaler.transform(test_object)          # <- scale before predict
predict = model.predict(test_scaled)

print(f"\nA new game ~30 hours and 90 rating:", predict)

test_object = [[108000, 65, 1]]
test_scaled = scaler.transform(test_object)          # <- scale before predict
predict = model.predict(test_scaled)

print(f"\nAn old game ~30 hours and 65 rating:", predict)

# get top 10 games from my backlog I may probably complete
games = []
for item in data['lines']:
    release_year = pd.to_datetime(item["release_world"], errors="coerce")
    is_old_game_int = int (release_year.year < 2016)

    # is_old_game_int = 1

    if item['game_type'] == "game" and item['list_comp'] == 0 and item['list_replay'] == 0:
        test_object = [[item["comp_all"], item["review_score_g"], is_old_game_int]]
        test_scaled = scaler.transform(test_object)  # <- scale before predict
        predict = model.predict(test_scaled)
        if predict == 1:
            games.append(item)

games.sort(key=lambda g: g["review_score_g"], reverse=True)
for item in games:
    print(f'You should try to play this game:', item['custom_title'], ' rating: ', item['review_score_g'],
          ' time to beat: ~', item['comp_all'] / 3600, 'hours')