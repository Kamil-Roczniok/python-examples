import numpy as np
import pandas as pd
import json

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import zscore
from sklearn.pipeline import Pipeline
from sklearn.base import clone

#data from how long to beat
# {
#   "lines": [
#     {"game_type": "game",    "comp_all": "180000",  "review_score_g": 90, "list_comp": 1},
#     {"game_type": "endless", "comp_all": 75000,   "review_score_g": 75, "list_comp": 0}
#   ]
# }

#genres = ['Fighting', 'Shooter','Isometric','First-person shooter','Platform', 'Strategy', 'Open World','Role-Playing', 'Horror', 'Point-and-Click', 'Third-Person','Hack and Slash', 'Sports', 'City-Building', 'Survival', 'First-Person', 'Management','Virtual Reality', 'Stealth', 'Racing/Driving', 'Turn-Based', 'Tactical', 'Sandbox']
genres = ['First-person shooter', 'Open World','Role-Playing','Turn-Based', 'Isometric']
#genres = []

with open("data_new_with_genres.json") as f:
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

# compute z-scores (handle constant column)
if vals.std(ddof=0) == 0 or vals.isna().all():
    df_filtered = df.copy()  # nothing to filter
else:
    z = (vals - vals.mean()) / vals.std(ddof=0)
    df_filtered = df.loc[z.abs() <= 3].copy()

# optional: reset index
df_filtered.reset_index(drop=True, inplace=True)
df = df_filtered

cols = ["comp_all", "review_score_g"] + genres
# features and target (iris-like: X numeric matrix, y class labels)
X = df[cols].to_numpy()
y = df["list_comp"].astype(int).to_numpy()

# KFold (stratified recommended for imbalanced classes)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# model pipeline: scaler is inside pipeline so it's fit on TRAIN fold only each split
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", probability=True, random_state=42))
])

fold_scores = []
fold_idx = 0
for train_idx, val_idx in kf.split(X, y):
    fold_idx += 1
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # clone pipeline to have a fresh estimator (not strictly required if you call fit)
    pipe = clone(pipeline)
    pipe.fit(X_train, y_train)                 # scaler.fit only on X_train
    y_val_pred = pipe.predict(X_val)           # scaler.transform on X_val uses training fit
    acc = accuracy_score(y_val, y_val_pred)
    fold_scores.append(acc)
    print(f"Fold {fold_idx} accuracy: {acc:.4f}")

print(f"Mean CV accuracy: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")
final_pipe = clone(pipeline)
final_pipe.fit(X, y)

def predict_single(item):
    comp = float(item.get("comp_all", np.nan) or 0.0)
    review = float(item.get("review_score_g", np.nan) or 0.0)
    genre_flags = [1.0 if item.get(g) else 0.0 for g in genres]
    feat = np.array([[comp, review] + genre_flags], dtype=float)
    return final_pipe.predict(feat), final_pipe.predict_proba(feat)[:, 1]

# choose the 10 best games from my backlog list that I'll probably manage to finish
games = []
for item in data['lines']:
     if item['game_type'] == "game" and item['list_comp'] == 0 and item['list_replay'] == 0:
        prediction = predict_single(item)
        # print(prediction)
        if prediction[0] == 1:
            games.append(item)

games.sort(key=lambda g: g["review_score_g"], reverse=True)
i = 0;
for item in games:
    if i < 100:
        print(f'You should try to play this game:', item['custom_title'], ' rating: ', item['review_score_g'], ' time to beat: ~', round(item['comp_all'] / 3600, 2), 'hours ', item['genre'])
    i = i + 1