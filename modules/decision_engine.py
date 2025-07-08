from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def train_models(X_train, y_train):
    models = {
        "DecisionTree": DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    }

    best_model = None
    best_auc = 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_train)[:, 1]
        auc = roc_auc_score(y_train, y_proba)
        print(f"{name} AUC: {auc:.4f}")
        if auc > best_auc:
            best_model = model
            best_auc = auc

    return best_model