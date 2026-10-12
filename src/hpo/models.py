"""Build a model from decoded settings and score it (validation AUC)."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

from . import config


def build(model: str, params: dict):
    """Create an untrained SVM, Random Forest or XGBoost model with the given settings."""
    if model == "svm":
        # RBF fixed. No probability=True: it would run a hidden 5-fold CV per fit.
        return SVC(kernel="rbf", C=params["C"], gamma=params["gamma"],
                   cache_size=500, random_state=config.MODEL_SEED)
    if model == "rf":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"], max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            max_features=params["max_features"],
            n_jobs=1, random_state=config.MODEL_SEED)
    if model == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=params["n_estimators"], max_depth=params["max_depth"],
            learning_rate=params["learning_rate"], subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            tree_method="hist", n_jobs=1, random_state=config.MODEL_SEED,
            eval_metric="logloss", verbosity=0)
    raise ValueError(model)


def score(model: str, est, X_val, y_val) -> float:
    """Validation AUC. SVM uses decision_function; trees use predicted probability."""
    if model == "svm":
        s = est.decision_function(X_val)
    else:
        s = est.predict_proba(X_val)[:, 1]
    return float(roc_auc_score(y_val, s))


def fit_and_score(model: str, params: dict, data) -> float:
    """Train the model on the training data and return its AUC on the validation data."""
    est = build(model, params)
    est.fit(data.X_train, data.y_train)
    return score(model, est, data.X_val, data.y_val)
