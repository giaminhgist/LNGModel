"""Model factory: classical ML baselines and the basic FNN.

Classical methods mirror the traditional baselines of the EMS benchmark
(EDB_* / ESR_* in the paper) plus standard additions. All sklearn models
receive standardized features (median imputation + StandardScaler fitted on
the training subjects only). The run seed is threaded through every estimator
with a random_state (SVC, RF, LR, PCA, QDA) — no module-level SEED constant.

FNN: a basic feed-forward network 128 -> 64 -> 32 (BN/ReLU/Dropout 0.3),
trained by the shared generic trainer under the unified Phase-3 checkpoint
policy (best outer-validation AUC, earliest tie).
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# --------------------------------------------------------------------------- #
# Classical ML
# --------------------------------------------------------------------------- #

def make_ml_pipeline(method, seed):
    """Return a sklearn Pipeline (impute -> scale -> estimator) per method.

    `seed` reaches every stochastic estimator (SVC Platt scaling, RF, LR,
    PCA, QDA). Deterministic estimators (GNB, KNN) ignore it — identical
    results across seeds are valid for those methods and recorded as such.
    """
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    if method == "svm_rbf":
        est = SVC(C=1.0, kernel="rbf", gamma="scale", probability=True,
                  random_state=seed)
    elif method == "svm_linear":
        est = SVC(C=1.0, kernel="linear", probability=True, random_state=seed)
    elif method == "rf":
        # n_jobs=1: deterministic given random_state and worker-friendly
        # (the matrix runner runs several single-threaded workers in parallel)
        est = RandomForestClassifier(n_estimators=500, random_state=seed,
                                     n_jobs=1)
    elif method == "qda":
        # QDA requires n_samples > n_features per class; with 91 features and
        # ~60 train samples per class we first reduce to 20 PCA components.
        from sklearn.decomposition import PCA
        return Pipeline([("imputer", imputer), ("scaler", scaler),
                         ("pca", PCA(n_components=20, random_state=seed)),
                         ("est", QuadraticDiscriminantAnalysis(reg_param=0.5))])
    elif method == "gnb":
        est = GaussianNB()
    elif method == "lr":
        est = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    elif method == "lr_l1":
        est = LogisticRegression(C=1.0, penalty="l1", solver="liblinear",
                                 max_iter=2000, random_state=seed)
    elif method == "knn":
        est = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError(f"unknown method {method}")
    return Pipeline([("imputer", imputer), ("scaler", scaler), ("est", est)])


def fit_predict_ml(method, X_train, y_train, X_eval, seed):
    """Fit ONCE on train, return (pipe, prob_train, prob_eval).

    The pipeline is fitted exactly once; train and eval probabilities come
    from the same fit (needed for the fit_complete train/val metric record).
    """
    pipe = make_ml_pipeline(method, seed)
    pipe.fit(np.asarray(X_train, dtype=np.float64), np.asarray(y_train))
    if hasattr(pipe["est"], "predict_proba"):
        p_tr = pipe.predict_proba(np.asarray(X_train, dtype=np.float64))[:, 1]
        p_ev = pipe.predict_proba(np.asarray(X_eval, dtype=np.float64))[:, 1]
    else:
        p_tr = pipe.decision_function(np.asarray(X_train, dtype=np.float64))
        p_ev = pipe.decision_function(np.asarray(X_eval, dtype=np.float64))
    return pipe, p_tr, p_ev


# --------------------------------------------------------------------------- #
# Basic FNN
# --------------------------------------------------------------------------- #

class BasicFNN(nn.Module):
    """Basic feed-forward network: Linear -> BN -> ReLU -> Dropout blocks.

    Input  : (batch, in_dim)  — subject-level hand-crafted feature vector
    Hidden : 128 -> 64 -> 32 with BatchNorm1d / ReLU / Dropout(0.3)
    Output : (batch, 1) sigmoid — P(SZ)
    """

    def __init__(self, in_dim, hidden=(128, 64, 32), dropout=0.3):
        super().__init__()
        layers, d_in = [], in_dim
        for d_out in hidden:
            layers += [nn.Linear(d_in, d_out), nn.BatchNorm1d(d_out),
                       nn.ReLU(), nn.Dropout(dropout)]
            d_in = d_out
        layers += [nn.Linear(d_in, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, y=None):
        # y argument ignored; the shared trainer passes it, the model only
        # needs x. aux keys mirror the proposal contract.
        return torch.sigmoid(self.net(x)), {"emb": None}
