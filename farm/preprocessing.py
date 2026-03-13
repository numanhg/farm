from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import QuantileTransformer


def preprocess_data(X, variance_threshold=1e-5):
    """Preprocess features with variance filtering and quantile scaling."""
    vt = VarianceThreshold(threshold=variance_threshold)
    qt = QuantileTransformer()
    X_processed = vt.fit_transform(X)
    X_processed = qt.fit_transform(X_processed)
    return X_processed, vt, qt
