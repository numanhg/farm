from sklearn.model_selection import train_test_split


def split_train_val_test(X, y, test_size=0.2, val_size=0.25, random_state=42):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
