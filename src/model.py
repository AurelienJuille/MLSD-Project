import pickle
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline


def split_data(df):
    """
    Prepare data for training

    Returns:
        X_train: training data
        X_test: test data
        y_train: training labels
        y_test: test labels
    """

    X = df.drop(columns=["blueWin"]) # Features
    y = df["blueWin"] # Labels
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) # Split data
    
    return X_train, X_test, y_train, y_test


def feature_selection(model, X_train, y_train):
    """
    Select best features

    Args:
        model: model to be used
        X_train: training data
        y_train: training labels
        
    Returns:
        selected_features_indices: indices of selected features
    """

    # For the moment, no feature selection is done
    
    pass


def train_model_with_random_search(model, param_grid, X_train, y_train, cv):
    """
    Trains a model using RandomizedSearchCV to perform hyperparameter optimization.
    Uses cross-validation to evaluate performance and minimizes the negative log loss.
    Args:
        model: The machine learning model to be trained.
        param_grid (dict): The hyperparameters.
        X_train: The training input samples.
        y_train: The target values (class labels) for the training data.
    Returns:
        tuple: A tuple containing:
            - best_model: The model with the best hyperparameters.
            - best_param (dict): The best hyperparameter combination found during the search.
    Prints:
        - The best training score achieved during the search.
        - The best hyperparameter combination.
    """

    print("Starting training...")

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10,
                                       scoring='neg_log_loss', n_jobs=-1, cv=cv)

    # Fit RandomizedSearchCV
    random_search.fit(X_train, y_train)

    print("Training completed")
    print()

    # Get best estimator
    best_model = random_search.best_estimator_
    # Get best training score
    best_score = random_search.best_score_
    print('Best Training Score: ', best_score)
    # Get best param
    best_param = random_search.best_params_
    print('Best Parameters: ', best_param)
    print()

    # Return model
    return best_model, best_param


def compute_log_loss(model, X_test, y_test):
    """
    Computes the log loss of a model's predictions on a dataset.
    Args:
        model: A trained classification model.
        X_test: The test data
        y_test: The test labels
    Returns:
        float: The computed log loss value.
    Prints:
        - The log loss value to the console.
    """

    print("Model Evaluation: ")

    # Predict on test set
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Compute Log Loss
    loss = log_loss(y_test, y_pred_prob)
    print('Log Loss: ', loss)
    print()

    return loss


if __name__ == '__main__':

    # Load data
    df = pd.read_csv("data/processed_dataset.csv", index_col=0)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df)

    # Preprocess data
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Define model and hyperparameters
    param_grid = {
        'n_estimators': [200, 300, 500],
        'criterion': ['entropy'],
        'max_depth': [None, 3, 4, 5, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    model = RandomForestClassifier(n_estimators=300,
                                criterion='entropy',
                                max_depth=10,
                                min_samples_split=5,
                                min_samples_leaf=2,
                                n_jobs=-1,
                                random_state=0)


    # Define cross-validation strategy
    cv = KFold(n_splits=5, shuffle=True, random_state=0)

    # Train the model
    trained_model, param_sample = train_model_with_random_search(model, param_grid, X_train, y_train, cv)
    
    # Evaluate model
    loss = compute_log_loss(trained_model, X_test, y_test)

    # Save model
    pickle.dump(trained_model, open("model.pkl", "wb"))
    pickle.dump(scaler, open("scaler.pkl", "wb"))
    