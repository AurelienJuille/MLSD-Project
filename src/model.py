import pickle
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline

"""
Prepare data for training

Returns:
    X_train: training data
    X_test: test data
    y_train: training labels
    y_test: test labels
"""
def prepare_data(df):
    X = df.drop(columns=["blueWin"]) # Features
    y = df["blueWin"] # Labels
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) # Split data
    
    # Standardize data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

"""
Find best K features using SelectKBest

Args:
    model: model to be used
    X_train: training data
    X_test: test data
    y_train: training labels
    y_test: test labels
    
Returns:
    selected_features_indices: indices of selected features
"""
def feature_selection(model, X_train, X_test, y_train, y_test):
    k = -1 
    max_score = 0
  
    for i in range(1, 16, 2):
        selector = SelectKBest(k=i)
        pipeline = Pipeline([('selector', selector), ('model', model)])
        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)
        
        if score > max_score:
            k = i
            max_score = score
            selected_features_indices = selector.get_support(indices=True)
    
    print(f"Selected {k} features")
    print(selected_features_indices)
    
    return X_train[:, selected_features_indices], X_test[:, selected_features_indices]

"""
Train model with RandomizedSearchCV

Args:
    model: model to be used
    param_grid: hyperparameters
    X_train: training data
    y_train: training labels
    
Returns:
    best_model: best model
    best_score: best score
    best_param: best hyperparameters
"""
def train_model_with_random_search(model, param_grid, X_train, y_train):
    cv = KFold(n_splits=5, shuffle=True, random_state=0)

    random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, 
                                       scoring='accuracy', n_jobs=-1, cv=cv)

    # Fit RandomizedSearchCV
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    best_score = random_search.best_score_
    best_param = random_search.best_params_

    return best_model, best_score, best_param

"""
Evaluate model with accuracy, f1 and roc_auc scores

Args:
    model: trained model
    X_test: test data
    y_test: test labels

Returns:
    accuracy: accuracy score
    f1: f1 score
    roc_auc: roc_auc score
"""
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    return accuracy, f1, roc_auc

if __name__ == '__main__':
    df = pd.read_csv("data/processed_dataset.csv", index_col=0)
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Model
    model = RandomForestClassifier()
    
    # Feature selection
    X_train, X_test = feature_selection(model, X_train, X_test, y_train, y_test)
    
    # Train model with RandomizedSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'criterion': ['gini'],
        'max_depth': [None, 3, 4, 5, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    trained_model, best_score, best_param = train_model_with_random_search(model, param_grid, X_train, y_train)
    
    # Save model
    pickle.dump(trained_model, open("model.pkl", "wb"))
    
    # Evaluate model
    accuracy, f1, roc_auc = evaluate_model(trained_model, X_test, y_test)
    