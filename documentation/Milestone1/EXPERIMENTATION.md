# Model training and experimentation

We used a methodical step by step approach to ensure good practices and results.
We selected Random Forest to make predictions because it is a powerful and reliable model that can handle the complex dataset and nature of our problem.

## Train-test split

Since there is plenty of data avaliable (24k samples), reserving **20% for the test set** is acceptable.

## Scaling

**Scaling** is an important step when working with ML models. In this case, we are using **Random Forest**, which is based on Decision Trees. Whith this type of model feature scaling is not as important, since DT use thresholds rather than distances to separate the data. However, it is still a good practice and can improve efficiency.

We used a **Standard Scaler**, mainly because our data does not have many **outliers** and they may also convey important information that we want to preserve.

## Feature selection

Even though Random Forest is a powerful model that can manage many features, it still takes advantage of the simplification of feature selection so that it can focus on the most important features. We implemented a function that, for different values of k, selects **the best k features** for the model to use.

## Hyper-Parameter tuning

For this, we used scipy's **RandomizedSearch** with a **5-split cross validation**. We tested the following hyperparameter values:

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'criterion': ['gini'],
    'max_depth': [None, 3, 4, 5, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
```

## Model Evaluation

We computed different metrics for evaluation of the model. Precisely:

- Accuracy
- F1 score
- Area under the ROC curvedocumentation/EXPERIMENTATION.md
## Results

With this setup, we achieved consistently an **accuracy of 75%** on the test set.

For feature selection, we imposed a **limit on 15 features** (out of 38), as we saw that the difference in performance was minimal once enough features were used. This way we ensure that data collection will be easier to implement without damaging performance.

We noticed that the model often selected both features like *redTeamTotalKills*, *blueTeamTotalKills* and summary features like *diffTotalKills*. We believe that if the model is choosing to include such redundant information is because it helps it focus more in the importance of that feature and the impact of the advantage of one team over another.

These are the **hyperparameters** chosen in one execution:

```Python
'n_estimators': 200,
'min_samples_split': 2,
'min_samples_leaf': 1,
'max_depth': None,
'criterion': 'gini',
'bootstrap': True
```