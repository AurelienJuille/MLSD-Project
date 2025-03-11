# Review on previous work: Experimentation

After the first milestone, we used the feedback provided by the teachers to better refine our work. In this revision, we motivate more our decisions and made some important changes.

## Model Evaluation

Since the output of our predictions is going to be a probability rather than a binary classification, we chose log-loss (cross-entropy loss) as the main metric that we are going to test for. Log loss directly penalizes confident incorrect predictions and evaluates how well the predicted probabilities align with the actual outcomes. It is particularly useful when you need to judge the quality of probability estimates rather than just the classification accuracy.
This ensures that the model not only distinguishes between wins and losses but also provides reliable probability estimates.

We still kept different other metrics for the record, but our decisions are not based on them. Precisely:

- Accuracy
- F1 score
- Area under the ROC curve
- ROC curve plot

## Addressing feature importance

Tree-based models have a built-in feature that lets us rank and visualize the importance that the model assigned to each feature after training to make its predictions, providing insight into which features are more relevant. We have to keep in mind though that this ranking is not objective and can change if some correlated features are removed making the the model rely more in others.
We now plot this ranking at the end of each training.

## Feature selection

We are using a very powerful model that can manage very well a large number of features. For this reason, we compared the results of performing or not different kinds of feature selection to justify this choice. Also, we included another function to perform feature selection using Recursive Feature Elimination with Cross Validation.

### Results

More detailed results are avaliable. This is a summary:

| Method | Log-loss | Number of features |
|----------|----------|----------|
| No feature selection  | 0.4860   | All - 38 features |
| Using RFECV (step=3)   | 0.4865 | 29 |
| Using K-best (step=3)  | 0.4866 | 31 |
| Keeping only 6 most important features and related ones   | 0.4875 | 6x3=18 |
| Keeping only 3 most important features and related ones <sup>1</sup>   | 0.4965 | 3x3=9 |


<sup>1</sup> In this example, the chosen features are: 
       'diffTotalGold', 'diffXp', 'diffTotalKills',
       'blueTeamTotalGold', 'redTeamTotalGold', 
       'redTeamXp', 'blueTeamXp',
       'blueTeamTotalKills','redTeamTotalKills',

### Conclusions

It appears that feature selection **does not affect significantly** the model performance. A more precise statistical test would be needed to determine if the observed variations in the log-loss score of the table above is statistically significant or not. However, even if the values indeed were significant, it is still a **very small difference** (0.0105 in the worst case) and will probably not make a big difference in the quality of our predictions.

For this reason, we conclude that **we will use feature selection** if the simplification of the data gathering process adds any value (faster response times, smoother experience for the user if imputting manually); and **will not use it** if the integration with the Riot API allows for easy acess to all the features.

## Other motivations

### Why use 'entropy' over 'gini' as the DT split criteria?

Entropy usually produces more balanced splits, at the cost of computational complexity. For this use case, we are not in need of very low latencies and the difference in response time is not significant.

### Why add the "diff" features if they are redundant information?

These features might add value to the model by highlighting the advantage of one team over the other in that category. Although these relations can still be learnt by the model without adding these features, they help the model capture that pattern more easily, reducing complexity.

Adding the features decreases the model's log-loss from ***0.4898*** to ***0.4860*** (0.0038 difference). Although it is not a big difference at all and it may not be statistically significant, that paired with the fact that automatic feature selection did not remove those features and they are computationally easy to compute makes enough evidence to keep them.