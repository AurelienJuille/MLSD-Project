# Data Analysis

For the data analysis, we followed a systematic **step by step approach** to guarantee clarity and good results.
We first loaded the dataset and assigned names to each feature.

## Exploring basic attributes

At first glance, we see that the data is a **24000 x 29 table with integer values**. Most of them represent **counts** even though some features have very few possible values in practice. The output column is **categorical**, either a 1 ("blue wins") or a 0 ("red wins).

## Handling duplicates and missing values

We noticed that the last column does not contain any information and is merely a marker of the end of the tuple. For this reason we removed it.

Even though duplicates in this context **could be meaningful** (multiple matches with the same statistics at 15 minutes), it is **not the case** because the *matchID* is also duplicated, meaning that the repeated samples are referring to the same match. For this reason, we **removed** them.

There are no missing values in this dataset.

## Exploring data distribution

We checked that the dataset is in fact **balanced**, with roughly half the samples belonging to one class and the other half to the other. This is expected because belonging to either the blue or the red team in a match is a random event with 50% chance.

We used a data analysis tool called Y-Data profiling to perform an exhaustive statistical and correlation analysis both to gain insight on the distribution and properties of features and have a record of the data used for documentation.

## Handling outliers

We noticed that outliers are not very sparse and carry significant information. For this reason, we decided to keep them and only mitigate their impact through data scaling.

## Creating new features

Almost all of the observations have one feature for the blue team and another for the red team. For this reason, it is easy to create a **summary feature** that encapsulates some of the information provided by both features by computing the difference among both teams.

These features might not seem very valuable at first because they only introduce redundancy. However, we believe they can be useful for the model after feature selection by reducing dimensionality and simplifying the amount of features needed.
