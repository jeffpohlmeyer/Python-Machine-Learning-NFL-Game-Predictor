# Python Machine Learning NFL Game Predictor
An attempt to use scrapy to pull historical NFL game data and to use a supervised learning algorithm to attempt to predict the results of games


The algorithm will be developed using historical data, and then starting in week 5 the home team's average offensive and defensive stats, as well as the away team's stats, will be used to calculate a home team spread and point total.

The values used to train and eventually score the classifiers is differential values.  This was the simplest way to account for both teams' attributes.  I couldn't simply say that the home team's offensive yards was the same as the away team's defensive yards for these purposes as this could cause a matrix singularity issue.  Thus, using differential in the past, as well as calculating differentials in the predicting phase (taking the home team's average up to that point minus the away team's average) would account for both.

The first algo will be a simple binary classifier, telling us if the home team wins or loses.

The second algo will provide confidence levels that can be converted into spreads.  At first it seems that logistic regression is the way to go but a bit more research is needed.

The final algo will be a test of applying a neural network to the problem.  I don't yet know if/how I would be able to apply a neural net to the problem, but I want to first see just how well the first two steps work.