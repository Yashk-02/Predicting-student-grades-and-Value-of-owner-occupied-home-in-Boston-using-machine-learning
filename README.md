# Predicting-student-grades-using-machine-learning

The code present in the repository is written in R language.

The main purpose of this code is to Fit a multiple regression model to predict the response 
using all of the predictors and further applying different regularisation methods such as 
Ridge Regression(l2)
lasso (l1)
and Elastic net (l1 and l2)

Now each of the above regularisation approach along with predicting the response variable is performed using Support Vector Machine
and Multi layer perceptron(a type of supervised neural network) to further investigate which algorithm acts superior based on the quality of dataset.

The three algorithms along with regularisation approach is studied on
1. Boston dataset (built in dataset n R package(MASS))
2. Students Dataset, where target variable is to predict the students grade and analyse the influencing factors which correlates with the students grade.

Also, to train the dataset well and to estimate the model quality, k-cross validation is applied wherever appropriate.
