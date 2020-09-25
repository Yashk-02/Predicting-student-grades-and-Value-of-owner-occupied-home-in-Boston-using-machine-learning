# R script for the support vector machine classifier on the Boston dataset
# installing the caret package may take a few minutes depending on the machine
# install.packages('caret')
# install.packages('sparseSVM')
# install.packages('kernlab')
# install.packages('caTools')
# install.packages('readxl')


library(MASS)
library(e1071)
library(sparseSVM)
library(caTools)
library(caret)
library(kernlab)

# Splitting the data into training and testing
splitter <- sample.split(Boston$medv, SplitRatio = 0.75)

train_data <- subset(Boston, splitter == TRUE)
test_data <- subset(Boston, splitter == FALSE )

x_train = train_data[,-14]
y_train = train_data$medv
x_test = test_data[,-14]
y_test = test_data$medv

# alternative method for cross validation if the implemented measure due to version differences
# training_model <- trainControl(method='cv', number = 10)
# model_fit <- train(x_train, y_train, trControl=training_model, method="svmRadial")

# Fitting the model and obtaining predictions
model_fit <- svm(x_train, y_train, cross = 10)

predictions <- predict(model_fit, x_test)

len_predictions <- length(predictions)

x <- seq(1, len_predictions, 1)

plot(x, predictions, main = "Comparision of predicted (red line) vs actual (green line) values ", type = "l", col = "red")
lines(x, y_test, col = "green")
legend(1, 40, legend=c("Predicted", "Actual"),
       col=c("red", "green"), lty=1:2, cex=0.6)


# Statistics calculated for regression problem 
train_rmse <- RMSE(predictions, y_train)
rmse <- RMSE(predictions, y_test)

train_mse <- train_rmse^2
mse <- rmse^2

# Calculations for the r squared metric
train_ssr <- sum((predictions - y_train)^2)
train_sst <- sum((y_train - mean(y_train))^2)
train_rsquared <- 1 - (train_ssr / train_sst)

ssr <- sum((predictions - y_test)^2)
sst <- sum((y_test - mean(y_test))^2)
rsquared <- 1 - (ssr / sst)

# Printing the calculated statistics
print("Training RMSE")
print(train_rmse)

print("Test RMSE")
print(rmse)

print("Training MSE")
print(train_mse)

print("Test MSE")
print(mse)

print("Training R Squared value")
print(train_rsquared)

print("R Squared Value")
print(rsquared)

copy_x_train <- x_train
copy_y_train <- y_train
copy_x_test <- x_test
copy_y_test <- y_test

# threshold used to convert regression problem into classification problem 
threshold <- mean(Boston$medv)

# Mapping the problem to a binary classification problem
copy_y_train[copy_y_train < threshold] <- 0
copy_y_train[copy_y_train >= threshold] <- 1

copy_y_test[copy_y_test < threshold] <- 0
copy_y_test[copy_y_test >= threshold] <- 1
copy_y_train <- unlist(copy_y_train)

copy_x_train <-  matrix(unlist(copy_x_train), nrow = nrow(copy_x_train), ncol = ncol(copy_x_train) )
copy_x_test <- matrix(unlist(copy_x_test), nrow = nrow(copy_x_test), ncol = ncol(copy_x_test))


# Fit the model for the lasso regularization and obtain predictions
lasso_fit <- cv.sparseSVM(copy_x_train, copy_y_train, alpha = 1, nfolds = 10)

lasso_predictions <- predict(lasso_fit, copy_x_test, type = 'class')


lasso_confusion_matrix <- confusionMatrix(factor(lasso_predictions), 
                factor(copy_y_test))

# Print the resulting confusion matrix
print("Lasso confusion matrix")
print(lasso_confusion_matrix)

# Fit the model for the elastic net, where the value for alpha is 0.5
elastic_fit <- cv.sparseSVM(copy_x_train, copy_y_train, alpha = 0.5, nfolds = 10)

elastic_predictions <- predict(elastic_fit, copy_x_test, type = 'class')

# Get the resulting confusion matrix
elastic_net_confusion_matrix <- confusionMatrix(factor(elastic_predictions), 
                factor(copy_y_test))

# Print the resulting confusion matrix
print("Elastic Net confusion matrix")
print(elastic_net_confusion_matrix)