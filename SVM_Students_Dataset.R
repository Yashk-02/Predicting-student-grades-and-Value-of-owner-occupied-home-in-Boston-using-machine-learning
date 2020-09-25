# R script for the support vector machine classifier on the Students grades dataset
# installing the caret package may take a few minutes depending on the machine

# install.packages('caret')
# install.packages('sparseSVM')
# install.packages('kernlab')
# install.packages('caTools')
# install.packages('readxl')

library("readxl")
library(caTools)
library(MASS)
library(e1071)
library(sparseSVM)
library(caTools)
library(caret)
library(kernlab)

# Read in student dataset, this can be done without a file path providing that the file is in the same directory as r library
students_data <- read.csv("Student_Grades_Data_Cleaned.csv")

# create dummy variables(issues arise with character variables)
dmy <- dummyVars(" ~ .", data = students_data)
students <- data.frame(predict(dmy, newdata = students_data))

# Splitting the training/testing data
splitter <- sample.split(students$G3, SplitRatio = 0.75)

train_data <- subset(students, splitter == TRUE)
test_data <- subset(students, splitter == FALSE)

# allocating the training and testing data
x_train = train_data[,-44]
y_train = train_data$G3
x_test = test_data[,-44]
y_test = test_data$G3

# train the model and obtain predictions
model_fit <- svm(x_train, y_train, cross = 10)

predictions <- predict(model_fit, x_test)

# plot the predictions alongisde actual values
len_predictions <- length(predictions)

x <- seq(1, len_predictions, 1)

plot(x, predictions, main = "Comparision of predicted (red line) vs actual values (green line) ", type = "l", col = "red")
lines(x, y_test, col = "green")
# text(locator(), labels = c("red line", "green line)"))
legend(1, 5, legend=c("Predicted", "Actual"),
       col=c("red", "green"), lty=1:2, cex=0.6)

# Statistics calculated for regression problem 
train_rmse <- RMSE(predictions, y_train)
rmse <- RMSE(predictions, y_test)

train_mse <- train_rmse^2

mse <- rmse^2

mae <- sum(mean(abs(predictions - y_test)))

train_ssr <- sum((predictions - y_train)^2)
train_sst <- sum((y_train - mean(y_train))^2)

train_rqsuared <- 1 - (train_ssr / train_sst)

ssr <- sum((predictions - y_test)^2)
sst <- sum((y_test - mean(y_test))^2)

rsquared <- 1 - (ssr / sst)

# Printing the calculated statistics
print("RESULTS USING RIDGE REGRESSION")
print("Training RMSE")
print(train_rmse)

print("Test RMSE")
print(rmse)

print("Training MSE")
print(train_mse)

print("Test MSE")
print(mse)

print("Training R Squared Value")
print(train_rqsuared)

print("Test R Squared Value")
print(rsquared)

# data preparation required for the 'SparseSVM' package
copy_x_train <- x_train
copy_y_train <- y_train
copy_x_test <- x_test
copy_y_test <- y_test
# threshold used to convert regression problem into classification problem 
threshold <- mean(students$G3)

copy_y_train[copy_y_train < threshold] <- 0
copy_y_train[copy_y_train >= threshold] <- 1

copy_y_test[copy_y_test < threshold] <- 0
copy_y_test[copy_y_test >= threshold] <- 1

copy_y_train <- unlist(copy_y_train)
copy_x_train <-  matrix(unlist(copy_x_train), nrow = nrow(copy_x_train), ncol = ncol(copy_x_train) )
copy_x_test <- matrix(unlist(copy_x_test), nrow = nrow(copy_x_test), ncol = ncol(copy_x_test))

# model fitting using lasso 
lasso_fit <- cv.sparseSVM(copy_x_train, copy_y_train, alpha = 1, nfolds = 10)

lasso_predictions <- predict(lasso_fit, copy_x_test, type = 'class')
# Printing the confusion matrix
lasso_confusion_matrix <- confusionMatrix(factor(lasso_predictions), 
                factor(copy_y_test))

print("Lasso confusion matrix")
print(lasso_confusion_matrix)

# model fitting using elastic net value of 0.5
elastic_fit <- cv.sparseSVM(copy_x_train, copy_y_train, alpha = 0.5, nfolds = 10)

elastic_predictions <- predict(elastic_fit, copy_x_test, type = 'class')

# Getting the confusion matrix
elastic_fit_confusion_matrix <- confusionMatrix(factor(elastic_predictions), 
                factor(copy_y_test))

print("Elastic Net confusion matrix")
print(elastic_fit_confusion_matrix)

