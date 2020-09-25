##install relevant packages at start and corresponding libraries, can take up to 15 minutes to load caret pacakge

install.packages('caret', dependencies = TRUE)
install.packages('xfun', dependencies = TRUE)
install.packages('generics', dependencies = TRUE)
install.packages('gower', dependencies = TRUE)

library(plyr)
library(readr)
library(dplyr)
library(caret)
library(ggplot2)
library(repr)
library(tibble)
library(MASS)
library(ISLR)
library(glmnet)

## for the purposes of MLR we have cleaned data at source rather than applying cleaning techniques in R
## changing Yes / No values to 1 / 0 and other binary outcomes
## SVM uses cleaning in R to show flexibility
##CSV file needs to be stored in same directory as R script to run as below

Students <- read_csv('Student_Grades_Data_Cleaned.csv')
names(Students)

dmy <- dummyVars(" ~ .", data = Students)
Students_h <- data.frame(predict(dmy, newdata = Students))
Students_h

##names(Students_h) not needed, useful to see list of column names
##glimpse(Students_h) preview of the data structures


##split the dataset into training and test data, assuming 25% training data
index = sample(1:nrow(Students_h), 0.75*nrow(Students_h)) 

train = Students_h[index,] # Create the training data 
test = Students_h[-index,] # Create the test data

dim(train)
dim(test)

##name the individual variables from 'Students_h' file here
##exclude G3 being the dependent variable

cols = c('schoolGP0MS2',     'sexF0M1',          'age',              'addressR0U1',     
        'famsizeLE30GT31',  'PstatusA0T1',      'Medu',             'Fedu',            
        'Mjobat_home',      'Mjobhealth',       'Mjobother',        'Mjobservices',    
        'Mjobteacher',      'Fjobat_home',      'Fjobhealth',       'Fjobother',       
        'Fjobservices',     'Fjobteacher',      'reasoncourse',     'reasonhome',      
        'reasonother',      'reasonreputation', 'guardianfather',   'guardianmother',  
        'guardianother',    'traveltime',       'studytime',        'failures',        
        'schoolsup',        'famsup',           'paid',             'activities',      
        'nursery',          'higher',           'internet',         'romantic',        
        'famrel',           'freetime',         'goout',            'Dalc',            
        'Walc',             'health',           'absences')    




pre_proc_val <- preProcess(train[,cols], method = c('center', 'scale'))


train[,cols] = predict(pre_proc_val, train[,cols])
test[,cols] = predict(pre_proc_val, test[,cols])

## get some summary statistics on the variables from the training dataset

summary(train)


## execute a linear regression with all predictors on training dataset

lr = lm(G3~.,data = train)

summary(lr)

##lm(formula = G3 ~ ., data = train) just to see in coefficient terms

##Create output metrics

#Step 1 - create the evaluation metrics function

eval_metrics = function(model, df, predictions, target){
  resids = df[,target] - predictions
  resids2 = resids**2
  N = length(predictions)
  r2 = as.character(round(summary(model)$r.squared, 2))
  adj_r2 = as.character(round(summary(model)$adj.r.squared, 2))
  print(adj_r2) #Adjusted R-squared
  print(as.character(round(sqrt(sum(resids2)/N), 2))) #RMSE
}

# Step 2 - predicting and evaluating the model on train data
predictions = predict(lr, newdata = train)
eval_metrics(lr, train, predictions, target = 'G3')

# Step 3 - predicting and evaluating the model on test data
predictions = predict(lr, newdata = test)
eval_metrics(lr, test, predictions, target = 'G3')

cols_reg = c('schoolGP0MS2',     'sexF0M1',          'age',              'addressR0U1',     
             'famsizeLE30GT31',  'PstatusA0T1',      'Medu',             'Fedu',            
             'Mjobat_home',      'Mjobhealth',       'Mjobother',        'Mjobservices',    
             'Mjobteacher',      'Fjobat_home',      'Fjobhealth',       'Fjobother',       
             'Fjobservices',     'Fjobteacher',      'reasoncourse',     'reasonhome',      
             'reasonother',      'reasonreputation', 'guardianfather',   'guardianmother',  
             'guardianother',    'traveltime',       'studytime',        'failures',        
             'schoolsup',        'famsup',           'paid',             'activities',      
             'nursery',          'higher',           'internet',         'romantic',        
             'famrel',           'freetime',         'goout',            'Dalc',            
             'Walc',             'health',           'absences', 'G3')

dummies <- dummyVars(G3 ~ ., data = Students_h[,cols_reg])

train_dummies = predict(dummies, newdata = train[,cols_reg])

test_dummies = predict(dummies, newdata = test[,cols_reg])

print(dim(train_dummies)); print(dim(test_dummies))

##execute the ridge regression
## per below you'll note for Ridge that alpha is set to 0


x = as.matrix(train_dummies)
y_train = train$G3

x_test = as.matrix(test_dummies)
y_test = test$G3

lambdas <- 10^seq(2, -3, by = -.1)
ridge_reg = glmnet(x, y_train, nlambda = 25, alpha = 0, family = gaussian, lambda = lambdas)

summary(ridge_reg)

cv_ridge <- cv.glmnet(x, y_train, alpha = 0, lambda = lambdas)
optimal_lambda <- cv_ridge$lambda.min
optimal_lambda

##Getting Ridge regression outputs
# Compute R^2 from true and predicted values
eval_results <- function(true, predicted, df) {
  SSE <- sum((predicted - true)^2)
  SST <- sum((true - mean(true))^2)
  R_square <- 1 - SSE / SST
  RMSE = sqrt(SSE/nrow(df))
  
  
  # Model performance metrics
  data.frame(
    RMSE = RMSE,
    Rsquare = R_square
  )
  
}

# Prediction and evaluation on train data
predictions_train <- predict(ridge_reg, s = optimal_lambda, newx = x)
eval_results(y_train, predictions_train, train)

# Prediction and evaluation on test data
predictions_test <- predict(ridge_reg, s = optimal_lambda, newx = x_test)
eval_results(y_test, predictions_test, test)


##Lasso attempt

lambdas <- 10^seq(2, -3, by = -.1)

# Setting alpha = 1 implements lasso regression
lasso_reg <- cv.glmnet(x, y_train, alpha = 1, lambda = lambdas, standardize = TRUE, nfolds = 5)

# Best 
lambda_best <- lasso_reg$lambda.min 
lambda_best

##note for lasso we've now set aplha to 1
lasso_model <- glmnet(x, y_train, alpha = 1, lambda = lambda_best, standardize = TRUE)

predictions_train <- predict(lasso_model, s = lambda_best, newx = x)
eval_results(y_train, predictions_train, train)

predictions_test <- predict(lasso_model, s = lambda_best, newx = x_test)
eval_results(y_test, predictions_test, test)

##plot ridge and lasso results for report on Student Grades
plot(lasso_reg)
plot(cv_ridge)



########### Section Change, below is the same code base applied to Boston


names(Boston)

dmy <- dummyVars(" ~ .", data = Boston)
Boston_h <- data.frame(predict(dmy, newdata = Boston))

set.seed(100) 

index = sample(1:nrow(Boston_h), 0.75*nrow(Boston_h)) 

train = Boston_h[index,] # Create the training data for Boston
test = Boston_h[-index,] # Create the test data for Boston

dim(train)
dim(test)

##name the individual variables from Boston file here
##exclude medv being the dependent variable for Boston

cols = c('crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','black','lstat') 

pre_proc_val <- preProcess(train[,cols], method = c('center', 'scale'))

train[,cols] = predict(pre_proc_val, train[,cols])
test[,cols] = predict(pre_proc_val, test[,cols])

##summary(train) - create summary statistics on the training data, not needed

## execute a linear regression with all predictors on training dataset

lr = lm(medv~.,data = train)

summary(lr)

##execute the same model without age or indus classifiers

lr2 = lm(medv~. -age -indus, data = train)
summary(lr2)

##lr3 gets the MLR outputs on the full dataset, not split into test & train
lr3 = lm(medv~., data = Boston_h)
summary(lr3)


##lm(formula = medv ~ ., data = train) ##same output except in different format looking at the coefficients

##Create output metrics

#Step 1 - create the evaluation metrics function
# we will be assessing the models on the basis of adjusted r-squared and RMSE
# this function is used to create out puts or RMSE & R^2
eval_metrics = function(model, df, predictions, target){
  resids = df[,target] - predictions
  resids2 = resids**2
  N = length(predictions)
  r2 = as.character(round(summary(model)$r.squared, 2))
  adj_r2 = as.character(round(summary(model)$adj.r.squared, 2))
  print(adj_r2) #Adjusted R-squared
  print(as.character(round(sqrt(sum(resids2)/N), 2))) #RMSE
}

# Step 2 - predicting and evaluating the model on train data
predictions = predict(lr, newdata = train)
eval_metrics(lr, train, predictions, target = 'medv')

# Step 3 - predicting and evaluating the model on test data
predictions = predict(lr, newdata = test)
eval_metrics(lr, test, predictions, target = 'medv')

##change column names here
cols_reg = c('crim','zn','chas','nox','rm','dis','rad','tax','ptratio','black','lstat','medv')
##removing 'indus', 'age',
dummies <- dummyVars(medv ~ ., data = Boston_h[,cols_reg])

train_dummies = predict(dummies, newdata = train[,cols_reg])

test_dummies = predict(dummies, newdata = test[,cols_reg])

print(dim(train_dummies)); print(dim(test_dummies))

##execute the ridge regression using the glmnet library functions

##library(glmnet) removing this as already imported at start

x = as.matrix(train_dummies)
y_train = train$medv

x_test = as.matrix(test_dummies)
y_test = test$medv

lambdas <- 10^seq(2, -3, by = -.1)
ridge_reg = glmnet(x, y_train, nlambda = 25, alpha = 0, family = gaussian, lambda = lambdas)

summary(ridge_reg)

cv_ridge <- cv.glmnet(x, y_train, alpha = 0, lambda = lambdas)
optimal_lambda <- cv_ridge$lambda.min
optimal_lambda


##Getting Ridge regression outputs
# Compute R^2 from true and predicted values
eval_results <- function(true, predicted, df) {
  SSE <- sum((predicted - true)^2)
  SST <- sum((true - mean(true))^2)
  R_square <- 1 - SSE / SST
  RMSE = sqrt(SSE/nrow(df))
  
  
  # Model performance metrics
  data.frame(
    RMSE = RMSE,
    Rsquare = R_square
  )
  
}

# Prediction and evaluation on train data
predictions_train <- predict(ridge_reg, s = optimal_lambda, newx = x)
eval_results(y_train, predictions_train, train)

# Prediction and evaluation on test data
predictions_test <- predict(ridge_reg, s = optimal_lambda, newx = x_test)
eval_results(y_test, predictions_test, test)

##Lasso attempt

lambdas <- 10^seq(2, -3, by = -.1)

# Setting alpha = 1 implements lasso regression
lasso_reg <- cv.glmnet(x, y_train, alpha = 1, lambda = lambdas, standardize = TRUE, nfolds = 5)

# Best 
lambda_best <- lasso_reg$lambda.min 
lambda_best


lasso_model <- glmnet(x, y_train, alpha = 1, lambda = lambda_best, standardize = TRUE)

predictions_train <- predict(lasso_model, s = lambda_best, newx = x)
eval_results(y_train, predictions_train, train)

predictions_test <- predict(lasso_model, s = lambda_best, newx = x_test)
eval_results(y_test, predictions_test, test)

##plot ridge and lasso results for report on Boston
plot(lasso_reg)
plot(cv_ridge)


