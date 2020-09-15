# R script for Mulilayerperceptron applied on Boston dataset
# installing the caret package may take a few minutes depending on the machine
# referencing for the below code can be found in the reference file attached during submission

install.packages("caTools")   # To split the dataset into Train and Test
install.packages("ANN2")      # To Train data using neural network
install.packages("Metrics")   # To calculate mae value
install.packages("neuralnet") # To plot the NN
install.packages('caret')     # To calculate RMSE





library(MASS)   # To access Boston dataset
library(caTools)
library(neuralnet)
library(ANN2)
library(caret)
library(Metrics)


# To get the information about dataset
?Boston


# to get the names of all the columns present in the datset
names(Boston)

# To get the statistics for each columns present in the dataset
summary(Boston)
dim(Boston)



# Plotting the Boston dataset in the form of ANN architecture
nn_D = neuralnet(medv~crim + zn + indus + chas + nox + rm + age + dis + rad + tax + ptratio + black + lstat, data = Boston, act.fct = 'logistic', hidden = c(5,5), err.fct = "sse", linear.output = FALSE)
plot(nn_D)


############
#Independent variables = 13
#Dependent variable = 14->medv
############


## Splitting dataset into Train and test in the ratio 75:25

splitter <- sample.split(Boston$medv, SplitRatio = 0.75)
splitter
train_data <- subset(Boston, splitter == TRUE)
test_data <- subset(Boston, splitter == FALSE )
x_train = train_data[,-14]
y_train = train_data$medv
x_test = test_data[,-14]
y_test = test_data$medv

### Creating Function to train the model using Multilayer perceptron learner with parameters
multilayerperceptron = function(inputdata, varpos,  cvfold, reg, loss, activefunc, learnrate, l1, l2, batch ){
  
# 10 cross validation can be done using caret function
#kfolding <- trainControl(method = "repeatedcv", number = 10)
#Training can be done either by using caret train function or neuralnet
# Preferred one is neuralnetwork as it supports all the regularisation by handling both training and testing datasets but needs manual cross validation.

##########
#Performing 10 cross validation manually
##########

# 10 folds generation from given dataset

set.seed(1) # To reproduce results
k <- cvfold
Foldingdata <- inputdata[,varpos]
class(Foldingdata)
kfolding<- data.frame(Foldingdata)
kfolds <- rep_len(1:k, nrow(kfolding))
kfolds <- sample(kfolds, nrow(kfolding))


# Iterating foldings 10 times
neu <- sapply(1:k, FUN = function(i){    # For loop can be applied but sapply is better for computation time
  set.seed(1)
  
  # Partition of data using 10 foldings
  testin <- which(kfolds == i, arr.ind = TRUE)
  testANN <- inputdata[testin,]
  trainANN <- train_data[-testin,]
  y_train_B <- trainANN[, varpos]
  x_train_B <- trainANN[, -varpos]
  y_test_B <<- testANN[,varpos]
  x_test_B <- testANN[, -varpos]
  
  # Training data using neural network function with various parameters given by the user
  Mlptrain_B <- neuralnetwork(x_train_B,y_train_B,hidden.layers = c(6,6),regression = reg, standardize = TRUE, loss.type = loss, activ.functions = activefunc , optim.type = "adam", learn.rates = learnrate, L1 = l1, L2 = l2, sgd.momentum = 0, batch.size = 20, verbose = TRUE, random.seed = 1)
  
  # Prediction of data using trained model and test data from folding
  y_pred_B <<- unlist(predict(Mlptrain_B, newdata = x_test_B)) # converting prediction using test dataset into vectors
  
  y_pred2_B <- unlist(predict(Mlptrain_B, newdata = x_train_B)) # Converting prediction using train dataset into vectors

  
  # Calculating various parameters to check the accuracy of the model
  # To check the RMSE and MSE value for Testing data
  RMSE1_B <<- RMSE(y_test_B, y_pred_B)
  MSE1_B <<- RMSE1_B^2

  # To check the RMSE and MSE for Training data # to compare with testing RMSE to check overfitting
  RMSE2_B <<- RMSE(y_train_B, y_pred2_B)
  MSE2_B <<- RMSE2_B^2
  
  # R2 value for Testing dataset
  rss1_B <- sum((y_pred_B - y_test_B) ^2)
  tss1_B <- sum((y_test_B - mean(y_test_B)) ^ 2)
  Rsquare1_B <<- 1- rss1_B/tss1_B
  # R2 value for Training dataset
  rss2_B <- sum((y_pred2_B - y_train_B) ^2)
  tss2_B <- sum((y_train_B - mean(y_train_B)) ^ 2)
  Rsquare2_B <<- 1- rss2_B/tss2_B
  Mlptrain_B<<- Mlptrain_B # to make Global variable
  Mae <- mae(y_test_B,y_pred_B)
  #print(Rsquare2_B)
  print(Rsquare1_B)
  #print(RMSE2_B)
})
print(neu)
print(mean(neu)) # Taking the mean of all the 10 values of Rsquare
plot(Mlptrain_B)

# Condition to plot Actual vs Predictor values
if(l1 == 0 & l2 == 0){
  plot(y_test_B, y_pred_B, col = "Blue", main = 'Real vs Predicted with No Regularisation', pch = 1, cex = 0.9, type = "p", xlab = "Actual", ylab = "Predicted")
  abline(0,1,col = "black")
}else if(l1 == 1 & l2 == 0){
  plot(y_test_B, y_pred_B, col = "Red", main = 'Real vs Predicted with L1 Regularisation', pch = 1, cex = 0.9, type = "p", xlab = "Actual", ylab = "Predicted")
  abline(0,1,col = "black")
}else if(l1 ==0 & l2 == 1){
  plot(y_test_B, y_pred_B, col = "Dark green", main = 'Real vs Predicted with L2 Regularisation', pch = 1, cex = 0.9, type = "p", xlab = "Actual", ylab = "Predicted")
  abline(0,1,col = "black")
}else if (l1 ==1 & l2 == 1){
  plot(y_test_B, y_pred_B, col = "Brown", main = 'Real vs Predicted with elastic net Regularisation', pch = 1, cex = 0.9, type = "p", xlab = "Actual", ylab = "Predicted")
  abline(0,1,col = "black")
}
}



## Calling Function

##################
# Arguments and their values required to call multilayerperceptron function
#Detailed information related to the argument can be found at ?neuralnetwork
# Further information are included in the Report attached
#1. Data
#2. variable position, in our case it is 14
#3. cvfold -> 10
#4. reg = TRUE ( For regression problem otherwise FALSE)
#4. loss function -> "squared", "absolute" ( other values include "log", "quadratic")
#5. activefunc -> relu performed the best for simplemlp (Other values are "sigmoid", "tanh", "ramp")
#6. learnrate -> 0.01 
#7. l1 and l2 are lasso and Ridge regularisation respectively and combination of both is elasticnet
#8. Batchsize = 30
##################

#SimpleMLP
SimpleMLP <- multilayerperceptron(train_data, 14, 10, TRUE, "squared", "relu", 0.01, 0, 0, 30 )

#MLP with lasso regularisation, l1 = 1, l2 = 0
l1 <- multilayerperceptron(train_data, 14, 10,  TRUE, "squared", "sigmoid", 0.01, 1, 0, 30 )

#MLP with Ridge regularisation, l1 = 0, l2 = 1
l2 <- multilayerperceptron(train_data, 14, 10,  TRUE, "squared", "sigmoid", 0.01, 0, 1, 30 )

#MLP with elastic regularisation, l1 = 1, l2 = 1
elasticnet <- multilayerperceptron(train_data, 14, 10,  TRUE, "squared", "sigmoid", 0.01, 1, 1, 30 )

plot(Mlptrain_B)

####### Extra analysis
cor(Boston)
plot(Boston$medv, Boston$rm)
plot(Boston$medv, Boston$lstat)
plot(Boston$medv, Boston$nox)
plot(Boston$medv, Boston$age)
plot(Boston$rad, Boston$medv)
plot(Boston$tax, Boston$medv)
plot(Boston$ptratio, Boston$medv)
plot(Boston$rad, Boston$medv)
plot(Boston$zn, Boston$medv)
plot(Boston$chas, Boston$medv)
plot(Boston$black, Boston$medv)
results = lm(medv~., Boston)
summary(results)




#####Plots
attach(Boston)
plot(density(medv))
results = lm(medv~ crim + zn + nox + rm + dis + rad + tax + ptratio + black + lstat, Boston)
summary(results)
names(Boston)
plot(rm, medv)
lines(lowess(rm, medv), col = "red", lwd = 2)
fit.rm = lm(medv ~ rm)
points(rm, fitted(fit.rm), col = "blue", pch = 20)
#summary(fit.rm)
fit.rmpoly = lm(medv ~ poly(rm,6))
points(rm, fitted(fit.rmpoly), col = "darkgreen", pch = 20)
lm.fit=lm(medv~rm)
plot(rm ,medv)
abline(lm.fit)
points(rm, fitted(lm.fit), col = "yellow", lwd = 2)

