# R script for Mulilayerperceptron applied on Boston dataset
# installing the caret package may take a few minutes depending on the machine
# referencing for the below code can be found in the reference file attached during submission

install.packages("caTools")   # To split the dataset into Train and Test
install.packages("ANN2")      # To Train data using neural network
install.packages("Metrics")   # To calculate mae value
install.packages("neuralnet") # To plot the NN
install.packages('caret')     # To calculate RMSE
install.packages("Boruta")    # To perform Feature selection




library(Boruta)
library(caTools)
library(neuralnet)
library(ANN2)
library(Metrics)
library(caret)

# Read the dataset file(CSV file)
mydata <- read.csv('Student Grades Data.csv')


#Creating Dummy variable to convert character to numeric variable(One hot encoding) Since neural model supports data in a numeric form.
dmy <- dummyVars(" ~ .", data = mydata)
Students_h <- data.frame(predict(dmy, newdata = mydata))
dim(Students_h)
sum(is.na(Students_h))


# Feature selection to address overfitting issue
set.seed(1)
boruta <- Boruta(G3~., data = Students_h, doTrace = 2)
#boruta <- Boruta(G3~., data = mydata, doTrace = 2)
attStats(boruta)
plot(boruta, las = 2, cex.axis = 0.7)
plotImpHistory(boruta)

#Tentative fix
bor <- TentativeRoughFix(boruta)
print(bor)
getNonRejectedFormula(boruta)
getConfirmedFormula(boruta)


#Selecting variables based on the results given by feature selection
names(Students_h)
#mydatanew <- mydata[c(1, 3, 7, 12, 15, 16, 21, 26, 30,31)]
mydatanew <- Students_h[c(5, 12, 14, 30, 33, 34, 35, 44, 45, 52, 56, 57 )]
#mydatanew <- mydata[c(1,2, 3, 7,9, 12, 15, 16, 21, 26,27, 30)]
names(mydatanew)
sum(is.na(mydatanew))



####### Plotting the Boston dataset in the form of ANN architecture
nn_D = neuralnet(G3~ age + Medu + Mjobat_home + guardianother + failures + schoolsupno + schoolsupyes + higherno + higheryes + goout + absences, data = mydatanew, act.fct = 'logistic', hidden = c(5,5), err.fct = "sse", linear.output = FALSE)
plot(nn_D)



# to get the names of all the columns present in the datset
names(mydatanew)

# To get the statistics for each columns present in the dataset
summary(mydatanew)
dim(mydatanew)

############
#Independent variables = 11
#Dependent variable = 12->G3
############


## Splitting dataset into Train and test in the ratio 75:25
splitter_D <- sample.split(mydatanew$G3, SplitRatio = 0.75)
splitter_D
train_data_D <- subset(mydatanew, splitter_D == TRUE)
test_data_D <- subset(mydatanew, splitter_D == FALSE )
dim(train_data_D)
sum(is.na(train_data_D))
class(train_data_D)
x_train_D = train_data_D[,-12]
y_train_D = train_data_D$G3
x_test_D = test_data_D[,-12]
y_test_D = test_data_D$G3

### Creating Function to train the model using Multilayer perceptron learner
multilayerperceptron = function(inputdata, varpos,  cvfold, reg, loss, activefunc, learnrate, l1, l2, batch ){
  
# 10 cross validation can be done using caret function
#kfolding <- trainControl(method = "repeated", number = 10)
#Training can be done either by using caret train function or neuralnet
# Preferred one is neuralnetwork as it supports all the regularisation by handling both training and testing datasets but needs manual cross validation.
  
##########
#Performing 10 cross validation manually
##########

# 10 folds generation from given dataset

set.seed(1) # To reproduce results
k <- cvfold
Foldingdata <- inputdata[ , varpos]
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
  trainANN <- inputdata[-testin,]
  y_train_D <- trainANN[, varpos]
  x_train_D <- trainANN[, -varpos]
  y_test_D <- testANN[,varpos]
  x_test_D <- testANN[, -varpos]
  sum(is.na(x_train_D))
  sum(is.na(x_test_D))
  summary(x_train_D)
  
  # Training data using neural network function with parameters given by users
  Mlptrain_D <- neuralnetwork(x_train_D,y_train_D,hidden.layers = c(5,5),regression = reg, standardize = TRUE, loss.type = loss, activ.functions = activefunc , optim.type = "adam", learn.rates = learnrate, L1 = l1, L2 = l2, sgd.momentum = 0, batch.size = batch, verbose = TRUE, random.seed = 1)
  
  # Prediction of data using trained model and test data from folding
  y_pred_D <- unlist(predict(Mlptrain_D, newdata = x_test_D)) # converting prediction using test dataset into vectors
  
  y_pred2_D <- unlist(predict(Mlptrain_D, newdata = x_train_D)) # Converting prediction using train dataset into vectors
  
  
  # Claculating various parameters to check the accuracy of the model
  # To check the RMSE and MSE vallue for Testing data
  RMSE1_D <<- RMSE(y_test_D, y_pred_D)
  MSE1_D <<- RMSE1_D^2
  
  # To check the RMSE and MSE for Training data # to compare with testing RMSE to check overfitting
  RMSE2_D <<- RMSE(y_train_D, y_pred2_D)
  MSE2_D <<- RMSE2_D^2
  
  # R2 value using Testing data set
  rss1_D <- sum((y_pred_D - y_test_D) ^2)
  tss1_D <- sum((y_test_D - mean(y_test_D)) ^ 2)
  Rsquare1_D <<- 1- rss1_D/tss1_D
  # R2 value for Training dataset
  rss2_D <- sum((y_pred2_D - y_train_D) ^2)
  tss2_D <- sum((y_train_D - mean(y_train_D)) ^ 2)
  Rsquare2_D <<- 1- rss2_D/tss2_D
  Mae <- mae(y_test_B,y_pred_B)
  #print(Rsquare2_D)
  #print(Rsquare1_D)
  #print(RMSE2_D)
  #Mlptrain_D<<- Mlptrain_D # to make Global variable
  print(Rsquare1_D)
  #print(RMSE1_D)
})
print(neu)
print(mean(neu)) # Taking the mean of all the 10 values of Rsquare
#plot(Mlptrain_D)

#Condition to plot the Actual vs predictor value given by the model
if(l1 == 0 & l2 == 0){
  plot(y_test_B, y_pred_B, col = "Blue", main = 'Real vs Predicted with No Regularisation(Selected dataset)', pch = 1, cex = 0.9, type = "p", xlab = "Actual", ylab = "Predicted")
  abline(0,1,col = "black")
}else if(l1 == 1 & l2 == 0){
  plot(y_test_B, y_pred_B, col = "Red", main = 'Real vs Predicted with L1 Regularisation(Selected Dataset)', pch = 1, cex = 0.9, type = "p", xlab = "Actual", ylab = "Predicted")
  abline(0,1,col = "black")
}else if(l1 ==0 & l2 == 1){
  plot(y_test_B, y_pred_B, col = "Dark green", main = 'Real vs Predicted with L2 Regularisation(Selected Dataset)', pch = 1, cex = 0.9, type = "p", xlab = "Actual", ylab = "Predicted")
  abline(0,1,col = "black")
}else if (l1 ==1 & l2 == 1){
  plot(y_test_B, y_pred_B, col = "Brown", main = 'Real vs Predicted with elastic net Regularisation(Selected Dataset)', pch = 1, cex = 0.9, type = "p", xlab = "Actual", ylab = "Predicted")
  abline(0,1,col = "black")
}
}

#calling Functon for SimpleMlp with no regularisation
SimpleMLP <- multilayerperceptron(train_data_D, 12, 20, TRUE, "squared", "sigmoid", 0.01, 0, 0, 30 )

#calling function for l1 regularisation(lasso regularisation)
 l1 <- multilayerperceptron(train_data_D, 12, 20,  TRUE, "squared", "sigmoid", 0.01, 1, 0, 30)

 #calling function for l2 regularisation(Ridge regularisation)
l2 <- multilayerperceptron(train_data_D, 12, 20,  TRUE, "squared", "sigmoid", 0.01, 0, 1, 30 )

#calling function for l1&l2 regularisation(Elastic net regularisation)
elasticnet <- multilayerperceptron(train_data_D, 12, 20,  TRUE, "squared", "sigmoid", 0.01, 1, 1, 30 )
