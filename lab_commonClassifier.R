# Clear the workspace
rm(list=ls())
cat("\014")
# install.packages('caret')
# more information about the package can be found at http://topepo.github.io/caret/index.html
# and https://cran.r-project.org/web/packages/caret/caret.pdf
library(caret)

# Use menu /Session/Set Working Directory/Choose Directory Or command below to set working directory
setwd("/Users/sivasrinivasnarra/Desktop/Academic/SPRING_2024/BAR/6.Classification_II")

# load the data
bank.df <- read.csv("UniversalBank.csv")
# convert output as factor
bank.df$Personal.Loan <- as.factor(bank.df$Personal.Loan)
# treat Education as categorical (R will create dummy variables)
bank.df$Education <- factor(bank.df$Education, levels = c(1, 2, 3), 
                            labels = c("Undergrad", "Graduate", "Advanced/Professional"))

# split the data into training and test data sets
set.seed(2)   # for reproducible results
train <- sample(1:nrow(bank.df), (0.6)*nrow(bank.df))
train.df <- bank.df[train,]
test.df <- bank.df[-train,]


########### 1. Logistic Regression ########### 
# use glm() (general linear model) with family = "binomial" to fit a logistic 
logit.reg <- glm(Personal.Loan ~ Age + Experience + Income + Family + CCAvg + Education 
                 + Mortgage + Securities.Account + CD.Account + Online + CreditCard, 
                 data = train.df, family = "binomial") 
summary(logit.reg)

# use predict() with type = "response" to compute predicted probabilities. 
logitPredict <- predict(logit.reg, test.df, type = "response")
# we choose 0.5 as the cutoff here for 1 vs. 0 classes
logitPredictClass <- ifelse(logitPredict > 0.5, 1, 0)

# evaluate classifier on test.df
actual <- test.df$Personal.Loan
predict <- logitPredictClass
cm <- table(predict, actual)
# consider class "1" as positive
tp <- cm[2,2]
tn <- cm[1,1]
fp <- cm[2,1]
fn <- cm[1,2]
# accuracy
(tp + tn)/(tp + tn + fp + fn)
# TPR = Recall = Sensitivity
tp/(fn+tp)
# TNR = Specificity
tn/(fp+tn)
# FPR
fp/(fp+tn)
# FNR
fn/(fn+tp)

########### alternative way of doing "Logistic Regression" using caret ########### 
# 10-fold cross-validation
ctrl <- trainControl(method="cv", number=10) 
# method = 'glmnet' & family = 'binomial'
logit.CV <- train(Personal.Loan ~ Age + Experience + Income + Family + CCAvg + Education 
                  + Mortgage + Securities.Account + CD.Account + Online + CreditCard, 
                  data = train.df, 
                  method = 'glmnet',
                  trControl = ctrl,
                  family = 'binomial' )

# evaluate classifier on test.df
actual <- test.df$Personal.Loan
predict <- predict(logit.CV, test.df, type = "raw") # use predict() with type = "raw"/"prob" for class labels/class probabilities,
cm <- table(predict, actual)
confusionMatrix(cm, positive = "1") # "yes", "default"


########### 2. K-Nearest Neighbors ########### 
# Checking distribution of outcome classes -> very few class = "1"
prop.table(table(bank.df$Personal.Loan)) * 100
prop.table(table(train.df$Personal.Loan)) * 100
prop.table(table(test.df$Personal.Loan)) * 100

# 10-fold cross-validation
ctrl <- trainControl(method="cv", number=10) 
# use preProcess to to normalize the predictors
# "center" subtracts the mean; "scale" divides by the standard deviation
knnFit <- train(Personal.Loan ~ Age + Experience + Income + Family + CCAvg + Education 
                + Mortgage + Securities.Account + CD.Account + Online + CreditCard, 
                data = train.df, method = "knn", trControl = ctrl, preProcess = c("center","scale"),
                tuneGrid = expand.grid(k = 1:10))
# print the knn fit for different values of k
# Kappa is a more useful measure to use on problems that have imbalanced classes.
knnFit
# plot the # of neighbors vs. accuracy (based on repeated cross validation)
plot(knnFit)
ggplot(data=knnFit$results, aes(k, Accuracy)) + geom_line() + scale_x_continuous(breaks=1:10)
ggplot(data=knnFit$results, aes(k, Kappa)) + geom_line() + scale_x_continuous(breaks=1:10)

# Evaluate classifier performance on testing data
actual <- test.df$Personal.Loan
knnPredict <- predict(knnFit, test.df)
cm <- table(knnPredict, actual)
cm 
# alternative way to get a comprehensive set of statistics
confusionMatrix(knnPredict, actual, positive="1")
# or
confusionMatrix(cm, positive = "1") # "yes", "default"


########### 3. Naive Bayes Classifier ########### 
# install the following package for building naive Bayes classifier
#install.packages("e1071")
library(e1071)

# run naive bayes
fit.nb <- naiveBayes(Personal.Loan ~ Age + Experience + Income + Family + CCAvg + Education 
                     + Mortgage + Securities.Account + CD.Account + Online + CreditCard, 
                     data = train.df)

# Evaluate Performance using Confusion Matrix
actual <- test.df$Personal.Loan
# predict class probability
nbPredict <- predict(fit.nb, test.df, type = "raw")
# predict class membership
nbPredictClass <- predict(fit.nb, test.df, type = "class")
cm <- table(nbPredictClass, actual)
cm 
# alternative way to get confusion matrix
confusionMatrix(nbPredictClass, actual, positive="1")

########### alternative way of building "Naive Bayes Classifier" using caret ########### 
fit.nb.CV <- train(Personal.Loan ~ Age + Experience + Income + Family + CCAvg + Education 
                   + Mortgage + Securities.Account + CD.Account + Online + CreditCard, 
                   data = train.df, 
                   method="naive_bayes", 
                   trControl=trainControl(method="cv", number=10))
# Evaluate Performance using Confusion Matrix
actual <- test.df$Personal.Loan
# predict class membership
nbPredictClass <- predict(fit.nb.CV, test.df, type = "raw")
cm <- table(nbPredictClass, actual)
cm
confusionMatrix(nbPredictClass, actual, positive="1")


########### 4. Ensemble Methods ########### 
# More details of the package can be found at https://cran.r-project.org/web/packages/adabag/adabag.pdf
#install.packages("adabag")
# See more ensemble methods in caret at https://topepo.github.io/caret/train-models-by-tag.html#ensemble-model
library(adabag)
library(rpart) 
library(caret)

# single tree
fit.tree <- rpart(Personal.Loan ~ ., data = train.df)
pred <- predict(fit.tree, test.df, type = "class")
cm1 <- confusionMatrix(pred, test.df$Personal.Loan)

# bagging
# "mfinal" an integer, the number of trees to use.
fit.bagging <- bagging(Personal.Loan ~ ., data = train.df, mfinal = 20)
pred <- predict(fit.bagging, test.df, type = "class")
# should convert pred$class to factor, same as the original data Personal.Loan
cm2 <- confusionMatrix(as.factor(pred$class), test.df$Personal.Loan)

# boosting
# "mfinal" an integer, the number of iterations for which boosting is run 
fit.boosting <- boosting(Personal.Loan ~ ., data = train.df, mfinal = 20)
pred <- predict(fit.boosting, test.df, type = "class")
# should convert pred$class to factor, same as the original data Personal.Loan
cm3 <- confusionMatrix(as.factor(pred$class), test.df$Personal.Loan)

# compare across different methods
result <- rbind(cm1$overall["Accuracy"], cm2$overall["Accuracy"], cm3$overall["Accuracy"])
row.names(result) <- c("single tree", "bagging", "boosting")
result
# TPR (sensitivity, recall)
result <- rbind(cm1$byClass['Sensitivity'], cm2$byClass['Sensitivity'], cm3$byClass['Sensitivity'])
row.names(result) <- c("single tree", "bagging", "boosting")
result




