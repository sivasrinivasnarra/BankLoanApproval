# Clear the workspace
rm(list=ls())
cat("\014")
# Use menu /Session/Set Working Directory/Choose Directory Or command below to set working directory
setwd("/Users/ShujingSun/Desktop/Teaching/")
# install.packages("ROCit")
# more about the package can be found at https://cran.r-project.org/web/packages/ROCit/ROCit.pdf
library(ROCit)
library(caret)

# load the data
bank.df <- read.csv("UniversalBank.csv")
# convert output as factor
bank.df$Personal.Loan <- as.factor(bank.df$Personal.Loan)
# treat Education as categorical 
bank.df$Education <- factor(bank.df$Education, levels = c(1, 2, 3), 
                            labels = c("Undergrad", "Graduate", "Advanced/Professional"))

# split the data into training and test data sets
set.seed(2)   # for reproducible results
train <- sample(1:nrow(bank.df), (0.6)*nrow(bank.df))
train.df <- bank.df[train,]
test.df <- bank.df[-train,]

# check the balance of the training and testing data set
prop.table(table(bank.df$Personal.Loan))   
prop.table(table(test.df$Personal.Loan))      


########### 1. Logistic Regression ########### 
logit.reg <- glm(Personal.Loan ~ Age + Experience + Income + Family + CCAvg + Education 
                 + Mortgage + Securities.Account + CD.Account + Online + CreditCard, 
                 data = train.df, family = "binomial") 

# compute predicted probabilities of Personal.Loan=1
pred_logit <- predict(logit.reg, test.df, type = "response")
summary(pred_logit)
actual <- test.df$Personal.Loan
summary(actual)

# create ROC curve
roc_logit <- rocit(score = pred_logit, class = actual) 

# check AUC, Cutoff, TPR, FPR(=1-Specificity)
result_logit <- data.frame(cbind(AUC=roc_logit$AUC, Cutoff=roc_logit$Cutoff, 
                          TPR=roc_logit$TPR, FPR=roc_logit$FPR))
head(result_logit)
tail(result_logit)

# find the optimal point (Youden Index point)
result_logit$diff <- result_logit$TPR - result_logit$FPR
bestcutoff <- result_logit[which.max(result_logit[, c("diff")]), ]
bestcutoff$Cutoff

# we choose cutoff based on Youden Index point
logitPredictClass <- ifelse(pred_logit > bestcutoff$Cutoff, 1, 0)
confusionMatrix(table(pred=logitPredictClass, actual), positive='1')

# what about using 0.5 as the cutoff?
logitPredictClass <- ifelse(pred_logit > 0.5, 1, 0)
confusionMatrix(table(pred=logitPredictClass, actual), positive='1')

# plot ROC 
plot(roc_logit)  # default
plot(roc_logit, YIndex = T, col = c(2,4)) # Changing color
plot(roc_logit, YIndex = F, col = c(2,4),legend = F) # disable legend



########### 2. Naive Bayes Classifier ########### 
library(e1071)
# run naive bayes
fit.nb <- naiveBayes(Personal.Loan ~ Age + Experience + Income + Family + CCAvg + Education 
                     + Mortgage + Securities.Account + CD.Account + Online + CreditCard, 
                     data = train.df)

# compute predicted probabilities
pred_nb <- predict(fit.nb, test.df, type = "raw")
summary(pred_nb) 
actual <- test.df$Personal.Loan
summary(actual)

# create ROC curve: pred_nb contains probabilities of both class labels, column 2 corresponds to class label "1"
roc_nb <- rocit(score = pred_nb[,2], class = actual) 

# check AUC, Cutoff, TPR, FPR(=1-Specificity)
result_nb <- data.frame(cbind(AUC=roc_nb$AUC, Cutoff=roc_nb$Cutoff, 
                          TPR=roc_nb$TPR, FPR=roc_nb$FPR))

# find the optimal point (Youden Index point)
result_nb$diff <- result_nb$TPR - result_nb$FPR
bestcutoff <- result_nb[which.max(result_nb[, c("diff")]), ]
bestcutoff$Cutoff

# we choose cutoff based on Youden Index point
nbPredictClass <- ifelse(pred_nb[,2] > bestcutoff$Cutoff, 1, 0)
confusionMatrix(table(pred=nbPredictClass, actual), positive='1')


# plot multiple ROC curves
plot(roc_logit, col = c("blue", "black"), legend = FALSE, YIndex = FALSE)
lines(roc_nb$TPR ~ roc_nb$FPR, col = "red")
legend("bottomright", col = c("blue","red"), c("ROC for logit", "ROC for NaiveBayes"), lwd = 2) #lwd for line width

# compare AUC for different model: logistic regression is better than NaiveBayes because AUC is larger
roc_logit$AUC  
roc_nb$AUC


