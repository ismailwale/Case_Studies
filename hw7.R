########
##  Hand Writing Recognision
##  Tajudeen Abdulazeez
##  IST 707 Homework 7
##
##   
##  SVM, KNN and Random Forest
##
##
######################################

## Packages
library(ggplot2)
library(RWeka)
library(e1071)
library(rpart)
library(randomForest)
library(caret)


############################################# Data Prep #################################

digit_train <- read_csv('../data/wk7-data/Kaggle-digit-train.csv')
digit_test <- read_csv('../data/wk7-data/Kaggle-digit-test.csv')

digit_train$label[1:10]
summary(digit_train)
dim(digit_train)

## 
str(digit_train)

# change the label to factor
digit_train$label <- as.ordered(digit_train$label)


# Data partinsion
ind <- sample(2, nrow(digit_train), replace = TRUE,prob = c(0.7,0.3))

train <- digit_train[ind==1,]
eval <- digit_train[ind==2,]

summary(train)

# train distribution
table(train$label)

#eval
table(eval$label)


#eval without label
eval_with_no_label <- eval[,-1]
eval_label <- eval[,1]





############################################# SVM #######################################





############################################ KNN ########################################3



######################################### Random Forest ####################################








