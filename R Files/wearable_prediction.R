###########################
# File: wearable_prediction.R
# Description: Coursera Data Science Class 9 Project, Prediction Assignment
# Date: 2/22/2016
# Author: Joel Polanco (joel.g.polanco@intel.com)
# Notes:
# To do:
###########################

#set working directory
setwd("C:/Users/jgpolanc/Desktop/Coursera/C9/Human_Activity_Recognition")

#Load libraries

library(dplyr)
library(caret)
library(corrgram)
library(randomForest)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)


#reproducibility
set.seed(1234)


#Load Data
validation<- read.csv(paste(getwd(),"/data/pml-testing.csv", sep=""),na.strings=c("NA","#DIV/0!",""))
train<- read.csv(paste(getwd(),"/data/pml-training.csv", sep=""),na.strings=c("NA","#DIV/0!",""))

##reorder columns and prep data
train<- train[ ,c(160,1:159)]
validation<- validation[ ,c(160,1:159)]
train[, c(9:160)] <- sapply(train[, c(9:160)], as.numeric)
validation[, c(9:160)] <- sapply(validation[, c(9:160)], as.numeric)
str(train);str(validation)


#data partitioning for training
train_1 <- createDataPartition(y=train$classe, p=0.6, list=FALSE)
train_dat  <- train[train_1, ];test_dat<- train[-train_1, ]
dim(train_dat);dim(test_dat)

#data cleaning, remove zero variability 
train_subset1<-train_dat[,c(1:8)]
train_subset<-train_dat[,c(9:130)]
NZVdat <- nearZeroVar(train_subset, saveMetrics=TRUE)
NZVdat[NZVdat[,"zeroVar"] + NZVdat[,"nzv"] > 0, ]
nzv <- nearZeroVar(train_subset)
train_subset <- train_subset[,-nzv]
filtered_train<-cbind(train_subset1,train_subset)
dim(train_dat); dim(filtered_train)


#identify correlated (numeric predictors)
##-- Notes split filtered train into classe+non-numerics vs numerics and run the code chunk below, then run the high corr chunk and cbind them back together
descrCor <-  cor(train_subset,use="complete.obs")
summary(descrCor[upper.tri(descrCor)])
highlyCorDescr <- findCorrelation(descrCor, cutoff = .75)
filteredDescr <- train_subset[,-highlyCorDescr]
descrCor2 <- cor(filteredDescr,use="complete.obs")
summary(descrCor2[upper.tri(descrCor2)])
highlyCorDescr = sort(highlyCorDescr)
reduced_Data = train_subset[,-c(highlyCorDescr)]

## Check to see if there are any linear combinations of data present
reduced_Data2<- reduced_Data[complete.cases(reduced_Data),]
comboInfo <- findLinearCombos(reduced_Data2)
comboInfo

##There are no linear combinations present
final_train<-cbind(train_subset1,reduced_Data)

# Removing X variable so that it does not interfer with ML Algorithms
final_train <- final_train[c(-2)]

##Remove Variables with too many NAs. Removing variables with a  60% threshold of NA's
trainingV1 <- final_train #creating another subset to iterate in loop
for(i in 1:length(final_train)) { #for every column in the training dataset
  if( sum( is.na( final_train[, i] ) ) /nrow(final_train) >= .6 ) { #if n?? NAs > 60% of total observations
    for(j in 1:length(trainingV1)) {
      if( length( grep(names(final_train[i]), names(trainingV1)[j]) ) ==1)  { #if the columns are the same:
        trainingV1 <- trainingV1[ , -j] #Remove that column
      }   
    } 
  }
}

#To check the new N?? of observations
dim(trainingV1)

#Setting back to our set:
final_train <- trainingV1
rm(trainingV1)

#Perform the exact same  transformations for testing(test_dat) and valiation(validation) data sets.
clean1 <- colnames(final_train)
clean2 <- colnames(final_train[, -1]) #already with classe column removed
test_dat <- test_dat[clean1]
validation <- validation[clean2]

#To check the new N?? of observations
dim(test_dat)

#To check the new N?? of observations
dim(validation)

#In order to ensure proper functioning of Decision Trees and especially RandomForest Algorithm with the Test data set (data set provided), we need to coerce the data into the same type.

for (i in 1:length(test_dat) ) {
  for(j in 1:length(final_train)) {
    if( length( grep(names(final_train[i]), names(test_dat)[j]) ) ==1)  {
      class(test_dat[j]) <- class(final_train[i])
    }      
  }      
}

#In order to ensure proper functioning of Decision Trees and especially RandomForest Algorithm with the Test data set (data set provided), we need to coerce the data into the same type.
for (i in 1:length(validation) ) {
  for(j in 1:length(final_train)) {
    if( length( grep(names(final_train[i]), names(validation)[j]) ) ==1)  {
      class(validation[j]) <- class(final_train[i])
    }      
  }      
}

#Double Check it worked
validation <- rbind(final_train[1, -1] , validation) #note row 2 does not mean anything, this will be removed right.. now:
validation <- validation[-1,]

#Using ML algorithms for prediction: Decision Tree

Fit1 <- rpart(classe ~ ., data=final_train, method="class")

##To view the decision tree with fancy :
fancyRpartPlot(Fit1)


#Predicting:
  predict1 <- predict(Fit1, test_dat, type = "class")

##Using confusion Matrix to test results:
  confusionMatrix(predict1, test_dat$classe)


#Using ML algorithms for prediction: Random Forests
  Fit2 <- randomForest(classe ~. , data=final_train)

  ##Predicting in-sample error:
    
  predict2 <- predict(Fit2, test_dat, type = "class")

##Using confusion Matrix to test results:
    confusionMatrix(predict2, test_dat$classe)
    
#Generating Output for quiz submission
##For Random Forests we use the following formula, which yielded a much better prediction in in-sample:
  predict3 <- predict(Fit2, validation, type = "class")
  predict3

    