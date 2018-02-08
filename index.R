library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)
library(mice)
library(h2o)
library(mxnet)

training<-read_csv("pml-training.CSV")
testing<-read_csv("pml-testing.CSV")

set.seed(0)

str(testing)
training<-training[,!sapply(testing,anyNA)]
testing<-testing[,!sapply(testing,anyNA)]
str(testing)

factors<-c("X1","user_name","raw_timestamp_part_1","raw_timestamp_part_2","new_window","num_window")

training<-training[,!names(training) %in% factors]
testing<-testing[,!names(testing) %in% c(factors,"problem_id")]

sum(is.na(training))

mod<-mice(training)
training<-complete(mod)

inTrain<-createDataPartition(training$classe,p=.7,list=F)

train<-training[inTrain,]
validation<-training[-inTrain,]

rm(inTrain,training,factors,mod)


#random Forest  
rfmodel<-train(classe~.,model='rf',train)
rfresult<-predict(rfmodel,validation)
confusionMatrix(rfresult$,validation$classe)

#h2o
h2o.train<-as.h2o(train)
h2o.validation<-as.h2o(validation)

h2o.model<-h2o.deeplearning(x=setdiff(colnames(train),"classe"),
                            y="classe",
                            training_frame = h2o.train,
                            standardize = T,
                            hidden = c(100,100),
                            rate = 0.05,
                            epochs = 100,
                            seed=0
)

h2oresult<-h2o.predict(h2o.model,h2o.validation)
confusionMatrix(h2oresult$prediction,validation$classe)

#xgboost


#xmnet