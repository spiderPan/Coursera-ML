library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)
library(mice)
library(h2o)
library(mxnet)
library(GGally)

training<-read_csv("pml-training.csv")
testing<-read_csv("pml-testing.csv")

set.seed(0)

str(testing)
training<-training[,!sapply(testing,anyNA)]
testing<-testing[,!sapply(testing,anyNA)]
str(testing)

factors<-c("X1","user_name","raw_timestamp_part_1","raw_timestamp_part_2","new_window","num_window","cvtd_timestamp")

training<-training[,!names(training) %in% factors]
testing<-testing[,!names(testing) %in% c(factors,"problem_id")]

sum(is.na(training))

mod<-mice(training)
training<-complete(mod)

inTrain<-createDataPartition(training$classe,p=.7,list=F)

train<-training[inTrain,]
validation<-training[-inTrain,]

rm(inTrain,training,factors,mod)
train$classe<-as.factor(train$classe)
validation$classe<-as.factor(validation$classe)

train<-as.data.frame(lapply(train,as.numeric))
validation<-as.data.frame(lapply(validation,as.numeric))

train.x<-data.matrix(train[,-c(5,6,7,53)])
validation.x<-data.matrix(validation[,-c(5,6,7,53)])
train.y<-as.factor(train[,53])
train.x<-t(scale(train.x))
validation.x<-t(scale(validation.x))


m2.data<-mx.symbol.Variable("data")

## 1st convoluntional layer
m2.conv1<-mx.symbol.Convolution(m2.data,kernel=c(5,5),num_filter=16)
m2.bn1<-mx.symbol.BatchNorm(m2.conv1)
m2.act1<-mx.symbol.Activation(m2.bn1,act_type="relu")
m2.pool1<-mx.symbol.Pooling(m2.act1,pool_type='max',kernel=c(2,2),stride=c(2,2))
m2.drop1<-mx.symbol.Dropout(m2.pool1,p=.5)


## 2nd convoluntional layer
m2.conv2<-mx.symbol.Convolution(m2.drop1,kernel=c(3,3),num_filter=32)
m2.bn2<-mx.symbol.BatchNorm(m2.conv2)
m2.act2<-mx.symbol.Activation(m2.bn2,act_type="relu")
m2.pool2<-mx.symbol.Pooling(m2.act2,pool_type='max',kernel=c(2,2),stride=c(2,2))
m2.drop2<-mx.symbol.Dropout(m2.pool2,p=.5)
m2.flatten<-mx.symbol.Flatten(m2.drop2)

## 4 Fully Connected layer
m2.fc1<-mx.symbol.FullyConnected(m2.flatten,num_hidden=1024)
m2.ac3<-mx.symbol.Activation(m2.fc1,act_type="relu")

m2.fc2<-mx.symbol.FullyConnected(m2.ac3,num_hidden=512)
m2.ac4<-mx.symbol.Activation(m2.fc2,act_type="relu")

m2.fc3<-mx.symbol.FullyConnected(m2.ac4,num_hidden=256)
m2.ac5<-mx.symbol.Activation(m2.fc3,act_type="relu")

m2.fc4<-mx.symbol.FullyConnected(m2.ac5,num_hidden=5)
m2.softmax<-mx.symbol.SoftmaxOutput(m2.fc4)

train.array <- train.x
dim(train.array) <- c(7, 7, 1, ncol(train.x))

validation.array <- validation.x
dim(validation.array) <- c(7, 7, 1, ncol(validation.x))

m2 <- mx.model.FeedForward.create(m2.softmax, 
                                  X = train.array, 
                                  y = train.y,
                                  num.round = 1, # This many will take a couple of hours on a CPU
                                  array.batch.size = 500,
                                  array.layout="colmajor",
                                  learning.rate = 0.01,
                                  momentum = 0.91,
                                  wd = 0.00001,
                                  eval.metric = mx.metric.accuracy,
                                  initializer = mx.init.uniform(0.07))


m2.pred<-predict(m2,validation.array)
m2.predict.value<-max.col(t(m2.pred))
m2.predict.value%>%summary()