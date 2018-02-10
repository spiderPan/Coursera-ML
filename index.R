library(caret)
library(ranger)
library(tidyverse)
library(GGally)
library(corrplot)
library(xgboost)
library(vtreat)
library(magrittr)

train_data <- read_csv('pml-training.csv')

summary(train_data)

train_cols <- c("accel_forearm_x",
                "accel_forearm_y",
                "accel_forearm_z",
                "accel_arm_x",
                "accel_arm_y",
                "accel_arm_z",
                "accel_belt_x",
                "accel_belt_y",
                "accel_belt_z",
                "accel_dumbbell_x",
                "accel_dumbbell_y",
                "accel_dumbbell_z",
                "classe")

train_data<- train_data[,colnames(train_data) %in% train_cols]
head(train_data)

#ggpairs(train_data[,c(1:7,13)],aes(col=classe,alpha=.4))

#corrplot(train_data,method='square')

inTrain = createDataPartition(train_data$classe, p = 3/4)[[1]]
training = train_data[inTrain,]
testing = train_data[-inTrain,]

treatplan<-designTreatmentsZ(training,train_cols[-13],verbose = F)
new_vars<-treatplan%>%
  use_series(scoreFrame)%>%
  filter(code %in% c("lev","clean"))%>%
  use_series(varName)

training.treat<-prepare(treatplan,training,varRestriction = new_vars)
testing.treat<-prepare(treatplan,testing,varRestriction = new_vars)

training$classe<-training$classe %>% as.factor()

training$classe<-training$classe %>% as.numeric()

testing$classe<-testing$classe %>% as.factor() %>% as.numeric()

xgb_model <- xgboost(data = as.matrix(training.treat), # training data as matrix
                          label = training$classe-1,  # column of outcomes
                          nrounds = 1000,       # number of trees to build
                          objective          = "multi:softmax",   # default = "reg:linear"
                          num_class          = 5,
                          eta = .3,
                          depth = 6,
                          verbose = 0  # silent
)

xgb_result<- predict(xgb_model,as.matrix(testing.treat))+1

confusionMatrix(xgb_result,testing$classe)
