library(caret)
library(ranger)
library(Matrix)
library(xgboost)
library(tidyverse)

train_data <- read.csv('pml-training.csv')

selected_cols <- c("accel_forearm_x",
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

train_data<-train_data[,colnames(train_data) %in% selected_cols]

summary(train_data)
head(train_data$accel_dumbbell_x)

inTrain = createDataPartition(train_data$classe, p = 3/4)[[1]]
training = train_data[inTrain,]
testing = train_data[-inTrain,]

colnames(training)

mod_rf <- train(classe~., data=training, method="ranger")
sparse_matrix <- sparse.model.matrix(classe~.-1, data = training)
testing_matrix <- sparse.model.matrix(classe~.-1, data = testing)
training[,'classe'] <- as.numeric(factor(training[,'classe']))
testing[,'classe'] <- as.numeric(factor(testing[,'classe']))
mod_xgboost <- xgboost(data = sparse_matrix, label = training$classe, max_depth = 6, eta = 0.3, nthread = 2, nrounds = 1000, objective = "multi:softmax", num_class=6)
#pred_rf <- predict(mod_rf,testing)
pred_rf <- predict(mod_xgboost,testing_matrix)
confusionMatrix(pred_rf,testing$classe)
