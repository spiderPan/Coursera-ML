library(caret)
train_data <- read.csv('pml-training.csv')

summary(train_data)
head(train_data$accel_dumbbell_x)

inTrain = createDataPartition(train_data$classe, p = 3/4)[[1]]
training = train_data[inTrain,]
testing = train_data[-inTrain,]

mod_rf <- train(classe~., data=training, method="rf")

pred_rf <- predict(mod_rf,testing)

confusionMatrix(pred_rf,testing$classe)
