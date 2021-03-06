library(caret)
library(ranger)

train_data <- read.csv('pml-training.csv')

summary(train_data)
head(train_data$accel_dumbbell_x)

inTrain = createDataPartition(train_data$classe, p = 3/4)[[1]]
training = train_data[inTrain,]
testing = train_data[-inTrain,]

colnames(training)

mod_rf <- train(classe~accel_forearm_x+accel_forearm_y+accel_forearm_z+
                  accel_arm_x+accel_arm_y+accel_arm_z+
                  accel_belt_x+accel_belt_y+accel_belt_z+
                  accel_dumbbell_x+accel_dumbbell_y+accel_dumbbell_z
  , data=training, method="ranger")

pred_rf <- predict(mod_rf,testing)

confusionMatrix(pred_rf,testing$classe)
