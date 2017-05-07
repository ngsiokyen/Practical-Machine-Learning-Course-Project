# Practical Machine Learning Course Project
Ng Siok Yen  

# Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, our goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.


#Loading Libraries
The necessary libraries are loaded.

```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(rpart)
library(rattle)
```

```
## Rattle: A free graphical interface for data mining with R.
## XXXX 4.1.0 Copyright (c) 2006-2015 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
library(rpart.plot)
library(randomForest)
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```


# Loading Data

```r
#Download the data
if(!file.exists("pml-training.csv")){download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = "pml-training.csv")}

if(!file.exists("pml-testing.csv")){download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = "pml-testing.csv")}


#Read the training data and replace empty values by NA
TrainSet<- read.csv("pml-training.csv", sep=",", header=TRUE, na.strings = c("NA","",'#DIV/0!'))
TestSet<- read.csv("pml-testing.csv", sep=",", header=TRUE, na.strings = c("NA","",'#DIV/0!'))
dim(TrainSet)
```

```
## [1] 19622   160
```

```r
dim(TestSet)
```

```
## [1]  20 160
```

# Data Cleaning
Data cleaning is carried out by removing variables that have a nearly zero variance, variables that are almost always NA and variables that don't make intuitive sense for prediction.

```r
# Remove variables with Nearly Zero Variance
NZV <- nearZeroVar(TrainSet)
TrainSet <- TrainSet[, -NZV]
TestSet  <- TestSet[, -NZV]
dim(TrainSet)
```

```
## [1] 19622   124
```

```r
dim(TestSet)
```

```
## [1]  20 124
```

```r
# Remove variables that are mostly NA
AllNA    <- sapply(TrainSet, function(x) mean(is.na(x))) > 0.95
TrainSet <- TrainSet[, AllNA==FALSE]
TestSet  <- TestSet[, AllNA==FALSE]
dim(TrainSet)
```

```
## [1] 19622    59
```

```r
dim(TestSet)
```

```
## [1] 20 59
```

```r
# Remove identification only variables (columns 1 to 6)
TrainSet <- TrainSet[, -(1:6)]
TestSet  <- TestSet[, -(1:6)]
dim(TrainSet)
```

```
## [1] 19622    53
```

```r
dim(TestSet)
```

```
## [1] 20 53
```

With the cleaning process above, the number of variables for the analysis has been reduced to 53.

# Partition Data for Cross Validation
cross validation dataset is created to compare the model created by the training subset.

```r
# Create a partition with the training dataset
set.seed(100)
inTrain  <- createDataPartition(TrainSet$classe, p=0.7, list=FALSE)
TrainingSet <- TrainSet[inTrain, ]
ValidationSet  <- TrainSet[-inTrain, ]
```

# Prediction Model Building
Two methods will be applied to model the regressions (in the Train dataset) and the best one (with higher accuracy when applied to the Validation dataset) will be used for the quiz predictions. The methods are: Decision Tree and Random Forests as described below.

### a) Method: Decision Trees
5-fold cross validation (default setting in trainControl function is 10) is considered when implementing the algorithm to save a little computing time. Since data transformations may be less important in non-linear models like classification trees, we do not transform any variables.

```r
# Model fit
set.seed(200)
control <- trainControl(method="cv", number = 5)
mod_rpart <- train(classe ~ ., data = TrainingSet, method = "rpart", trControl = control)
mod_rpart$finalModel
```

```
## n= 13737 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 13737 9831 A (0.28 0.19 0.17 0.16 0.18)  
##    2) roll_belt< 129.5 12496 8634 A (0.31 0.21 0.19 0.18 0.11)  
##      4) pitch_forearm< -26.65 1212   52 A (0.96 0.043 0 0 0) *
##      5) pitch_forearm>=-26.65 11284 8582 A (0.24 0.23 0.21 0.2 0.12)  
##       10) magnet_dumbbell_y< 436.5 9496 6852 A (0.28 0.18 0.24 0.19 0.1)  
##         20) roll_forearm< 122.5 5844 3470 A (0.41 0.18 0.18 0.17 0.058) *
##         21) roll_forearm>=122.5 3652 2411 C (0.074 0.17 0.34 0.23 0.18) *
##       11) magnet_dumbbell_y>=436.5 1788  875 B (0.032 0.51 0.041 0.23 0.19) *
##    3) roll_belt>=129.5 1241   44 E (0.035 0 0 0 0.96) *
```

```r
fancyRpartPlot(mod_rpart$finalModel)
```

![](Practical_Machine_Learning_Course_Project_files/figure-html/unnamed-chunk-5-1.png)<!-- -->

```r
# Predict outcomes using validation set
predict_rpart <- predict(mod_rpart, ValidationSet)

# Show prediction result
(confmat_rpart <- confusionMatrix(ValidationSet$classe, predict_rpart))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1493   33  118    0   30
##          B  506  369  264    0    0
##          C  504   35  487    0    0
##          D  438  170  356    0    0
##          E  152  141  287    0  502
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4845          
##                  95% CI : (0.4716, 0.4973)
##     No Information Rate : 0.5256          
##     P-Value [Acc > NIR] : 1               
##                                           
##                   Kappa : 0.3256          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.4827   0.4933  0.32209       NA   0.9436
## Specificity            0.9352   0.8501  0.87674   0.8362   0.8916
## Pos Pred Value         0.8919   0.3240  0.47466       NA   0.4640
## Neg Pred Value         0.6200   0.9201  0.78905       NA   0.9938
## Prevalence             0.5256   0.1271  0.25692   0.0000   0.0904
## Detection Rate         0.2537   0.0627  0.08275   0.0000   0.0853
## Detection Prevalence   0.2845   0.1935  0.17434   0.1638   0.1839
## Balanced Accuracy      0.7089   0.6717  0.59942       NA   0.9176
```

```r
# Accuracy of the model
(accuracy_rpart <- confmat_rpart$overall[1])
```

```
## Accuracy 
## 0.484452
```

```r
# Out of sample error estimate
(out_of_sample_error_rpart <- 1 - as.numeric(accuracy_rpart))
```

```
## [1] 0.515548
```

From the confusion matrix, the accuracy rate is 0.48, and so the out-of-sample error rate is 0.52. Using decision tree does not predict the outcome classe very well.


### b) Method: Random Forest
Since decision tree method does not perform well, we try random forest method instead.

```r
# Model fit
set.seed(200)
mod_rf <- train(classe ~ ., data=TrainingSet, method="rf", trControl=control)
mod_rf
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 10990, 10991, 10988, 10988, 10991 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9902457  0.9876607
##   27    0.9895173  0.9867389
##   52    0.9826017  0.9779883
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

```r
# Predict outcomes using validation set
predict_rf <- predict(mod_rf, ValidationSet)

# Show prediction result
(confmat_rf <- confusionMatrix(ValidationSet$classe, predict_rf))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    1    0    0    0
##          B    7 1132    0    0    0
##          C    0    5 1018    3    0
##          D    0    0   15  948    1
##          E    0    0    0    1 1081
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9944          
##                  95% CI : (0.9921, 0.9961)
##     No Information Rate : 0.2855          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9929          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9958   0.9947   0.9855   0.9958   0.9991
## Specificity            0.9998   0.9985   0.9984   0.9968   0.9998
## Pos Pred Value         0.9994   0.9939   0.9922   0.9834   0.9991
## Neg Pred Value         0.9983   0.9987   0.9969   0.9992   0.9998
## Prevalence             0.2855   0.1934   0.1755   0.1618   0.1839
## Detection Rate         0.2843   0.1924   0.1730   0.1611   0.1837
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9978   0.9966   0.9919   0.9963   0.9994
```

```r
# Accuracy of the model
(accuracy_rf <- confmat_rf$overall[1])
```

```
##  Accuracy 
## 0.9943925
```

```r
# Out of sample error estimate
(out_of_sample_error_rf <- 1 - as.numeric(accuracy_rf))
```

```
## [1] 0.005607477
```

For this dataset, random forest method is way better than decision tree method. The accuracy rate is 0.9944, and so the out-of-sample error rate is 0.0056. This may be due to the fact that many predictors are highly correlated. Random forests chooses a subset of predictors at each split and decorrelate the trees. This leads to high accuracy, although this algorithm is sometimes difficult to interpret and computationally inefficient.


# Prediction on Testing Set
In this case, random forests is used to predict the outcome variable classe for the testing set.

```r
(predict(mod_rf, TestSet))
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
