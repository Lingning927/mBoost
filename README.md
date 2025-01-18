# mBoost

**mBoost** is a statistical framework designed to comprehensively evaluate the performance of predictive models, focusing on their optimality during training and their applicability to new datasets. This package implemented the evaluation metric \( p_s \) for structure improvement and the train-test dataset similarity metric Coverage Ratio(CR). For structure improvement evaluation, we support the continuous or binary phenotypes.

## Installation

To install the latest development builds directly from GitHub, run this instead:

```r
if (!require("devtools"))
  install.packages("devtools")
devtools::install_github("Lingning927/mBoost")
```

## Using mBoost
Determine whether the model structure can be improved. If \( p_s < 0.05 \), we need to use more complex model.
```r
library("mBoost")

#generate data
set.seed(100)
n <- 200
X <- runif(n, 0, 1)
e <- rnorm(n, 0, 1)
Y <- 10 * (X - 0.2)^2 + e

#true model
X_true <- seq(0, 1, length.out = 200)
Y_true <- 10 * (X_true - 0.2)^2

#linear model
linear_model <- lm(Y ~ X)

#model eval
summary(linear_model)
Residuals <- linear_model$residuals
Y2 <- linear_model$fitted.values
ps_linear <- ps_continuous(X, Y, Y2)
print(ps_linear)

quadratic_model <- lm(Y ~ X + I(X^2))
summary(quadratic_model)
Y3 <- quadratic_model$fitted.values
ps_quadratic <- ps_continuous(X, Y, Y3)
print(ps_quadratic)
```

```r
#data generation
set.seed(123)
n <- 200
theta1 <- runif(n, -pi / 4, 3*pi/4)
x1 <- 5 * cos(theta1) + rnorm(n, sd = 0.5)
y1 <- 10 * sin(theta1) + rnorm(n, sd = 0.5)
class1 <- rep(0, n)

theta2 <- runif(n, -pi / 4, 3*pi/4)
x2 <- 5 * cos(theta2 + pi) + rnorm(n, 5, sd = 0.5)
y2 <- 10 * sin(theta2 + pi) + rnorm(n, -5, sd = 0.5)
class2 <- rep(1, n)

X <- rbind(cbind(x1, y1), cbind(x2, y2))
y <- c(class1, class2)
colnames(X) <- c("X1", "X2")

trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
trainData <- X[trainIndex, ]
trainLabels <- y[trainIndex]
testData <- X[-trainIndex, ]
testLabels <- y[-trainIndex]


lr_model <- glm(as.factor(y) ~ ., data = data.frame(X, y = as.factor(y)), family = "binomial")

# SVM with radial
svm_rbf <- svm(as.factor(y) ~ ., data = data.frame(X, y = as.factor(y)), kernel = "radial", probability = TRUE)

# randomForest
rf_model <- randomForest(as.factor(y) ~ ., data = data.frame(X, y = as.factor(y)))

# predicted probilities
lr_train_prob <- predict(lr_model, type = "response", newdata = data.frame(trainData))

svm_rbf_train_prob <- attr(predict(svm_rbf, newdata = data.frame(trainData), probability = TRUE), "probabilities")[,2]

rf_train_prob <- predict(rf_model, type = "prob", newdata = data.frame(trainData))[,2]

#model evaluation
lr_pred <- predict(lr_model, newdata = data.frame(testData), type = "response")
lr_acc <- mean((lr_pred > 0.5) == testLabels)

svm_rbf_pred <- predict(svm_rbf, newdata = data.frame(testData))
svm_rbf_acc <- mean(svm_rbf_pred == testLabels)

rf_pred <- predict(rf_model, newdata = data.frame(testData))
rf_acc <- mean(rf_pred == testLabels)
cat("Accuracy on test set for linear model:", lr_acc, "\n")
cat("Accuracy on test set for SVM model:", svm_rbf_acc, "\n")
cat("Accuracy on test set for RF model:", rf_acc, "\n")
p_linar <- ps_probs(trainData, trainLabels, lr_train_prob)
p_svm <- ps_probs(trainData, trainLabels, svm_rbf_train_prob)
p_rf <- ps_probs(trainData, trainLabels, rf_train_prob)
cat("PS of Linear:", p_linar, "\n")
cat("PS of SVM:", p_svm, "\n")
cat("PS of RF:", p_rf, "\n")

```

Measure the similarity between the training set and the test set. If CR < 0.5, we believe that this test set is significantly different from the training set and may not be applicable to the corresponding model.

```r
X_train <- matrix(rnorm(2000), 200, 10)
X_test <- matrix(rnorm(1000, 1), 100, 10)
coverage_ratio(X_train, X_test)
```
