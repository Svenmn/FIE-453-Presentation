# FIE 453 Presentation

#--------------Loading Packages-----------------------
library(readr)
library(dplyr)
library(caret)
library(ggplot2)
library(gbm)
library(xgboost)
library(lightgbm)
library(catboost)
library(Matrix)
library(Metrics)
library(pROC)

#--------------Loading Data---------------------------

#The compustat dataset
raw_data <- read_csv("compustat.csv")

#The field explanations as a Dataframe
sorted_fields <- read.delim("compustat-fields.txt", sep = "\t", header = TRUE)

#--------------Examining Data--------------

#Function for indicating whether EPS is positive or negative
split_epspxq_indicator <- function(data) {
  data %>%
    mutate(
      epspxq_sign = case_when(
        epspxq > 0  ~ 1L,
        epspxq < 0  ~ 0L,
        epspxq == 0 ~ NA_integer_,  # Assign NA for zero values
        TRUE        ~ NA_integer_
      )
    )
}

# Filter data
filtered_data <- raw_data %>%
  split_epspxq_indicator() %>%
  filter(!is.na(epspxq_sign), fyearq %in% c(2018:2019)) %>%
  mutate(atq_ltq = atq / ltq) %>% 
  select(-epspxq) # Exclude EPS to prevent leakage
  

#positive_eps, xintq npatq, atq_ltq, dlttq, icaptq, opvolq

# Select predictor variables (excluding variables that can compute EPS directly)
predictor_vars <- c("xintq", #Interest and Related Expense- Total
                    #"npatq", #Non-Performing Assets
                    "atq_ltq", #Current Assets - Total / Current Liabilites Total
                    "dlttq", #Long-Term Debt - Total
                    "icaptq" #Invested Capital - Total - Quarterly
                    #"optvolq") # Option volotility
)
# Prepare the dataset
data_model <- filtered_data %>%
  select(epspxq_sign, all_of(predictor_vars))
    # Remove rows with missing values

#--------------Data Splitting -------------
N <- nrow(data_model)
N
# Optional: Shuffle the data to ensure randomness
set.seed(123)  # For reproducibility
data_model <- data_model[sample(N), ]

train.indices <- 1:floor(N*0.6);
valid.indices <- floor(N*0.6+1):floor(N*0.8);
test.indices  <- floor(N*0.8+1):N

# Split the data using the indices
train_data <- data_model[train.indices, ]
val_data   <- data_model[valid.indices, ]
test_data  <- data_model[test.indices, ]

# Separate features and target variable
train_x <- train_data %>% select(-epspxq_sign)
train_y <- train_data$epspxq_sign

val_x <- val_data %>% select(-epspxq_sign)
val_y <- val_data$epspxq_sign

test_x <- test_data %>% select(-epspxq_sign)
test_y <- test_data$epspxq_sign


#--------------Model Training and Evaluation--------------

# Define a function to calculate performance metrics
evaluate_performance <- function(preds, true_labels) {
  preds_class <- ifelse(preds > 0.5, 1, 0)
  confusionMatrix(factor(preds_class), factor(true_labels), positive = "1")
}

#-------------Functions ---------------------------------

calculate_mse <- function(predictions, actual_values) {
  mse <- mean((predictions - actual_values)^2)
  return(mse)
}

#-------------Dumb Model------------------------------

dumb_pred <- sum(filtered_data$epspxq_sign[train.indices]) / nrow(train_x)

#rep(sum(filtered_data$epspxq_sign[train.indices]) / nrow(train_x), nrow(train_x))

dumb_preds <- rep(dumb_pred, nrow(val_x))
valid_values <- filtered_data$epspxq_sign[valid.indices]

calculate_mse(dumb_preds, valid_values)

#-------------GBM Model----------------------------------------
gbm_model <- gbm(
  formula = epspxq_sign ~ .,
  data = train_data,
  distribution = "bernoulli",
  n.trees = 2000,
  interaction.depth = 3,
  shrinkage = 0.05,
  cv.folds = 5,
  verbose = FALSE
)


#--------------Find the Optimal tree count------------
best_iter <- gbm.perf(gbm_model, method = "cv", plot.it = TRUE)

# Predict on validation set
gbm_preds_val <- predict(gbm_model, val_x, n.trees = best_iter, type = "response")
gbm_conf_mat_val <- evaluate_performance(gbm_preds_val, val_y)

print("GBM Model Validation Results:")
print(gbm_conf_mat_val)


roc_obj_val <- roc(val_y, gbm_preds_val)
auc_val <- auc(roc_obj_test)

#-------------GBM: Predicting on Test Set-------------------

# Use the optimal number of trees found during validation (`best_iter`)
gbm_preds_test <- predict(gbm_model, test_x, n.trees = best_iter, type = "response")

# Convert predictions to binary classification (0 or 1)
# (You can adjust the threshold if needed, here we use 0.5)
gbm_preds_test_class <- ifelse(gbm_preds_test > 0.5, 1, 0)

# Evaluate the performance on the test set
gbm_conf_mat_test <- confusionMatrix(factor(gbm_preds_test_class), factor(test_y), positive = "1")

# Print the confusion matrix and performance metrics
print("GBM Model Test Results:")
print(gbm_conf_mat_test)

# Optionally, compute AUC for the test set
roc_obj_test <- roc(test_y, gbm_preds_test)
auc_test <- auc(roc_obj_test)


#--------------Comparing Validation vs Test Set Metrics-------------------

cat("\nPerformance Comparison:\n") %>% 
cat("Accuracy (Validation): ", gbm_conf_mat_val$overall['Accuracy'], "\n") %>% 
cat("Accuracy (Test): ", gbm_conf_mat_test$overall['Accuracy'], "\n") %>% 
cat("No Information Rate: ", dumb_pred, "\n") %>% 
cat("AUC (Validation): ", auc_val, "\n") %>% 
cat("AUC (Test): ", auc_test, "\n")


# Plot ROC curve for the validation and test sets
plot(roc_obj_val, col = "blue", main = "ROC Curve Comparison", lwd = 2)
plot(roc_obj_test, col = "red", add = TRUE, lwd = 2)  
legend("bottomright", legend = c("Validation", "Test"), col = c("blue", "red"), lwd = 2)

#---------------CatBoost------------

train_pool <- catboost.load_pool(data = as.matrix(train_x), label = train_y)
val_pool <- catboost.load_pool(data = as.matrix(val_x), label = val_y)
testing_pool <- catboost.load_pool(data = as.matrix(test_x), label = test_y)

# Train CatBoost model with early stopping
cat_model <- catboost.train(
  train_pool,
  params = list(
    fold_permutation_block = 5,
    use_best_model = TRUE,
    loss_function = "Logloss",
    iterations = 2000,
    depth = 3,
    learning_rate = 0.05,
    eval_metric = "Logloss",
    train_dir = "catboost_info"
  ),
  test_pool = val_pool  
)

# Best iteration
best_nrounds <- cat_model$best_iteration


cat_model



#----------------Performance Metrics---------

# Predict probabilities on the validation set
cat_preds_val <- catboost.predict(cat_model, val_pool, prediction_type = "Probability") 
cat_preds_val_class <- ifelse(cat_preds_val > 0.5, 1, 0)  
val_labels <- val_y  
conf_matrix <- confusionMatrix(factor(cat_preds_val_class), factor(val_labels), positive = "1")

# Display the confusion matrix
print("Confusion Matrix for Validation Set:")
print(conf_matrix)

# Display the accuracy of the model
accuracy <- conf_matrix$overall['Accuracy']
cat("Accuracy of CatBoost Model on Validation Set: ", accuracy, "\n")


# Predict probabilities on the test set
cat_preds_test <- catboost.predict(cat_model, testing_pool, prediction_type = "Probability")
cat_preds_test_class <- ifelse(cat_preds_test > 0.5, 1, 0)
test_labels <- test_y
conf_matrix_test <- confusionMatrix(factor(cat_preds_test_class), factor(test_labels), positive = "1")

# Display the confusion matrix for the test set
print("Confusion Matrix for Test Set:")
print(conf_matrix_test)

# Display the accuracy of the model on the test set
accuracy_test <- conf_matrix_test$overall['Accuracy']
cat("Accuracy of CatBoost Model on Test Set: ", accuracy_test, "\n")

