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

dumb_preds <- rep(pred, nrow(val_x))
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
print(gbm_conf_mat)


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

# Display the comparison of Accuracy, AUC, and other relevant metrics
cat("\nPerformance Comparison:\n") %>% 
cat("Accuracy (Validation): ", gbm_conf_mat_val$overall['Accuracy'], "\n") %>% 
cat("Accuracy (Test): ", gbm_conf_mat_test$overall['Accuracy'], "\n") %>% 
cat("No Information Rate: ", dumb_pred, "\n") %>% 
cat("AUC (Validation): ", auc_val, "\n") %>% 
cat("AUC (Test): ", auc_test, "\n")


# Plot the ROC curve for the validation and test sets
plot(roc_obj_val, col = "blue", main = "ROC Curve Comparison", lwd = 2)
plot(roc_obj_test, col = "red", add = TRUE, lwd = 2)  
legend("bottomright", legend = c("Validation", "Test"), col = c("blue", "red"), lwd = 2)

