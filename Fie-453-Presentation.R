# FIE 453 Presentation

#--------------Loading Packages-----------------------
library(readr)
library(dplyr)
library(caret)
library(ggplot2)

#--------------Loading Data---------------------------

#The compustat dataset
raw_data <- read_csv("compustat.csv")

#The field explanations as a Dataframe
sorted_fields <- read.delim("compustat-fields.txt", sep = "\t", header = TRUE)

#--------------Examining Data--------------

#Function for indicating wether EPS is positive or negative
split_epspiq_indicator <- function(data) {
  data %>%
    mutate(
      epspiq_sign = case_when(
        epspiq > 0  ~ "Positive",
        epspiq < 0  ~ "Negative",
        epspiq == 0 ~ "Zero",
        TRUE        ~ NA_character_
      )
    )
}

# Filter data
filtered_data <- raw_data %>%
  filter(costat == "A", !is.na(epspiq)) %>%
  select(epspiq, datadate)

# Apply the function
result_data <- split_epspiq_indicator(filtered_data)
result_data

sign_counts <- result_data %>%
  count(epspiq_sign)

  print(sign_counts)


  
  