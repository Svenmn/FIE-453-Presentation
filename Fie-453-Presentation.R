# FIE 453 Presentation

library(readr)
library(dplyr)
library(caret)
library(ggplot2)


#The compustat dataset
raw_data <- read_csv("compustat.csv")

#The field explanations as a Dataframe
sorted_fields <- read.delim("compustat-fields.txt", sep = "\t", header = TRUE)

