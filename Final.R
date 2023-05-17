setwd("/Users/nvgenomics/Desktop/projects/ logistic regression part and the kmeans clustering part")
data <- read.csv("DataSet5.csv")
head(data)
###################################################################################
# Calculate mean
mean_age <- mean(data$Age)
mean_icu_hours <- mean(data$ICU.Hours)
mean_bmi <- mean(data$BMI)
mean_creatinine <- mean(data$Pre.Op.Creatinine)
# Calculate standard deviation
sd_age <- sd(data$Age)
sd_icu_hours <- sd(data$ICU.Hours)
sd_bmi <- sd(data$BMI)
sd_creatinine <- sd(data$Pre.Op.Creatinine)
# Print mean, standard deviation, and median
cat("Variable \t\t\t Mean +/- SD \t\t\t Mean \n")
cat("Age \t\t\t", round(mean_age, 2), "+/-", round(sd_age, 2), "\t\t\t", round(mean_age, 2), "\n")
cat("ICU Hours \t\t", round(mean_icu_hours, 2), "+/-", round(sd_icu_hours, 2), "\t\t\t", round(mean_icu_hours, 2), "\n")
cat("BMI \t\t\t", round(mean_bmi, 2), "+/-", round(sd_bmi, 2), "\t\t\t", round(mean_bmi, 2), "\n")
cat("Pre.Op.Creatinine \t", round(mean_creatinine, 2), "+/-", round(sd_creatinine, 2), "\t\t\t", round(mean_creatinine, 2), "\n")
# Create histogram for Age
hist(data$Age, main = "Age Distribution", xlab = "Age", ylab = "Frequency")
# Create histogram for Total.Days.In.Hospital
hist(data$Total.Days.In.Hospital, main = "Total Days in Hospital Distribution", xlab = "Total Days in Hospital", ylab = "Frequency")
###################################################################################

# Create a frequency table for Sex
sex_freq <- table(data$sex)
# Create a frequency table for Diabetes
diabetes_freq <- table(data$Diabetes.General)
# Create a frequency table for Renal Impairment
renal_impairment_freq <- table(data$Renal.Impairment)
# Display the frequency tables
cat("Frequency Table for Sex:\n")
print(sex_freq)
cat("\nFrequency Table for Diabetes:\n")
print(diabetes_freq)
cat("\nFrequency Table for Renal Impairment:\n")
print(renal_impairment_freq)

##########################################################################################
# Create a boxplot for Sex
boxplot(Total.Days.In.Hospital ~ Sex, data = data, 
        main = "Total Days in Hospital by Sex", xlab = "Sex", ylab = "Total Days in Hospital")

#####################################################################################################
install.packages('knitr', repos = c('http://rforge.net', 'http://cran.rstudio.org'),
                 type = 'source')
library(knitr)
# Build the linear regression model
model <- lm(Total.Days.In.Hospital ~ Age + Renal.Impairment, data = data)

# Extract the model coefficients
intercept <- coef(model)[1]
age_coeff <- coef(model)[2]
renal_moderate_coeff <- coef(model)[3]
renal_severe_coeff <- coef(model)[4]
renal_unknown_coeff <- coef(model)[5]

# Extract the standard errors
se_intercept <- summary(model)$coefficients[1, 2]
se_age <- summary(model)$coefficients[2, 2]
se_renal_moderate <- summary(model)$coefficients[3, 2]
se_renal_severe <- summary(model)$coefficients[4, 2]
se_renal_unknown <- summary(model)$coefficients[5, 2]

# Extract the p-values
p_intercept <- summary(model)$coefficients[1, 4]
p_age <- summary(model)$coefficients[2, 4]
p_renal_moderate <- summary(model)$coefficients[3, 4]
p_renal_severe <- summary(model)$coefficients[4, 4]
p_renal_unknown <- summary(model)$coefficients[5, 4]

library(knitr)

# Create the regression table
regression_table <- data.frame(
  Variable = c("Intercept", "Age", "Renal Impairment", "Renal Impairment", "Renal Impairment"),
  Level = c("", "", "Moderately Impaired", "Severely Impaired", "Unknown"),
  Estimate = c(intercept, age_coeff, renal_moderate_coeff, renal_severe_coeff, renal_unknown_coeff),
  Std.Error = c(se_intercept, se_age, se_renal_moderate, se_renal_severe, se_renal_unknown),
  P.value = c(p_intercept, p_age, p_renal_moderate, p_renal_severe, p_renal_unknown)
)

# Print the regression table
kable(regression_table, align = "c", format.args = list(digits = 3), 
      col.names = c("Variable", "Level", "Estimate", "Std. Error", "P-value"), 
      caption = "Table 3. Linear Regression Model")
#################################################################################################################
# Perform linear regression using forward selection
final_model <- step(lm(Total.Days.In.Hospital ~ ., data = train_data), direction = "forward")

# Extract the model coefficients
coef_table <- coef(final_model)

# Extract the standard errors
se_table <- summary(final_model)$coefficients[, 2]

# Extract the p-values
p_table <- summary(final_model)$coefficients[, 4]

# Create the regression table
regression_table <- data.frame(
  Variable = names(coef_table),
  Level = "",
  Estimate = coef_table,
  `St. Error` = se_table,
  `P-value` = p_table
)

# Print the regression table
kable(regression_table, align = "c", format.args = list(digits = 3), 
      col.names = c("Variable", "Level", "Estimate", "St. Error", "P-value"), 
      caption = "Table 4. Final Linear Regression Model")

###################################################################################################################
# Convert Total.Days.In.Hospital to numeric
data$Total.Days.In.Hospital <- as.numeric(as.character(data$Total.Days.In.Hospital))
# Calculate predicted values
predicted_values <- predict(model)

# Calculate MSE
mse <- mean((data$Total.Days.In.Hospital - predicted_values)^2)

# Calculate MAE
mae <- mean(abs(data$Total.Days.In.Hospital - predicted_values))

# Calculate R-squared
ss_total <- sum((data$Total.Days.In.Hospital - mean(data$Total.Days.In.Hospital))^2)
ss_residual <- sum((data$Total.Days.In.Hospital - predicted_values)^2)
r_squared <- 1 - (ss_residual / ss_total)

# Print evaluation metrics
cat("Mean Squared Error (MSE):", mse, "\n")
cat("Mean Absolute Error (MAE):", mae, "\n")
cat("R-squared:", r_squared, "\n")

library(pROC)

# Convert Total.Days.In.Hospital to numeric
data$Total.Days.In.Hospital <- as.numeric(as.character(data$Total.Days.In.Hospital))

# Set a threshold for converting the problem into binary classification
threshold <- 7

# Create binary labels based on the threshold
binary_labels <- ifelse(data$Total.Days.In.Hospital > threshold, 1, 0)

# Calculate predicted probabilities
predicted_probs <- predict(model, type = "response")

# Create ROC curve
roc_obj <- roc(binary_labels, predicted_probs)

# Plot ROC curve
plot(roc_obj, main = "ROC Curve", xlab = "False Positive Rate", ylab = "True Positive Rate")

# Add AUC value to the plot
auc_value <- auc(roc_obj)
text(0.5, 0.3, paste0("AUC = ", round(auc_value, 3)), cex = 1.2)

# Add a legend
legend("bottomright", legend = "Linear Regression", col = "black", lwd = 2)

#####################################################################################################################
#K MEANS CLUSTERING

library(cluster)

subset_data <- data[, c("Age", "ICU.Hours", "BMI", "Pre.Op.Creatinine", "Total.Days.In.Hospital")]
View(subset_data)
subset_data <- na.omit(subset_data)
scaled_data <- scale(subset_data)
wss <- numeric(10)

for (k in 1:10) {
  kmeans_model <- kmeans(scaled_data, centers = k, nstart = 10)
  wss[k] <- sum(kmeans_model$withinss)
}

plot(1:10, wss, type = "b", xlab = "Number of Clusters", ylab = "Within-Cluster Sum of Squares", main = "Elbow Plot")
sil_width <- silhouette(kmeans_model$cluster, dist(scaled_data))
sil_avg <- aggregate(sil_width[, "sil_width"], by = list(kmeans_model$cluster), FUN = mean)

plot(sil_avg[, "Group.1"], sil_avg[, "x"], type = "b", xlab = "Number of Clusters", ylab = "Average Silhouette Width", main = "Silhouette Plot")
kmeans_control <- function(n) {
  kmeans(nstart = 10, maxiter = n)
}

gap_stat <- clusGap(scaled_data, FUN = kmeans_control, K.max = 10, B = 100)

plot(gap_stat, main = "Gap Statistic")
k <- 3  # Choose the desired number of clusters based on the analysis

kmeans_model <- kmeans(scaled_data, centers = k, nstart = 10)
cluster_labels <- kmeans_model$cluster

# Plot the variables against Total.Days.In.Hospital
par(mfrow = c(2, 2))
plot(subset_data$Age, subset_data$Total.Days.In.Hospital, pch = cluster_labels, xlab = "Age", ylab = "Total Days in Hospital", main = "Age vs. Total Days in Hospital")
plot(subset_data$ICU.Hours, subset_data$Total.Days.In.Hospital, pch = cluster_labels, xlab = "ICU Hours", ylab = "Total Days in Hospital", main = "ICU Hours vs. Total Days in Hospital")
plot(subset_data$BMI, subset_data$Total.Days.In.Hospital, pch = cluster_labels, xlab = "BMI", ylab = "Total Days in Hospital", main = "BMI vs. Total Days in Hospital")
plot(subset_data$Pre.Op.Creatinine, subset_data$Total.Days.In.Hospital, pch = cluster_labels, col = cluster_labels, xlab = "Pre.Op.Creatinine", ylab = "Total.Days.In.Hospital", main = "Scatter Plot - Pre.Op.Creatinine vs. Total.Days.In.Hospital")
legend("topright", legend = unique(cluster_labels), col = unique(cluster_labels), pch = unique(cluster_labels), title = "Cluster Labels")

##############################################################################################################################################################################################################################################################
