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
# Recode Prolonged_Hospital_Stay as a binary variable
data$Prolonged_Hospital_Stay <- ifelse(data$Total.Days.In.Hospital > 11, 1, 0)

# Univariate logistic regression analysis
univariate_model <- glm(Prolonged_Hospital_Stay ~ Renal.Impairment, data = data, family = "binomial")
univariate_results <- summary(univariate_model)$coefficients[, c(1, 4)]

# Multivariate logistic regression analysis
multivariate_model <- glm(Prolonged_Hospital_Stay ~ Age + Renal.Impairment + Diabetes.General, data = data, family = "binomial")
multivariate_results <- summary(multivariate_model)$coefficients[, c(1, 4)]

# Create the odds ratio table
odds_ratio_table <- data.frame(
  Variable = c("Renal Impairment", "Age", "Renal Impairment", "Diabetes.General"),
  Level = c("Moderately Impaired", "", "Severely Impaired", ""),
  OR = c(univariate_results[2, 1], multivariate_results[2:4, 1]),
  P.value = c(univariate_results[2, 2], multivariate_results[2:4, 2]),
  stringsAsFactors = FALSE
)

# Print the odds ratio table
knitr::kable(odds_ratio_table, align = "c",
             col.names = c("Variable", "Level", "OR", "P-value"),
             caption = "Table: Logistic Regression Results - Unadjusted and Adjusted Odds Ratios")
###################################################################################################################
# Load the required libraries
library(caret)
# Create a new variable based on Total.Days.In.Hospital
data$Prolonged_Hospital_Stay <- ifelse(data$Total.Days.In.Hospital > 11, "Prolonged", "Not Prolonged")
# Split the data into training and testing sets
set.seed(123)  # For reproducibility
train_indices <- createDataPartition(data$Prolonged_Hospital_Stay, p = 0.7, list = FALSE)
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]
unique(train_data$Prolonged_Hospital_Stay)
class(train_data$Prolonged_Hospital_Stay)
train_data$Prolonged_Hospital_Stay <- as.factor(train_data$Prolonged_Hospital_Stay)
# Perform univariate logistic regression analysis
univariate_model <- glm(Prolonged_Hospital_Stay ~ ., data = train_data, family = "binomial", maxit = 100)
univariate_results <- summary(univariate_model)$coefficients[, c(1, 4)]
# Perform multivariate logistic regression analysis
multivariate_model <- stepAIC(univariate_model, direction = "forward", trace = FALSE)
multivariate_results <- summary(multivariate_model)$coefficients[, c(1, 4)]
# Create the odds ratio table
odds_ratio_table <- data.frame(
  Variable = rownames(univariate_results),
  Level = "",
  Unadjusted_OR = univariate_results[, 1],
  Unadjusted_Pvalue = univariate_results[, 4],
  Adjusted_OR = multivariate_results[, 1],
  Adjusted_Pvalue = multivariate_results[, 3],
  stringsAsFactors = FALSE
)
# Print the odds ratio table
print(odds_ratio_table)

##################################################################################################################
# Predict probabilities on test data
test_prob <- predict(univariate_model, newdata = test_data, type = "response")

# Create a binary outcome based on a threshold
threshold <- 0.5
test_pred <- ifelse(test_prob > threshold, 1, 0)

# Create a confusion matrix
confusion_matrix <- table(test_data$Prolonged_Hospital_Stay, test_pred)

# Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
TPR <- confusion_matrix[2, 2] / sum(confusion_matrix[2, ])
FPR <- confusion_matrix[1, 2] / sum(confusion_matrix[1, ])

# Plot the ROC curve
library(pROC)
roc_obj <- roc(test_data$Prolonged_Hospital_Stay, test_prob)
plot(roc_obj, main = "ROC Curve", xlab = "False Positive Rate (1 - Specificity)", ylab = "True Positive Rate (Sensitivity)")

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
library(cluster)

gap_stat <- clusGap(scaled_data, FUN = kmeans, nstart = 10, K.max = 10, B = 50)

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
