# Report 

# Introduction 
CaTHI, an abbreviation that stands for cardiac, cardiology, and thoracic health information, is a
comprehensive dataset that provides information on a variety of heart and lung problems. It is
common practise for healthcare practitioners, researchers, and policymakers to utilise the CaTHI
dataset in order to assess the quality of service, determine areas in need of improvement, and
create innovative treatment options. The study of the CaTHI dataset can give insights into the
epidemiology, risk factors, and outcomes of these ailments, which is particularly useful in light of the
rising incidence of cardiovascular diseases and lung problems around the globe. The patient
database contains information on people who have been given diagnoses for a range of ailments,
including lung cancer, heart failure, arrhythmias, pulmonary embolism, and coronary artery disease
(Kim et al., 2019). The collection also includes data on individuals&#39; medical histories and clinical
outcomes throughout time, allowing researchers to follow the development of diseases and gauge
the efficacy of treatments (Hofmann et al., 2019).
As an example, CaTHI datasets include the National Cardiovascular Data Registry (NCDR), which
records information on millions of cardiac operations performed annually in the United States.
Health care practitioners and researchers use the NCDR to assess the quality of cardiovascular
treatment and find ways to enhance it by analysing data on patient demographics, procedural
specifics, and results (Brindis et al., 2010).
For a number of reasons, it is crucial to analyse the Cardiac, Cardiology, and Thoracic Health
Information (CaTHI) dataset. It can first aid in the identification of risk factors for cardiovascular
illnesses (CVD), including as coronary artery disease, heart failure, and arrhythmias. Researchers can
find patterns and trends in patient data by evaluating the dataset, which may point to CVD risk
factors. Age, gender, hypertension, and diabetes, for instance, were all identified in one study
utilising the CaTHI dataset as major risk factors for developing heart failure (Zhang et al., 2019).
They can also investigate the efficacy of various CVD therapies and interventions and find areas for
improvement. Additionally, the CaTHI dataset may be used to examine trends and changes in CVD
epidemiology across time and across communities, informing public health policies and treatments
(American College of Cardiology, n.d). According to a study utilising the CaTHI dataset, treatment for
heart failure that is advised by guidelines is linked to a decreased chance of passing away or being
admitted to the hospital(Shah et al., 2019). Another study utilising the CaTHI dataset revealed that
medication adherence was linked to a lower risk of cardiovascular events and all-cause
death(Stefanovic et al., 2019)..
The primary objective of this study is to calculate the typical duration that a patient is required to
remain hospitalised following heart surgery. The data set includes a wide variety of parameters and
datapoints in order for us to find our objective, to extract patterns and conduct analysis from this
information, we will be employing a number of different approaches. At the beginning, we are going
to characterise the dataset by employing descriptive statistics in order to ensure that all the
information regarding the dataset is well understood. After that, for the analysis, we are going to
begin utilising a linear regression model to predict the total amount of time spent at the hospital.
Second, we will also make use of logistic regression in order to forecast which patients will have a
length of stay of 11 days or fewer, and which patients will remain in the hospital for more than 11

days. In the end, we will use a technique called K-means clustering to group the patients together
according to the amount of time they have spent in the hospital.
Hence for our report, The CaTHI dataset may be used to estimate the duration of stay for heart
surgery patients, which can assist healthcare practitioners manage resources more efficiently and
offer patients and their families with a more realistic picture of the recovery process. Given the
rising usage of machine learning algorithms in healthcare, this strategy offers the potential to
enhance patient outcomes while also lowering healthcare expenditures and more importantly
patient stay durations over time (Krittanawong et al, 2019).
<hr> 

## Summary of the code

## Data Loading and Exploration:

1. The code reads a CSV file named "DataSet5.csv" and assigns it to the variable data.

2. The head(data) function displays the first few rows of the dataset.
Mean and standard deviation calculations are performed for several variables, including Age, ICU Hours, BMI, and Pre.Op.Creatinine.
3. Histograms are created to visualize the distributions of Age and Total.Days.In.Hospital variables.

## Frequency Analysis:

1. Frequency tables are generated for variables such as Sex, Diabetes.General, and Renal.Impairment.

## Boxplot:

1. A boxplot is created to examine the relationship between Total.Days.In.Hospital and Sex.

## Linear Regression:

1. The code builds a linear regression model (lm) to predict Total.Days.In.Hospital using Age and Renal.Impairment as predictors.
2. Model coefficients, standard errors, and p-values are extracted and stored in variables.
3. A regression table is created to display the model's results using the kable function.

## Logistic Regression:

1. The code performs logistic regression analysis to predict Prolonged_Hospital_Stay.
2. Both univariate and multivariate logistic regression models are built using the glm function.
3. Odds ratio tables are created to display the results.

## Model Training and Evaluation:

1. The dataset is split into training and testing sets using the createDataPartition function from the caret package.
2. Univariate logistic regression analysis is performed on the training data.
3. Multivariate logistic regression analysis is performed using the stepAIC function for feature selection.
4. The odds ratio table is created to display both unadjusted and adjusted odds ratios.

## Prediction and Evaluation:

1. Probabilities are predicted on the test data using the logistic regression model.
2. A binary outcome is created based on a threshold of 0.5.
3. A confusion matrix is generated to evaluate the model's performance.
4. True Positive Rate (TPR) and False Positive Rate (FPR) are calculated.
5. An ROC curve is plotted using the pROC package.

## K-means Clustering:

1. The code performs k-means clustering on a subset of variables from the dataset.
2. Elbow plot and silhouette plot are generated to determine the optimal number of clusters.
3. The chosen number of clusters (k) is used to create the final k-means clustering model.
4. Scatter plots are created to visualize the relationship between variables and Total.Days.In.Hospital, colored by cluster labels.
