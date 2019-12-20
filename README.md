# Weather Data Project

## Description
This project investigates the performance between SVM and Fully Connected Neural Networks in predicting Precipitation values from the HCN Dataset.

More information can be found in our presentation slides `FINAL_aqngo_ellin_ML_presentation.pdf`.

## Lab Notebook

### 12/17/2019 (3 hrs)
- Jason: Finish generating training and testing sets for nearby stations experiments
- Emily: Determine best hyperparameters for SVC for nearby stations;
  Get confusion matrix for one and nearby stations; plot validation curves.

### 12/15/2019 (2 hrs)
- Jason: Get started on getting the two nearest stations for a given station
- Emily: Determine best hyperparameters for SVC for one station;

### 12/11/2019 (2 hrs)
- Jason: Finish pairing the dataset by two day increments
- Emily: Code to generate confusion matrices and heatmaps.

### 12/9/2019 (2 hrs)
- Jason: Pair training data.
- Emily: Convert labels from continuous to categorical. Output confusion matrix.

### 12/5/2019 (1 hr)
- Jason: Converting training data to contain data for 2 days.
- Emily: Plot SVR results for different kernels.

### 12/2/2019 (2 hrs)
- Jason: Extract data for one station
- Emily: Split data into train and test sets;
  write code for svc models and predictions.

### 11/28/2019 (5 hrs)
- Jason: finish cleaning data. The dataset is now stored in `final.csv` with
  the number of instances `n=1048185` and number of features `p=7`.

### 11/25/2019 (2.5 hrs)
- Jason:
- Emily: Set up starter files for Support Vector Regression and Neural Network
  using sklearn and tensorflow documentation.

## References
[1] Aravind, “Confusion Matrix as a Heatmap with Python,” Data Fiction, 12-Jun-2019.  
[2] M. J. Menne, I. Durre, R. S. Vose, B. E. Gleason, and T. G. Houston, “An overview of the global historical climatology network-daily database,” Journal of Atmospheric and Oceanic Technology, vol. 29, no. 7, pp. 897–910, 2012.  
[3] N. Sharma, “Splitting CSV Into Train And Test Data,” Medium, 10-Oct-2018. [Online].   
[4] “Basic classification: Classify images of clothing | TensorFlow Core,” TensorFlow. [Online].  
[5] “Confusion matrix — scikit-learn 0.22 documentation.” 