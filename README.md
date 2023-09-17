# Credit-Card-Fraud-Detection-ML
Credit Card Fraud Detection using Logistic Regression 
* Name - Ayush Agarwal 
* Project - Credit Card Fraud Detection 
* Skills - Logistic regression , Support Vector Machine, K Nearest Neighbours, F1 Score, ROC-AUC Curve, Data Visualisation , Exploratory Data Analysis , Data Science application in Finance , Machine Learning 
* Tools - Google Colab , Jupyter Notebooks , Python , Numpy , Pandas , Matplotlib , Seaborn , Sklearn 

## Code :

The code is availaible at : https://github.com/ayush-agarwal-0502/Credit-Card-Fraud-Detection-ML/blob/main/credit_card_fraud_detection_project.ipynb (in this repository itself ) .

## The Dataset :

The data was taken from Kaggle site : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud .

The Columns do not have physical significance directly visible since as per the source (Kaggle) , the data was compressed using Principle Component Analysis (PCA) in order to protect the privacy of the individuals while making a realistic secnario dataset availaible to public . 

## Data Preprocessing and Visualisation :

![image](https://user-images.githubusercontent.com/86561124/174430662-5a302491-d9ca-4705-b5c3-2d263563f564.png)

### Correlations :

![image](https://user-images.githubusercontent.com/86561124/174430686-86a03acf-d2b0-4888-bd7b-ca0a13df524e.png)
![image](https://user-images.githubusercontent.com/86561124/174430691-a1e9a345-2924-4a7f-9160-daab07f58af4.png)

The columns do not seem to have correlations with each other , and seem to have great correlation with the Class and time variables , hence being a great indicator that simple models would be helpful here , and neural networks wont be needed hopefully . 

### Relation between target variables and columns :
![image](https://user-images.githubusercontent.com/86561124/174430766-c090f225-59c0-4ddb-9a6f-dd8f05389e23.png)
![image](https://user-images.githubusercontent.com/86561124/174430773-7d1943c5-15b6-4daa-a6b9-574cdd7cbe8e.png)
![image](https://user-images.githubusercontent.com/86561124/174430829-b846dd24-fef8-42fc-b532-42b01c520e69.png)
![image](https://user-images.githubusercontent.com/86561124/174430842-95865a87-7cb9-48b0-b379-622d6c4f3107.png)
![image](https://user-images.githubusercontent.com/86561124/174430848-c63a5e3d-fd7f-4b86-b803-50ed30acab6f.png)
![image](https://user-images.githubusercontent.com/86561124/174430854-7961a3bb-fb66-474b-89db-31b70c0c2e8e.png)
![image](https://user-images.githubusercontent.com/86561124/174430860-bdc69fbf-beb8-4844-a7d0-c16b0974399e.png)
![image](https://user-images.githubusercontent.com/86561124/174430867-2fd707eb-c370-4ea8-b95b-2c740242903a.png)
![image](https://user-images.githubusercontent.com/86561124/174430871-c720db9c-1573-4efb-b166-270a305a1ac3.png)
![image](https://user-images.githubusercontent.com/86561124/174430875-5a17d646-ec37-4cec-9eeb-b8051ee54fc7.png)
![image](https://user-images.githubusercontent.com/86561124/174430881-3932b5bb-f2ae-4e9e-abbd-7818cf2c910c.png)
![image](https://user-images.githubusercontent.com/86561124/174430883-e9520d29-d7a6-435c-98c3-878827cd8c12.png)
![image](https://user-images.githubusercontent.com/86561124/174430887-a31417db-12bb-4c7e-92a1-9ac70221001f.png)
![image](https://user-images.githubusercontent.com/86561124/174430890-f8754898-f87a-4d1c-bcef-8a1a11401189.png)
![image](https://user-images.githubusercontent.com/86561124/174430895-5c969eff-1f74-44ff-8790-bda64463d73e.png)

A plot between different columns and amount along with different colours for target variable show that our output classes are separable by linear boundary even in case of graphing variables alone , hence __LOGISTIC REGRESSION__ will help separate the multivariable data into 2 classes .

### Class Imbalance in dataset :

![image](https://user-images.githubusercontent.com/86561124/174430979-7d6dbbfa-8949-43bc-acdb-96daa1587309.png)

This shows that we have way way less data for fraud cases than for non fraud cases , which is expected from the dataset . 

To cure imbalance , we can use undersampling or oversampling . Here , I have decided to use SMOTE to counter the class imbalance in the dataset . 

![image](https://user-images.githubusercontent.com/86561124/174431044-03c757fa-758b-4a1d-a3d7-8746f07290e8.png)

### Training the model :

![image](https://user-images.githubusercontent.com/86561124/174431091-84847f62-0628-4f7f-83e4-b5d0c4295477.png)

I have trained a Logistic Regression Model here . The model was showing a not converging warning , so I read its documentation and added the code to make it run for 150 iterations . 

### Results from part 1 :

![image](https://user-images.githubusercontent.com/86561124/174431703-3b4657d4-d52d-48d3-a43b-bb6bb982f3a4.png)

The F1 score came 0.99 meaning the Classifier is working great . It managed to catch 91 out of 101 frauds , thus preventing frauds 90% of the time . 
The confusion matrix , precision , recall and F1 score has been displayed for your convenience . The confusion matrix readings and the F1 show the success of the project .

![image](https://user-images.githubusercontent.com/86561124/174431788-0c9feb90-de29-4477-a6ad-25c0f6481663.png)

### Results from part 2 :

I have also uploaded some raw code to this repository , here are the conclusions derived from it :

Frauds are time independent so we can drop time : 

![image](https://github.com/ayush-agarwal-0502/Credit-Card-Fraud-Detection-ML/assets/86561124/64bfdfa6-17db-47f3-87f7-25d0aa2ecc70)

__Lower Dimension Visualization__ is beautiful : 

![image](https://github.com/ayush-agarwal-0502/Credit-Card-Fraud-Detection-ML/assets/86561124/e6084542-6721-4b8b-b6ab-43d17141db21)

I also took advice from my seniors, decided to __undersample__ the dataset since significance of the data would be more realistic if there was no synthetic dataset. 
I also decided to choose the ML model with most recall , reason being that I realized later that as a business, labelling a Non Fraud datapoint as fraudulent would be much more worse for the company, __since nobody would like their card to decline__ and people would literally stop using that credit card, so we must __focus more on achieving lower recall than only blindly improving F1 score__ . 
So I got Logistic regression as the winner again with the following results : 

![image](https://github.com/ayush-agarwal-0502/Credit-Card-Fraud-Detection-ML/assets/86561124/e7a10968-4eac-4a9d-85f0-8a5cbbc91f67)

Other models weren't much far behind regarding performance too , but I decided to keep the final code clean and keep the trial and error part in the "raw_code" file . 


