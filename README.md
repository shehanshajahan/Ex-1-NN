<H3>ENTER YOUR NAME : SHEHAN SHAJAHAN</H3>
<H3>ENTER YOUR REGISTER NO. : 212223240154</H3>
<H3>EX. NO.1</H3>
<H3>DATE : 31/07/2026</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
Step 1: Import the required libraries such as Pandas, MinMaxScaler, and train_test_split for data handling, normalization, and dataset splitting.
<br>
Step 2: Read the Churn_Modelling.csv file using pd.read_csv() and store it in the DataFrame df. Display the dataset.
<br>
Step 3: Separate the input features (X) by selecting all columns except the last column, and separate the target variable (Y) by selecting the last column.
<br>
Step 4: Check for missing values using isnull().sum() and replace missing numerical values with their respective column mean using fillna().
<br>
Step 5: Check for duplicate records using duplicated() and use describe() to obtain statistical details of the EstimatedSalary column.
<br>
Step 6: Select only the numerical columns and apply Min-Max Scaling using MinMaxScaler() to convert the values into a range between 0 and 1.
<br>
Step 7: Split the input and output data into training and testing sets using train_test_split(), with 80% for training and 20% for testing.
<br>
Step 8: Display the training and testing data along with their sizes. The processed dataset is now ready to be used for building and evaluating a machine learning model.
<br>
##  PROGRAM:
```
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df=pd.read_csv(r"/content/Churn_Modelling.csv")
print(df)

x=df.iloc[:,:-1].values
print(x)

y=df.iloc[:,-1].values
print(y)

print(df.isnull().sum())

df.fillna(df.mean(numeric_only=True).round(1), inplace=True)
print(df.isnull().sum())
y=df.iloc[:,-1].values
print(y)

df.duplicated()

print(df['EstimatedSalary'].describe())

# Select only numeric columns for scaling
numeric_cols = df.select_dtypes(include=['number']).columns
df_numeric = df[numeric_cols]

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(df_numeric))
print(df1)
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

print(X_train)
print(len(X_train))
print(X_test)
print(len(X_test))
```
## OUTPUT:
<img width="840" height="557" alt="image" src="https://github.com/user-attachments/assets/00e29429-85bc-4765-b5aa-fe10e2af6ca6" />
<br>
<img width="523" height="827" alt="image" src="https://github.com/user-attachments/assets/7280eef0-84f6-4acc-8727-ea32abcf3262" />
<br>
<img width="810" height="822" alt="image" src="https://github.com/user-attachments/assets/4fdbe986-a83a-4551-8cc7-0a146ceb8048" />
<br>
<img width="540" height="685" alt="image" src="https://github.com/user-attachments/assets/71e5f4fa-df13-4209-bcc8-6ce142435c81" />

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


