import streamlit as st

# Center-align the project title
st.markdown("<h1 style='text-align: center;'>Fraud Analysis</h1>", unsafe_allow_html=True)

# Layout for logo and team members
col1, col2 = st.columns([1,1])

with col1:
    st.image("E:\\clg logo.jpg", width=150)  # Adjust the path and width of the image as needed

with col2:
    st.markdown("""
    ### Team Members:
    - Aditya Kadam
    - Kunal Jadhav
    - Saurabh Hinge
    """)


# Title
st.markdown("<h1 style='text-align: center;'>WARRANTY CLAIM</h1>", unsafe_allow_html=True)


# Subtitle
st.subheader("Business Statement:")
st.write("Fraud can take place in various forms, and it affects the industry economically, although not in equal measure. The sectors that deal with it uses various techniques to overcome the fraudulent cases.")

# Objective
st.subheader("Objective:")
st.write("The objective of the analysis is to predict when an item is sold, what is the probability that a customer would file a fraudulent or genuine warranty claim and to understand the important factors associated with them.")

# Attributes
st.subheader("Attributes:")
st.write("- Region: Customer region details")
st.write("- State: Current location of customer")
st.write("- Area: Urban/rural")
st.write("- City: Customer's current located city")
st.write("- Consumer_profile: Customer's work profile")
st.write("- Product_category: Product category")
st.write("- Product_type: Type of the product (TV/Air Conditioner)")
st.write("- AC_1001_Issue: Failure of Compressor in Air Conditioner")
st.write("- AC_1002_Issue: Failure of Condenser Coil in Air Conditioner")
st.write("- AC_1003_Issue: Failure of Evaporator Coil in Air Conditioner")
st.write("- TV_2001_Issue: Failure of power supply in TV")
st.write("- TV_2002_Issue: Failure of Inverter in TV")
st.write("- TV_2003_Issue: Failure of Motherboard in TV")
st.write("- claim_value: Customer's claim amount in Rs")
st.write("- Service_Centre: 7 Different service centers")
st.write("- Product_Age: Duration of the product purchased by customer")
st.write("- Purchased_from: From where product is purchased")
st.write("- Call_details: Call duration in mins")
st.write("- Purpose: Purpose (complaint raised by customer, claimed for the product, other)")

# Fraud definition
st.write("Fraud: '1' - fraudulent claim, '0' - Genuine claim")
st.write("Note: '0' means to replace the component, '1' means partial damage of the component with servicing, '2' means no issue with the component.")

#-------------------------------------------------------------------------------------------------------------------

#Import Libraries
# Data Manipulation
import numpy as np
import pandas as pd
# import missingno
import copy as cp

# Visualization 
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

#Cross-validation
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold, cross_val_score

# Machine learning
# import catboost
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
# from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
# from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
# from catboost import CatBoostClassifier, Pool, cv
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score, classification_report ,roc_auc_score ,f1_score


#ignore warnings
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',None)

#-------------------------------------------------------------------------------------------------------------------

#Load Data
df = pd.read_csv("train.csv")

# Display some basic information about the loaded DataFrame
st.markdown("<h2 style='text-align: center;'>Import Data</h2>", unsafe_allow_html=True)
st.write("Shape of the DataFrame:", df.shape)
st.write("First few rows of the DataFrame:")
st.write(df.head())

#-------------------------------------------------------------------------------------------------------------------

#Handling Missing Values and removing unnecessary column

st.markdown("<h2 style='text-align: center;'>Handling Missing Values & Removing Unnecessary Column</h2>", unsafe_allow_html=True)
st.subheader("Before Replacing Null Values")
st.write(df.isnull().sum())
st.write("No null values encountered in the dataset!")

#Removing unnecessary columns

df.drop(['Unnamed: 0'],axis = 1,inplace = True)
st.write("Column Unnamed: 0 Removed from the dataset since it does not impact the Model and was of no use ! ")
st.write(df.head())


#-------------------------------------------------------------------------------------------------------------------

#Data Insights
st.markdown("<h2 style='text-align: center;'>Data Insights</h2>", unsafe_allow_html=True)

# Display information about the DataFrame
st.subheader("DataFrame Info:")
st.write("This section provides general information about the DataFrame.")
st.write(df.info())

# Display summary statistics of the DataFrame
st.subheader("Summary Statistics:")
st.write("This section provides summary statistics of the DataFrame.")
st.write(df.describe())

st.write("Unique Values of Respective Column")
st.write(pd.DataFrame(df.nunique()))

st.write(print())
st.subheader("Component Statistics")
st.write("0: means to replace the component")
st.write("1: means partial damage of the component and with servicing component good work")
st.write("2: no issue with the component.")


st.write("**Value Counts for Product Category:**")
st.write(df['Product_category'].value_counts())

st.write("**Value Counts for AC_1001_Issue:**")
st.write(df['AC_1001_Issue'].value_counts())

st.write("**Value Counts for AC_1002_Issue:**")
st.write(df['AC_1002_Issue'].value_counts())

st.write("**Value Counts for AC_1003_Issue:**")
st.write(df['AC_1003_Issue'].value_counts())

st.write("**Value Counts for TV_2001_Issue:**")
st.write(df['TV_2001_Issue'].value_counts())

st.write("**Value Counts for TV_2002_Issue:**")
st.write(df['TV_2002_Issue'].value_counts())

st.write("**Value Counts for TV_2003_Issue:**")
st.write(df['TV_2003_Issue'].value_counts())

st.write("**Claim ,Complent and Other Queries Count**")
df.loc[(df.Purpose == "claim"), "Purpose"] = "Claim"  
st.write(df['Purpose'].value_counts())



# Displaying descriptive statistics for 'Claim_Value'
st.write("Descriptive Statistics for Claim Value:")
st.write(df['Claim_Value'].describe())

# Calculating and displaying the skewness of 'Claim_Value'
skewness = df['Claim_Value'].skew()
st.write("Skewness of Claim Value:", skewness)

# Interpreting the skewness value
if skewness > 1 or skewness < -1:
    st.write("The data is highly skewed.")
    # For highly skewed data, the median is a better measure of central tendency.
    st.write("Preferred measure: Median")
elif (skewness > 0.5 and skewness < 1) or (skewness < -0.5 and skewness > -1):
    st.write("The data is moderately skewed.")
    # For moderately skewed data, either mean or median could be considered, but median is often more reliable.
    st.write("Preferred measure: Median")
else:
    st.write("The data is approximately symmetric.")
    # For symmetric data, the mean is a preferred measure of central tendency.
    st.write("Preferred measure: Mean")

st.write("**The skewness for a normal distribution is zero, and any symmetric data should have a skewness near zero.**")
st.write("**Mean-It is preferred if data is numeric and not skewed.**")
st.write("**Median-It is preferred if data is numeric and skewed.**")

#Imputing missing values with median of claim_value variable    
df["Claim_Value"].fillna(10000,inplace=True)
#-----------------------------------------------------------------------------------------------------------------------

#Outlier Handling

st.markdown("<h2 style='text-align: center;'>Cheking Outliers Using IQR Method</h2>", unsafe_allow_html=True)

int_col = list(df.select_dtypes(['int64']).columns)#List of all Integer Values
float_col = list(df.select_dtypes(['float64']).columns)#list of all float values


for i in float_col:
    Q3, Q1 = np.percentile(df[i], [75 ,25])
    iqr= Q3 - Q1
    low = Q1 - (1.5*iqr)
    high = Q3 + (1.5*iqr)
    x=df[i][df[i]>high]
    y=df[i][df[i]<low]
    st.write(i, 'has ===>', x.shape[0] + y.shape[0], 'outliers')


for i in float_col:
    Q3, Q1 = np.percentile(df[i], [75 ,25])
    iqr= Q3 - Q1
    low = Q1 - (1.5*iqr)
    high = Q3 + (1.5*iqr)
    st.write("Low:", low)
    st.write("High:", high)
    st.write("IQR:", iqr)

q3 = df['Claim_Value'].quantile(0.75)

st.write("75th percentile of Claim_Value:", q3)

outlier=['Claim_Value']

p0_ = []
p100_ = []
iqr_ = []
high_ = []
low_ = []

for i in outlier:
    p0 =df[i].min()
    p100=df[i].max()
    
    p0_.append(p0)
    p100_.append(p100)

    q1=df[i].quantile(0.25)
    q2=df[i].quantile(0.5)
    q3=df[i].quantile(0.75)
    
    iqr=q3-q1
    iqr_.append(iqr)
    
    low: int = q1 - (1.5 * iqr)
    high: int = q3 + (1.5 * iqr)
    
    low_.append(low)
    high_.append(high)

    
st.write("p0_:", p0_)
st.write("p100_:", p100_)
st.write("high_ (Q3):", high_)
st.write("low_ (Q1):", low_)

#-----------------------------------------------------------------------------------------------------------------------

st.subheader("Removing Outliers Using Flooring and Capping ")

count = 0
for i in outlier:
    df[i] = df[i].clip(upper = high_[count], lower = low_[count])
    count+=1

for i in outlier:
    Q3, Q1 = np.percentile(df[i], [75 ,25])
    iqr= Q3 - Q1
    low = Q1 - (1.5*iqr)
    high = Q3 + (1.5*iqr)
    x=df[i][df[i]>high]
    y=df[i][df[i]<low]
    print(i,'has ===>  ',x.shape[0]+y.shape[0],'outliers')


st.write(df['Claim_Value'].describe())


#-----------------------------------------------------------------------------------------------------------------------

def app():
    st.subheader('Visualization of Call Details')

    # Create a figure for matplotlib
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))

    # First subplot
    sns.distplot(df['Call_details'], ax=ax[0])
    ax[0].set_title("Call details", fontsize=14)

    # Second subplot
    sns.boxplot(x=df['Call_details'], ax=ax[1])
    ax[1].set_title('Call details Boxplot', fontsize=14)

    # Display the plot in Streamlit
    st.pyplot(fig)

    st.write("**Therer is no Outlier present in Call Details Column**")

if __name__ == "__main__":
    app()

#----------------------------------------------------------------------------------------------------------------------------

#EDA


st.markdown("<h2 style='text-align: center;'>Exploratory Data Analysis</h2>", unsafe_allow_html=True)


# Visualization 1: Region Distribution
st.markdown("<h4 style='text-align: center;'>Region Distribution</h4>", unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(10,4))
sns.countplot(y="Region", data=df, palette="Spectral", ax=ax)
ax.set_title('Region Distribution')
st.pyplot(fig)
st.write("Most customers are from south of the states")

# Visualization 2: Region Distribution with Claims
st.markdown("<h4 style='text-align: center;'>Region Distribution with Claim</h4>", unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(8, 4))
pd.crosstab(df.Region, df.Fraud).plot(kind="bar", ax=ax)
ax.set_title('Region Distribution with Claims')
st.pyplot(fig)
st.write("In North East region there is least fraud claim")

# Visualization 3: Statewise Distribution of Customers
st.markdown("<h4 style='text-align: center;'>State wise Distribution of Customers</h4>", unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(22,8))
sns.countplot(x="State", data=df, palette="Spectral", ax=ax)
plt.title('Statewise Distribution of Customers')
st.pyplot(fig)
st.write("Andhrapradesh have most customers with J&K least customers")

# Visualization 4: Area wise Distribution of Customers
st.markdown("<h4 style='text-align: center;'>Area wise Distribution of Customers</h4>", unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(5,5))
explode=[0.02]*2
df.Area.value_counts().plot(kind='pie', radius=1, autopct='%.1f%%', explode=explode, ax=ax)
plt.legend()
plt.title('Area wise Distribution of Customers')
st.pyplot(fig)
st.write("Most buyers are from urban area\n")
st.write("Urban section has most bought products up to 5319")

# Visualization 5: Area wise Distribution of Customers with Claim
st.markdown("<h4 style='text-align: center;'>Area wise Distribution of Customers with Claim</h4>", unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(6, 4))
pd.crosstab(df.Area, df.Fraud).plot(kind="bar", ax=ax)
plt.title('Area wise Distribution of Customers with Claim')
st.pyplot(fig)
st.write("Urban has more fraud claim than rural")

# Visualization 6: Customers Consumer profile
st.markdown("<h4 style='text-align: center;'>Customers Consumer profile</h4>", unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(6,4))
sns.countplot(x="Consumer_profile", data=df, palette="Spectral", ax=ax)
plt.title('Customers Consumer profile')
st.pyplot(fig)
st.write("There are 4973 and 3368 Personal and Business Customers respectively")

# Visualization 7: Customer profile Distribution of Customers with Claim
st.markdown("<h4 style='text-align: center;'>Customer profile Distribution of Customers with Claim</h4>", unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(6, 4))
pd.crosstab(df.Consumer_profile, df.Fraud).plot(kind="bar", ax=ax)
plt.title('Customer profile Distribution of Customers with Claim')
st.pyplot(fig)



# Visualization 9: Purpose of customers
st.markdown("<h4 style='text-align: center;'>Purpose of customers</h4>", unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(6,4))
sns.countplot(x="Purpose", data=df, palette="Spectral", ax=ax)
plt.title('Purpose of customers')
st.pyplot(fig)

# Visualization 10: Service center Distribution
st.markdown("<h4 style='text-align: center;'>Service center Distribution</h4>", unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(10,2))
sns.countplot(y='Service_Centre', data=df, ax=ax)
plt.title('Service center Distribution ')
st.pyplot(fig)

# Visualization 11: Purchased from(seller) to customers
st.markdown("<h4 style='text-align: center;'>Purchased from(seller) to customers</h4>", unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(6, 4))
pd.crosstab(df.Purchased_from, df.Fraud).plot(kind="bar", ax=ax)
plt.title('Purchased from(seller) to customers')
st.pyplot(fig)
st.write("There is least fraud claim after purchasing from Internet")



# Visualizing Fraud vs. Genuine Claims
st.subheader("Fraud vs. Genuine Claims")
fraud_counts = df.Fraud.value_counts()
st.write(f"There are {fraud_counts[1]} fraud claims out of {len(df)} entries.")

# Plot for Fraud/Genuine Claims
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x="Fraud", data=df, palette="Spectral", ax=ax)
plt.title('Number of Fraud/Genuine Claims')
st.pyplot(fig)

# Claim Value Distribution
st.subheader("Claim Value Distribution")
st.write("10k is the most commonly claimed value after warranty. Claim value ranges between 0-40000 Rs.")

# Plot for Claim Value Distribution
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(x='Claim_Value', data=df, ax=ax)
plt.title('Claim Value for Warranty')
st.pyplot(fig)

# Product Age Distribution
st.subheader("Product Age Distribution")
st.write("The age of the product varies from 1 up to 991 days.")

# Plot for Product Age Distribution
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(x='Product_Age', data=df, ax=ax)
plt.title('Product Age Distribution')
st.pyplot(fig)


#----------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------

#Label Encoding 

# Define the columns you want to encode
cols_to_encode = ['Region', 'State', 'Area', 'City', 'Consumer_profile', 'Product_category', 'Product_type', 'Purchased_from', 'Purpose']

# Apply LabelEncoder to the specified columns
df_copy = cp.deepcopy(df)
label_encoder = LabelEncoder()

df_copy[cols_to_encode] = df_copy[cols_to_encode].apply(lambda col: label_encoder.fit_transform(col))

# Display the modified DataFrame
st.markdown("<h4 style='text-align: center;'>Label Encoded DataFrame</h4>", unsafe_allow_html=True)
st.write(print())
st.write("**Performed Label Encoding on the below columns :**")
st.write("Region', 'State','Area','City','Consumer_profile','Product_category','Product_type','Purchased_from','Purpose'")
st.write(df_copy.head())
st.write(f'Shape of DataFrame after encoding: {df_copy.shape}')


#--------------------------------------------------------------------------------------------------------------------------------

#Feature Selection

st.title('Correlation Matrix')

# Display the heatmap using Streamlit and seaborn
fig, ax = plt.subplots(figsize=(14, 7))
sns.heatmap(df_copy.corr(), cmap="Reds", annot=True, ax=ax)
st.pyplot(fig)
st.write("**No pair is having corrrelation coefficient exactly equal to 1. Therefore there is no perfect multi-collinearity. 'claim_value','component_issues' are mostly correlated with fraud attribute. Column 'service_centre' is least correlated.**")


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

st.markdown("<h4 style='text-align: center;'>Feature selection Chi sqaure Method</h4>", unsafe_allow_html=True)
st.write("**In feature selection, we aim to select the features which are highly dependent on the response.**")
st.write("Null Hypothesis (H0): Two variables are independent.")
st.write("Alternate Hypothesis (H1): Two variables are not independent.")


X = df_copy.loc[:, df.columns != 'Fraud']
y = df_copy[['Fraud']]

# Drop rows with missing values
X.dropna(inplace=True)
y = y.reindex(X.index)  # Reindex y to match the rows after dropping NaNs in X

# Feature selection using SelectKBest with chi2
selector = SelectKBest(chi2, k=15)
selector.fit(X, y)
X_new = selector.transform(X)

# Important features selected by SelectKBest
imp_feature = X.columns[selector.get_support(indices=True)]
st.subheader('Important Features Selected by using Chi Square Method')
st.write("Important Features:", imp_feature)

# Shape of the transformed data
st.write(f'Shape of Transformed Data: {X_new.shape}')

# Chi-Square scores and p-values
chi_scores = chi2(X, y)
st.title('Chi-Square Scores and p-values')
st.write("Chi-Square Scores:", chi_scores[0])
st.write("p-values:", chi_scores[1])


st.write("Higher the Chi-Square value the feature is more dependent on the response and it can be selected for model training.")
st.write("Here first array represents chi square values and second array represnts p-values")


# Plot p-values
plt.figure(figsize=(6, 4))
p_values = pd.Series(chi_scores[1], index=X.columns)
p_values.sort_values(ascending=False, inplace=True)
st.title('Chi-Square p-values')
st.bar_chart(p_values)

st.write("**Since first 3 columns has higher the p-value, it says that this variables is independent of the response and can not be considered for model training**")

st.write("**'TV_2001_Issue', 'Service_Centre', 'Purpose'**")

#Removing Columns with high p-value

# Drop specified columns
df_copy.drop(columns=['TV_2001_Issue', 'Service_Centre', 'Purpose'], inplace=True)

# Display the modified DataFrame
st.subheader('Modified DataFrame after dropping specified columns')
st.write(df_copy.head())
st.write(f'Shape of Modified DataFrame: {df_copy.shape}')


# Display the value counts of the 'Fraud' column
st.subheader('Value Counts of the Fraud Column')
st.write(df_copy['Fraud'].value_counts())

#-------------------------------------------------------------------------------------------

#K- Fold Cross Validation Before oversampling

st.markdown("<h4 style='text-align: center;'>K-Fold Cross Validation Before Oversampling</h4>", unsafe_allow_html=True)

# Separate input features and target
X1= df_copy.drop(['Fraud'],axis = 1)
y1= df_copy.Fraud

# setting up testing and training sets
X_train, X_val, y_train, y_val = train_test_split(X1, y1, test_size=0.2, random_state = 42)

# Create  classifier object.
RF = RandomForestClassifier(n_estimators=101,criterion='gini',n_jobs=-1)
  
# Create StratifiedKFold object.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=41)
 

# partition data
for train_index, test_index in skf.split(X1, y1):
    # specific ".loc" syntax for working with dataframes
    X_train, X_val = X1.loc[train_index], X1.loc[test_index]
    y_train, y_val = y1[train_index], y1[test_index]


#-------------------------------------------------------------------------------------------------------------------------

#Balancing data: Random_oversampling

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=41)

#----------------------------------------------------------------------------------------------------
#Defining independent and dependent variable

st.markdown("<h4 style='text-align: center;'Defining independent and dependent variable</h4>", unsafe_allow_html=True)

#Separate input features and target
X= df_copy.drop(['Fraud'],axis = 1)
y= df_copy.Fraud
st.subheader("Independent Data")
st.write(X)
st.subheader("Dependent Data")
st.write(y)

X, y = ros.fit_resample(X, y)




#-------------------------------------------------------------------------------------------------------------------

#Split Data into train and test
st.write("Split Data info Train and Test")

# setting up testing and training sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state = 42)


st.write("Split Successfull !")
#-------------------------------------------------------------------------------------------------------------

# #Scaling

st.write("List of Columns where Scaling is required")
st.write(X.columns)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
 
X= scaler.fit_transform(X.to_numpy())
X= pd.DataFrame(X, columns=['Region', 'State', 'Area', 'City', 'Consumer_profile',
       'Product_category','Product_type', 'AC_1001_Issue', 'AC_1002_Issue', 'AC_1003_Issue',
       'TV_2002_Issue', 'TV_2003_Issue', 'Claim_Value',
       'Product_Age', 'Purchased_from', 'Call_details'])

st.write("**Scaled Dataset Using MinMaxScaler**")
st.write(X.head())

st.write("**Scaled Successfully !**")

#------------------------------------------------------------------------------------------------------------

#K-Fold cross validation after oversampling
st.markdown("<h4 style='text-align: center;'>K-Fold cross validation after oversampling</h4>", unsafe_allow_html=True)

# Create  classifier object.
RF = RandomForestClassifier(n_estimators=101,criterion='gini',n_jobs=-1)
  
# Create StratifiedKFold object.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=41)
  

# partition data
for train_index, test_index in skf.split(X, y):
    # specific ".loc" syntax for working with dataframes
    X_train, X_val = X.loc[train_index], X.loc[test_index]
    y_train, y_val = y[train_index], y[test_index]




#------------------------------------------------------------------------------------------------------------
#Test Data Processing


st.title("Test Dateset")

test=pd.read_csv("test_1.csv")
st.write(test.head())

st.write(test.describe())


st.write("Drop unnecessary columns 'Unnamed: 0','TV_2001_Issue','Service_Centre','Purpose'")

test.drop(['Unnamed: 0','TV_2001_Issue','Service_Centre','Purpose'],axis = 1,inplace = True)


cols = ['Region', 'State','Area', 'City', 'Consumer_profile','Product_category','Purchased_from','Product_type']
#
# Encode labels of multiple columns at once
#
test[cols] = test[cols].apply(LabelEncoder().fit_transform)
#
# Print head
#
st.write(test.head())


st.write(test.isna().sum())
test.fillna(test['Claim_Value'].median(),inplace = True)

test_df= cp.deepcopy(test)



st.title("Scaling Test Data")

scaler = MinMaxScaler() # use min-max standardization for numerical features
 
test= scaler.fit_transform(test.to_numpy())
test= pd.DataFrame(test, columns=['Region', 'State', 'Area', 'City', 'Consumer_profile',
       'Product_category','Product_type', 'AC_1001_Issue', 'AC_1002_Issue', 'AC_1003_Issue',
       'TV_2002_Issue', 'TV_2003_Issue', 'Claim_Value',
       'Product_Age', 'Purchased_from', 'Call_details'])
 
st.write("**Scaled Dataset Using MinMaxScaler**")
st.write(test.head())


st.write("Test Data Scaled Successfully ! ")


#-----------------------------------------------------------------------------------------------------------
#Logistic Regression 

st.subheader("Logistic Regression")

# Check for NaN values in X_train
missing_values = X_train.isnull().sum()
st.write("Missing Values in X_train:")
st.write(missing_values)

# Impute missing values with SimpleImputer
from sklearn.impute import SimpleImputer  # Import SimpleImputer
imputer = SimpleImputer(strategy='mean')  # You can choose another strategy if needed
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)

# Create a Logistic Regression model
model_lr = LogisticRegression()

# Fit the model on imputed training data
model_lr.fit(X_train_imputed, y_train)

# Predict on imputed validation data
y_predict1 = model_lr.predict(X_val_imputed)





from sklearn import metrics
mat = metrics.confusion_matrix(y_val, y_predict1)
st.write("Confusion Matix : \n",mat)
f1_log=(f1_score(y_val, y_predict1))
st.write("\n")
st.write("F1_Score :",f1_log*100)
acc_log=accuracy_score(y_val, y_predict1)*100
st.write('Accuracy :',acc_log)

#calculating precision and reall
precision = precision_score(y_val, y_predict1)
recall = recall_score(y_val, y_predict1)
 
st.write('Precision: ',precision)
st.write('Recall: ',recall)


#----------------------------------------------------------------------------------


st.subheader("KNN")

# Check for NaN values in X_train
missing_values = X_train.isnull().sum()
st.write("Missing Values in X_train:")
st.write(missing_values)

# Impute missing values with SimpleImputer
imputer = SimpleImputer(strategy='mean')  # You can choose another strategy if needed
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)

# Create a KNN model
model_knn = KNeighborsClassifier(n_neighbors=11)

# Fit the model on imputed training data
model_knn.fit(X_train_imputed, y_train)

# Predict on imputed validation data
y_predict2 = model_knn.predict(X_val_imputed)


# Calculate confusion matrix
mat = confusion_matrix(y_val, y_predict2)
st.write("Confusion Matrix:")
st.write(mat)

# Calculate F1 score
f1_knn = f1_score(y_val, y_predict2)
st.write("F1 Score:", f1_knn * 100)

# Calculate accuracy score
acc_knn = accuracy_score(y_val, y_predict2)
st.write("Accuracy:", acc_knn * 100)


#-------------------------------

#Random forest

st.subheader("Random Forest")
# Handle missing values
imputer = SimpleImputer(strategy='mean')  # You can use other strategies like 'median', 'most_frequent'
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)

# Train the model
Model_rf = RandomForestClassifier(n_estimators=100)
Model_rf.fit(X_train_imputed, y_train)

# Make predictions
y_predict3 = Model_rf.predict(X_val_imputed)


# Calculate evaluation metrics
mat = confusion_matrix(y_val, y_predict3)
st.write("Confusion Matrix:")
st.write(mat)

f1_rf = f1_score(y_val, y_predict3)
st.write("F1 Score:", f1_rf * 100)

acc_rf = accuracy_score(y_val, y_predict3)
st.write("Accuracy:", acc_rf * 100)

#------------------------------------------------------------------------------------------------------

st.subheader("SVM")

# Handle missing values
imputer = SimpleImputer(strategy='mean')  # You can use other strategies like 'median', 'most_frequent'
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)

# Train the model
Model_svc = SVC()
Model_svc.fit(X_train_imputed, y_train)

# Make predictions
y_predict4 = Model_svc.predict(X_val_imputed)

# Calculate evaluation metrics
mat = confusion_matrix(y_val, y_predict4)
st.write("Confusion Matrix:")
st.write(mat)

f1_svm = f1_score(y_val, y_predict4)
st.write("F1 Score:", f1_svm * 100)

acc_svm = accuracy_score(y_val, y_predict4)
st.write("Accuracy:", acc_svm * 100)


#----------------------------------
st.subheader("Decision Tree")


# Handle missing values
imputer = SimpleImputer(strategy='mean')  # You can use other strategies like 'median', 'most_frequent'
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)

# Train the model
Model_dt = DecisionTreeClassifier()
Model_dt.fit(X_train_imputed, y_train)

# Make predictions
y_predict5 = Model_dt.predict(X_val_imputed)

# Calculate evaluation metrics
mat = confusion_matrix(y_val, y_predict5)
st.write("Confusion Matrix:")
st.write(mat)

f1_dt = f1_score(y_val, y_predict5)
st.write("F1 Score:", f1_dt * 100)

acc_dt = accuracy_score(y_val, y_predict5)
st.write("Accuracy:", acc_dt * 100)


#-------------------------------------------------------------------------------------------

st.markdown("<h4 style='text-align: center;'>Model Evaluation</h4>", unsafe_allow_html=True)


# # Update F1 scores and accuracies for each model
# f1_scores = [f1_log, f1_knn, f1_rf, f1_svm, f1_dt]
# accuracies = [acc_log, acc_knn, acc_rf, acc_svm, acc_dt]

# # Update the models DataFrame
# models['F1 Score'] = f1_scores
# models['Accuracy'] = accuracies

# # Sort models by F1 Score in descending order
# models_sorted = models.sort_values(by='F1 Score', ascending=False)


# Create a DataFrame to display model evaluation results
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'K-Nearest Neighbors', 'Random Forest', 'Support Vector Machine', 'Decision Tree'],
    'F1 Score': [f1_log, f1_knn, f1_rf, f1_svm, f1_dt],
    'Accuracy': [acc_log, acc_knn, acc_rf, acc_svm, acc_dt]
})

# Display the DataFrame
st.write(models)


#------------------------------------------------------------------------------
#Confusion Matrix

# import streamlit as st
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# st.title('Confusion Matrix Display')

# # Compute the confusion matrix using predictions and actual labels
# conf_matrix = confusion_matrix(actual_labels, predicted_labels)

# # Change figure size and increase dpi for better resolution
# fig, ax = plt.subplots(figsize=(4, 4), dpi=100)

# # Initialize ConfusionMatrixDisplay with the confusion matrix and display labels
# display = ConfusionMatrixDisplay(conf_matrix, display_labels=Model_svc.classes_)

# # Plot the confusion matrix on the Streamlit app
# display.plot(ax=ax)

# # Set the plot title
# ax.set_title('Confusion Matrix')

# # Show the plot in Streamlit
# st.pyplot(fig)

#----------------------------------------------------------------------------------------------


#ROC Curve on test data

#-----------------------------------------------------------------------------------------------


#Prediction on Test Data


#Random Forest

st.markdown("<h4 style='text-align: center;'>Prediction on Test Data (Random Forest)</h4>", unsafe_allow_html=True)

st.write("First 5 rows of Test Data")

st.write(test.head())

predicted_class = Model_rf.predict(test)
st.write(predicted_class)


test_id = pd.read_csv("test_1.csv")

test_df= cp.deepcopy(test_id)

test_df.fillna(test_df['Claim_Value'].median(),inplace = True)

test_df.rename(columns = {'Unnamed: 0':'Id'}, inplace = True)

last_predict = pd.DataFrame({'Id': test_id['Unnamed: 0'], 'Fraud': predicted_class})

st.write("**Predicted Fraud claim on test data**")

st.write(last_predict.head(10))

st.write(last_predict['Fraud'].value_counts())

st.subheader("Final Result")

st.subheader("Final Result : There are 320 fraudulent claims & 3256 Genuine Claim")


final=pd.merge(test_df,last_predict,how="outer", on=["Id"])
# st.write(final.head())


# Create a figure and set its size
plt.figure(figsize=(6, 4))

# Create the countplot
sns.countplot(x="Fraud", data=final, palette="Spectral")

# Set the title
plt.title('Fraud Count')

# Display the plot using Streamlit
st.pyplot(plt)



#------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------
#KNN

st.markdown("<h4 style='text-align: center;'>Prediction On Test DataSet  (KNN)</h4>", unsafe_allow_html=True)

st.write("First 5 rows of Test Data")

st.write(test.head())

predicted_class = model_knn.predict(test)
st.write(predicted_class)


test_id = pd.read_csv("test_1.csv")

test_df= cp.deepcopy(test_id)

test_df.fillna(test_df['Claim_Value'].median(),inplace = True)

test_df.rename(columns = {'Unnamed: 0':'Id'}, inplace = True)

last_predict = pd.DataFrame({'Id': test_id['Unnamed: 0'], 'Fraud': predicted_class})

st.write("**Predicted Fraud claim on test data**")

st.write(last_predict.head(10))

st.write(last_predict['Fraud'].value_counts())


st.subheader("Final Result : There are 307 fraudulent claims & 3269 Genuine Claim")

final=pd.merge(test_df,last_predict,how="outer", on=["Id"])

# st.write(final.head())

# Create a figure and set its size
plt.figure(figsize=(6, 4))

# Create the countplot
sns.countplot(x="Fraud", data=final, palette="Spectral")

# Set the title
plt.title('Fraud Count')

# Display the plot using Streamlit
st.pyplot(plt)



#-------------------------------------------------------------------------------------------------


st.markdown("<h4 style='text-align: center;'>Prediction on Test Data (SVM)</h4>", unsafe_allow_html=True)

st.write("First 5 rows of Test Data")

st.write(test.head())

predicted_class = Model_svc.predict(test)
st.write(predicted_class)


test_id = pd.read_csv("test_1.csv")

test_df= cp.deepcopy(test_id)

test_df.fillna(test_df['Claim_Value'].median(),inplace = True)

test_df.rename(columns = {'Unnamed: 0':'Id'}, inplace = True)

last_predict = pd.DataFrame({'Id': test_id['Unnamed: 0'], 'Fraud': predicted_class})

st.write("**Predicted Fraud claim on test data**")

st.write(last_predict.head(10))

st.write(last_predict['Fraud'].value_counts())

st.subheader("Final Result")


st.subheader("Final Result : There are 533 fraudulent claims & 3043 Genuine Claim")

final=pd.merge(test_df,last_predict,how="outer", on=["Id"])
# st.write(final.head())


# Create a figure and set its size
plt.figure(figsize=(6, 4))

# Create the countplot
sns.countplot(x="Fraud", data=final, palette="Spectral")

# Set the title
plt.title('Fraud Count')

# Display the plot using Streamlit
st.pyplot(plt)



#-------------------------------------------------------------------------------------------------------

# from sklearn import metrics
# import matplotlib.pyplot as plt
# import numpy as np  # Assuming you're using numpy arrays for your data


# Model_rf = lambda x: np.random.rand(len(x), 2)  # Dummy model, replace with your actual model

# # Your original code with minor adjustments for Streamlit
# def plot_roc_curve(X_val, y_val):
#     y_pred_proba = Model_rf(X_val)[:, 1]
#     fpr, tpr, _ = metrics.roc_curve(y_val,  y_pred_proba)
#     auc = round(metrics.roc_auc_score(y_val, y_pred_proba), 3)
#     plt.figure(figsize=(8, 4))
#     plt.plot(fpr, tpr, label="RF (auc=" + str(auc) + ')', color='red')
#     plt.plot([0, 1], [0, 1], label='Random (auc = 0.5)', linestyle='--')
#     plt.title('ROC Curve on Test Data')
#     plt.xlabel('FPR Rate')
#     plt.ylabel('TPR Rate')
#     plt.legend(loc=4)
#     st.pyplot(plt)

# # Streamlit interface
# st.markdown("<h4 style='text-align: center;'>ROC Curve on Test Data</h4>", unsafe_allow_html=True)

# if st.button('Plot ROC Curve'):
#     plot_roc_curve(X_val, y_val)



# st.write("**ROC :- Receiver Operator Char Curve**")
# st.write("It shows the performance of the classification model")

#-------------------------------------------------------------------------------------------
