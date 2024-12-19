#!/usr/bin/env python
# coding: utf-8

# ## DATA CLEANING OR FEATURE ENGINEERING

# ### TASKS TO BE PERFORMED
# 
# #### 1. DATA CLEANING
# #### 2. HANDLING DATASET 
# #### 3. PROBLEM STATEMENT

# ### importing the libraries for the dataset

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
import scipy
import matplotlib.image as mpimg
from pandas.plotting import scatter_matrix
from scipy.stats import norm
from numpy.random import randn
from statsmodels.stats.proportion import proportions_ztest
import plotly.graph_objects as go
from scipy.stats import spearmanr
from scipy.stats import shapiro
from scipy.stats import chi2_contingency

warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# ### reading the dataset 

# In[2]:


df1 = pd.read_csv(r"C:\Users\Hp\Downloads\archive\housing.csv")


# ### making a new copy of the dataset

# In[3]:


df = df1.copy()


# ## 1. DATA CLEANING 

# In[4]:


df.info()


# ### 1.1 missing value handling of total_bedrooms

# In[5]:


# comment - checking the missing values or nan values in total_bedrooms are 207 entries.
# observation - data has missing values only in total_bedrooms and are 207 entries.
df.isnull().sum()


# ### 1.1.1 we will fill the missing values by total_bedroom median values

# In[6]:


# comment - median of total_bedroom feature
# observation - the missing value need to be filled by its median.
                   # I use median because it is less attractive to outliers and best way to fill missing values.
df['total_bedrooms'].median()


# In[7]:


# comment - using fillna inbuilt function of pandas
# observation - filling nan values by median.
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())


# In[8]:


# comment - after filling values checking is the total_bedrooms values are all filled or not
# observation - zero values interpret that now data is cleaned and no missing values are there.
df.isnull().sum()

# categorical features and numerical features segregating
# In[9]:


categorical_features = [fea for fea in df.columns if df[fea].dtypes == 'O']
print(f'we have {categorical_features} as our categorical feature')


# In[10]:


numerical_features = [fea for fea in df.columns if df[fea].dtypes != 'O']
print(f'we have {numerical_features} as our numerical feature')


# ### now converting the datatypes of 7 numeric features all together.

# In[11]:


# comment - earlier column index of original data has float as its datatypes
# observation - now features of index 2 to 9 are all converted to int datatype.
for col in numerical_features[2:9]:
    df[col] = df[col].astype('int')


# In[12]:


df


# In[13]:


# comment - now segregating data with only int and float datatypes.
# observation - first two index of data are of float datatypes.
                   # rest index 2 to 8 are of int datatypes.
df[df.dtypes[(df.dtypes == 'float64' )| (df.dtypes == 'int')].index]


# ### new info of the data and its types

# In[14]:


df.info()


# ## 2. handling data

# ### 2.1 handling scaling

# In[15]:


# comment - scaling or normalizing datatypes of categorical features.
# observation - clear picture that 44% of the people are preferring <1H ocean location.
                          ## less preferrence at ISLAND location.
for col in categorical_features:
    print(df[col].value_counts(normalize=True) * 100)


# In[16]:


# comment -  scaling or normalizing datatypes of numerical features.
for col in numerical_features:
    print(df[col].value_counts(normalize=True) * 100)


# ### 2.2 handling outliers

# In[17]:


# comment - checking outliers in the data by using boxplot from seaborn .
# observation - numerical features index from 3 to 8 all have outliers in the data.
for i in numerical_features:
    sns.boxplot(df[i])
    plt.show()


# ### handling outliers of features index [3:9]

# ### while removing outliers we tried to show 
# ##### 1. outliers before
# ##### 2. outliers after
# ##### 3. and how many left to remove.

# In[18]:


for col in numerical_features[4:6]:
    Q1 = np.percentile(df[col], 1,
                   interpolation = 'midpoint')
 
    Q3 = np.percentile(df[col], 99,
                   interpolation = 'midpoint')
    IQR = Q3 - Q1
    upper = Q3 +1.5*IQR

    lower = Q1 - 1.5*IQR
    upper, lower
    new_df = df.loc[(df[col] > upper) | (df[col] < lower)]
    print(f'before removing outliers {len(df)}')
    print(f'after removing outliers {len(new_df)}')
    print('new outliers', len(df) - len(new_df))
    sns.boxplot(new_df[col])
    plt.show()


# In[19]:


for col in numerical_features[6:7]:
    Q1 = np.percentile(df[col], 1.95,
                   interpolation = 'midpoint')
 
    Q3 = np.percentile(df[col], 98.05,
                   interpolation = 'midpoint')
    IQR = Q3 - Q1
    upper = Q3 +1.5*IQR

    lower = Q1 - 1.5*IQR
    upper, lower
    new_df = df.loc[(df[col] > upper) | (df[col] < lower)]
    print(f'before removing outliers {len(df)}')
    print(f'after removing outliers {len(new_df)}')
    print('new outliers', len(df) - len(new_df))
    sns.boxplot(new_df[col])
    plt.show()


# In[20]:


for col in numerical_features[7:8]:
    Q1 = np.percentile(df[col], 15,
                   interpolation = 'midpoint')
 
    Q3 = np.percentile(df[col], 85,
                   interpolation = 'midpoint')
    IQR = Q3 - Q1
    upper = Q3 +1.5*IQR

    lower = Q1 - 1.5*IQR
    upper, lower
    new_df = df.loc[(df[col] > upper) | (df[col] < lower)]
    print(f'before removing outliers {len(df)}')
    print(f'after removing outliers {len(new_df)}')
    print('new outliers', len(df) - len(new_df))
    sns.boxplot(new_df[col])
    plt.show()


# In[21]:


for col in numerical_features[8:9]:
    Q1 = np.percentile(df[col], 40,
                   interpolation = 'midpoint')
 
    Q3 = np.percentile(df[col], 60,
                   interpolation = 'midpoint')
    IQR = Q3 - Q1
    upper = Q3 +1.5*IQR

    lower = Q1 - 1.5*IQR
    upper, lower
    new_df = df.loc[(df[col] > upper) | (df[col] < lower)]
    print(f'before removing outliers {len(df)}')
    print(f'after removing outliers {len(new_df)}')
    print('new outliers', len(df) - len(new_df))
    sns.boxplot(new_df[col])
    plt.show()


# In[22]:


for col in numerical_features[3:4]:
    Q1 = np.percentile(df[col], .5,
                   interpolation = 'midpoint')
 
    Q3 = np.percentile(df[col], 99.5,
                   interpolation = 'midpoint')
    IQR = Q3 - Q1
    upper = Q3 +1.5*IQR

    lower = Q1 - 1.5*IQR
    upper, lower
    new_df = df.loc[(df[col] > upper) | (df[col] < lower)]
    print(f'before removing outliers {len(df)}')
    print(f'after removing outliers {len(new_df)}')
    print('new outliers', len(df) - len(new_df))
    sns.boxplot(new_df[col])
    plt.show()


# ### 2.3 transformation of skewed data distribution 

# In[23]:


# comment - separate distribution of each and every numerical features
# observation - the index of numerical features from 3 to 7 are right skewed also known as log distribution and index 8 is right-skewed.
              # the index 0,1 are bimodal distributed and 2 is multimodal disrtibuted
plt.figure(figsize=(15, 15))
plt.suptitle('Univariate Analysis of Numerical Features', fontsize=20, fontweight='bold', alpha=0.8, y=1.)

for i in range(0, len(numerical_features)):
    plt.subplot(5, 3, i+1)
    sns.kdeplot(x=df[numerical_features[i]],shade=True, color='g')
    plt.xlabel = (numerical_features[i])
    plt.tight_layout()


# ### we need to transform the right skewed data into log transformation for normality of data.

# In[24]:


# comment - selecting list of features which need to log transformed.
log_var = ['total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']


# In[25]:


# comment - log transformation of selected skewed data features.
# observation - the data of these features are now normally distributed.
fig = plt.figure(figsize = (24,10))
for j in range(len(log_var)):
    var = log_var[j]
    transformed = 'log_' + var
    df[transformed] = np.log10(df[var] + 1)
    sub = fig.add_subplot(2,5, j+1)
    sub.set_xlabel(var)
    df[transformed].plot(kind = 'hist')


# ### 2.4 scaling 

# ### importing libraries for scaling data.

# In[26]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# In[27]:


def log_features(scale_var):
    scale_var = ['median_house_value', 'housing_median_age', 'latitude', 'longitude']
    scalers_list = [MinMaxScaler(), StandardScaler()]
    for i in range(len(scalers_list)):
        scaler = scalers_list[i]
        fig = plt.figure(figsize = (26,5))
        plt.title(scaler, fontsize = 20)
        for j in range(len(scale_var)):
            var = scale_var[j]
            scaled_var = 'scaled_' + var
            model = scaler.fit(df[var].values.reshape(-1,1))
            df[scaled_var] = model.transform(df[var].values.reshape(-1,1))
    
            sub = fig.add_subplot(2,4, j+1)
            sub.set_xlabel(var)
            df[scaled_var].plot(kind = 'hist')
    return(log_features('median_house_value', 'housing_median_age', 'latitude', 'longitude'))


# In[28]:


# comment - we have used two scaling libraries MinMaxScaler has [0-1] scale.
# observation - by scaling we have transformed the features of remaining datatypes into normal too.
scale_var = ['median_house_value', 'housing_median_age', 'latitude', 'longitude']
scalers_list = [MinMaxScaler(), StandardScaler()]
for i in range(len(scalers_list)):
    scaler = scalers_list[i]
    fig = plt.figure(figsize = (26,5))
    plt.title(scaler, fontsize = 20)
    for j in range(len(scale_var)):
        var = scale_var[j]
        scaled_var = 'scaled_' + var
        model = scaler.fit(df[var].values.reshape(-1,1))
        df[scaled_var] = model.transform(df[var].values.reshape(-1,1))
    
        sub = fig.add_subplot(2,4, j+1)
        sub.set_xlabel(var)
        df[scaled_var].plot(kind = 'hist')


# ### 2.5 imbalance data

# In[29]:


# comment - checking where the location of houses is highest near the ocean_proximity
# observation - the analysis shows that population or households prefers <1H ocean location as the most preferred one.
                # second preference of people of California will be INLAND.
                # less to NEAR OCEAN and than NEAR BAY respectively
                # shows the data is imbalanced as island as less or nrgligible preferences.
plt.figure(figsize=(10, 10))
plt.suptitle('Univariate Analysis of Categorical Features', fontsize=20, fontweight='bold', alpha=0.8, y=1.)

for i in range(0, len(categorical_features)):
    plt.subplot(1, 1, i+1)
    sns.countplot(x = df[categorical_features[i]])
    plt.xticks(rotation=45)
    plt.xlabel = (categorical_features[i])
    plt.tight_layout()


# ### creating dummies for having the imbalanced data to balanced data by giving equal numeric data presentation

# In[30]:


# comment - one hot-encode all categorical features 
ohe = pd.get_dummies(df[categorical_features])


# In[31]:


ohe


# ### 2.7 handling multicollinearity

# In[32]:


# comment - chi square test shows the relation between features and here we are checking for categorical with all numerical features
# observation - for index 0,1,2 and 8 we are able to reject H0 (which is good) as independent features will not affect relation.
                ## for index 3 to 7 we fail to reject as these factors might affect the relation value (not good) .
chi2_test = []
for feature in numerical_features:
    if chi2_contingency(pd.crosstab(df['ocean_proximity'], df[feature]))[1] < 0.05:
        chi2_test.append('Reject Null Hypothesis')
    else:
        chi2_test.append('Fail to Reject Null Hypothesis')
result = pd.DataFrame(data=[numerical_features, chi2_test]).T
result.columns = ['Column', 'Hypothesis Result']
result


# In[33]:


# comment - data has mulicollinaerity this means that some features are highly correlated that they affect the data
# observation - we will handle by VIF and
                     ## if VIF > 5 means multicollinearity and need to be removed.
                     ## if VIF < 5 than no high correlation and good for the data.
# load statmodels functions
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# compute the vif for all given features
def compute_vif(considered_features):
    
    X = df[considered_features]
    # the calculation of variance inflation requires a constant
    X['intercept'] = 1
    
    # create dataframe to store vif values
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) 
    for i in range(X.shape[1])]
    vif = vif[vif['Variable']!='intercept']
    return vif


# In[34]:


# comment - the feature index from 3 to 6(including) have failed the test and shows that high idependent variable correlation.
# observation - households has high correlation and this means that we need to remove it for VIF to be less than 5.
considered_features = ['total_rooms','total_bedrooms','population','households','median_income']
# compute vif 
compute_vif(considered_features).sort_values('VIF', ascending=False)


# In[35]:


# comment - removed the housedhold feature
# observation - now total_rooms has high multicollinearity and need to be removed.

# compute vif values after removing a feature
considered_features.remove('households')
compute_vif(considered_features)


# In[36]:


# comment - compute vif values after removing another feature total_rooms
# observation - now the multicollinearity is all under 5 and hence all errors are removed and fixed the high correlation problem.
considered_features.remove('total_rooms')
compute_vif(considered_features)


# ## 3. problem statement and conditional problems

# ### 3.1 answering various observation via equation and graphs
1.   Where the maximum population have houses near the ocean location ?
# In[37]:


# observation - max population resides or prefers ocean location as <1H OCEAN.
df[df['population'] == max(df['population'])]['ocean_proximity']

2.   Among population and households who has the maximum median income ?
# In[38]:


# observation - max income population earns is 2 ten thousands USD.
df[df['population'] == max(df['population'])]['median_income']


# In[39]:


# observation - same result as population are households residing in a block and earns max income as 2 ten thousands USD.
df[df['households'] == max(df['households'])]['median_income']


# In[40]:


# observation - towards north population more resides in a block.
df[df['population'] == max(df['population'])]['latitude']

3.  Which location around ocean has highest median house value and where the house is costlier?
# In[41]:


# observation - Except ISLAND the houses at all other locations has almost same cost.
                  ## at first sight we might think that ISLAND has cheaper house, no, because
                       ## population prefers less this location and still this price is high even though it is less than others.
house= df.groupby('ocean_proximity').median_house_value.max()
house=house.to_frame().sort_values('median_house_value',ascending=False)
house


# In[42]:


##  observation - graphical analysis for better understanding.
plt.subplots(figsize=(14,7))
sns.barplot(x=house.index, y= house.median_house_value,ec = "black",palette="Set1")
plt.title("where house is costlier", weight="bold",fontsize=20, pad=20)
plt.ylabel("median_house_value", weight="bold", fontsize=15)#plt.xlabel("ocean_proximity", weight="bold", fontsize=16)
plt.xticks(rotation=90)
plt.show()


# In[46]:


# observation - the ISLAND has highest house value ranges and all other have all types of range house value at ocean location.
sns.FacetGrid(df[numerical_features]).map(plt.scatter, x = df['median_house_value'], y = df['ocean_proximity'])
plt.suptitle('BiVariate Analysis of area wise house value', fontsize=20, fontweight='bold', alpha=0.8, y=1.)
plt.xlabel ('median_house_value')
plt.ylabel ('ocean_proximity')
plt.tight_layout()

4.  Where the population is dense near ocean locations ?
# In[45]:


# observation - the population is densely residing more in the block of ocean location (that is at <1H ocean location).
plt.subplots(figsize=(14,7))
sns.histplot(x='ocean_proximity', y='population', bins = 5,  data=df ,palette="Set1_r")
plt.title("ocean locations dense population", weight="bold",fontsize=20, pad=20)
plt.ylabel("population", weight="bold", fontsize=15)
plt.xlabel("ocean_proximity", weight="bold", fontsize=12)
plt.show()


# In[47]:


plt.subplots(figsize=(14,7))
sns.histplot(x='ocean_proximity', y='median_house_value', bins = 5, data=df ,palette="Set1_r")
plt.title("costlier house at which ocean location", weight="bold",fontsize=20, pad=20)
plt.ylabel("median_house_value", weight="bold", fontsize=15)
plt.xlabel("ocean_proximity", weight="bold", fontsize=12)
plt.show()

5.  How many peope have less than 10 (thousand USD) and still can afford the houses and where?
# In[48]:


# observation - out of 20640 total 20331 population has less than 10 thousand USD.
df[(df['median_income']< 10)]


# In[49]:


df[(df['median_income'] < 10) & (df['median_house_value']< 20000)]

6.  How population manages to purchase houses even at less income than house value ?
# In[50]:


# observation - we can estimate new column from the data about other_income_source feature.
df['other_income_source'] = df['median_house_value'] - df['median_income']


# In[51]:


# observation - datatype chnanged to int.
df['other_income_source'] = df['other_income_source'].astype('int')


# In[52]:


df[['other_income_source']]


# In[53]:


# comment - checking how this new is correlated to other features
# observation - obvious it will show the high positive correlation with median_house_value and median_income.
corr_matrix = df.corr()
corr_matrix['other_income_source'].sort_values(ascending = False)


# In[54]:


# contour plot for seeing how the z variable is at each x and y variables by just moving house at the graph.
import plotly.graph_objects as go

fig = go.Figure(data =
    go.Contour(
        z= ((df['median_house_value'])- (df['median_income'])),
        x= np.linspace(14999, 500000, 500001),
        y = np.linspace(0.4999, 14, 15) # vertical axis
    ))
fig.show()

7.  How many total rooms and bedrooms are in the blocks have under the every house value
# In[55]:


# observation - the data shows the count of rooms and bedrooms in a block of that location.
                    ## so, we can see that at the highest house value (last index) has highest
                             ### number of rooms and bedrooms, nearly positive realtion related to value and high rooms and bedrooms
df.groupby('median_house_value')['total_rooms','total_bedrooms'].count()

8.  Analysis of ecah population that where they reside more north or west along with how much they earn at that location?
# In[56]:


# observation - latitude means north location and longitude is west location
   ## this shows that first index of below data shows less earning opportunity and that's why less population at this block.
     ## location plays important role along with population residing over there.
df.groupby('population')['longitude','latitude','median_income',].sum()


# In[ ]:




