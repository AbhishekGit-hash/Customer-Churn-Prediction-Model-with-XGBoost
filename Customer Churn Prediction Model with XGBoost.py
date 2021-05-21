#!/usr/bin/env python
# coding: utf-8

# <br>
# <br>
# In this project based the dataset used is of an electricity power company that supplies electricity utility to coorporates, SME and residential customers. <br>A significant amount of churn in customers is happening in the SME customer segments which is decreasing the revenue of the company. At a high level research the customer churn among SME segment is driven by price sensitivity.<br>
# <br>
# The motive of this project is to develop a predictive model that will predict the customers likely to churn and from a strategic perspective to decrease the churn rate of customers, some monetary benefits may be provided to the predicted customers.
# <br>
# <br>
# <b>Datasets used:<br></b>
# 1. Customer data - which should include characteristics of each client, for example, industry, historical
# electricity consumption, date joined as customer etc<br>
# 2. Churn data - which should indicate if customer has churned<br>
# 3. Historical price data – which should indicate the prices the client charges to each customer for both
# electricity and gas at granular time intervals<br>
# <br>
# From the XGBoost model it was observed that other factors apart from price sensitivity like Yearly consumption, Forecasted Consumption and net margin were drivers of customer churn.<br><br>
# <b>Recommendations :</b> The strategy of monetary benefits is effective. However it should be appropiately targeted to high-valued customers with high churn probability. If not administered properly then the company may face a hard impact on their revenue<br>
# <br>
# <b>Table of Contents</b><br><br>
# <b> 
# 1. Loading Dataset<br>
# 2. Data Quality Assessment<br>
# &ensp; 2.1. Data Types<br>
# &ensp; 2.2. Descriptive Statistics<br>
# &ensp; 2.3. Descriptive Statistics<br>
# 3. Exploratory Data Analysis<br>
# 4. Data Cleaning<br>
# &ensp;4.1. Missing Data<br>
# &ensp;4.2. Duplicates<br>
# &ensp;4.3. Formatting Data<br>
# &emsp;4.3.1. Missing Dates<br>
# &emsp;4.3.2. Formatting dates - customer churn data and price history data<br>
# &emsp;4.3.3. Negative data points<br>
# 5. Feature Engineering
# &ensp;5.1. New Feature Creation<br>
# &ensp;5.2. Boolean Data Transformation<br>
# &ensp;5.3. Categorical data and dummy variables<br>
# &ensp;5.4. Log Transformation<br>
# &ensp;5.5. High Correlation Features<br>
# &ensp;5.6. Outliers Removal<br>
# 6. Churn Prediction Model with XGBoost<br>
# &ensp;6.1. Splitting Dataset<br>
# &ensp;6.2. Modelling<br>
# &ensp;6.3. Model Evaluation<br>
# &ensp;6.4. Stratiﬁed K-fold validation<br>
# &ensp;6.5. Model Finetuning<br>
# &emsp;6.5.1. Grid search with cross validation<br>
# 7. Model Understanding<br>
# &ensp;7.1. Feature Importance<br>

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('fivethirtyeight')
import seaborn as sns
import shap
import datetime
import warnings
warnings.filterwarnings("ignore")

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb


# In[2]:


pd.set_option('display.max_columns',500)


# ## 1. Loading Datasets

# In[3]:


cust_data = pd.read_csv('customer_data.csv')
hist_price = pd.read_csv('historical_price_data.csv')
churned_data = pd.read_csv('churn_data.csv')


# <b>Customer Data</b>

# In[4]:


cust_data.head()


# From the above data set we see that there are null values. We will replace with appropiate values or remove columns

# <b>Churn Indicator Data</b>

# In[5]:


churned_data.head()


# The churn data is in correct format. 0 stands for not churn and 1 stands for churn.

# <b> Historical Price Data</b>

# In[6]:


hist_price.head()


# A lot of values are 0 in the historical price dataset.

# <b>Merging the customer data and churn data</b>

# In[7]:


print('Total records in customer dataset :{}'.format(cust_data.shape[0]))
print('Total records in churn dataset :{}'.format(churned_data.shape[0]))


# In[8]:


cust_churn = pd.merge(cust_data, churned_data, left_on='id', right_on='id', how='inner')


# In[9]:


cust_churn.head()


# ## 2. Data Quality Assessment and Data Cleaning

# ### 2.1. Data Types

# The dates in cust_churn dataframe are not datetime types yet, which means we might need to convert them. In addition, we can see that the churn is full of integers so we can keep it in that form.

# In[10]:


cust_churn.info()


# In[11]:


hist_price.info()


# ### 2.2. Descriptive Statistics

# In[12]:


cust_churn.describe()


# Based on the above statistics the following points can be outlined :
# 
# 1. The min consumption and forecasted values for elictricity and gas are negative. This could mean that the client companies are producing energy and likely to be returned.
# It is unlikely to consider this data to be corrupted.
# 
# 2. The campaign_disc_ele column contains onlu null values
# 3. The metric columns in are highly skewed, looking at the percentiles.

# In[13]:


round(cust_data['campaign_disc_ele'].isnull().mean()*100)


# In[14]:


hist_price.describe()


# The historical price dataset is looking fine. <br>Even though the prices are negative. The prices will be made positive in the data cleaning step.

# ### 2.3. Missing Values

# There are a lot of missing data so we can check the percentage of missing values.

# In[15]:


missing_data = (cust_churn.isnull().mean()*100).reset_index()
missing_data.rename(columns={0 : 'Percentage of Missing Values'})


# <b>Columns having missing values greater than 75% will be dropped.</b>

# In[16]:


(hist_price.isnull().mean()*100).reset_index()


# Here the missing values are less so appropiate values may be imputed.

# ## 3. Exploratory Data Analysis

# ### Churn

# In[17]:


churn_data = cust_churn[['id', 'churn']]
churn_data.rename(columns={'id':'Companies'}, inplace=True)
churn_count = churn_data.groupby(['churn']).count().reset_index()
churn_count.rename(columns={'Companies' : 'Num'},inplace=True)
churn_count['Num'] = round((churn_count['Num']/churn_count['Num'].sum())*100,1)


# In[18]:


churn_count


# In[19]:


plt.figure(figsize=(5,5))
churn_count.transpose().drop('churn', axis=0).plot(y=[0,1], kind='bar', stacked=True)

plt.ylabel('Companies Base (%)')
plt.legend(['Retention', 'churn'], loc='upper right')
plt.title('Churning Status')


# 10% of the customers have churned from service.

# ###  SME Activity

# Lets see the category of the company's activity in relation to companies clients and churn

# In[20]:


sme_activity = cust_churn[['id', 'activity_new', 'churn']]


# In[21]:


sme_activity.head()


# In[22]:


# Number of companies under SME Activity

num_comp_per_sme_activity = sme_activity.groupby(['activity_new', 'churn'])['id'].count().unstack(level=1)


# In[23]:


num_comp_per_sme_activity.head(10)


# In[24]:


num_comp_per_sme_activity.plot(kind='bar', figsize=(20,10), width=2, stacked=True, title='Number of Companies under SME Activity')

plt.ylabel('Number of Companies')
plt.xlabel('SME Activity')

plt.legend(['Retention', 'Churn'])
plt.xticks([])
plt.show()


# In[25]:


sme_activity_total = num_comp_per_sme_activity.fillna(0)[0]+num_comp_per_sme_activity.fillna(0)[1]
sme_activity_total_percentage =  num_comp_per_sme_activity.fillna(0)[1]/(sme_activity_total)*100
pd.DataFrame({'Churn Percentage': sme_activity_total_percentage, 'Number of Companies':sme_activity_total}).sort_values(
by='Churn Percentage', ascending=False).head(20)


# Our predictive model is likely to struggle accurately predicting the the SME activity due to the large number of categories and lownumber of companies belonging to each category.

# In[26]:


# Function to plot stacked bars with annotations

def plot_stack_bars(df, title_,  y_label, size_=(20,10), rot_=0, legend_='upper_right'):
    
    ax = df.plot(kind='bar', stacked=True, figsize=size_, rot=rot_, title=title_)
    
    annotate_plot(ax, textsize=15)
    
    plt.legend(['Retention', 'Churn'], loc=legend_)
    plt.ylabel(y_label)
    plt.show()

def annotate_plot(ax, pad=1, colour='white', textsize=14):
    
    for i in ax.patches:
        
        val = str(round(i.get_height(),1))
        
        if val=='0.0':
            continue
        ax.annotate(val , ((i.get_x()+i.get_width()/2)*pad-0.05, (i.get_y()+i.get_height()/2)*pad), color=colour, size=textsize)


# ### Sales channel

# The sales channel seems to be an important feature when predecting the churning of a user. It is not the same if the sales were through email ortelephone.

# In[27]:


sales_channel = cust_churn[['id','channel_sales','churn']]


# In[28]:


sales_channel = sales_channel.groupby(['channel_sales', 'churn'])['id'].count().unstack(level=1).fillna(0)


# In[29]:


sales_channel_churn = (sales_channel.div(sales_channel.sum(axis=1), axis=0)*100).sort_values(by=[1], ascending=False)


# In[30]:


plot_stack_bars(sales_channel_churn, 'Sales Channel Chart', 'Company %', rot_=30)


# In[31]:


# Percentage Wise

sales_channel_total = sales_channel.fillna(0)[0]+sales_channel.fillna(0)[1]
sales_channel_total_percentage = sales_channel.fillna(0)[1]/(sales_channel_total)*100
pd.DataFrame({'Churn Percenatge': sales_channel_total_percentage, 'Number of Companies':sales_channel_total}).sort_values(by='Churn Percenatge', ascending=False).head(10)


# ### Consumption

# In[32]:


cust_churn.columns


# In[33]:


consumption = cust_churn[['id', 'cons_12m', 'cons_gas_12m','cons_last_month','imp_cons', 'has_gas', 'churn']]


# In[34]:


# Functions to plot Histograms

def plot_histogram(df, col, ax):
    
    data_hist = pd.DataFrame({'Retention':df[df['churn']==0][col], 'Churn' : df[df['churn']==1][col]})
    data_hist[['Retention', 'Churn']].plot(kind='hist', bins=50, ax=ax, stacked=True)
    ax.set_xlabel(col)


# In[35]:


fig, axs = plt.subplots(nrows=4, figsize=(20,25))
plot_histogram(consumption, 'cons_12m', axs[0])
plot_histogram(consumption[consumption['has_gas']=='t'], 'cons_12m', axs[1])
plot_histogram(consumption, 'cons_last_month', axs[2])
plot_histogram(consumption, 'imp_cons', axs[3])


# <b>Consumption</b> data is highly skewed to the right, presenting a very long right-tail towards the higher values of thedistribution.

# The values on the higher end and lower ends of the distribution are likely to be outliers. We can use a standard plot to visualise the outliers in moredetail. A boxplot is a standardized way of displaying the distribution of data based on a five number summary (“minimum”, first quartile (Q1), median,third quartile (Q3), and “maximum”). It can tell us about our outliers and what their values are. It can also tell us if our data is symmetrical, how tightlyour data is grouped, and if and how our data is skewed.

# In[36]:


fig, axs = plt.subplots(nrows=4, figsize=(18,25))
sns.boxplot(consumption['cons_12m'], ax=axs[0])
sns.boxplot(consumption[consumption['has_gas']=='t']['cons_12m'], ax=axs[1])
sns.boxplot(consumption['cons_last_month'], ax=axs[2])
sns.boxplot(consumption['imp_cons'], ax=axs[3])

for ax in axs :
    ax.ticklabel_format(style='plain', axis='x')

axs[0].set_xlim(-200000,2000000)
axs[1].set_xlim(-200000,2000000)
axs[2].set_xlim(-200000,1000000)
plt.show()


# We have a highly skewed distribution, and several outliers.

# ### Dates

# In[37]:


dates = cust_churn[['id', 'date_activ', 'date_end', 'date_modif_prod', 'date_renewal', 'churn']].copy()


# In[38]:


dates['date_activ'] = pd.to_datetime(dates['date_activ'] , format='%Y-%m-%d')
dates['date_end'] = pd.to_datetime(dates['date_end'] ,  format='%Y-%m-%d')
dates['date_modif_prod'] = pd.to_datetime(dates['date_modif_prod'] ,  format='%Y-%m-%d')
dates['date_renewal'] = pd.to_datetime(dates['date_renewal'] ,  format='%Y-%m-%d')


# In[39]:


# Function to plot monthly churn and retention distribution

def plot_dates(df, col , fontsize_=12) :
    
    date_df = df[[col, 'churn', 'id']].set_index(col).groupby([pd.Grouper(freq='M'), 'churn']).count().unstack(level=1)
    
    ax = date_df.plot(kind='bar', stacked=True, figsize=(18,10), rot=0)
    ax.set_xticklabels(map(lambda x: line_format(x), date_df.index))
    plt.xticks(fontsize = fontsize_)
    plt.ylabel('Num of Companies')
    plt.legend(['Retention', 'Churn'], loc='upper_right')
    plt.show()


# In[40]:


# Function to convert time label to the format of pandas line plot

def line_format(label):
    
    month = label.month_name()[:1]
    if label.month_name()=='January':
        month+=f'\n{label.year}'
    return month


# In[41]:


plot_dates(dates, 'date_activ', fontsize_=8)


# In[42]:


plot_dates(dates, 'date_end', fontsize_=8)


# In[43]:


plot_dates(dates, 'date_modif_prod', fontsize_=8)


# In[44]:


plot_dates(dates, 'date_renewal', fontsize_=8)


# We can visualize the distribution of churned companies according to the date. However, this does not provide us with any usefulinsight. We will create a new feature using the raw dates provided in the next exercise.

# ### Forecast

# In[45]:


forecast_churn = cust_churn[['id' , 'forecast_base_bill_ele', 'forecast_base_bill_year',
       'forecast_bill_12m', 'forecast_cons', 'forecast_cons_12m',
       'forecast_cons_year', 'forecast_discount_energy','forecast_meter_rent_12m', 'forecast_price_energy_p1',
       'forecast_price_energy_p2', 'forecast_price_pow_p1', 'churn']]


# In[46]:


fig , axs = plt.subplots(nrows=11, figsize=(20,55))
plot_histogram(forecast_churn, 'forecast_base_bill_ele', axs[0])
plot_histogram(forecast_churn, 'forecast_base_bill_year', axs[1])
plot_histogram(forecast_churn, 'forecast_bill_12m', axs[2])
plot_histogram(forecast_churn, 'forecast_cons', axs[3])
plot_histogram(forecast_churn, 'forecast_cons_12m', axs[4])
plot_histogram(forecast_churn, 'forecast_cons_year', axs[5])
plot_histogram(forecast_churn, 'forecast_discount_energy', axs[6])
plot_histogram(forecast_churn, 'forecast_meter_rent_12m', axs[7])
plot_histogram(forecast_churn, 'forecast_price_energy_p1', axs[8])
plot_histogram(forecast_churn, 'forecast_price_energy_p2', axs[9])
plot_histogram(forecast_churn, 'forecast_price_pow_p1', axs[10])


# Similarly to the consumption plots, we can observe that a lot of the variables are highly skewed to the right, creating a very long tail on the highervalues.
# We will make some transformations to correct for this skewness

# ### Contract type (electricity, gas)

# In[47]:


contract_type = cust_churn[['id', 'has_gas', 'churn']]


# In[48]:


contract_type_count = contract_type.groupby(['has_gas', 'churn'])['id'].count().unstack(level=1)


# In[49]:


contract_type_percent = (contract_type_count.div(contract_type_count.sum(axis=1), axis=0)*100).sort_values(by=[1], ascending=False)


# In[50]:


contract_type_percent


# In[51]:


plot_stack_bars(contract_type_percent, 'Contract Type (gas)', 'Company %')


# ### Margin

# In[52]:


margin_churn = cust_churn[['id', 'margin_gross_pow_ele', 'margin_net_pow_ele', 'net_margin']]


# In[53]:


fig, axs = plt.subplots(nrows=3, figsize=(18,20))
sns.boxplot(margin_churn['margin_gross_pow_ele'] , ax=axs[0])
sns.boxplot(margin_churn['margin_net_pow_ele'] , ax=axs[1])
sns.boxplot(margin_churn['net_margin'] , ax=axs[2])

plt.show()


# We can observe a few outliers in here as well.

# ### Subscribed power

# In[54]:


subs_power_churn = cust_churn[['id', 'pow_max' , 'churn']].fillna(0)


# In[55]:


fig, axs = plt.subplots(nrows=1, figsize=(20,10))
plot_histogram(subs_power_churn, 'pow_max' , axs)


# ### Other Features

# In[56]:


other_feat = cust_churn[['id', 'nb_prod_act', 'num_years_antig', 'origin_up', 'churn']]


# In[57]:


num_products = other_feat.groupby(['nb_prod_act' , 'churn'])['id'].count().unstack(level=1)


# In[58]:


num_products_percent = (num_products.div(num_products.sum(axis=1) , axis=0)*100).sort_values(by=[1] , ascending=False)


# In[59]:


plot_stack_bars(num_products_percent, 'Number of products', 'Company %')


# In[60]:


years_antiquity = other_feat.groupby(['num_years_antig' , 'churn'])['id'].count().unstack(level=1)
years_antiquity_percent = (years_antiquity.div(years_antiquity.sum(axis=1) , axis=0)*100)
plot_stack_bars(years_antiquity_percent, 'Number of Years of Antiquity', 'Company %')


# In[61]:


origin = other_feat.groupby(['origin_up' , 'churn'])['id'].count().unstack(level=1)
origin_percent = (origin.div(origin.sum(axis=1) , axis=0)*100)
plot_stack_bars(origin_percent, 'Origin', 'Company %')


# ## 4. Data cleaning

# ### 4.1. Missing data

# In[62]:


cust_churn.isnull().mean()*100


# In[63]:


(cust_churn.isnull().mean()*100).plot(kind='bar' , figsize=(20,10))
plt.xlabel('Variables')
plt.ylabel('% of Missing Values')
plt.show()


# The columns with more than 60% of the values missing, will be dropped.

# In[64]:


cust_churn.drop(columns=['campaign_disc_ele', 'date_first_activ', 'forecast_base_bill_ele', 'forecast_base_bill_year',
                        'forecast_bill_12m', 'forecast_cons'], inplace=True)


# ### 4.2. Duplicates

# There are no duplicate rows in dataset.

# In[65]:


cust_churn[cust_churn.duplicated()]


# ### 4.3. Formatting Data

# ### 4.3.1. Missing dates

# There could be several ways in which we could deal with the missing dates.<br>
# One way, we could "engineer" the dates from known values. For example, the
# date_renewal
# is usually the same date as the
# date_modif_prod
# but one year ahead.
# The simplest way, we will replace the missing values with the median (the most frequent date). For numerical values, the built-in function
# .median()
# can be used, but this will not work for dates or strings, so we will use a workaround using
# .valuecounts()

# In[66]:


cust_churn.loc[cust_churn['date_modif_prod'].isnull(), 'date_modif_prod'] = cust_churn['date_modif_prod'].value_counts().index[0]
cust_churn.loc[cust_churn['date_end'].isnull(), 'date_end'] = cust_churn['date_end'].value_counts().index[0]
cust_churn.loc[cust_churn['date_renewal'].isnull(), 'date_renewal'] = cust_churn['date_renewal'].value_counts().index[0]


# We might have some prices missing for some companies and months

# In[67]:


(hist_price.isnull().mean()*100).plot(kind='bar', figsize=(20,10))
plt.xlabel('Variables')
plt.ylabel('Percentage of Missing Values %')
plt.show()


# There is not much data missing. Instead of removing the entries that are empty we will simply substitute them with the
# median

# In[68]:


hist_price.columns


# In[69]:


hist_price.loc[hist_price['price_p1_var'].isnull(), 'price_p1_var']=hist_price['price_p1_var'].median()
hist_price.loc[hist_price['price_p2_var'].isnull(), 'price_p2_var']=hist_price['price_p2_var'].median()
hist_price.loc[hist_price['price_p3_var'].isnull(), 'price_p3_var']=hist_price['price_p3_var'].median()
hist_price.loc[hist_price['price_p1_fix'].isnull(), 'price_p1_fix']=hist_price['price_p1_fix'].median()
hist_price.loc[hist_price['price_p2_fix'].isnull(), 'price_p2_fix']=hist_price['price_p2_fix'].median()
hist_price.loc[hist_price['price_p3_fix'].isnull(), 'price_p3_fix']=hist_price['price_p3_fix'].median()


# In order to use the dates in our churn prediction model we are going to change the representation of these dates. Instead of using the date itself, wewill be transforming it in number of months. In order to make this transformation we need to change the dates to
# datetime
# and create a
# reference date
# which will be January 2016 

# ### 4.3.2. Formatting dates - customer churn data and price history data

# In[70]:


cust_churn['date_activ'] = pd.to_datetime(cust_churn['date_activ'] , format='%Y-%m-%d')
cust_churn['date_end'] = pd.to_datetime(cust_churn['date_end'] , format='%Y-%m-%d')
cust_churn['date_modif_prod'] = pd.to_datetime(cust_churn['date_modif_prod'] , format='%Y-%m-%d')
cust_churn['date_renewal'] = pd.to_datetime(cust_churn['date_renewal'] , format='%Y-%m-%d')


# In[71]:


hist_price['price_date'] = pd.to_datetime(hist_price['price_date'], format='%Y-%m-%d')


# ### 4.3.3. Negative data points

# In[72]:


hist_price.describe()


# We can see that there are negative values for
# price_p1_fix
# ,
# price_p2_fix
# and
# price_p3_fix
# .
# Further exploring on those we can see there are only about
# 10
# entries which are negative. This is more likely to be due to corrupted data rather thana "price discount".
# We will replace the negative values with the
# median
# (most frequent value)

# In[73]:


hist_price[(hist_price['price_p1_fix'] < 0) | (hist_price['price_p2_fix'] < 0) | (hist_price['price_p3_fix'] < 0)]


# In[74]:


hist_price.loc[hist_price['price_p1_fix'] < 0 , 'price_p1_fix'] = hist_price['price_p1_fix'].median()
hist_price.loc[hist_price['price_p2_fix'] < 0 , 'price_p2_fix'] = hist_price['price_p2_fix'].median()
hist_price.loc[hist_price['price_p3_fix'] < 0 , 'price_p3_fix'] = hist_price['price_p3_fix'].median()


# ## 5. Feature engineering

# ### 5.1. New Feature Creation

# We will create new features using the average of the year, the last six months, and the last three months to our model beacuse we have the consumption data for each of the companies for the year 2015.

# In[75]:


mean_year = hist_price.groupby(['id']).mean().reset_index()


# In[76]:


mean_6m = hist_price[hist_price['price_date'] > '2015-06-01'].groupby(['id']).mean().reset_index()
mean_3m = hist_price[hist_price['price_date'] > '2015-10-01'].groupby(['id']).mean().reset_index()


# In[77]:


mean_year = mean_year.rename(index = str, columns={'price_p1_var' : 'mean_year_price_p1_var',
                                                  'price_p2_var' : 'mean_year_price_p2_var',
                                                  'price_p3_var' : 'mean_year_price_p3_var',
                                                  'price_p1_fix' : 'mean_year_price_p1_fix',
                                                  'price_p2_fix' : 'mean_year_price_p2_fix',
                                                  'price_p3_fix' : 'mean_year_price_p3_fix'})


# In[78]:


mean_year['mean_year_price_p1'] = mean_year['mean_year_price_p1_var'] + mean_year['mean_year_price_p1_fix']
mean_year['mean_year_price_p2'] = mean_year['mean_year_price_p2_var'] + mean_year['mean_year_price_p2_fix']
mean_year['mean_year_price_p3'] = mean_year['mean_year_price_p3_var'] + mean_year['mean_year_price_p3_fix']


# In[79]:


mean_6m = mean_6m.rename(index = str, columns={'price_p1_var' : 'mean_6m_price_p1_var',
                                                  'price_p2_var' : 'mean_6m_price_p2_var',
                                                  'price_p3_var' : 'mean_6m_price_p3_var',
                                                  'price_p1_fix' : 'mean_6m_price_p1_fix',
                                                  'price_p2_fix' : 'mean_6m_price_p2_fix',
                                                  'price_p3_fix' : 'mean_6m_price_p3_fix'})


# In[80]:


mean_6m['mean_6m_price_p1'] = mean_6m['mean_6m_price_p1_var'] + mean_6m['mean_6m_price_p1_fix']
mean_6m['mean_6m_price_p2'] = mean_6m['mean_6m_price_p2_var'] + mean_6m['mean_6m_price_p2_fix']
mean_6m['mean_6m_price_p3'] = mean_6m['mean_6m_price_p3_var'] + mean_6m['mean_6m_price_p3_fix']


# In[81]:


mean_3m = mean_3m.rename(index = str, columns={'price_p1_var' : 'mean_3m_price_p1_var',
                                                  'price_p2_var' : 'mean_3m_price_p2_var',
                                                  'price_p3_var' : 'mean_3m_price_p3_var',
                                                  'price_p1_fix' : 'mean_3m_price_p1_fix',
                                                  'price_p2_fix' : 'mean_3m_price_p2_fix',
                                                  'price_p3_fix' : 'mean_3m_price_p3_fix'})


# In[82]:


mean_3m['mean_3m_price_p1'] = mean_3m['mean_3m_price_p1_var'] + mean_3m['mean_3m_price_p1_fix']
mean_3m['mean_3m_price_p2'] = mean_3m['mean_3m_price_p2_var'] + mean_3m['mean_3m_price_p2_fix']
mean_3m['mean_3m_price_p3'] = mean_3m['mean_3m_price_p3_var'] + mean_3m['mean_3m_price_p3_fix']


# We create a new feature,  <b> tenure = date_end - date_activ</b>

# In[83]:


cust_churn['tenure'] = ((cust_churn['date_end'] - cust_churn['date_activ'])/np.timedelta64(1, "Y")).astype(int)


# In[84]:


tenure = cust_churn[['tenure', 'churn' , 'id']].groupby(['tenure', 'churn'])['id'].count().unstack(level=1)
tenure_percentage = (tenure.div(tenure.sum(axis=1) , axis=0)*100)


# In[85]:


tenure.plot(kind = 'bar' , figsize=(20,10) , stacked=True, rot=0, title='Tenure')

plt.legend(['Retention', 'Churn'], loc='upper_right')
plt.ylabel('Number of Companies')
plt.xlabel('Number of years')
plt.show()


# The churn is very low for companies which joined recently or that have made the contract a long time ago. With the higher number of churners within the 3-7 years of tenure.

# Need to transform the date columns to gain more insights.<br>
# months_activ : Number of months active until reference date (Jan 2016)<br>
# months_to_end : Number of months of the contract left at reference date (Jan 2016)<br>
# months_modif_prod : Number of months since last modification at reference date (Jan 2016)<br>
# months_renewal : Number of months since last renewal at reference date (Jan 2016)

# In[86]:


def get_months(ref_date, df , col):
    
    time_diff = ref_date-df[col]
    months = (time_diff / np.timedelta64(1, "M")).astype(int)
    
    return months


# In[87]:


ref_date = datetime.datetime(2016,1,1)


# In[88]:


cust_churn['months_activ'] = get_months(ref_date, cust_churn , 'date_activ')
cust_churn['months_end'] = -get_months(ref_date, cust_churn , 'date_end')
cust_churn['months_modif_prod'] = get_months(ref_date, cust_churn , 'date_modif_prod')
cust_churn['months_renewal'] = get_months(ref_date, cust_churn , 'date_renewal')


# In[89]:


def plot_monthly_churn(df, col):
    
    churn_per_month = df[[col, 'churn', 'id']].groupby([col, 'churn'])['id'].count().unstack(level=1)
    churn_per_month.plot(kind = 'bar', figsize=(20,10) , stacked=True, rot=0, title=col)
    plt.legend(['Retention', 'Churn'], loc='upper_right')
    plt.ylabel('Number of companies')
    plt.ylabel('Number of Months')
    plt.show()


# In[90]:


plot_monthly_churn(cust_churn, 'months_activ')


# In[91]:


plot_monthly_churn(cust_churn, 'months_end')


# In[92]:


plot_monthly_churn(cust_churn, 'months_modif_prod')


# In[93]:


plot_monthly_churn(cust_churn, 'months_renewal')


# Removing date columns

# In[94]:


cust_churn.drop(columns=['date_activ', 'date_end', 'date_modif_prod', 'date_renewal'], inplace=True)


# ### 5.2. Boolean Data Transformation

# For the column has_gas, we will replace t for True or 1 and f for False or 0 (onehot encoding)

# In[95]:


cust_churn['has_gas'] = cust_churn['has_gas'].replace(['t', 'f'], [1,0])


# ### 5.3. Categorical data and dummy variables

# ### Categorical data channel_sales

# Categorical data channel_sales
# What we are doing here relatively simple, we want to convert each category into a new dummy variable which will have 0 s and 1 s depending
# whether than entry belongs to that particular category or not
# First of all let's replace the Nan values with a string called null_values_channel

# In[96]:


cust_churn['channel_sales'] = cust_churn['channel_sales'].fillna('null_channels')


# Now transform the channel_sales column into categorical data type

# In[97]:


cust_churn['channel_sales'] = cust_churn['channel_sales'].astype('category')


# In[98]:


cust_churn['channel_sales'].value_counts().reset_index()


# So that means we will create 8 different dummy variables . Each variable will become a different column.

# In[99]:


# Dummy Variables
channels_category = pd.get_dummies(cust_churn['channel_sales'] , prefix='channel')


# In[100]:


channels_category.columns = [col[:11] for col in channels_category.columns]


# In[101]:


channels_category.head(10)


# Multicollinearity can affect our models so we will remove one of the columns.

# In[102]:


channels_category.drop(columns=['channel_nul'] , inplace=True)


# ### Categorical data origin_up

# In[103]:


cust_churn['origin_up'] = cust_churn['origin_up'].fillna('null_origin')


# In[104]:


cust_churn['origin_up'] = cust_churn['origin_up'].astype('category')


# In[105]:


cust_churn['origin_up'].value_counts().reset_index()


# In[106]:


origin_categories = pd.get_dummies(cust_churn['origin_up'] , prefix='origin')
origin_categories.columns = [col[:11] for col in origin_categories.columns]


# In[107]:


origin_categories.head(10)


# In[108]:


origin_categories.drop(columns=['origin_null'] , inplace=True)


# ### Categorical data activity_new

# In[109]:


cust_churn['activity_new'] = cust_churn['activity_new'].fillna('null_activity')


# In[110]:


cat_activity = cust_churn['activity_new'].value_counts().reset_index().rename(columns={'activity_new' : 'Activity_Counts', 
                                                                                       'index' : 'Activity'})
cat_activity


# As we can see below there are too many categories with very few number of samples. So we will replace any category with less than 75 samples as
# null_values_category

# In[111]:


cat_activity[cat_activity['Activity']=='null_activity']


# In[112]:


#to_replace = list(cat_activity[cat_activity['Activity_Counts'] <= 75].index)
to_replace = list(cat_activity[cat_activity['Activity_Counts'] <= 75]['Activity'])


# In[113]:


cust_churn['activity_new'] = cust_churn['activity_new'].replace(to_replace, 'null_activity')


# In[114]:


cat_activity = pd.get_dummies(cust_churn['activity_new'], prefix='activity')
cat_activity.columns = [col[:12] for col in cat_activity.columns] 


# In[115]:


cat_activity.head(10)


# In[116]:


cat_activity.drop(columns = ['activity_nul'], inplace=True)


# We will merge all the new categories into our main dataframe and remove the old categorical columns

# In[117]:


cust_churn =  pd.merge(cust_churn, channels_category , left_index=True, right_index=True)
cust_churn =  pd.merge(cust_churn, origin_categories , left_index=True, right_index=True)
cust_churn =  pd.merge(cust_churn, cat_activity , left_index=True, right_index=True)


# In[118]:


cust_churn.drop(columns=['channel_sales', 'origin_up', 'activity_new'], inplace=True)


# ### 5.4. Log transformation

# There are several methods in which we can reduce skewness such as square root , cube root , and log . In this case, we will use a log
# transformation which is usually recommended for right skewed data.

# In[119]:


cust_churn.describe()


# Columns having large standard deviation std need log trnsformation for skewness. Log transformation doesnot work with negative data, in such case we will convert the values to Nan. Also for 0 data we will add 1 then apply Log transformation.

# In[120]:


# Removing negative data

cust_churn.loc[cust_churn['cons_12m'] < 0 , 'cons_12m'] = np.nan
cust_churn.loc[cust_churn['cons_gas_12m'] < 0 , 'cons_gas_12m'] = np.nan
cust_churn.loc[cust_churn['cons_last_month'] < 0 , 'cons_last_month'] = np.nan
cust_churn.loc[cust_churn['forecast_cons_12m'] < 0 , 'forecast_cons_12m'] = np.nan
cust_churn.loc[cust_churn['forecast_cons_year'] < 0 , 'forecast_cons_year'] = np.nan
cust_churn.loc[cust_churn['forecast_meter_rent_12m'] < 0 , 'forecast_meter_rent_12m'] = np.nan
cust_churn.loc[cust_churn['imp_cons'] < 0 , 'imp_cons'] = np.nan


# In[121]:


# Applying Log Transformation

cust_churn['cons_12m'] = np.log10(cust_churn['cons_12m']+1)
cust_churn['cons_gas_12m'] = np.log10(cust_churn['cons_gas_12m']+1)
cust_churn['cons_last_month'] = np.log10(cust_churn['cons_last_month']+1)
cust_churn['forecast_cons_12m'] = np.log10(cust_churn['forecast_cons_12m']+1)
cust_churn['forecast_cons_year'] = np.log10(cust_churn['forecast_cons_year']+1)
cust_churn['forecast_meter_rent_12m'] = np.log10(cust_churn['forecast_meter_rent_12m']+1)
cust_churn['imp_cons'] = np.log10(cust_churn['imp_cons']+1)


# In[122]:


fig, axs = plt.subplots(nrows=7, figsize=(20,60))
sns.distplot((cust_churn['cons_12m'].dropna()), ax=axs[0])
sns.distplot((cust_churn[cust_churn['has_gas']==1]['cons_gas_12m'].dropna()), ax=axs[1])
sns.distplot((cust_churn['cons_last_month'].dropna()), ax=axs[2])
sns.distplot((cust_churn['forecast_cons_12m'].dropna()), ax=axs[3])
sns.distplot((cust_churn['forecast_cons_year'].dropna()), ax=axs[4])
sns.distplot((cust_churn['forecast_meter_rent_12m'].dropna()), ax=axs[5])
sns.distplot((cust_churn['imp_cons'].dropna()), ax=axs[6])
plt.show()


# In[123]:


fig, axs = plt.subplots(nrows=7, figsize=(20,60))
sns.boxplot((cust_churn['cons_12m'].dropna()), ax=axs[0])
sns.boxplot((cust_churn[cust_churn['has_gas']==1]['cons_gas_12m'].dropna()), ax=axs[1])
sns.boxplot((cust_churn['cons_last_month'].dropna()), ax=axs[2])
sns.boxplot((cust_churn['forecast_cons_12m'].dropna()), ax=axs[3])
sns.boxplot((cust_churn['forecast_cons_year'].dropna()), ax=axs[4])
sns.boxplot((cust_churn['forecast_meter_rent_12m'].dropna()), ax=axs[5])
sns.boxplot((cust_churn['imp_cons'].dropna()), ax=axs[6])
plt.show()


# In[124]:


cust_churn.describe()


# From the boxplots we can still see some values are quite far from the range ( outliers ).

# ### 5.5. High Correlation Features

# In[125]:


features = mean_year

correlation = features.corr()


# In[126]:


plt.figure(figsize=(20,15))

sns.heatmap(correlation,  xticklabels=correlation.columns.values, yticklabels=correlation.columns.values, annot=True,
           annot_kws={'size' : 10})

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()


# We can remove highly correlated variables.

# In[127]:


correlation = cust_churn.corr()
correlation


# As expected, num_years_antig has a high correlation with months_activ (it provides us the same information).<br>
# We can remove variables with very high correlation.

# In[128]:


cust_churn.drop(columns=['num_years_antig', 'forecast_cons_year'], inplace=True)


# ### 5.6. Outliers Removal

# The consumption data has several outliers. Need to remove those outliers

# The most common way to identify an outlier are either:<br>
# 1. Data point that falls outside of 1.5 times of an interquartile range above the 3rd quartile and below the 1st quartile OR, <br>
# 2. Data point that falls outside of 3 standard deviations.

# We will replace the outliers with the mean (average of the values excluding outliers).

# In[129]:


# Replace outliers with the mean values using the Z score.
# Nan values are also replaced with the mean values.

def replace_z_score(df, col, z=3):
    
    from scipy.stats import zscore
    
    temp_df = df.copy(deep=True)
    temp_df.dropna(inplace=True, subset=[col])
    
    temp_df["zscore"] = zscore(df[col])
    mean_=temp_df[(temp_df['zscore'] > -z) & (temp_df['zscore'] < z)][col].mean()
    
    df[col] = df[col].fillna(mean_)
    df['zscore']=zscore(df[col])
    no_outlier=df[(df['zscore'] < -z) | (df['zscore'] > z)].shape[0]
    df.loc[(df['zscore'] < -z) | (df['zscore'] > z) , col] = mean_
    
    print('Replaced : {} outliers in {}'.format(no_outlier, col))
    return df.drop(columns='zscore')


# In[130]:


for feat in features.columns:
    
    if feat!='id':
        features = replace_z_score(features, feat)


# In[131]:


features.reset_index(drop=True, inplace=True)


# When carrying out the log transformation , the dataset has several outliers.

# What are the criteria to identify an outlier?
# The most common way to identify an outlier are:
# 1. Data point that falls outside of 1.5 times of an interquartile range above the 3rd quartile and below the 1st quartile
# OR
# 2. Data point that falls outside of 3 standard deviations.
# Once, we have identified the outlier, What do we do with the outliers?
# There are several ways to handle with those outliers such as removing them (this works well for massive datasets) or replacing them with sensible data
# (works better when the dataset is not that big).
# We will replace the outliers with the mean (average of the values excluding outliers).

# In[132]:


def find_outliers_iqr(df, column):
    
    col = sorted(df[column])
    
    q1, q3 = np.percentile(col, [25,75])
    iqr = q3-q1
    lower_bound = q1 - (1.5*iqr)
    upper_bound = q3 + (1.5*iqr)
    
    results_ouliers = {'iqr' : iqr , 'lower_bound' : lower_bound , 'upper_bound' : upper_bound}
    
    return results_ouliers


# In[133]:


def remove_ouliers_iqr(df, column):
    
    outliers = find_outliers_iqr(df,column)
    removed_outliers = df[(df[col] < outliers['lower_bound']) | (df[col] > outliers['upper_bound'])].shape
    
    df = df[(df[col] > outliers['lower_bound']) | (df[col] < outliers['upper_bound'])] 
    print('Removed {} outliers'.format(removed_outliers[0]))
    return df


# In[134]:


def remove_outliers_zscore(df, col, z=3):
    
    from scipy.stats import zscore
    
    df["zsscore"]=zscore(df[col])
    removed_outliers = df[(df["zscore"] < -z) | (df["zscore"] > z)].shape
    df = df[(df["zscore"] > -z) | (df["zscore"] < z)]
    
    print('Removed: {} otliers of {}'.format(removed_outliers[0], col))
    
    return df.drop(columns="zscore")


# In[135]:


def replace_outliers_z_score(df, col, z=3):
    
    from scipy.stats import zscore
    
    temp_df = df.copy(deep=True)
    #temp_df.dropna(inplace=True, subset=[col])
    
    temp_df["zscore"] = zscore(df[col])
    mean_=temp_df[(temp_df["zscore"] > -z) & (temp_df["zscore"] < z)][col].mean()
    
    num_outliers = df[col].isnull().sum()
    df[col] = df[col].fillna(mean_)
    df["zscore"]=zscore(df[col])
    df.loc[(df["zscore"] < -z) | (df["zscore"] > z) , col] = mean_
    
    print('Replaced : {} outliers in {}'.format(num_outliers, col))
    return df.drop(columns="zscore")


# In[136]:


cust_churn = replace_outliers_z_score(cust_churn , 'cons_12m')
cust_churn = replace_outliers_z_score(cust_churn , 'cons_gas_12m')
cust_churn = replace_outliers_z_score(cust_churn , 'cons_last_month')
cust_churn = replace_outliers_z_score(cust_churn , 'forecast_cons_12m')
cust_churn = replace_outliers_z_score(cust_churn , 'forecast_discount_energy')
cust_churn = replace_outliers_z_score(cust_churn , 'forecast_meter_rent_12m')
cust_churn = replace_outliers_z_score(cust_churn , 'forecast_price_energy_p1')
cust_churn = replace_outliers_z_score(cust_churn , 'forecast_price_energy_p2')
cust_churn = replace_outliers_z_score(cust_churn , 'forecast_price_pow_p1')
cust_churn = replace_outliers_z_score(cust_churn , 'imp_cons')
cust_churn = replace_outliers_z_score(cust_churn , 'margin_gross_pow_ele')
cust_churn = replace_outliers_z_score(cust_churn , 'margin_net_pow_ele')
cust_churn = replace_outliers_z_score(cust_churn , 'net_margin')
cust_churn = replace_outliers_z_score(cust_churn , 'pow_max')
cust_churn = replace_outliers_z_score(cust_churn , 'months_activ')
cust_churn = replace_outliers_z_score(cust_churn , 'months_end')
cust_churn = replace_outliers_z_score(cust_churn , 'months_modif_prod')
cust_churn = replace_outliers_z_score(cust_churn , 'months_renewal')


# In[137]:


cust_churn.reset_index(drop=True, inplace=True)


# Let's see how the boxplots changed!

# In[138]:


fig, axs = plt.subplots(nrows=6, figsize=(20,60))
sns.boxplot((cust_churn['cons_12m'].dropna()), ax=axs[0])
sns.boxplot((cust_churn[cust_churn['has_gas']==1]['cons_gas_12m'].dropna()), ax=axs[1])
sns.boxplot((cust_churn['cons_last_month'].dropna()), ax=axs[2])
sns.boxplot((cust_churn['forecast_cons_12m'].dropna()), ax=axs[3])
sns.boxplot((cust_churn['forecast_meter_rent_12m'].dropna()), ax=axs[4])
sns.boxplot((cust_churn['imp_cons'].dropna()), ax=axs[5])
plt.show()


# In[139]:


# Loading JS visualization 
shap.initjs()


# In[140]:


train = pd.merge(cust_churn, hist_price, on='id')


# In[141]:


pd.DataFrame({'Columns' : train.columns})


# ## 6. Churn Prediction Model with XGBoost

# ### 6.1. Splitting Dataset

# In[142]:


y = train['churn']
X = train.drop(labels=['id', 'churn', 'price_date'], axis=1)


# In[143]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# ### 6.2. Modelling

# In[144]:


model = xgb.XGBClassifier(learning_rate=0.1, max_depth=6, n_estimators=500, n_jobs=-1)
result = model.fit(X_train, y_train)


# ### 6.3. Model Evaluation

# We are going to evaluate our Logistic Regression model on our test data (which we did not use for training) using the evalution metrics <b>Accuracy, Precision, Recall</b>

# In[145]:


def evaluation(model_, X_test_, y_test_):
    
    predict_test = model_.predict(X_test_)
    results = pd.DataFrame({'Accuracy' : [metrics.accuracy_score(y_test_, predict_test)],
                           'Precision' : [metrics.precision_score(y_test_, predict_test)],
                           'Recall' : [metrics.recall_score(y_test_, predict_test)]})
    
    print(metrics.classification_report(y_test_, predict_test))
    
    return results


# In[146]:


evaluation(model, X_test, y_test)


# <b>ROC-AUC :</b> Receiver Operating Characteristic(ROC) curve is a plot of the true positive rate against the false positive rate. It shows the tradeoff between sensitivityand specificity.

# In[147]:


def calculation_roc_auc(model_, X_test_, y_test_): 

    # Get the model predictions, class 1 -> churn
    prediction_test_ = model_.predict_proba(X_test_)[:,1]

    # Computing roc-auc
    fpr, tpr, thresholds = metrics.roc_curve(y_test_, prediction_test_)
    score = pd.DataFrame({"ROC-AUC" : [metrics.auc(fpr, tpr)]})

    return fpr, tpr, score

def plot_roc_auc(fpr,tpr):
    f, ax = plt.subplots(figsize=(14,8))

    # Plot ROC
    roc_auc = metrics.auc(fpr, tpr) 
    ax.plot(fpr, tpr, lw=2, alpha=0.3,
    label="AUC = %0.2f" % (roc_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=3, color='r', label="Random", alpha=.8)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05]) 
    ax.set_xlabel("False Positive Rate (FPR)") 
    ax.set_ylabel("True Positive Rate (TPR)") 
    ax.set_title("ROC-AUC") 
    ax.legend(loc="lower right")
    plt.show()


# In[148]:


fpr, tpr, auc_score = calculation_roc_auc(model, X_test, y_test)


# In[149]:


auc_score


# In[150]:


plot_roc_auc(fpr, tpr) 
plt.show()


# ### 6.4. Stratiﬁed K-fold validation

# In[151]:


def plot_roc_curve(fprs, tprs): 

    tprs_interp = [] 
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    f, ax = plt.subplots(figsize=(18,10))

    # Plot ROC for each K-Fold and compute AUC scores.
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)): 
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr)) 
        tprs_interp[-1][0] = 0.0
        roc_auc = metrics.auc(fpr, tpr) 
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=2, alpha=0.3, label="ROC fold %d (AUC = %0.2f)" % (i, roc_auc))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=3, color='r', label="Random", alpha=.8)

    # Plot the mean ROC.
    mean_tpr = np.mean(tprs_interp, axis=0) 
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr) 
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b', label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc), 
            lw=4, alpha=.8)

    # Plot the standard deviation around the mean ROC.
    std_tpr = np.std(tprs_interp, axis=0) 
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1) 
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=.2, label=r"$\pm$ 1 std. dev.")

    # Fine tune and show the plot.
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05]) 
    ax.set_xlabel("False Positive Rate (FPR)") 
    ax.set_ylabel("True Positive Rate (TPR)") 
    ax.set_title("ROC-AUC") 
    ax.legend(loc="lower right")
    plt.show()
    return (f, ax)


# In[152]:


def compute_roc_auc(model_, index):
    y_predict = model_.predict_proba(X.iloc[index])[:,1]
    fpr, tpr, thresholds = metrics.roc_curve(y.iloc[index], y_predict) 
    auc_score = metrics.auc(fpr, tpr)
    return fpr, tpr, auc_score


# In[153]:


cv = StratifiedKFold(n_splits=3, random_state=42, shuffle=True) 
fprs, tprs, scores = [], [], []


# In[154]:


for (train, test), i in zip(cv.split(X, y), range(3)):
    model.fit(X.iloc[train], y.iloc[train])
    _, _, auc_score_train = compute_roc_auc(model, train) 
    fpr, tpr, auc_score = compute_roc_auc(model, test) 
    scores.append((auc_score_train, auc_score)) 
    fprs.append(fpr)
    tprs.append(tpr)


# In[155]:


plot_roc_curve(fprs, tprs) 
plt.show()


# ### 6.5. Model Finetuning

# ### 6.5.1. Grid search with cross validation

# In[156]:


from sklearn.model_selection import GridSearchCV

# Parameter grid based on the results of random search
param_grid ={'subsample': [0.7],
'scale_pos_weight': [1],
'n_estimators': [1100],
'min_child_weight': [1],
'max_depth': [12, 13, 14],
'learning_rate': [0.005, 0.01],
'gamma': [4.0],
'colsample_bytree': [0.6]}


# In[157]:


# Create model
xg = xgb.XGBClassifier(objective='binary:logistic',silent=True, nthread=1)


# In[158]:


# Grid search model
grid_search = GridSearchCV(estimator = xg, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2, scoring = "roc_auc")


# In[159]:


# Fit the grid search to the data
grid_search.fit(X_train,y_train)


# In[160]:


best_grid = grid_search.best_params_ 
best_grid


# In[165]:


# Model with the parameters found
model_grid = xgb.XGBClassifier(objective='binary:logistic',silent=True, nthread=1, **best_grid) 


# In[166]:


fprs, tprs, scores = [], [], []

for (train, test), i in zip(cv.split(X, y), range(3)): 
    model_grid.fit(X.iloc[train], y.iloc[train])
    _, _, auc_score_train = compute_roc_auc(model_grid, train) 
    fpr, tpr, auc_score = compute_roc_auc(model_grid, test) 
    scores.append((auc_score_train, auc_score)) 
    fprs.append(fpr)
    tprs.append(tpr)


# In[167]:


plot_roc_curve(fprs, tprs) 
plt.show()


# ## 7. Model Understanding

# ### 7.1. Feature Importance

# Feature importance is done by counting the number of times each feature is split on across all boosting rounds (trees) in the model, and then visualizing the result as a bar graph, with the features ordered according to how many times they appear.

# In[168]:


fig, ax = plt.subplots(figsize=(15,20)) 
xgb.plot_importance(model_grid, ax=ax)


# In the feature importance graph above we can see that cons_12m, forecast_meter_rent_12m, forecast_cons_12m, margin_gross_pow_ele and net_margin are the features that appear the most in our model and we could infere that these two features have a signiﬁcant importnace in our model
