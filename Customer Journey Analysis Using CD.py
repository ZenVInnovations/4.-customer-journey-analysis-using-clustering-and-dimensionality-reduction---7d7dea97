#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install scikit-learn


# In[4]:


pip install streamlit


# ## Libraries like numpy, pandas etc have been imported. A dataframe has been created for the Customer behaviour Tourism csv file.

# In[2]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# In[3]:


df=pd.read_csv('CustomerbehaviourTourism.csv')


# In[4]:


df.head()


# In[5]:


df.columns


# In[6]:


print(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns")


# In[7]:


df.info()


# # Data Cleaning

# The dataset contails 11760 rows and 17 columns. There are 7 categorical and 10 numeric variables. Few columns has missing values in it. Taken Product is the target Variable

# In[8]:


cat_columns= df.select_dtypes(exclude=['int64', 'float64'] )

for col in cat_columns:
    print(f"Value counts for column '{col}':")
    print(df[col].unique())
    print()


# In[9]:


device_mapping= {'iOS and Android': 'Mobile',
          'iOS' : 'Mobile',
          'ANDROID' : 'Mobile',
          'Android': 'Mobile',
          'Android OS' : 'Mobile',
          'Other': 'Mobile',
          'Others' : 'Mobile',
          'Tab' : 'Mobile'}
df['preferred_device'] = df['preferred_device'].replace(device_mapping)
df['preferred_device'].unique()
     


# In[10]:


device_mapping= {'iOS and Android': 'Mobile',
                        'iOS' : 'Mobile',
                        'ANDROID' : 'Mobile',
                        'Android': 'Mobile',
                        'Android OS' : 'Mobile',
                        'Other': 'Mobile',
                        'Others' : 'Mobile',
                        'Tab' : 'Mobile'}
if 'preferred_device' in df.columns:
    df['preferred_device'] = df['preferred_device'].replace(device_mapping)
    print("Successfully replaced preferred_device values.")
    print(df['preferred_device'].unique())
else:
    print("Error: 'preferred_device' column not found in the DataFrame.")


# In[11]:


df['yearly_avg_Outstation_checkins']= df['yearly_avg_Outstation_checkins'].replace('*',np.nan)
df['yearly_avg_Outstation_checkins'] = pd.to_numeric(df['yearly_avg_Outstation_checkins'], errors='coerce', downcast='integer')

df['yearly_avg_Outstation_checkins'].unique()


# In[12]:


page_mapping= {'Yes': 1,
          'No' : 0,
          'Yeso' : 1}
df['following_company_page'] = df['following_company_page'].replace(page_mapping)
df['following_company_page'].unique()
     


# In[13]:


df['member_in_family']= df['member_in_family'].replace('Three',3)
     


# In[14]:


df['member_in_family'] = pd.to_numeric(df['member_in_family'], errors='coerce', downcast='integer')
df['member_in_family'].unique()


# In[15]:


# "following_company_page" column has "Three", change it to 3
df['working_flag']= df['working_flag'].replace('0','No')
df['working_flag'].unique()
     


# In[16]:


df[["travelling_network_rating", "Adult_flag"]]= df[["travelling_network_rating", "Adult_flag"]].astype("object")


# In[17]:


df.info()


# In[18]:


num_columns = df.select_dtypes(exclude=['object']).drop(columns=['UserID'])


# In[19]:


negative_values = (num_columns < 0).any()
print("Columns with negative values:")
print(negative_values[negative_values].index)


# # Missing Values

# We will impute all missing in categorical using Mode. For all Numeric, we will use Median

# In[20]:


df = df.applymap(lambda x: np.nan if x == 'nan' else x)


# In[21]:


RED, BOLD, RESET = '\033[91m', '\033[1m','\033[0m'
total_missing = df.isnull().sum().sum()
total_cells = df.size
missing_percentage = (total_missing / total_cells) * 100
print(f"The total number of missing values are {BOLD}{RED}{total_missing}{RESET}, which is {BOLD}{RED}{missing_percentage:.2f}%{RESET} of total data.")
     


# In[22]:


missing = df.columns[df.isna().any()].tolist()
total_rows = len(df)
for column in missing:
    missing_count = df[column].isna().sum()
    missing_percentage = (missing_count / total_rows) * 100
    print(f"{BOLD}{column}{RESET} has {BOLD}{RED}{missing_count}{RESET} missing values, which is {BOLD}{RED}{missing_percentage:.2f}%{RESET} of the column.")
     


# In[23]:


cat_columns = ['preferred_device', 'preferred_location_type', 'following_company_page', 'working_flag', 'Adult_flag']
for i in cat_columns:
    df[i].fillna(df[i].mode()[0], inplace = True)


# In[24]:


num_columns = ['Yearly_avg_view_on_travel_page', 'total_likes_on_outstation_checkin_given', 'yearly_avg_Outstation_checkins', 'Yearly_avg_comment_on_travel_page', 'Daily_Avg_mins_spend_on_traveling_page']
for column in num_columns:
    median_value = df[column].median()
    df[column].fillna(median_value, inplace=True)


# In[25]:


print('Missing Values in the dataset after treatment :', df.isnull().sum().sum())


# # Duplicate Values

# In[26]:


df.duplicated().sum()


# In[27]:


df[["travelling_network_rating", "Adult_flag"]]= df[["travelling_network_rating", "Adult_flag"]].astype("object")
df.info()
     


# # Descriptive Summary

# 1. User Engagement Metrics: Users exhibit varied engagement levels on the platform, as reflected in metrics such as yearly average views on the travel page, total likes on outstation check-ins, and yearly average comments on the travel page.
# 2. User Demographics: User demographics indicate an average family size of approximately 2.88 members, with variability observed.
# 3. Check-in Behavior: Users tend to engage in an average of 8.10 outstation check-ins per year, with some variability in behavior. The duration since the last outstation check-in shows variability, potentially influencing current engagement levels.
# 4. Social Interaction: Social interaction plays a significant role, as seen in the metrics for total likes on out-of-station check-ins received and monthly comments on the company page. These metrics shows the importance of user interactions within the platform.
# 5. Platform Usage Patterns: Users spend an average of 13.64 minutes daily on the traveling page, with variability in daily engagement.

# In[28]:


df.describe().T


# In[29]:


df.select_dtypes(include = ['object']).describe().T


# # Data Visualization

# In[32]:


import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.subplots as sp


# In[31]:


get_ipython().system('pip install plotly')


# In[33]:


fig = px.pie(
    df.assign(ClassMap=df.Taken_product.map({'No': "Not Taken", "Yes": "Taken"})),
    names="ClassMap", hole=0.5,color_discrete_sequence=["#79a5db", "#e0a580"])
fig.update_layout(height=450,width=600, font_color="#28838a",title_font_size=16,  showlegend=False,)
fig.add_annotation( x=0.5, y=0.5, align="center", xref="paper",yref="paper", showarrow=False, font_size=20, text="TargetOverview",)
fig.update_traces(hovertemplate=None, textposition="outside", texttemplate="%{label}%{value} - %{percent}",
    textfont_size=16,rotation=-20, marker_line_width=25,  marker_line_color='#ffffff',)
fig.show()


# # Distribution of Numerical variables

# # Distribution of Categorical variables

# In[35]:


df.travelling_network_rating.value_counts()


# In[36]:


df= df[df['travelling_network_rating'] != 10]
     


# In[37]:


cat_colums = df.select_dtypes(include = ['object'])
def univariateAnalysis_category(cols):
    print("Distribution of", cols)
    print("----------------------------------------------------------------")
    colors = ['#79a5db', '#e0a580', '#6fab90', '#896ca8', '#ADD8E6']
    value_counts = cat_colums[cols].value_counts()
    # Count plot
    fig = px.bar(
        value_counts,
        x=value_counts.index,
        y=value_counts.values,
        title=f'Distribution of {cols}',
        labels={'x': 'Categories', 'y': 'Count'},color_discrete_sequence=[colors])
    fig.update_layout(width=700)
    fig.update_layout(plot_bgcolor='#ffffff', paper_bgcolor='#ffffff')
    fig.show()
    # Donut chart
    percentage = (value_counts / value_counts.sum()) * 100
    fig = px.pie(
        values=percentage, names=value_counts.index,
        labels={'names': 'Categories', 'values': 'Percentage'}, hole=0.5,color_discrete_sequence=colors)
    fig.add_annotation(
        x=0.5, y=0.5, align="center", xref="paper",
        yref="paper", showarrow=False, font_size=15, text=f'{cols}')
    fig.update_layout(legend=dict(x=0.9, y=0.5))
    fig.update_layout(width=700)
    fig.show()
    print("       ")
for x in cat_colums:
    univariateAnalysis_category(x)


# In[38]:


fig = plt.figure(figsize=[32, 15])
fig.suptitle('Bivariate Analysis : Distribution of Columns with Product Taken ', fontsize=18, fontweight='bold')
fig.subplots_adjust(top=0.92)
fig.subplots_adjust(hspace=0.5, wspace=0.4)
for i, col in enumerate(num_columns):
    a = fig.add_subplot(3, 3, i+1)
    sns.distplot(df[df['Taken_product'] == "No"][col], color='#142863', ax=a, hist=False, label='Not Taken')
    sns.distplot(df[df['Taken_product'] == "Yes"][col], color='#f2634e', ax=a, hist=False, label='Taken')
    a.set_title(col, fontdict=axtitle_dict)
    a.legend(fontsize=15)
     


# In[39]:


#Correlation heatmap
corr = df[num_columns].corr(method='pearson')
fig = plt.subplots(figsize=(12, 6))
ax = sns.heatmap(corr, annot=True, fmt='.2f', cbar=None, linewidth=0.9)
ax.set_xticklabels([label.get_text().replace('_', '\n') for label in ax.get_xticklabels()], rotation=0, horizontalalignment='center')
ax.set_title(' Correlation Matrix', fontdict=axtitle_dict)
plt.show()


# In[40]:


#Outliers in each Columns
plt.rcParams['axes.facecolor'] = 'white'
fig = plt.figure(figsize=[32,24])
fig.suptitle('BOXPLOT OF ALL COLUMNS', fontsize=18, fontweight='bold')
fig.subplots_adjust(top=0.92);
fig.subplots_adjust(hspace=0.5, wspace=0.4);
for i ,col in enumerate(num_columns):
    ax1 = fig.add_subplot(6,3, i+1);
    ax1 = sns.boxplot(data = df, x=col ,  color= colours[i]);
    ax1.set_title(f'{col}', fontdict=axtitle_dict)
    ax1.set_xlabel(f'{col}', fontdict=axlab_dict)


# In[41]:


#Checking numbers of observations beyond Upper & Lower Limit
Q5 = df[num_columns].quantile(0.05)
Q95 = df[num_columns].quantile(0.95)
UL = Q95
LL = Q5
outliers = ((df[num_columns] > UL) | (df[num_columns] < LL)).sum()
print("Number of Observations Beyond Upper & Lower Limit for Each Column:")
display(outliers)


# In[42]:


def treat_outlier(col):
    q5  , q95 = np.percentile(col, [5, 95])
    return q5, q95

for i in num_columns:
    LR, UR  = treat_outlier(df[i])
    df[i] = np.where(df[i] > UR, UR, df[i])
    df[i] = np.where(df[i] < LR, LR, df[i])


# In[43]:


plt.rcParams['axes.facecolor'] = 'white'
fig = plt.figure(figsize=[32,24])
fig.suptitle('BOXPLOT OF ALL COLUMNS POST TREATMENT (SCALED)', fontsize=18, fontweight='bold')
fig.subplots_adjust(top=0.92);
fig.subplots_adjust(hspace=0.5, wspace=0.4);
for i ,col in enumerate(num_columns):
    ax1 = fig.add_subplot(6,3, i+1);
    ax1 = sns.boxplot(data = df, x=col ,  color= colours[i]);
    ax1.set_title(f'{col}', fontdict=axtitle_dict)
    ax1.set_xlabel(f'{col}', fontdict=axlab_dict)


# # Inferences

# 1.Device Preference: Give mobile optimization top priority for all digital material, including ads, websites, and applications, as 90.58% of users prefer mobile devices.
# 
# 2.Improving Engagement: There may be a void in user engagement as the bulk of users (71.93%) do not follow the company page. Create plans to grow the number of followers on your corporate page by posting insightful and pertinent material. Encourage community and loyalty among users by involving them in interactive posts, polls, and promotions.
# 
# 3.Recognizing the Non-Working Majority: Marketing techniques should be tailored in light of the preponderance of non-working users (84.62%). Think of providing this user demographic with exclusive specials, off-peak pricing, or flexible vacation packages.
# 
# 4.Resolving Travelling Network Ratings: The ratings distribution emphasizes the significance of answering customer comments, especially with a notable number at '3' (31.23%) and '4' (29.38%). Make adjustments to give a better user experience and increase overall satisfaction.
# 
# 5.User Segment Demographic Tailoring: Given the wide range of distribution in the Adult Flag category, with notable percentages in the '0.0' (42.92%) and '1.0' (40.55%) categories, adjusting services and promotions according to user age groups can result in more focused and successful marketing campaigns.
# 
# 

# In[ ]:




