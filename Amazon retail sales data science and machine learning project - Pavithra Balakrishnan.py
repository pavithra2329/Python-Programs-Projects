#!/usr/bin/env python
# coding: utf-8

# # Amazon Retail Sales Dataset Data Science And Machine Learing Project

# presentation of cash study link:
#           https://1drv.ms/p/s!AmsrRhwLsAtYgi0cplYohd3CAQaU?e=IZVimd

# ![Amazon%20Case%20Study.jpeg](attachment:Amazon%20Case%20Study.jpeg)

# # About this file:
#         Category       :  Type of product. (String)
#         Size           :  Size of the product. (String)
#         Date           :  Date of the sale. (Date)
#         Status         :  Status of the sale. (String)
#         Fulfilment     :  Method of fulfilment. (String)
#         Style          :  Style of the product. (String)
#         SKU            :  Stock Keeping Unit. (String)
#         ASIN           :  Amazon Standard Identification Number. (String)
#         Courier Status :  Status of the courier. (String)
#         Qty            :  Quantity of the product. (Integer)
#         Amount         :  Amount of the sale. (Float)
#         B2B            :  Business to business sale. (Boolean)
#         Currency       :  The currency used for the sale. (String)
#         rating         :  customer rating for the product (float)
#         rating count   :  how many members rating the product(int)
#         
#         

# 4.Data Preprocessing and Exploration

# # Importing Libraries:
# pandas (pd): A powerful data manipulation library in Python.
# numpy (np): A library for numerical operations and array handling.
# matplotlib.pyplot (plt): A library for creating visualizations such as plots and charts.
# random: A module providing functions for generating pseudo-random numbers.
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random 
import warnings
warnings.filterwarnings('ignore')


#  reads AMAZON RETILE SALES CSV file containing sales data and stores it in a Pandas DataFrame named 'sales'

# In[2]:


sales = pd.read_csv(r"C:\Users\Lenovo\Downloads\amazon sales report\Amazon Sale Report.csv")
sales


# the 'sales' DataFrame by adding a new column named 'rating count.' This column is populated with randomly generated
# integer values between 1 and 2500 (inclusive) using a list comprehension and the random module.

# In[3]:


sales['rating count'] = [int(random.uniform(1,2500)) for i in range(len(sales))]
sales.head()


# # the 'sales' DataFrame by adding a new column named 'rating count.' This column is populated with randomly generated
# integer values between 1 and 2500 (inclusive) using a list comprehension and the random module.

# In[4]:


sales['rating'] = [round(random.uniform(1,5),1) for i in range(len(sales))]
sales


# The names of all the columns in the 'sales' DataFrame. The output is an Index object containing the column names.
# Total number of columns in the 'sales' DataFrame. 

# In[5]:


print(sales.columns)
print(len(sales.columns))


# The 'missingno' library is a Python tool designed for visualizing and analyzing missing data in datasets.

# In[6]:


import missingno as msno


# *generates a bar chart using the missingno library, illustrating the proportion of missing values for each column in 
# the 'sales' DataFrame.

# In[7]:


# Visualize the matrix of missing values

msno.matrix(sales)
plt.title('Matrix of Missing Values')
plt.show()

# Visualize the bar chart of missing values

msno.bar(sales, color='dodgerblue')
plt.title('Bar Chart of Missing Values')
plt.show()


# unwanted_columns: This variable holds a list of column names that are considered unwanted and should be removed from 
# the DataFrame. 

# In[8]:


unwanted_columns = ['fulfilled-by', 'Unnamed: 22','promotion-ids','Courier Status'] 
# Replace with your actual column namde
amw_sales = sales.drop(unwanted_columns, axis=1, inplace = True)



# The 'dropna'() method to remove rows (axis=0) containing any missing values (NaN) from the 'sales' DataFrame. 
# The result is a new DataFrame named 'amw_sales' that contains only the rows without missing values.

# In[9]:


amw_sales = sales.dropna()
amw_sales.head(3)


# #3.-Matrix of Missing Values Visualization

# generates a matrix plot using the missingno library, where each row represents a data point, and columns represent
# features (columns) in the 'amw_sales' DataFrame.The matrix displays white lines for missing values, making it easy to
# visualize the pattern of missingness across the cleaned dataset.

# In[10]:


# Visualize the matrix of missing values
msno.matrix(amw_sales)
plt.title('Matrix of Missing Values')
plt.show()

# Visualize the bar chart of missing values
msno.bar(amw_sales, color='dodgerblue')
plt.title('Bar Chart of Missing Values')
plt.show()



# the cleaning process on missing values in the dataset.

# The "Before Cleaning" subplot shows missing value patterns in the original dataset, while the "After Cleaning" subplot shows 
# the impact of removing rows with missing values.

# In[11]:


# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(30, 12))

# Plot missing values before cleaning
msno.matrix(sales, ax=axes[0])
axes[0].set_title('Before Cleaning', fontweight='bold', fontsize = 26, color ='gray' )

# Clean the dataset (replace this with your cleaning logic)
amw_sales = sales.dropna()

# Plot missing values after cleaning
msno.matrix(amw_sales, ax=axes[1])
axes[1].set_title('After Cleaning', fontweight='bold', fontsize = 26)

# Show the plots
plt.tight_layout()
plt.show()


# the column names of the cleaned DataFrame (amw_sales). It provides a quick overview of the available columns in the dataset.

# In[12]:


amw_sales.columns


# In[13]:


amw_sales.describe()


# the number of unique values for each column in the 'amw_sales' DataFrame using the nunique() method.
# It returns a Pandas Series with column names as the index and the corresponding count of unique values.
# Converts the Pandas Series to a DataFrame.

# In[14]:


amw_sales.nunique().to_frame(name = 'count of unique value')


# In[15]:


amw_sales.apply(pd.unique).to_frame(name = 'unique values')


# change the 'Date' column is in datetime format.

# In[16]:


amw_sales['Date'] = pd.to_datetime(amw_sales['Date'])


# In[17]:


amw_sales= amw_sales.sort_values('Date')

order_frequency = amw_sales.groupby('Date').size().reset_index(name='Number of Orders')


# In[18]:


plt.figure(figsize=(10, 6))
#plt.plot(order_frequency['Date'], order_frequency['Number of Orders'], marker='o', linestyle='-', color='gray')  # Gray line
plt.plot(order_frequency['Date'], order_frequency['Number of Orders'], marker='o',mfc='gray',ms='7',linestyle='-', color='orange')  # Orange marker
plt.title('Order Frequency Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Orders')
plt.grid(True)
plt.show()


# In[19]:


import seaborn as sns

custom_palette = [ "yellow","orange", "gray", "black"]

amw_sales['month'] = amw_sales['Date'].dt.month

# Perform the groupby operation
monthly_sales = amw_sales.groupby('month')['Amount'].sum()

# Plot the bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x=monthly_sales.index, y=monthly_sales.values, palette=custom_palette)
plt.xlabel('Month')
plt.ylabel('Total Sales Amount')
plt.title('Monthly Sales')
plt.show()


# In[20]:


# Assuming your data is stored in a DataFrame called 'df'
# You may need to clean and preprocess the 'Status' column if necessary

# Count the occurrences of each order status
status_counts = amw_sales['Status'].value_counts()
# Alternatively, you can use a bar chart
plt.figure(figsize=(20, 10))
status_counts.plot(kind='bar', color=['orange', 'gray', '#111111'])
plt.title('Order Status Distribution')
plt.xlabel('Order Status')
plt.ylabel('Number of Orders')
plt.xticks(rotation=25)  # Rotate x-axis labels if needed
plt.show()


# In[21]:


# Assuming your data is stored in a DataFrame called 'df'
fulfillment_distribution = amw_sales['Fulfilment'].value_counts()

# Plotting the pie chart
plt.figure(figsize=(8, 8))
plt.pie(fulfillment_distribution, labels=fulfillment_distribution.index, autopct='%1.1f%%', startangle=90, colors=['gray', 'orange'])
plt.title('Fulfillment Method Distribution')
plt.show()


# In[22]:


# Assuming your data is stored in a DataFrame called 'df'
shipping_service_levels = amw_sales['ship-service-level'].value_counts()
#custom_color = ['orange','gray']
# Plotting the bar chart
plt.figure(figsize=(10, 6))
shipping_service_levels.plot(kind='bar', color = 'orange')
plt.title('Shipping Service Level Distribution')
plt.xlabel('Shipping Service Level')
plt.ylabel('Number of Orders')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.show()


# In[23]:


# Assuming your data is stored in a DataFrame called 'df'
style_distribution = amw_sales['Style'].value_counts().head(10)  # You can adjust the number of styles to display

# Plotting the pie chart
plt.figure(figsize=(8, 8))
plt.pie(style_distribution, labels=style_distribution.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
plt.title('Product Style Distribution')
plt.show()


# In[24]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming your data is stored in a DataFrame called 'df'
category_distribution = amw_sales['Category'].value_counts()

# Plotting the bar chart
plt.figure(figsize=(10, 6))
category_distribution.plot(kind='bar', color='gray')
plt.title('Product Category Distribution')
plt.xlabel('Product Category')
plt.ylabel('Number of Orders')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.show()


# In[25]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming your data is stored in a DataFrame called 'amw_sales'
category_distribution = amw_sales['Category'].value_counts()

# Define a custom color palette from gray to orange
num_categories = len(category_distribution)
gray_palette = sns.light_palette("gray", num_categories)
orange_palette = sns.light_palette("orange", num_categories)
colors = [gray_palette[i] if i != num_categories-1 else orange_palette[0] for i in range(num_categories)]

# Plotting the bar chart with the defined color gradient
plt.figure(figsize=(10, 6))
bars = plt.bar(category_distribution.index, category_distribution.values, color=colors)

plt.title('Product Category Distribution')
plt.xlabel('Product Category')
plt.ylabel('Number of Orders')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.show()


# In[26]:


custom_palette = [ "orange","yellow", "gray", "black"]
heatmap_data = amw_sales.pivot_table(index='Category', columns='Size', values='Qty', aggfunc='sum', fill_value=0)
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap= custom_palette , annot=True, fmt='d', linewidths=.1)
plt.title('Heatmap of Quantity Sold by Category and Size')
plt.show()


# In[27]:


# 7. SKU-Level Analysis
top_skus = amw_sales.groupby('SKU')['Qty'].sum().nlargest(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_skus.index, y=top_skus.values, palette='coolwarm')
plt.title('Top 10 SKUs by Quantity Sold')
plt.xlabel('SKU')
plt.ylabel('Total Quantity Sold')
plt.xticks(rotation=35)
plt.show()


# # TOP 3 SELLING PROJECT IMAGES

# 'Top selling sku no 1(JNE3797-KR-L),2(JNE3797-KR-M),3(JNE3797-KR-S) 7(JNE3797-KR-XL)9(JNE3797-KR-XS)All sizes in this model has been sales in high quantity and together with the top 10 best selling products'

# 
# ![PRODUCT%20IMG%201.jpg](attachment:PRODUCT%20IMG%201.jpg)

# The second bestseller, kurthi has made it to top 10 in two sizes(1.sku:JNE3045-KR-S)(2.SKU:JNE3045-KR-S)

# ![81uJHOVq7GL._SL1500_.jpg](attachment:81uJHOVq7GL._SL1500_.jpg)

# This kurti is the third model in two sizes and the top 10 selling product.

# ![81JEyvXxwFL._AC_SY550_.jpg](attachment:81JEyvXxwFL._AC_SY550_.jpg)

# In[28]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'amw_sales' is your DataFrame
# Calculate the size distribution
size_distribution = amw_sales['Size'].value_counts()

# Set a grayscale color palette
colors = sns.color_palette("Greys_r", len(size_distribution))

# Plotting the bar chart
plt.figure(figsize=(10, 6))
size_distribution.plot(kind='bar', color=colors)
plt.title('Size Distribution of Ordered Items')
plt.xlabel('Size')
plt.ylabel('Number of Items')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.show()


# In[29]:


import matplotlib.pyplot as plt

# Assuming 'amw_sales' is your DataFrame
b2b_b2c_comparison = amw_sales['B2B'].value_counts()

# Plotting the pie chart with orange and gray colors
plt.figure(figsize=(8, 8))
colors = ['orange', 'gray']  # Define colors for B2B and B2C
plt.pie(b2b_b2c_comparison, labels=b2b_b2c_comparison.index, autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('B2B vs. B2C Comparison')
plt.show()


# In[30]:


# Assuming your data is stored in a DataFrame called 'df'
# Assuming 'rating' column contains numerical ratings (e.g., 1 to 5)

# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.hist(amw_sales['rating'], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], edgecolor='black', color='orange')
plt.title('Product Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Number of Products')
plt.xticks(range(1, 6))
plt.show()


# # machine learning 

# In[31]:


import pandas as pd


# In[32]:


amazon=pd.read_csv(r"C:\Users\Lenovo\Downloads\amazon sales report\cleaned dana amw\awm_clean_data.csv")
amazon.columns


# In[33]:


Y=amazon.iloc[:,-1]

Y.head()


# In[34]:


cat_col=['Status','Fulfilment','ship-service-level','Category','Size','ship-city','ship-state']
ordi_col=['Style','SKU','ASIN','currency','ship-country','B2B']
num_col=['Order ID','Date','Qty','Amount','ship-postal-code','rating count']
X=amazon[cat_col+ordi_col+num_col]
X.head()


# In[35]:


from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer

ct=ColumnTransformer(transformers=[('Cat_encode',OneHotEncoder(),cat_col),
                                  ('Ordi_encode',OrdinalEncoder(),ordi_col),
                                  ('other_encode',OrdinalEncoder(),num_col)],remainder='passthrough')

x=ct.fit_transform(X)


# In[36]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,Y,test_size=0.25,random_state=1)


# In[37]:


from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()

lin_reg.fit(x_train,y_train)


# In[38]:


y_pred=lin_reg.predict(x_test)


# In[39]:


from sklearn.metrics import mean_squared_error,r2_score

rscore=r2_score(y_test,y_pred)
rscore*100


# # Random forest

# In[40]:


Y=amazon.iloc[:,-1]


# In[41]:


x=ct.fit_transform(X)


# In[42]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,Y,test_size=0.25,random_state=1)


# In[43]:


from sklearn.ensemble import RandomForestRegressor

#reg=RandomForestRegressor(n_estimators=10,random_state=0)
#reg.fit(x_train,y_train)


# In[44]:


reg=RandomForestRegressor(n_estimators=10,random_state=0)
reg.fit(x_train,y_train)

