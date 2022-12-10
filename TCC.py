#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import ast
import plotly.offline as py
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from   sklearn.linear_model import LinearRegression, Ridge
from   sklearn.metrics import r2_score
from sklearn import linear_model
from xgboost import XGBRegressor
from scipy import stats


# In[2]:


#Reading database 
movies = pd.read_csv("movies_metadata.csv")


# In[3]:


#Transforming release_date in date format
movies.release_date = pd.to_datetime(movies.release_date, format="%Y-%m-%d", errors='coerce')


# In[4]:


#Printing the firts and the last release date
print("Primeira data de lançamento:\n", movies.release_date.min())
print("Última data de lançamento:\n", movies.release_date.max())


# In[5]:


movies.info() #Chacking the movie data


# In[6]:


#Droping features that won't be necessary
movies = movies.drop(['homepage','imdb_id','overview','poster_path','tagline','video'], axis=1)


# In[7]:


#'belongs_to_collection' will be binary
movies.loc[movies.belongs_to_collection.isna(),'belongs_to_collection']= 0 #Replacing null values with 0
movies.loc[movies.belongs_to_collection != 0,'belongs_to_collection']= 1 #Replacing null values with 1


# In[8]:


#Checking title and origital_title column
movies[movies.original_title != movies.title][['title', 'original_title']].head()


# In[9]:


#From now on we'll use only original_title as the title information
movies = movies.drop(columns='original_title')


# In[10]:


#Droping movies without release_date information
movies = movies[movies.release_date.notna()]

#Transforming release_date onto year and month features
movies['year'] = pd.DatetimeIndex(movies.release_date).year
movies['month'] = pd.DatetimeIndex(movies.release_date).month

#Droping relase_date
movies = movies.drop(columns = 'release_date')


# In[11]:


#Checking revenue column
filter_revenue = movies.loc[movies.revenue == 0]
filter_revenue.shape


# In[12]:


#There 37969 movies with revenue as 0, this is a very important information for our analysis, so we're goint to drop this lines
movies = movies.drop(filter_revenue.index)


# In[13]:


#Checking the data one more time to search missing values
movies.info()


# In[14]:


#Tranforming some object features to float
movies.budget = movies.budget.astype(float)
movies.popularity = round(movies.popularity.astype(float),2)


# In[15]:


#Saving a csv file to work the visualization in DataStudio
movies.to_csv("movies.csv")


# DataStudio: https://datastudio.google.com/reporting/06a6bf61-f390-4a32-bef4-965d65f9a3df

# In[16]:


movies1 = movies.copy(deep=True) #Copying movies data to another dataframe


# In[17]:


#Analyzing the most profitable genres
movies1['genres'] = movies1['genres'].fillna('[]').apply(ast.literal_eval)
movies1['genres'] = movies1['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

s = movies1.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genres'

genres_df = movies1.drop('genres', axis=1).join(s)

genres_pivot = genres_df.groupby("genres").agg(Total = ('revenue','sum'), Avg = ('revenue','mean'), Number = ('revenue','count')).reset_index()

genres_pivot.sort_values('Total', ascending=False).head(10)


# In[18]:


genres_pivot.count() #Counting how many singular genres exists.


# In[19]:


#Analyzing genres frequency and percentual
genres_pivot1 = genres_df.groupby("genres").agg(Number = ('revenue','count')).reset_index()
genres_pivot1['Percentual'] = round((genres_pivot1.Number/7407)*100,2)

genres_pivot1.sort_values('Number', ascending=False).head(10)


# In[20]:


##Analyzing the most profitable production companies
movies1['production_companies'] = movies1['production_companies'].fillna('[]').apply(ast.literal_eval)
movies1['production_companies'] = movies1['production_companies'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

s = movies1.apply(lambda x: pd.Series(x['production_companies']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'production_companies'

comp_df = movies1.drop('production_companies', axis=1).join(s)

comp_pivot = comp_df.groupby("production_companies").agg(Total = ('revenue','sum'), Avg = ('revenue','mean'), Number = ('revenue','count')).reset_index()

comp_pivot.sort_values('Total', ascending=False).head(10)


# In[21]:


comp_pivot.count() #Counting how many production companies exists


# In[22]:


#Analyzing production companies frequency and percentual
comp_pivot1 = comp_df.groupby("production_companies").agg(Number = ('revenue','count')).reset_index()
comp_pivot1['Percentual'] = round((comp_pivot1.Number/7407)*100,2)

comp_pivot1.sort_values('Number', ascending=False).head(10)


# In[23]:


#Analyzing production countries frequency and percentual
movies1['production_countries'] = movies1['production_countries'].fillna('[]').apply(ast.literal_eval)
movies1['production_countries'] = movies1['production_countries'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

s = movies1.apply(lambda x: pd.Series(x['production_countries']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'production_countries'

con_df = movies1.drop('production_countries', axis=1).join(s)
con_df = pd.DataFrame(con_df['production_countries'].value_counts())
con_df['country'] = con_df.index
con_df.columns = ['num_movies','country']
con_df = con_df.reset_index().drop('index', axis=1)
con_df['percentual'] = round((con_df.num_movies/7405)*100,2)
con_df.head(10)


# In[24]:


#Ploting a map with the most frequency production countries without USA
con_df1 = con_df[con_df['country'] != 'United States of America']

data = [ dict(
        type = 'choropleth',
        locations = con_df1['country'],
        locationmode = 'country names',
        z = con_df1['num_movies'],
        text = con_df1['country'],
        colorscale = [[0,'rgb(230, 238, 255)'],[1,'rgb(0, 60, 179)']],
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Production Countries'),
      ) ]

layout = dict(
    title = 'Production Countries (Apart from US)',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='d3-world-map' )


# Some of features need a treatment to be used numerically in data visualization process and the modeling.

# In[25]:


#Transforming the spoken_languages features
movies1['spoken_languages'] = movies1['spoken_languages'].fillna('[]').apply(ast.literal_eval)
movies1['spoken_languages'] = movies1['spoken_languages'].apply(lambda x: [i['iso_639_1'] for i in x] if isinstance(x, list) else [])


# In[26]:


#Copying the movies1 in a new dataframe: movies2
movies2 = movies1.copy(deep=True)


# In[27]:


#Creating new features
movies2['genres_quantity'] = movies2['genres'].apply(lambda x: len(x))
movies2['is_english'] = movies2['original_language'].apply(lambda x: 1 if x=='en' else 0)
movies2['countries_quantity'] = movies2['production_countries'].apply(lambda x: len(x))
movies2['companies_quantity'] = movies2['production_companies'].apply(lambda x: len(x))
movies2['spoken_languages_quantity'] = movies2['spoken_languages'].apply(lambda x: len(x))
movies2['status_is_released'] = movies2['status'].apply(lambda x: 1 if x=='Released' else 0)


# In[28]:


#Droping the old features
movies2 = movies2.drop(columns=['adult','genres','original_language','production_countries','production_companies','spoken_languages','status'])


# In[29]:


#Printing the avarage of the new features about quantity
print("Genres quantity avg:",movies2.genres_quantity.mean())
print("Countries quantity avg:",movies2.countries_quantity.mean())
print("Companies quantity avg:",movies2.companies_quantity.mean())
print("Spoken languages quantity avg:",movies2.spoken_languages_quantity.mean())


# In[30]:


#Saving a csv file to work the visualization in DataStudio
movies2.to_csv("movies2.csv")


# In[31]:


#Working on bivariate analysis, considering revenue as target
#Corr matrix
mask = np.zeros_like(movies2.corr())
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(16, 8))
    ax = sns.heatmap(movies2.corr(), mask=mask, square=True, annot=True)


# In[32]:


#Ploting budget x revenue scatter plot
plt.scatter(movies2.budget, movies2.revenue)
plt.title("Budget x Revenue")
plt.xlabel("Budget")
plt.ylabel("Revenue")
plt.plot(np.unique(movies2.budget), np.poly1d(np.polyfit(movies2.budget, movies2.revenue, 1)) 
         (np.unique(movies2.budget)), color='red')


# In[33]:


#Ploting vote count x revenue scatter plot
plt.scatter(movies2.vote_count, movies2.revenue)
plt.title("Vote count x Revenue")
plt.xlabel("Vote count")
plt.ylabel("Revenue")
plt.plot(np.unique(movies2.vote_count), np.poly1d(np.polyfit(movies2.vote_count, movies2.revenue, 1)) 
         (np.unique(movies2.vote_count)), color='red')


# In[34]:


#Ploting popularity x revenue scatter plot
plt.scatter(movies2.popularity, movies2.revenue)
plt.title("Popularity x Revenue")
plt.xlabel("Popularity")
plt.ylabel("Revenue")
plt.plot(np.unique(movies2.popularity), np.poly1d(np.polyfit(movies2.popularity, movies2.revenue, 1)) 
         (np.unique(movies2.popularity)), color='red')


# In[35]:


#Feature engineering - Inputing in zero budgets with avarage
movies2.loc[movies2.budget == 0, "budget"]= round(movies2.budget.mean(),2)
#Transforming belongs to collection and id in int
movies2[['belongs_to_collection','id']] = movies2[['belongs_to_collection','id']].astype(int)


# In[36]:


title_df = movies2[['title','id']]
movies_new = movies2.drop(columns = ['title'])


# In[37]:


movies_new = movies_new[np.isfinite(movies_new).all(1)]


# In[38]:


#Spliting the dataset
X, y = movies_new.drop(['revenue','id'], axis=1), movies_new['revenue']
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=13)


# In[39]:


#Running the Linear Regression Model
reg = LinearRegression()
reg.fit(train_X,train_y)


# In[40]:


#R² score of train linear model
reg.score(train_X, train_y)


# In[41]:


#Printing the intercept and the coeficient of linear model
print(reg.intercept_, reg.coef_)


# In[42]:


#Printing the predict of the linear model
predict_linear = reg.predict(test_X)
predict_linear


# In[43]:


#R² score of test linear model
reg.score(test_X,test_y)


# In[44]:


#Residual analysis of Linear Regression Model
res_linear = test_y - predict_linear

plt.hist(res_linear, bins=15)
plt.title('Histograma dos Resíduos da Regressão')
plt.show()


# In[97]:


#Error graph Linear Regression
plt.scatter(y=res_linear, x=predict_linear)
plt.hlines(y=0, xmin=0, xmax=1000000000, color='red')
plt.ylabel('$\epsilon = y - \hat{y}$ - Resíduos')
plt.xlabel('$\hat{y}$ ou $E(y)$ - Predito')
plt.title('Gráfico de Erro da regressão')
plt.show()


# In[96]:


#Normal Test Linear Regression
name = ['Jarque-Bera', 'p-value']
test_linear = stats.jarque_bera(res_linear)
print(dict(zip(name, test_linear)))


# In[47]:


#Running the Lasso Regression Model
lasso = linear_model.Lasso(alpha=0.1) #Alpha default
lasso.fit(train_X, train_y)


# In[48]:


#Printing the intercept and coeficient of lasso model
print(lasso.intercept_, lasso.coef_)


# In[49]:


#Printing the lasso score for train model
lasso.score(train_X, train_y)


# In[50]:


#Printing the predict for lasso 
predict_lasso = lasso.predict(test_X)
predict_lasso


# In[51]:


#Printing the lasso score for test model
lasso.score(test_X, test_y)


# In[52]:


#Running Lasso Model with another value for alpha or lamba
lasso1 = linear_model.Lasso(alpha=.5) #Alpha = .5
lasso1.fit(train_X, train_y)


# In[53]:


#Printing the intercept and coeficient of lasso model 1
print(lasso1.intercept_, lasso1.coef_)


# In[54]:


#Printing the lasso score for train model 1
lasso1.score(train_X, train_y)


# In[72]:


#Residual analysis of Lasso Regression Model
res_lasso = test_y - predict_lasso

plt.hist(res_lasso, bins=15)
plt.title('Histograma dos Resíduos da Regressão')
plt.show()


# In[98]:


#Error graph Lasso Regression
plt.scatter(y=res_lasso, x=predict_lasso)
plt.hlines(y=0, xmin=0, xmax=1000000000, color='red')
plt.ylabel('$\epsilon = y - \hat{y}$ - Resíduos')
plt.xlabel('$\hat{y}$ ou $E(y)$ - Predito')
plt.title('Gráfico de Erro da regressão')
plt.show()


# In[91]:


#Normal Test Lasso Regression]
test_lasso = stats.jarque_bera(res_lasso)
print(dict(zip(name, test_lasso)))


# In[55]:


#Running the Ridge Regression Model
ridge = linear_model.Ridge(alpha=1.0)
ridge.fit(train_X, train_y)


# In[56]:


#Printing the intercept and coeficient of ridge model
print(ridge.intercept_, ridge.coef_)


# In[57]:


#Printing the ridge score for train model
ridge.score(train_X, train_y)


# In[58]:


#Printing the predict for ridge
predict_ridge = ridge.predict(test_X)
predict_ridge


# In[59]:


#Printing the ridge score for test model
ridge.score(test_X, test_y)


# In[93]:


#Residual analysis of Ridge Regression Model
res_ridge = test_y - predict_ridge

plt.hist(res_ridge, bins=15)
plt.title('Histograma dos Resíduos da Regressão')
plt.show()


# In[99]:


#Error graph Ridge Regression
plt.scatter(y=res_ridge, x=predict_ridge)
plt.hlines(y=0, xmin=0, xmax=1000000000, color='red')
plt.ylabel('$\epsilon = y - \hat{y}$ - Resíduos')
plt.xlabel('$\hat{y}$ ou $E(y)$ - Predito')
plt.title('Gráfico de Erro da regressão')
plt.show()


# In[95]:


#Normal Test Ridge Regression
test_ridge = stats.jarque_bera(res_ridge)
print(dict(zip(name, test_ridge)))


# In[60]:


#Running the XGBoost Model
xgb = XGBRegressor()
xgb.fit(train_X,train_y)


# In[61]:


#Printing the xgboost score for train model
xgb.score(train_X, train_y)


# In[62]:


#Printing the predict for xgboost
predict_xgb = xgb.predict(test_X)
predict_xgb


# In[63]:


#Printing the xgboost score for test model
xgb.score(test_X, test_y)


# In[64]:


#Our XGBoost model is overfitting, so we're going to use GridSearch and Cross-Validation to set the parameters of the model
parameters = {'nthread':[2,3,4],
              'objective':['reg:linear'],
              'learning_rate': [.05],
              'max_depth': [4,5, 6, 7,8,9],
              'min_child_weight': [2,3,4],
              'silent': [1],
              'subsample': [1],
              'colsample_bytree': [1],
              'n_estimators': [100]}

xgb_grid = GridSearchCV(xgb,
                        parameters,
                        cv = 10,
                        n_jobs = 5,
                        verbose=True)


# In[65]:


#Printing the xgboost score for train model with the new parameters
xgb_grid.fit(train_X, train_y)


# In[66]:


#Getting the best score and best parameters from the new xgboost model
print("Best Score:", xgb_grid.best_score_)
print("\nBest Paramaters:", xgb_grid.best_params_)


# In[67]:


#Running the new xgboost model
xgb1 = XGBRegressor(colsample_bytree=1, 
                    learning_rate= 0.05, 
                    max_depth= 6, 
                    min_child_weight= 4, 
                    n_estimators= 100, 
                    nthread= 2, 
                    silent= 1,
                    objective= 'reg:linear',
                    subsample= 1)
xgb1.fit(train_X,train_y)


# In[68]:


#Priting the new xgboost model score for train
xgb1.score(train_X, train_y)


# In[69]:


#Printing the predict from the new xgboost model 
predict_xgb1 = xgb1.predict(test_X)
predict_xgb1


# In[70]:


#Printing the test score for the new model
xgb1.score(test_X, test_y)

