#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


# data collection and pre-processing


# In[3]:


movies_data = pd.read_csv('/Users/kisha/OneDrive/Desktop/ML_project/movies.csv')


# In[4]:


# movies_data.head()


# In[5]:


#selectibg the relevent feAture for recommadation


# In[6]:


selected_feature = ['genres','keywords','tagline','cast','director']


# In[7]:


#replacing the null values with null string


# In[8]:


for feature in selected_feature:
    movies_data[feature] = movies_data[feature].fillna('')


# In[9]:


#combining all 5 selected feaTure..


# In[10]:


combined_feature = movies_data['genres'] + ''+movies_data['keywords'] + ''+movies_data['tagline'] + ''+movies_data['cast'] + ''+movies_data['director']
# print(combined_feature)


# In[11]:


#coverting the text data to feature vector..(numeric)
#using vectorizer


# In[11]:


vectorizer = TfidfVectorizer()
feature_vector = vectorizer.fit_transform(combined_feature)


# In[12]:


# print(feature_vector)


# In[14]:


#cosine_similarity to get score using cosine..


# In[13]:


similarity = cosine_similarity(feature_vector)


# In[14]:


# print(similarity)


# In[17]:


#creating a list with all the movies name given in dataset


# In[15]:


list_of_movies = movies_data['title'].tolist()


# In[19]:


#getting movies from user..


# In[20]:


movies_name = input("Enter your favorite movies name: ")


# In[21]:


#finding the close matching for the movies name given by the user


# In[22]:


find_closer = difflib.get_close_matches(movies_name,list_of_movies)


# In[23]:


print(find_closer)


# In[24]:


closed_match = find_closer[0]
print(closed_match)


# In[25]:


#finding index of the movie with title


# In[26]:


index_movie = movies_data[movies_data.title == closed_match]['index'].values[0]


# In[27]:


print(index_movie)


# In[28]:


#getting similar movies by similarity score..


# In[29]:


similarity_score = list(enumerate(similarity[index_movie]))


# In[30]:


# print(similarity_score)


# In[31]:


#sorted the movies based on their similarity..


# In[32]:


sorted_movies = sorted(similarity_score,key = lambda x:x[1],reverse = True)


# In[33]:


#printing similar movies based on the index..


# In[34]:


i = 0
for movies in sorted_movies:
    index = movies[0]
    title_from_index = movies_data[movies_data.index==index]['title'].values[0]
    if i<10:
        print(i,'.',title_from_index)
        i += 1


# In[ ]:




