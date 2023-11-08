#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd #To read csv files as dataframes
from sklearn.feature_extraction.text import TfidfVectorizer #To compute TFIDF sparse matrices 
from sklearn.metrics.pairwise import linear_kernel #To compute dot product with sparse matrix (cosine similarity)


# In[2]:


def read_data():
    '''Reads all the csv files into their own dataframes. Assumes all columns as strings.'''
    
    df_brand = pd.read_csv(r'data\brand_category.csv', dtype='str')
    df_cat = pd.read_csv(r'data\categories.csv', dtype='str')
    df_offers = pd.read_csv(r'data\offer_retailer.csv', dtype='str')
    
    return df_brand, df_cat, df_offers


# In[3]:


def user_input(df_brand, df_cat, df_offers):
    '''Asks the user for search term and checks to see if it is in the dataframes.'''
    
    x = input('Search term: ').upper()
    
    if x in set(df_offers.RETAILER):
        search_type = 'Retail'
    elif x in set(df_brand.BRAND):
        search_type = 'Brand'
    elif x.capitalize() in set(df_cat.PRODUCT_CATEGORY):
        x = x.capitalize()
        search_type = 'Category'
    else:
        print('Search term is not in the database. Please try again.')
        x, search_type = user_input(df_brand, df_cat, df_offers) # recursion until a valid search term is inputted
 
    
    return x, search_type


# In[4]:


def df_offer_tagged(df_brand, df_cat, df_offers):
    '''Gets all the tags for each offer from all the dataframes. The tags include retailer, brand, category, and subcategory. Returns a list of strings to be used in TFIDF.'''
    
    df_brand_cat = df_brand.merge(df_cat, how='left', left_on='BRAND_BELONGS_TO_CATEGORY', right_on='PRODUCT_CATEGORY')
    df_brands = df_brand.groupby(['BRAND'])['BRAND_BELONGS_TO_CATEGORY'].apply(', '.join).reset_index()
    df_cats = df_brand_cat.groupby(['BRAND'])['IS_CHILD_CATEGORY_TO'].apply(', '.join).reset_index()
    df_all = df_brands.merge(df_cats, on='BRAND')
    df = df_offers.merge(df_all, how='left', on='BRAND')
    df.fillna('None', inplace=True)
    res = [' '.join(ele).lower() for ele in df.values.tolist()]

    return res


# In[5]:


def get_retail_tags(keyword, df_brand, df_cat, df_offers):
    '''Gets the tags for Retail search in string format.'''

    df_offers1 = df_offers.dropna(subset=['RETAILER'])

    df1 = pd.merge(df_offers1[['RETAILER', 'BRAND']],
                   df_brand[['BRAND', 'BRAND_BELONGS_TO_CATEGORY']],
                   on='BRAND',
                   how='left')

    df2 = pd.merge(df1,
                   df_cat[['PRODUCT_CATEGORY', 'IS_CHILD_CATEGORY_TO']],
                   left_on='BRAND_BELONGS_TO_CATEGORY',
                   right_on='PRODUCT_CATEGORY',
                   how='left')
    df2.fillna('None', inplace=True)
    df3 = df2.groupby('RETAILER').agg({'BRAND': lambda x: ' '.join(set(x)),
                                       'BRAND_BELONGS_TO_CATEGORY': lambda x: ' '.join(set(x)),
                                       'IS_CHILD_CATEGORY_TO': lambda x: ' '.join(set(x))}
                                     )

    tags = keyword + ' ' + df3.loc[keyword, 'BRAND'] + ' ' + df3.loc[keyword, 'BRAND_BELONGS_TO_CATEGORY'] + ' ' + df3.loc[keyword, 'IS_CHILD_CATEGORY_TO']
    
    return tags


# In[6]:


def get_cat_tags(keyword, df_cat):
    '''Gets the tags for category search for the search term.'''
    
    df1 = df_cat.groupby('PRODUCT_CATEGORY').agg({'IS_CHILD_CATEGORY_TO': lambda x: ' '.join(set(x))}
                                     )
    tags = keyword + ' ' + df1.loc[keyword, 'IS_CHILD_CATEGORY_TO']
    return tags


# In[7]:


def get_brand_tags(keyword, df_brand, df_cat):
    '''Gets the tags for brand search for the search term.'''
    
    df_brand_cat = df_brand.merge(df_cat, how='left', left_on='BRAND_BELONGS_TO_CATEGORY', right_on='PRODUCT_CATEGORY')
    df_brands = df_brand.groupby(['BRAND'])['BRAND_BELONGS_TO_CATEGORY'].apply(', '.join).reset_index()
    df_cats = df_brand_cat.groupby(['BRAND'])['IS_CHILD_CATEGORY_TO'].apply(', '.join).reset_index()
    df_all = df_brands.merge(df_cats, on='BRAND')
    df_all.fillna('None', inplace=True)
    
    brand_ind = df_all[df_all.BRAND.str.match(keyword)].index
    
    tags = df_all.loc[brand_ind, 'BRAND'] + ' ' + df_all.loc[brand_ind, 'BRAND_BELONGS_TO_CATEGORY'] + ' ' + df_all.loc[brand_ind, 'IS_CHILD_CATEGORY_TO']
    
    return ' '.join(tags)


# In[8]:


def tfidf_offers(df_offer_tags, tags):
    '''Does word embbedding for the offer tags and search tags using TFIDF. Calculates the cosine similarity and returns top 10.'''
    
    vectorizer = TfidfVectorizer(stop_words="english") 
    # max_df, min_df possible parameters for larger datasets
    # max_df ignores very frequent words
    # min_df ignores very sparse words
       
    X = vectorizer.fit_transform(df_offer_tags)
    y = vectorizer.transform([tags])
    
    cosine_similarities = linear_kernel(y, X).flatten()
    related_offers_indices = cosine_similarities.argsort()[:-11:-1] # Top 10 most similiar
    
    return related_offers_indices, cosine_similarities


# In[9]:


def find_offers(df_offer_tags, keyword, search_type, df_brand, df_cat, df_offers):
    '''Finds the search term tags finds the most similar offer using cosine similarity.'''
    
    match search_type:
        case 'Brand':
            tags = get_brand_tags(keyword, df_brand, df_cat)
        case 'Retail':
            tags = get_retail_tags(keyword, df_brand, df_cat, df_offers)
        case 'Category':
            tags = get_cat_tags(keyword, df_cat)
        
    offer_inds, cos_sim = tfidf_offers(df_offer_tags, tags)
            
    return offer_inds, cos_sim


# In[10]:


def summary(offer_inds, cos_sim, df_offers, keyword, search_type):
    '''Prints a summary of the top offers with their cosine similarity'''
    
    print('Top {} offers for {}:'.format(len(offer_inds), keyword))
    print('Search Type: {} '.format(search_type))
    print()
    for i in offer_inds:
        print('{} || cosine similarity of {:.3f}.'.format(df_offers.loc[i, 'OFFER'], cos_sim[i]))


# In[11]:


def main():
    '''Main function that reads the data and finds the most similar offers to a search term'''
    
    df_brand, df_cat, df_offers = read_data()
    keyword, search_type = user_input(df_brand, df_cat, df_offers)
    df_offer_tags = df_offer_tagged(df_brand, df_cat, df_offers)
    offer_inds, cos_sim = find_offers(df_offer_tags, keyword, search_type, df_brand, df_cat, df_offers)
    summary(offer_inds, cos_sim, df_offers, keyword, search_type)


# In[13]:


main()


# In[ ]:




