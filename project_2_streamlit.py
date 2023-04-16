import streamlit as st 
import numpy as np 
import pandas as pd 
from gensim import corpora, models, similarities
import content_based as cbm
from PIL import Image
import requests
from scipy import stats

# ---------- Model -----------
stop_words = cbm.stop_words
# read data for content based
@st.cache_data
def load_data():
    df = pd.read_parquet("./data/Products_ThoiTrangNam_streamlit_final.parquet", engine='pyarrow')
    tfidf = models.TfidfModel.load("./models/gensim_tfidf_model.tfidf")
    corpus = corpora.MmCorpus('./models/gensim_corpus.mm')
    dictionary = corpora.Dictionary.load('./models/gensim_dictionary.dict')
    index = similarities.SparseMatrixSimilarity.load('./models/gensim_index.index')
    result = [df, tfidf, corpus, dictionary, index]
    return result
df, tfidf, corpus, dictionary, index = load_data()

@st.cache_data
def load_als_model():
    als_recommendation = pd.read_parquet('./data/Products_ThoiTrangNam_recommend_results_collaborative_streamlit_final.parquet/', engine='pyarrow')
    als_recommendation = als_recommendation[(~als_recommendation['image'].isnull()) & (~als_recommendation['link'].isnull())]
    return als_recommendation
als_recommendation = load_als_model()
# ----------- GUI ------------- 
# Add Sidebar
content_menu = ("Application Overview", "Start Recommendation")
content_select = st.sidebar.selectbox(
    "Application Content",
    content_menu
)
# Select recommendation method
recommendation_methods = ("Content-Based Filtering", "Collaborative Filtering")
with st.sidebar:
    if content_select == content_menu[1]:
        recommendation_methods_select = st.radio(
            "Choose a recommendation method:",
            recommendation_methods
        )
    else:
        recommendation_methods_select = None
# ---------
# TODO1: Application Overview
if content_select == content_menu[0]:
    st.write("# **Shopee Recommendation  System For Male Fashion**")
    st.image("./images/shopee.jpg")
    st.divider()

    # 1. Business Objectives 
    # Add header
    st.write("## **I. Business Objectives**")
    # Add content and image
    col1, col2 = st.columns(2)
    with col1:
        st.write("""Recommend relevant products to customers is a win-win scenario for both customers and e-commerce retailers.
        It helps users to get a better experience by helping customers find products they are looking for, personalize by buying experience.
        More importantly, this helps business to increase their revenue by offering their customers to browse and purchase more relevant products, hence increase sales and customer lifetime value.""")
    with col2:
        st.image("./images/money.jpg")
    st.divider()
    # 2. Methods
    st.write("## **II. Methods**")
    # Add content for Content-Based Filtering
    st.write("### **Content-Based Filtering:**")
    st.write("""Content-based filtering uses item features to recommend other items similar to 
    what the user likes, based on their previous actions or explicit feedback.
    The recommender system will recommend relevant products to the product that the user has searched.
    This helps business to increase sales by getting customers to buy more relevant products.""")
    st.image("./images/content_based.png")
    # Add content for Collaborative Filtering
    st.write("### **Collaborative Filtering:**")
    st.write("""Collaborative Filtering is a technique that can filter out items that a user might like on the basis of reactions by similar users. 
    It works by searching a large group of people and finding a smaller set of users with tastes similar to a particular user.
    Similarly to Content-Based Recommendation, Collaborative Filtering also help business to increase their sales by offering products that might fit a potential customer based on other customer's preferences.""")
    st.image("./images/colab.png")
# ---------
# TODO2: Content-Based Filtering

elif content_select == content_menu[1]:
    if recommendation_methods_select == recommendation_methods[0]:
        st.write("# Content-based filtering")
        num_product_per_row = 5
        search_method = st.radio(
            "Choose a search method:",
            ('Search', 'Product ID')
        )
        st.divider()
        if search_method == 'Search':
            with st.form("search_form"):
                user_search = st.text_input("Search Male Fashion:", placeholder="Type a product you need", key='user_search')
                num_product_show = st.select_slider('Select number of recommend products:', [5, 10, 15, 20, 30, 50, 100], value=5, key='num_product_show_1')
                submitted = st.form_submit_button("Recommend")
                st.divider()
                if submitted:
                    # 1. Reccommend by User's Search  
                    st.write("## Recommend By Search:")
                    # Recommend by user's input 
                    search_results = cbm.search_similar_product(df, index, tfidf, dictionary, user_search)
                    search_results = search_results[(~search_results['image'].isnull()) & (~search_results['link'].isnull())]
                    # Show on GUI
                    i = 0
                    for row in np.arange(num_product_show/num_product_per_row):
                        columns = st.columns(5)
                        for col in columns:
                            product_con = st.container()
                            with product_con:
                            # get each product's information
                                product_image = Image.open(requests.get(search_results['image'].iloc[i], stream=True).raw)
                                product_link = search_results['link'].iloc[i]
                                product_id = search_results['product_id'].iloc[i]
                                product_name = search_results['product_name'].iloc[i]
                                product_price = search_results['price'].iloc[i]
                                product_rating = search_results['rating'].iloc[i]
                                product_category = search_results['sub_category'].iloc[i]
                                # show product's info 
                                col.image(product_image)
                                col.write(f"[{str(product_name).capitalize()}]({product_link})")
                                col.write(f"Price: {product_price:,.0f} vnđ")
                                col.write(f"Rating: {product_rating}/5")
                                i += 1
        else:
        # 2. Reccommend by Product ID
            st.write("## Recommend By Product ID:")
            col1, col2 = st.columns(2)
            with col1:
                # Ask user to select category 
                categories = list(df['sub_category'].sort_values().unique())
                category_select = st.selectbox("Select a category:", categories, key='category_select')
                # Ask user to select a product in selected category 
                products_in_selected_category = df[(df['sub_category'] == category_select) & (~df['image'].isnull()) & (~df['link'].isnull())]['product_name'].to_list()
                product_select = st.selectbox("Select a product:", products_in_selected_category, key='product_select')
                with st.form("search_form"):
                    num_product_show = st.select_slider('Select number of recommend products:', [5, 10, 15, 20, 30, 50, 100], value=10, key='num_product_show_2')
                    submitted = st.form_submit_button("Recommend")
            # Show chosen product on GUI
            with col2:
                product_selected_df = df[df['product_name'] == product_select]
                product_con = st.container()
                with product_con:
                    # get each product's information
                    product_image = Image.open(requests.get(product_selected_df['image'].iloc[0], stream=True).raw)
                    product_link = product_selected_df['link'].iloc[0]
                    product_id = product_selected_df['product_id'].iloc[0]
                    product_name = product_selected_df['product_name'].iloc[0]
                    product_price = product_selected_df['price'].iloc[0]
                    product_rating = product_selected_df['rating'].iloc[0]
                    # show product's info 
                    product_con.image(product_image)
                    product_con.write(f"[{str(product_name).capitalize()}]({product_link})")
                    product_con.write(f"Price: {product_price:,.0f} vnđ")
                    product_con.write(f"Rating: {product_rating}/5")
            st.divider()
            if submitted:
                # Recommend by product id
                product_select_text = product_selected_df['product_name_description_wt']
                search_results = cbm.search_similar_product(df, index, tfidf, dictionary, product_select_text)
                search_results = search_results[(~search_results['image'].isnull()) & (~search_results['link'].isnull())]
                # Show on GUI   
                i = 0
                for row in np.arange(num_product_show/num_product_per_row):
                    columns = st.columns(5)
                    for col in columns:
                        # show product info 
                        product_con = st.container()
                        with product_con:
                        # get each product's information
                            product_image = Image.open(requests.get(search_results['image'].iloc[i], stream=True).raw)
                            product_link = search_results['link'].iloc[i]
                            product_id = search_results['product_id'].iloc[i]
                            product_name = search_results['product_name'].iloc[i]
                            product_price = search_results['price'].iloc[i]
                            product_rating = search_results['rating'].iloc[i]
                            product_category = search_results['sub_category'].iloc[i]
                            # show product's info 
                            col.image(product_image)
                            col.write(f"[{str(product_name).capitalize()}]({product_link})")
                            col.write(f"Price: {product_price:,.0f} vnđ")
                            col.write(f"Rating: {product_rating}/5")
                            i += 1
# --------- 
# TODO3: Collaborative Filtering
    elif recommendation_methods_select == recommendation_methods[1]:
        st.write("# Collaborative Filtering")
        # Reccomend by User ID
        num_product_per_row = 5
        with st.form("user_id_form"):
            user_id_select = st.number_input("Select User ID:", min_value=als_recommendation['user_id'].min(), max_value=als_recommendation['user_id'].max(), value=als_recommendation['user_id'].min(), step=1, key='user_id_select')
            num_product_show = st.select_slider('Select number of recommend products:', [5, 10, 15, 20, 30, 50, 100], value=5, key='num_product_show_3')
            submitted = st.form_submit_button("Recommend")
            st.divider()
            if submitted:
                if user_id_select in als_recommendation['user_id'].values:
                    # Show on GUI
                    als_recommendation_for_user = als_recommendation[als_recommendation['user_id'] == user_id_select]  
                    i = 0
                    for row in np.arange(num_product_show/num_product_per_row):
                        columns = st.columns(5)
                        for col in columns:
                            # show product info 
                            product_con = st.container()
                            with product_con:
                            # get each product's information
                                product_image = Image.open(requests.get(als_recommendation_for_user['image'].iloc[i], stream=True).raw)
                                product_link = als_recommendation_for_user['link'].iloc[i]
                                product_id = als_recommendation_for_user['product_id'].iloc[i]
                                product_name = als_recommendation_for_user['product_name'].iloc[i]
                                product_price = als_recommendation_for_user['price'].iloc[i]
                                product_rating = als_recommendation_for_user['rating'].iloc[i]
                                product_category = als_recommendation_for_user['sub_category'].iloc[i]
                                # show product's info 
                                col.image(product_image)
                                col.write(f"[{str(product_name).capitalize()}]({product_link})")
                                col.write(f"Price: {product_price:,.0f} vnđ")
                                col.write(f"Rating: {product_rating}/5")
                                i += 1
                else:
                    st.write(f"User ID: {user_id_select} is not in the system! Please try again.")


    else:
        st.write("# **Something went wrong! Please contact developers for more info.")
# --------- Exception ---------
else:
    st.write("# **Something went wrong! Please contact developers for more info.")