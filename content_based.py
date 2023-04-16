import re
import analyzer_cleaner as alc

stop_words = alc.stop_words

def prepare_gensim_input_for_text(text, dictionary):
  # Apply clean text for sample text 
  text_clean = alc.clean_text(text)
  # tokenizer(split) the sentences into words
  text_gem = text_clean.split()
  # remove numbers and stopwords from texts
  text_gem = [re.sub('[0-9]+', '', e) for e in text_gem]
  text_gem = [t for t in text_gem if not t in stop_words]
  # Obtain corpus based on dictionary 
  new_text_vector = [dictionary.doc2bow(text_gem, allow_update=True) for text in [text_gem]]
  return new_text_vector

def search_similar_product(df, similar_index, tfidf, dictionary, search_input):
  """This function will recommend similar product based on keywords"""
  # clean text
  search_key_clean = prepare_gensim_input_for_text(search_input, dictionary)
  # get similar product index in dataframe then sort relevent products
  result = similar_index[tfidf[search_key_clean]]
  result = list(enumerate(result[0, :]))
  result = sorted(result, key = lambda x: x[1], reverse=True)
  result_index = [i[0] for i in result]
  # recommend result
  recommend = df.iloc[result_index]
  return recommend
