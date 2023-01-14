import os
import numpy as np
import pandas as pd
import pickle
import streamlit as st
from google_play_scraper import Sort, reviews
import pandas as pd
import numpy as np
import re
import string
import nltk
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.models
from tensorflow.keras.models import model_from_json
import streamlit
import re
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
#import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, accuracy_score, classification_report
from zipfile import ZipFile
from tensorflow.keras.models import model_from_json
import pickle
import os

def main():

    # make sidebar
    with st.sidebar:
        st.header('Sentiment Analisis Aplikasi Google Play Store')
        st.image(Image.open('sentiment-analysis.webp'))
        st.caption('© sistem cerdas 2022-2023')

    # make tabs for Menu
    tab1,tab2,tab3,tab4,tab5 = st.tabs(["scraping","Pre-Processing & Labeling","Visualisi Data","Model Evaluasi", "lstm"])

    # Scraping Data
    with tab1:
        with st.form(key='my-form'):
            url = st.text_input('Enter Link Apps')
            counts = st.number_input('amount of data', min_value=50 ,step=1)
            submit = st.form_submit_button('Submit')

        if "submits" not in st.session_state:
            st.session_state.submits = False
        
        def callback():
            st.session_state.submits = False

        if submit or st.session_state.submits:
            st.session_state.submits = True
            try:
                result, continueation_token = reviews(
                    url,
                    lang='id',
                    country='id',
                    sort=Sort.NEWEST,
                    count=counts,
                    filter_score_with=None,
                )

                result, _ = reviews(
                    url,
                    continuation_token=continueation_token
                )

                df = pd.DataFrame(np.array(result),columns=['review'])
                df = df.join(pd.DataFrame(df.pop('review').tolist()))

                df = df[['userName','score','at','content']]
                df = df.copy().rename(columns={ 'score': 'star'})

                st.dataframe(df)
                st.download_button(label='Download CSV', data = df.to_csv(index=False, encoding='utf8'), file_name=url+'.csv', on_click=callback)
            
            except:
                st.write('Enter The Correct Link')
            callback()

    # Pre-Processing & Labeling
    with tab2:
        try:

            st.write("""
            ## LSTM

            Algoritma Long Short-Term Memory (LSTM) memiliki memori yang sangat besar, sehingga sangat cocok digunakan untuk data yang berbentuk sequence. 
            Modeling data dengan LSTM yaitu tahap memodelkan apakah data yang akan diuji termasuk kedalam sentimen positif, negatif, atau netral dan 
            mendapatkan akurasi.
            """)

            data_file = st.file_uploader("Upload CSV file",type=["csv"])            
            if data_file is not None :
                df = pd.read_csv(data_file)
                st.dataframe(df)

                proses = st.button('Start process')

                if "prosess" not in st.session_state:
                    st.session_state.prosess = False

                def callback():
                    st.session_state.prosess = False

                if proses or st.session_state.prosess:
                    st.session_state.prosess = True

                    # Cleaning Text
                    def cleansing(text):
                        #removing number
                        text = re.sub(r"\d+","", text)
                        # remove non ASCII (emoticon, chinese word, .etc)
                        text = text.encode('ascii', 'replace').decode('ascii')
                        # remove mention, link, hashtag
                        text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
                        #Alphabeth only, exclude number and special character
                        text = re.sub(r'[^a-zA-Z]', ' ', text)
                        text = re.sub(r'\b[a-zA-Z]\b', ' ', text)
                        # replace word repetition with a single occutance ('oooooooo' to 'o')
                        text = re.sub(r'(.)\1+', r'\1\1', text)
                        # replace punctations repetitions with a single occurance ('!!!!!!!' to '!')
                        text = re.sub(r'[\?\.\!]+(?=[\?.\!])', '', text)
                        #remove multiple whitespace into single whitespace
                        text = re.sub('\s+',' ',text)
                        #remove punctuation
                        text = text.translate(text.maketrans("","",string.punctuation))
                        # Remove double word
                        text = text.strip()
                        #text = ' '.join(dict.fromkeys(text.split()))
                        return text

                    # Case folding text
                    def casefolding(text):
                        text = text.lower()
                        return text

                    # Tokenize text
                    def tokenize(text):
                        text = word_tokenize(text)
                        return text

                    # Normalisasi text
                    normalizad_word = pd.read_excel("normal.xlsx")
                    normalizad_word_dict = {}
                    for index, row in normalizad_word.iterrows():
                        if row[0] not in normalizad_word_dict:
                            normalizad_word_dict[row[0]] = row[1]

                    def normalized_term(text):
                        return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in text]

                    # Filltering | stopwords removal
                    def stopword(text):
                        listStopwords = set(stopwords.words('indonesian'))
                        filtered = []
                        for txt in text:
                            if txt not in listStopwords:
                                filtered.append(txt)
                        text = filtered 
                        return text

                    # Stremming text 
                    def steamming(text):
                        factory = StemmerFactory()
                        stemmer = factory.create_stemmer()
                        text = [stemmer.stem(word) for word in text]
                        return text

                    # Remove punctuation
                    def remove_punct(text):
                        text = " ".join([char for char in text if char not in string.punctuation])
                        return text

                    # Deploy Function
                    st.write("===========================================================")
                    st.write("Start Pre-processing")

                    st.caption("| cleaning...")
                    df['cleansing'] = df['content'].apply(cleansing)

                    st.caption("| case folding...")
                    df['cleansing'] = df['cleansing'].apply(casefolding)

                    st.caption("| tokenizing...")
                    df['text_tokenize'] = df['cleansing'].apply(tokenize)

                    st.caption("| normalization...")
                    df['tweet_normalized'] = df['text_tokenize'].apply(normalized_term)

                    st.caption("| removal stopwords...")
                    df['text_stopword'] = df['tweet_normalized'].apply(stopword)

                    st.caption("| steamming...")
                    df['text_steamming'] = df['text_stopword'].apply(steamming)

                    # Remove Puct 
                    df['text_clean'] = df['text_steamming'].apply(lambda x: remove_punct(x))

                    # Remove NaN file
                    df['text_clean'].replace('', np.nan, inplace=True)
                    df.dropna(subset=['text_clean'],inplace=True)

                    # Reset index number
                    df = df.reset_index(drop=True)
                    st.write("Finish Pre-processing")
                    st.write("===========================================================")
                
                    # Determine sentiment polarity of doc using indonesia sentiment lexicon
                    st.write("Count Polarity and Labeling...")
                    st.caption("using indonesia sentiment lexicon")
                    lexicon = dict()
                    import csv
                    with open('modified3_full_lexicon.csv', 'r') as csvfile:
                        reader = csv.reader(csvfile, delimiter=',')
                        for row in reader:
                            lexicon[row[0]] = int(row[1])

                    # Function to determine sentiment polarity of tweets        
                    def sentiment_analysis_lexicon_indonesia(text):
                        #for word in text:
                        score = 0
                        for word in text:
                            if (word in lexicon):
                                score = score + lexicon[word]

                        polarity=''
                        if (score > 0):
                            polarity = 'positive'
                        elif (score < 0):
                            polarity = 'negative'
                        else:
                            polarity = 'neutral'
                        return score, polarity

                    dflabeled = df
                    st.dataframe(dflabeled)

                    results = df['text_steamming'].apply(sentiment_analysis_lexicon_indonesia)
                    results = list(zip(*results))
                    df['score'] = results[0]
                    df['sentiment'] = results[1]
                    st.text(df['sentiment'].value_counts())

                    st.dataframe(df)
                    st.download_button(label='Download CSV', data = df.to_csv(index=False, encoding='utf8'), file_name='data2.csv', on_click=callback)
        except:
            st.write('Select The Correct File')

    with tab3:
        try:
                dflabeled = df
                st.dataframe(dflabeled)

                positif = len(df[df['sentiment'] == "positive"])
                negatif = len(df[df['sentiment'] == "negative"])
                netral = len(df[df['sentiment'] == "neutral"])
                    
                docPositive = df[df['sentiment']=='positive'].reset_index(drop=True)
                docNegative = df[df['sentiment']=='negative'].reset_index(drop=True)
                docNeutral = df[df['sentiment']=='neutral'].reset_index(drop=True)
                
                st.write("========================================================================================")
                st.write('Document Positive Sentiment')
                st.caption(f"Positive = {positif}, {docPositive.shape[0]/df.shape[0]*100} % ")
                st.dataframe(docPositive)
               
                st.write("========================================================================================")
                st.write('Document Negative Sentiment')
                st.caption(f"Negative = {negatif}, {docNegative.shape[0]/df.shape[0]*100} % ")
                st.dataframe(docNegative)

                st.write("========================================================================================")            
                st.write('Document Neutral Sentiment')
                st.caption(f"Neutral = {netral}, {docNeutral.shape[0]/df.shape[0]*100} % ")
                st.dataframe(docNeutral)

                st.write("========================================================================================")
                try:
                    text = " ".join(df['text_clean'])
                    wordcloud = WordCloud(width = 600, height = 400, background_color = 'black', min_font_size = 10).generate(text)
                    fig, ax = plt.subplots(figsize = (8, 6))
                    ax.set_title('WordCloud of Comment Data', fontsize = 18)
                    ax.grid(False)
                    ax.imshow((wordcloud))
                    fig.tight_layout(pad=0)
                    ax.axis('off')
                    st.pyplot(fig)
                except:
                    st.write(' ')

                #WordCloud Positive
                try:
                    st.write('WordCloud Positive')
                    train_s0 = df[df["sentiment"] == 'positive']
                    text = " ".join((word for word in train_s0["text_clean"]))
                    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='black', width=700, height=400,colormap='Blues', mode='RGBA').generate(text)
                    fig, ax = plt.subplots(1,figsize=(13, 13))
                    ax.set_title('WordCloud Positive', fontsize = 18)
                    ax.imshow(wordcloud, interpolation = 'bilinear')
                    plt.axis('off')
                    st.pyplot(fig)
                except:
                    st.write('tidak ada sentiment positif pada data')

                #WordCloud Negative
                try:
                    st.write('WordCloud Negative')  
                    train_s0 = df[df["sentiment"] == 'negative']
                    text = " ".join((word for word in train_s0["text_clean"]))
                    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='black', width=700, height=400,colormap='Reds', mode='RGBA').generate(text)
                    fig, ax = plt.subplots(1,figsize=(13, 13))
                    ax.set_title('WordCloud Negative', fontsize = 18)
                    ax.imshow(wordcloud, interpolation = 'bilinear')
                    plt.axis('off')
                    st.pyplot(fig)
                except:
                    st.write('tidak ada sentiment negatif pada data')

                
                #pie chart
                try:    
                    st.write('Pie Chart')
                    def pie_chart(label, data, legend_title):
                        fig, ax = plt.subplots(figsize=(5,7), subplot_kw=dict(aspect='equal'))

                        labels = [x.split()[-1] for x in label]

                        def func(pct, allvals):
                            absolute = int(np.round(pct/100.*np.sum(allvals)))
                            return "{:.1f}% ({:d})".format(pct, absolute)

                        wedges, texts, autotexts = ax.pie(data, autopct = lambda pct: func(pct, data),
                            textprops = dict(color="w"))

                        ax.legend(wedges, labels, title = legend_title, 
                    loc = "center left", 
                    bbox_to_anchor=(1,0,0.25,1))
                        plt.setp(autotexts, size=6, weight="bold")
                        st.pyplot(fig)

                    label = ['Positif', 'Negatif','Neutral']
                    count_data =[positif, negatif, netral]

                    pie_chart(label, count_data, "status")
                except:
                    st.caption('')
                st.spinner(text="In progress...")


                #Word Frequency
                try:
                    st.write('Word Frequency')
                    top = 11
                    a = df['text_clean'].str.cat(sep=' ')
                    words = nltk.tokenize.word_tokenize(a)
                    Word_dist = nltk.FreqDist(words)
                    rslt = pd.DataFrame(Word_dist.most_common(top), columns=['Word', 'Frequency'])
                    
                    count = rslt['Frequency']

                    fig, x = plt.subplots(1,1,figsize=(11,8))
                    # create bar plot
                    plt.bar(rslt['Word'], count, color=['royalblue'])

                    plt.xlabel('\nKata', size=14)
                    plt.ylabel('\nFrekuensi Kata', size=14)
                    plt.title('Kata yang sering Keluar \n', size=16)
                    st.pyplot(fig)

                except:
                    st.write('error')


        except:
            st.write('Select The Correct File')

    
    
    with tab4:
            try:
                dflabeled = df
                st.dataframe(dflabeled)

                st.write("\n Counting SVM Accuracy...")
                    
                def score_sentiment(score):
                        if score == 'positive':
                            return 0
                        elif score == 'negative':
                            return -1
                        else:
                            return 1

                biner = df['sentiment'].apply(score_sentiment)    

                X_train, X_test, Y_train, Y_test = train_test_split(df['text_clean'], biner, test_size=0.2, stratify=biner, random_state=42)

                vectorizer = TfidfVectorizer()
                X_train = vectorizer.fit_transform(X_train)
                X_test = vectorizer.transform(X_test)

                clfsvm = svm.SVC(kernel="linear")
                clfsvm.fit(X_train,Y_train)
                predict = clfsvm.predict(X_test)

                st.write("SVM Accuracy score  -> ", accuracy_score(predict, Y_test)*100)
                st.write("SVM Recall Score    -> ", recall_score(predict, Y_test, average='macro')*100)
                st.write("SVM Precision score -> ", precision_score(predict, Y_test, average='macro')*100)
                st.write("SVM f1 score        -> ", f1_score(predict, Y_test, average='macro')*100)
                st.write("===========================================================")
                st.write('confusion matrix : \n', confusion_matrix(predict, Y_test))
                st.write("===========================================================")
                st.text('classification report : \n'+ classification_report(predict, Y_test, zero_division=0))
                st.write("===========================================================")

            except:
                    st.write('Select the correct')

            #ZipfileDotZip = "sample.zip"

            with open("sample.zip", "rb") as file:
                btn = st.download_button(
                    label="Download Zip File",
                    data=file,
                    file_name="sample.zip",
                    mime="application/zip"
                )

    with tab5: 
                    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

                    with open('tokenizer.pickle', 'rb') as tk:
                        tokenizer = pickle.load(tk)
                        json_file = open('lstm_model.json','r')
                        loaded_model_json = json_file.read()    
                        json_file.close()
                        lstm_model = model_from_json(loaded_model_json)
                        lstm_model.load_weights("lstm_model.h5")
                    def sentiment_prediction(review):
                        sentiment = []
                        input_review = [review]
                        input_review = [x.lower() for x in input_review]
                        input_review = [re.sub('[^a-zA-z0-9\s]','',x) for x in input_review]
                        input_feature = tokenizer.texts_to_sequences(input_review)
                        input_feature = pad_sequences(input_feature,46, padding="pre")
                        sentiment = lstm_model.predict(input_feature)[0]
                        if(np.argmax(sentiment) == 0):
                            pred = "Negative"
                        elif (np.argmax(sentiment) == 1):
                            pred = "Somewhat Negative"
                        elif (np.argmax(sentiment) == 2):
                            pred = "Neutral"
                        elif (np.argmax(sentiment) == 3):
                            pred = "Somewhat Positive"
                        else:
                            pred = "Positive"
                        return pred
                        
                    def run():
                        st.title("Sentiment Analysis by LSTM")
                        review = st.text_input("Enter the review")
                        #print(type(review))
                        pred = ""
                        if st.button("Predict Sentiment"):
                            pred = sentiment_prediction(review)
                        st.success("The sentiment predicted by Model : {}".format(pred))   
       
                    if __name__ == '__main__':
                        run()             

if __name__ == '__main__':
    main()
















