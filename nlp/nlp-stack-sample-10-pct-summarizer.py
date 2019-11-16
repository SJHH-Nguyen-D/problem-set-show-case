import nltk
import pandas as pd 
import spacy
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
QUESTIONS_PATH = "../input/Questions.csv"
ANSWERS_PATH = "../input/Answers.csv"
FEATURES_INCLUDE = ['Score', 'Title', 'Body']
nlp = spacy.load('en_core_web_lg')
PUNCTUATIONS = '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~'
STOPWORDS = nltk.corpus.stopwords.words('english')


def load_dataset_as_dataframe(path, n_obs, features_include):
    dataframe = pd.read_csv(path, 
                            nrows=n_obs, 
                            usecols=features_include,
                            encoding='latin1')
    dataframe = dataframe.dropna()
    return dataframe

    
def clean_text(text_series):
    """ also known as normalizing the text - this gets rid of the tags and punctuations """
    # replace <pre> tags with nothing
    cleaned_text_series = re.sub('<pre>.*?</pre>', '', text_series, flags=re.DOTALL)
    # replace <code> tags with nothing
    cleaned_text_series = re.sub('<code>.*?</code>', '', cleaned_text_series, flags=re.DOTALL)
    # replace unwanted punctuation with nothing
    cleaned_text_series = re.sub('<[^>]+>', '', cleaned_text_series, flags=re.DOTALL)
    return cleaned_text_series


def scrub_and_tokenize(documents, logging=False):
    texts = []
    # create documents, and disabled named-entity-recognition and parser
    documents = nlp(documents, disable=['parser', 'ner'])
    # lemmatize, lower-case, and strip words in documents
    tokens = [tok.lemma_.lower().strip() for tok in documents if tok.lemma_ != '-PRON-']
    # remove unwatned punctuations
    tokens = [tok for tok in tokens if tok not in STOPWORDS and tok not in PUNCTUATIONS]
    # combine all tokens
    tokens = ' '.join(tokens)
    texts.append(tokens)
    text_series = pd.Series(texts)
    return text_series


def all_process_text(dataframe):
    clean_text()
    pass


def generate_summarize_text(text_without_removing_dot, cleaned_text):
    """ pass in bodies of posts before the clean, and after the clean respectively to
    get an output of the text before and after summarization.
    """
    sample_text = text_without_removing_dot
    doc = nlp(sample_text)
    sentence_list=[]
    for idx, sentence in enumerate(doc.sents): # we are using spacy for sentence tokenization
        sentence_list.append(re.sub(r'[^\w\s]','',str(sentence)))

    stopwords = nltk.corpus.stopwords.words('english')

    word_frequencies = {}  
    for word in nltk.word_tokenize(cleaned_text):  
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1


    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():  
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)


    sentence_scores = {}  
    for sent in sentence_list:  
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]


    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)
    print("Original Text::::::::::::\n")
    print(text_without_removing_dot)
    print('\n\nSummarized text::::::::\n')
    print(summary)


def plot_distribution_of_scores(feature_series):
    """ plot a distribution plot from a Pandas Series """
    fig, ax = plt.subplots(1, figsize=(30, 5))
    sns.distplot(feature_series, kde=True, hist=True)
    plt.xlim(0,feature_series.max())
    plt.xlabel('Number of upvotes', fontsize=17)
    plt.ylabel('frequency', fontsize=17)
    plt.tick_params(labelsize=15)
    plt.title('Number of upvotes distribution', fontsize=17)
    plt.show()


def plot_scatter_of_body(dataframe, feature_1='Body', feature_2='Score'):
    fig, ax = plt.subplots(1, figsize=(30, 6))
    dataframe['Title_len'] = dataframe[feature_1].str.split().str.len()
    dataframe = dataframe.groupby('Title_len')[feature_2].mean().reset_index()

    x = dataframe['Title_len']
    y = dataframe['Score']

    sns.scatterplot(x=x, y=y, data=dataframe, legend='brief', ax=ax)
    plt.title("Average Upvote by Question Body Length")
    plt.yaxis("Average Upvote")
    plt.xaxis("Question Body Length")
    plt.plot()


def main():
    df_questions = load_dataset_as_dataframe(QUESTIONS_PATH, 1000, FEATURES_INCLUDE)
    df_questions['Body_Cleaned_1'] = df_questions['Body'].apply(clean_text)
    df_questions['Body_Cleaned'] = df_questions['Body_Cleaned_1'].apply(lambda x: scrub_and_tokenize(x, False))
    print(df_questions['Body'][0])
    plot_distribution_of_scores(df_questions['Score'])
    generate_summarize_text(df_questions['Body_Cleaned_1'][10], df_questions['Body_Cleaned'][10])


if __name__ == "__main__":
    main()