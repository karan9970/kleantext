import re
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

class TextPreprocessor:
    def __init__(self, remove_stopwords=True, perform_spellcheck=False, use_stemming=False, use_lemmatization=True):
        self.remove_stopwords = remove_stopwords
        self.perform_spellcheck = perform_spellcheck
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        # Lowercase
        text = text.lower()
        # Remove HTML tags
        text = re.sub(r"<.*?>", "", text)
        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        # Remove numbers
        text = re.sub(r"\d+", "", text)
        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()
        # Remove stopwords
        if self.remove_stopwords:
            text = " ".join([word for word in text.split() if word not in self.stop_words])
        # Apply stemming
        if self.use_stemming:
            text = " ".join([self.stemmer.stem(word) for word in text.split()])
        # Apply lemmatization
        if self.use_lemmatization:
            text = " ".join([self.lemmatizer.lemmatize(word) for word in text.split()])
        # Spellcheck
        if self.perform_spellcheck:
            text = str(TextBlob(text).correct())
        return text
