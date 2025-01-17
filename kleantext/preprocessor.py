import re
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from langdetect import detect
from googletrans import Translator
import emoji

class TextPreprocessor:
    def __init__(self, remove_stopwords=True, perform_spellcheck=False, use_stemming=False, use_lemmatization=True,
                 custom_stopwords=None, case_sensitive=False, detect_language=False, target_language="en"):
        self.remove_stopwords = remove_stopwords
        self.perform_spellcheck = perform_spellcheck
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.stop_words = set(stopwords.words("english"))
        self.stop_words.update(custom_stopwords or [])
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.case_sensitive = case_sensitive
        self.detect_language = detect_language
        self.target_language = target_language
        self.translator = Translator()

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        if not self.case_sensitive:
            text = text.lower()
        text = re.sub(r"<.*?>", "", text)  # Remove HTML
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
        text = re.sub(r"\d+", "", text)  # Remove numbers
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        text = emoji.demojize(text)  # Convert emojis
        text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
        if self.remove_stopwords:
            text = " ".join([word for word in text.split() if word not in self.stop_words])
        if self.use_stemming:
            text = " ".join([self.stemmer.stem(word) for word in text.split()])
        if self.use_lemmatization:
            text = " ".join([self.lemmatizer.lemmatize(word) for word in text.split()])
        if self.perform_spellcheck:
            text = str(TextBlob(text).correct())
        if self.detect_language:
            lang = detect(text)
            if lang != self.target_language:
                text = self.translator.translate(text, src=lang, dest=self.target_language).text
        return text
