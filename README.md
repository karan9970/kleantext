
# kleantext
A Python package for preprocessing textual data for machine learning and natural language processing tasks. It includes functionality for:

- Converting text to lowercase (optional case-sensitive mode)
- Removing HTML tags, punctuation, numbers, and special characters
- Handling emojis (removal or conversion to textual descriptions)
- Handling negations
- Removing or retaining specific patterns (hashtags, mentions, etc.)
- Removing stopwords (with customizable stopword lists)
- Stemming and lemmatization
- Correcting spelling (optional)
- Expanding contractions and slangs
- Named Entity Recognition (NER) masking (e.g., replacing entities with placeholders)
- Detecting and translating text to a target language
- Profanity filtering
- Customizable text preprocessing pipeline

---

## Installation
### Option 1: Clone or Download
1. Clone the repository using:
   ```bash
   git clone https://github.com/your-username/kleantext.git
   ```
2. Navigate to the project directory:
   ```bash
   cd kleantext
   ```

### Option 2: Install via pip (if published)
```bash
pip install kleantext
```

---

## Usage
### Quick Start
```python
from kleantext.preprocessor import TextPreprocessor

# Initialize the preprocessor with custom settings
preprocessor = TextPreprocessor(
    remove_stopwords=True,
    perform_spellcheck=True,
    use_stemming=False,
    use_lemmatization=True,
    custom_stopwords={"example", "test"},
    case_sensitive=False,
    detect_language=True,
    target_language="en"
)

# Input text
text = "This is an example! Isn't it great? Visit https://example.com for more 😊."

# Preprocess the text
clean_text = preprocessor.clean_text(text)
print(clean_text)  # Output: "this is isnt it great visit for more"
```

---

## Features and Configuration
### 1. Case Sensitivity
Control whether the text should be converted to lowercase:
```python
preprocessor = TextPreprocessor(case_sensitive=True)
```

### 2. Removing HTML Tags
Automatically remove HTML tags like `<div>` or `<p>`.

### 3. Emoji Handling
Convert emojis to text or remove them entirely:
```python
import emoji
text = emoji.demojize("😊 Hello!")  # Output: ":blush: Hello!"
```

### 4. Stopword Removal
Remove common stopwords, with support for custom lists:
```python
custom_stopwords = {"is", "an", "the"}
preprocessor = TextPreprocessor(custom_stopwords=custom_stopwords)
```

### 5. Slang and Contraction Expansion
Expand contractions like "can't" to "cannot":
```python
text = "I can't go"
expanded_text = preprocessor.clean_text(text)
```

### 6. Named Entity Recognition (NER) Masking
Mask entities like names, organizations, or dates using `spacy`:
```python
text = "Barack Obama was the 44th President of the USA."
masked_text = preprocessor.clean_text(text)
```

### 7. Profanity Filtering
Censor offensive words:
```python
text = "This is a badword!"
filtered_text = preprocessor.clean_text(text)
```

### 8. Language Detection and Translation
Detect the text's language and translate it:
```python
preprocessor = TextPreprocessor(detect_language=True, target_language="en")
text = "Bonjour tout le monde"
translated_text = preprocessor.clean_text(text)  # Output: "Hello everyone"
```

### 9. Tokenization
Tokenize text for further NLP tasks:
```python
from nltk.tokenize import word_tokenize
tokens = word_tokenize("This is an example.")
print(tokens)  # Output: ['This', 'is', 'an', 'example', '.']
```

---

## Advanced Configuration
Create a custom pipeline by enabling or disabling specific cleaning steps:
```python
pipeline = ["lowercase", "remove_html", "remove_urls", "remove_stopwords"]
preprocessor.clean_text(text, pipeline=pipeline)
```

---

## Testing
Run unit tests using:
```bash
python -m unittest discover tests
```

---

## License
This project is licensed under the MIT License.

---

## Contributing
Feel free to fork the repository, create a feature branch, and submit a pull request. Contributions are welcome!

---

## Snippets
### Full Preprocessing Example
```python
from kleantext.preprocessor import TextPreprocessor

# Initialize with default settings
preprocessor = TextPreprocessor(remove_stopwords=True, perform_spellcheck=False)

text = "Hello!!! This is, an example. Isn't it? 😊"
clean_text = preprocessor.clean_text(text)
print(clean_text)
```

### Profanity Filtering
```python
preprocessor = TextPreprocessor()
text = "This is a badword!"
clean_text = preprocessor.clean_text(text)
print(clean_text)  # Output: "This is a [CENSORED]!"
```

## Usage Examples

### 1. Converting Text to Lowercase (Optional Case-Sensitive Mode)
```python
preprocessor = TextPreprocessor(case_sensitive=True)
text = "Hello WORLD!"
cleaned_text = preprocessor.clean_text(text)
print(cleaned_text)  # Output: "Hello WORLD!"

preprocessor = TextPreprocessor(case_sensitive=False)
cleaned_text = preprocessor.clean_text(text)
print(cleaned_text)  # Output: "hello world"
```

---

### 2. Removing HTML Tags, Punctuation, Numbers, and Special Characters
```python
text = "This is a <b>bold</b> statement! Price: $100."
cleaned_text = preprocessor.clean_text(text)
print(cleaned_text)  # Output: "this is a bold statement price"
```

---

### 3. Handling Emojis (Removal or Conversion to Textual Descriptions)
```python
text = "I love Python! 😍🔥"
cleaned_text = preprocessor.clean_text(text)
print(cleaned_text)  # Output: "i love python :heart_eyes: :fire:"
```

---

### 4. Handling Negations
```python
text = "I don't like this movie."
cleaned_text = preprocessor.clean_text(text)
print(cleaned_text)  # Output: "i do not like this movie"
```

---

### 5. Removing or Retaining Specific Patterns (Hashtags, Mentions, etc.)
```python
text = "Follow @user and check #MachineLearning!"
cleaned_text = preprocessor.clean_text(text)
print(cleaned_text)  # Output: "follow user and check machinelearning"
```

---

### 6. Removing Stopwords (With Customizable Stopword Lists)
```python
preprocessor = TextPreprocessor(custom_stopwords={"example", "test"})
text = "This is an example test showing stopword removal."
cleaned_text = preprocessor.clean_text(text)
print(cleaned_text)  # Output: "this is an showing stopword removal"
```

---

### 7. Stemming and Lemmatization
#### With Stemming:
```python
preprocessor = TextPreprocessor(use_stemming=True, use_lemmatization=False)
text = "running flies better than walking"
cleaned_text = preprocessor.clean_text(text)
print(cleaned_text)  # Output: "run fli better than walk"
```
#### With Lemmatization:
```python
preprocessor = TextPreprocessor(use_stemming=False, use_lemmatization=True)
cleaned_text = preprocessor.clean_text(text)
print(cleaned_text)  # Output: "running fly better than walking"
```

---

### 8. Correcting Spelling (Optional)
```python
preprocessor = TextPreprocessor(perform_spellcheck=True)
text = "Ths is a tst sentnce."
cleaned_text = preprocessor.clean_text(text)
print(cleaned_text)  # Output: "This is a test sentence"
```

---

### 9. Expanding Contractions and Slang Handling
```python
from contractions import fix

text = "I'm gonna go, but I can't wait!"
text = fix(text)  
cleaned_text = preprocessor.clean_text(text)
print(cleaned_text)  # Output: "i am going to go but i cannot wait"
```

---

### 10. Named Entity Recognition (NER) Masking
```python
text = "John Doe lives in New York."
cleaned_text = preprocessor.clean_text(text)
print(cleaned_text)  # Output: "[PERSON] lives in [LOCATION]"
```

---

### 11. Detecting and Translating Text to a Target Language
```python
preprocessor = TextPreprocessor(detect_language=True, target_language="en")
text = "Bonjour! Comment ça va?"
cleaned_text = preprocessor.clean_text(text)
print(cleaned_text)  # Output: "Hello! How are you?"
```

---

### 12. Profanity Filtering
```python
text = "This is a f***ing great product!"
cleaned_text = preprocessor.clean_text(text)
print(cleaned_text)  # Output: "this is a **** great product"
```

---

### 13. Customizable Text Preprocessing Pipeline
```python
preprocessor = TextPreprocessor(
    remove_stopwords=True,
    perform_spellcheck=True,
    use_stemming=False,
    use_lemmatization=True,
    case_sensitive=False,
    detect_language=True,
    target_language="en"
)

text = "Ths is an amazng movi!! 😍🔥 <b>100%</b> recommended!"
cleaned_text = preprocessor.clean_text(text)
print(cleaned_text)  # Output: "this is an amazing movie heart_eyes fire recommended"
```

---

## Conclusion
KleanText is a robust and flexible text preprocessing library designed to clean and normalize text efficiently. You can customize the pipeline to fit your specific NLP needs. 🚀



