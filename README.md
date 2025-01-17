
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

