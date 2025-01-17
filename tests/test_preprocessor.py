import unittest
from text_preprocessor.preprocessor import TextPreprocessor

class TestTextPreprocessor(unittest.TestCase):
    def setUp(self):
        """Initialize the TextPreprocessor instance before each test."""
        self.preprocessor = TextPreprocessor(remove_stopwords=False)
    
    def test_clean_text_basic(self):
        """Test the cleaning of a basic text input."""
        text = "This is an example! Visit https://example.com"
        cleaned = self.preprocessor.clean_text(text)
        self.assertEqual(cleaned, "this is an example visit", 
                         "Basic text cleaning failed.")
    
    def test_clean_text_with_special_characters(self):
        """Test cleaning with special characters and punctuation."""
        text = "Hello!!! This... is, amazing? Right: @#test"
        cleaned = self.preprocessor.clean_text(text)
        self.assertEqual(cleaned, "hello this is amazing right test", 
                         "Special characters cleaning failed.")
    
    def test_clean_text_with_numbers(self):
        """Test cleaning with numerical data in text."""
        text = "The price is 100 dollars for 2 items."
        cleaned = self.preprocessor.clean_text(text)
        self.assertEqual(cleaned, "the price is dollars for items", 
                         "Number removal failed.")
    
    def test_clean_text_with_stopwords_removed(self):
        """Test cleaning when stopword removal is enabled."""
        self.preprocessor.remove_stopwords = True
        text = "This is an example sentence with stopwords."
        cleaned = self.preprocessor.clean_text(text)
        self.assertEqual(cleaned, "example sentence stopwords", 
                         "Stopword removal failed.")
    
    def test_clean_text_empty_input(self):
        """Test cleaning with an empty input."""
        text = ""
        cleaned = self.preprocessor.clean_text(text)
        self.assertEqual(cleaned, "", 
                         "Cleaning failed for empty input.")
    
    def test_clean_text_with_only_urls(self):
        """Test cleaning when text contains only URLs."""
        text = "https://example.com https://test.com"
        cleaned = self.preprocessor.clean_text(text)
        self.assertEqual(cleaned, "", 
                         "URL removal failed.")

if __name__ == "__main__":
    unittest.main()
