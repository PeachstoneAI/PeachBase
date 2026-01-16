"""Simple tokenizer for BM25 text processing.

Provides basic tokenization without heavy dependencies (no nltk, spacy, etc).
Suitable for AWS Lambda environments.
"""

import re
import unicodedata

# Common English stopwords (minimal set)
DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "will",
    "with",
}


class Tokenizer:
    """Simple tokenizer for text processing.

    Performs:
    - Lowercasing
    - Unicode normalization
    - Punctuation removal
    - Whitespace tokenization
    - Optional stopword filtering

    Args:
        lowercase: Whether to convert to lowercase (default: True)
        remove_punctuation: Whether to remove punctuation (default: True)
        stopwords: Set of stopwords to remove (default: DEFAULT_STOPWORDS)
        min_token_length: Minimum token length to keep (default: 2)
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        stopwords: set[str] | None = DEFAULT_STOPWORDS,
        min_token_length: int = 2,
    ) -> None:
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.stopwords = stopwords if stopwords is not None else set()
        self.min_token_length = min_token_length

        # Compile regex for punctuation removal
        if self.remove_punctuation:
            self._punct_pattern = re.compile(r"[^\w\s]", re.UNICODE)

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into list of tokens.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        if not text:
            return []

        # Unicode normalization (NFC form)
        text = unicodedata.normalize("NFC", text)

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Remove punctuation
        if self.remove_punctuation:
            text = self._punct_pattern.sub(" ", text)

        # Split on whitespace
        tokens = text.split()

        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            # Skip short tokens
            if len(token) < self.min_token_length:
                continue

            # Skip stopwords
            if token in self.stopwords:
                continue

            filtered_tokens.append(token)

        return filtered_tokens

    def tokenize_batch(self, texts: list[str]) -> list[list[str]]:
        """Tokenize multiple texts.

        Args:
            texts: List of texts

        Returns:
            List of token lists
        """
        return [self.tokenize(text) for text in texts]


# Default tokenizer instance
_default_tokenizer = Tokenizer()


def tokenize(
    text: str, lowercase: bool = True, remove_punctuation: bool = True
) -> list[str]:
    """Tokenize text using default tokenizer.

    Convenience function for one-off tokenization.

    Args:
        text: Input text
        lowercase: Whether to lowercase
        remove_punctuation: Whether to remove punctuation

    Returns:
        List of tokens

    Examples:
        >>> tokenize("Hello, World!")
        ['hello', 'world']
        >>> tokenize("Machine learning is fascinating")
        ['machine', 'learning', 'fascinating']
    """
    if lowercase and remove_punctuation:
        # Use cached default tokenizer
        return _default_tokenizer.tokenize(text)
    else:
        # Create custom tokenizer
        tokenizer = Tokenizer(
            lowercase=lowercase, remove_punctuation=remove_punctuation
        )
        return tokenizer.tokenize(text)
