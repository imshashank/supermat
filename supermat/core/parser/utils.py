import os

import nltk
import spacy
import tiktoken
from dotenv import find_dotenv, load_dotenv
from langchain.text_splitter import TokenTextSplitter
from nltk import word_tokenize
from nltk.tag import pos_tag

load_dotenv(find_dotenv())

TOKENIZER_MODEL_NAME = os.getenv("TOKENIZER_MODEL_NAME", "gpt-3.5-turbo")
SPACY_MODEL = os.environ.get("SPACY_MODEL", "en_core_web_sm")

try:
    nlp = spacy.load(SPACY_MODEL)
except OSError:
    spacy.cli.download(SPACY_MODEL)  # pyright: ignore [reportAttributeAccessIssue]
    nlp = spacy.load(SPACY_MODEL)


def extract_spacy_keywords(text: str) -> set[str]:
    doc = nlp(text)
    # Extract words with more than 4 characters, numerics, nouns, verbs, adverbs, and adjectives excluding pronouns
    keywords = [
        token.text
        for token in doc
        if ((token.is_alpha and len(token.text) > 4) or (token.is_digit))
        and token.pos_ in ["NUM", "NOUN", "VERB", "ADV", "ADJ"]
    ]
    return set(keywords)


def nltk_word_tokenize(text: str) -> list[str]:
    try:
        tokens = word_tokenize(text)
    except LookupError:
        nltk.download("punkt_tab")
        tokens = word_tokenize(text)
    return tokens


def nltk_pos_tag(tokens: list[str]) -> list[tuple[str, str]]:
    try:
        tagged_tokens = pos_tag(tokens)
    except LookupError:
        nltk.download("averaged_perceptron_tagger_eng")
        tagged_tokens = pos_tag(tokens)

    return tagged_tokens


def extract_meaningful_words(text: str) -> set[str]:
    """For given text, extract set of relevant keywords using nltk."""
    # Tokenize the sentence
    tokens = nltk_word_tokenize(text)
    # Perform POS tagging
    tagged_tokens = nltk_pos_tag(tokens)
    # Extract words with more than 4 characters, numerics, nouns, verbs, adverbs, and adjectives excluding pronouns
    keywords = [
        word
        for word, tag in tagged_tokens
        if ((tag.startswith(("NN", "VB", "JJ", "RB")) and len(word) > 4) or (tag == "CD")) and word.lower() != "i"
    ]
    return set(keywords)


def get_keywords(text: str) -> list[str]:
    """For given text, retrieve relevant list of keywords using spacy and nltk."""
    return list(extract_spacy_keywords(text) | extract_meaningful_words(text))


def split_text_into_token_chunks(text, max_tokens: int = 8000, model_name: str = TOKENIZER_MODEL_NAME) -> list[str]:
    """
    Splits a text into chunks based on token count using LangChain's token splitter.

    Args:
        text (str): The text to be split.
        max_tokens (int): The maximum number of tokens in each chunk.
        model_name (str): The LLM model name to determine tokenization rules.

    Returns:
        list: A list of text chunks, each with up to max_tokens tokens.
    """
    encoding = tiktoken.encoding_for_model(model_name)
    splitter = TokenTextSplitter(encoding_name=encoding.name, chunk_size=max_tokens, chunk_overlap=0)
    chunks = splitter.split_text(text)
    return chunks
