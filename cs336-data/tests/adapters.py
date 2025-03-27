#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Any
 
 # Imports for run_extract_text_from_html_bytes
import resiliparse.extract.html2text
import resiliparse.parse.encoding
import chardet

# Imports for run_identify_language
from langdetect import detect, detect_langs

# Imports for run_mask_emails
import re

# Imports for run_classify_nsfw
# from transformers import pipeline
# import traceback
import fasttext

# Imports for run_gopher_quality_filter
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import word_tokenize


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    # Extract plain text from an HTML byte string, handling various encodings.
    
    # First, try UTF-8 decoding
    try:
        html_str = html_bytes.decode('utf-8')
    except (UnicodeDecodeError, LookupError):
        # If UTF-8 fails, use chardet to detect encoding
        detected_encoding = chardet.detect(html_bytes)['encoding']

        # If no encoding detected, fallback to UTF-8
        if not detected_encoding:
            detected_encoding = 'utf-8'

        # Decode using detected encoding
        html_str = html_bytes.decode(detected_encoding, errors='replace')

    # Use resiliparse to extract plain text
    return resiliparse.extract.html2text.extract_plain_text(html_str)


def run_identify_language(text: str) -> tuple[Any, float]:
    # Identify the main language in a given Unicode string.

    # Handle empty or very short strings
    if not text or len(text.strip()) < 3:
        return 'en', 0.0
    
    try:
        # Detect the primary language
        lang = detect(text)

        # Get detailed language probabilities
        lang_probs = detect_langs(text)

        # Mapping for specific language codes if needed
        lang_mapping = {
            'zh-cn': 'zh',
            'zh-tw': 'zh'
        }

        # Apply mapping if exists, otherwise use original
        mapped_lang = lang_mapping.get(lang, lang)

        # Get the confidence score (first probability in the list)
        # Ensure the score is between 0 and 1
        confidence = max(0.0, min(1.0, lang_probs[0].prob))

        return mapped_lang, confidence
    
    except Exception:
        # Fallback to English with low confidence if detection fails
        return 'en', 0.0


def run_mask_emails(text: str) -> tuple[str, int]:
    # Mask out email addresses in the given text.

    # Comprehensive regex patterns for email addresses
    # This pattern covers most common email address formats
    email_pattern = r'\b[A-Za-z0-9._]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    # Find all email addresses
    email_matches = re.findall(email_pattern, text)

    # Replace email addresses with the mask
    masked_text = re.sub(
        email_pattern,
        '|||EMAIL_ADDRESS|||',
        text
    )

    # Return masked text and count of masked emails
    return masked_text, len(email_matches)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    # Mask out phone numbers in the given text.

    # Comprehensive regex patterns for various US phone number formats
    phone_patterns = [
        # Pattern with optional country code and various separators
        r'(?:\+?1[-.\s]?)?(?:\(?\d{3}\[-.\s]?)?\d{3}[-.\s]?\d{4}\b',

        # Patterns with optional paraentheses, different separators
        r'\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',

        # Patterns with explicit county code
        r'\+1\s*(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b',

        # Patterns with various separators
        r'\d{3}[.-]\d{3}[.-]\d{4}\b'
    ]

    # Combine all patterns
    combined_pattern = '|'.join(phone_patterns)

    # Find all phone number matches
    phone_matches = re.findall(combined_pattern, text)

    # Replace phone numbers with the mask
    masked_text = re.sub(
        combined_pattern,
        '|||PHONE_NUMBER|||',
        text
    )

    # Ensure we count unique matches (flatten potential nested matches)
    unique_matches = list(set(filter(bool, phone_matches)))

    return masked_text, len(unique_matches)


def run_mask_ips(text: str) -> tuple[str, int]:
    # Mask out IPv4 addresses in the given text.

    # Regex pattern for IPv4 addresses
    # Matches numbers between 0-255 separated by dots
    ipv4_pattern = r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[0-9]|[01]?[0-9][0-9]?)\b'
    
    # Find all IP address matches
    ip_matches = re.findall(ipv4_pattern, text)

    # Find all IP addresses with the mask
    masked_text = re.sub(
        ipv4_pattern,
        '|||IP_ADDRESS|||',
        text
    )

    # Count unique matches
    unique_matches = list(set(ip_matches))

    return masked_text, len(unique_matches)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    # Classify text as toxic or non-toxic.

    # # Path to the NSFW model
    # nsfw_model = "/Users/samanthakocher/Desktop/ece491b/s2025-assignment2-data/nsfw_model.bin"

    # try:
    #     # Load the fasttext model
    #     model = fasttext.load_model(nsfw_model)

    #     # Get prediction results
    #     labels, probabilities = model.predict(text)

    #     # Check if results were returned
    #     if labels and probabilities:
    #         # Extract the label and probability
    #         label = labels[0].replace("_label_", "").lower()
    #         score = float(probabilities[0])

    #         # Map the result to 'nsfw' or 'non-toxic'
    #         if label == 'nsfw' and score > 0.7:
    #             return "nsfw", score
    #         else:
    #             return "non-toxic", score

    #     # # Use a toxicity detection pipeline
    #     # classifier = pipeline(
    #     #     'text-classification',
    #     #     model='dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_nsfw_final.bin',
    #     #     top_k=None
    #     # )

    #     # # Classify the text
    #     # results = classifier(text)

    #     # # Interpret the results
    #     # if results and len(results[0]) > 0:
    #     #     # Find the highest scoring label
    #     #     top_result = max(results[0], key=lambda x: x['score'])

    #     #     # Threshold for classifying as NSFW
    #     #     NSFW_THRESHOLD = 0.7

    #     #     # Map the classifier's output to 'nsfw' or 'non-toxic'
    #     #     # Typically, LABEL_1 indicates toxic/inappropriate content
    #     #     if top_result['label'] == 'NEGATIVE' and top_result['score'] > NSFW_THRESHOLD:
    #     #         return 'nsfw', top_result['score']
    #     #     else:
    #     #         return 'non-toxic', top_result['score']
            
    #     # Fallback if no results
    #     return 'non-toxic', 0.0

    # except Exception as e:
    #     # Error handling
    #     print(f"Error in NSFW classification: {e}")
    #     return 'non-toxic', 0.0
    raise NotImplementedError


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    raise NotImplementedError


def run_classify_quality(text: str) -> tuple[Any, float]:
    raise NotImplementedError


def run_gopher_quality_filter(text: str) -> bool:
    """
    Apply Gopher quality filters to assess text quality.

    Filters include:
    1. Document length between 50 and 100,00 words
    2. Mean word length between 3 and 10 characters
    3. Less than 30% of lines ending with ellipsis
    4. At least 80% of words contain an alphabetic character
    """

    # Tokenize the text into words
    try:
        words = word_tokenize(text)
    except Exception:
        # Fallback tokenization if NLTK fails
        words = text.split()

    # Filter 1: Check document length (50-100,000 words)
    if len(words) < 50 or len(words) > 100000:
        return False
    
    # Filter 2: Check mean word length
    # Filter out empty strings and non-alphabetic tokens
    word_lengths = [len(word) for word in words if word.strip() and any(c.isalpha() for c in word)]

    if not word_lengths:
        return False
    
    mean_word_length = sum(word_lengths) / len(word_lengths)
    if mean_word_length < 3 or mean_word_length > 10:
        return False
    
    # Filter 3: Check lines ending with ellipsis
    lines = text.split('\n')
    ellipsis_lines = sum(1 for line in lines if line.strip().endswith('...'))

    if ellipsis_lines / len(lines) > 0.3:
        return False
    
    # Filter 4: Check alphabetic character percentage
    words_with_alpha = sum(1 for word in words if any(c.isalpha() for c in word))
    alpha_percentage = words_with_alpha / len(words)

    if alpha_percentage < 0.8:
        return False
    
    # If all filters pass
    return True



def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    raise NotImplementedError


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    raise NotImplementedError
