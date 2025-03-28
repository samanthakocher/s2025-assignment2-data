#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Any, List, Union
 
 # Imports for run_extract_text_from_html_bytes
import resiliparse.extract.html2text
import resiliparse.parse.encoding

# Imports for run_identify_language
from langdetect import detect, detect_langs

# Imports for run_mask_emails
import re

# Imports for run_classify_nsfw and run_classify_toxic_speech
import fasttext

# Load pre-trained models
nsfw_model_path = "/Users/samanthakocher/Desktop/ece491b/s2025-assignment2-data/nsfw_model.bin"
toxic_model_path = "/Users/samanthakocher/Desktop/ece491b/s2025-assignment2-data/toxic_speech_model.bin"

# Imports for run_gopher_quality_filter
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import word_tokenize

# Imports for run_exact_line_deduplication
import hashlib
from collections import Counter

# Imports for run_minhash_deduplication
import numpy as np
import unicodedata
import shutil


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    # Extract plain text from an HTML byte string, handling various encodings.
    
    # Try to detect the encoding
    try:
        # Attempt to detect encoding using resiliparse
        detected_encoding = resiliparse.parse.encoding.detect_encoding(html_bytes)

        # If no encoding detected, fallback to UTF-8
        encoding = detected_encoding or 'utf-8'

        # Decode the byte string using the detected or fallback encoding
        html_str = html_bytes.decode(encoding, errors='replace')
    except (UnicodeDecodeError, LookupError):

        # Fallback to UTF-8 with replacement if detection fails
        html_str = html_bytes.decode('utf-8', errors='replace')

    # Extract plain text using Resiliparse
    extracted_text = resiliparse.extract.html2text.extract_plain_text(html_str)

    return extracted_text


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
    # Classify whether the given text contains NSFW content.

    # Ensure the model is only loaded once
    if not hasattr(run_classify_nsfw, 'model'):
        run_classify_nsfw.model = fasttext.load_model(nsfw_model_path)

    # Preprocess the text (FastText requires lowercase)
    text = text.lower().strip()

    # Predict using the model
    predictions = run_classify_nsfw.model.predict(text, k=1)

    # Extract label and confidence
    label = predictions[0][0].replace('__label__', '')
    confidence = predictions[1][0]

    return ('nsfw' if label in ['toxic', 'nsfw', 'obscene'] else 'non-nsfw', confidence)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    # Classify whether the given text contains toxic speech.

    # Ensure the model is only loaded once
    if not hasattr(run_classify_toxic_speech, 'model'):
        run_classify_toxic_speech.model = fasttext.load_model(toxic_model_path)

    # Preprocess the text (FastText requires lowercase)
    text = text.lower().strip()

    # Predict using the model
    predictions = run_classify_toxic_speech.model.predict(text, k=1)

    # Extract label and confidence
    label = predictions[0][0].replace('__label__', '')
    confidence = predictions[1][0]

    return ('toxic' if label == 'toxic' else 'non-toxic', confidence)


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
    # Perform exact line deduplication across multiple input files.

    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # First pass: Count line frequencies
    line_counter = Counter()

    # Use hash of lines as keys to reduce memory footprint
    for input_file in input_files:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Strip whitespace and create hash
                line_hash = hashlib.md5(line.strip().encode('utf-8')).hexdigest()
                line_counter[line_hash] += 1

    # Second pass: Write unique lines for each file
    for input_file in input_files:
        # Determine output file path
        output_file = os.path.join(output_directory, os.path.basename(input_file))

        # Collecting unique lines for this file
        unique_lines = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Strip whitespace and create hash
                stripped_line = line.strip()
                line_hash = hashlib.md5(stripped_line.encode('utf-8')).hexdigest()

                # only keep line if it appears only once in the entire corpus
                if line_counter[line_hash] == 1:
                    unique_lines.append(line)

        # Write unique lines to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(unique_lines)

    return


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    # Perform fuzzy document deduplication using minhash and LSH.

    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Text normalization helper
    def normalize_text(text: str) -> str:
        text = unicodedata.normalize('NFD', text) # Decompose characters
        text = re.sub(r'[\u0300-\u036f]', '', text) # Remove accent marks
        text = text.lower() # Lowercase
        text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
        return re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
    
    # N-gram generation helper
    def generate_ngrams(text: str, n: int) -> List[str]:
        words = text.split()
        return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    
    # Read and normalize input fiels
    normalized_docs = []
    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        normalized_docs.append(normalize_text(text))

    # Minhash signature generation
    def minhash_signature(tokens: set, num_hashes: int) -> np.ndarray:
        np.random.seed(42)
        signature = np.full(num_hashes, np.inf)
        for token in tokens:
            hash_funcs = [
                hash(str(token) + str(seed))
                for seed in range(num_hashes)
            ]
            signature = np.minimum(signature, hash_funcs)
        return signature
    
    # Compute minhash signatures
    signatures = np.array([
        minhash_signature(set(generate_ngrams(doc, ngrams)), num_hashes)
        for doc in normalized_docs
    ])

    # Locality-Sensitive Hashing to find candidate duplicate pairs
    def lsh_candidate_pairs(signatures: np.ndarray, num_bands: int) -> Set[Tuple[int, int]]:
        rows_per_band = num_hashes // num_bands
        candidates = set()

        for band in range(num_bands):
            start = band * rows_per_band
            end = (band + 1) * rows_per_band
            band_signatures = signatures[:, start:end]

            band_hash_dict = {}
            for doc_idx, band_sig in enumerate(band_signatures):
                band_hash = hash(tuple(band_sig))

                if band_hash not in band_hash_dict:
                    band_hash_dict[band_hash] = []
                band_hash_dict[band_hash].append(doc_idx)

            # For each band hash with multiple documents, add all pairs as candidates
            for doc_indices in band_hash_dict.values():
                if len(doc_indices) > 1:
                    for i in range(len(doc_indices)):
                        for j in range(i + 1, len(doc_indices)):
                            candidates.add(tuple(sorted((doc_indices[i], doc_indices[j]))))

        return candidates

    # N-gram Jaccard similarity computation
    def ngram_jaccard_similarity(doc1: str, doc2: str, n: int) -> float:
        doc1_ngrams = set(generate_ngrams(doc1, n))
        doc2_ngrams = set(generate_ngrams(doc2, n))

        intersection = len(doc1_ngrams.intersection(doc2_ngrams))
        union = len(doc1_ngrams.union(doc2_ngrams))

        return intersection / union if union > 0 else 0.0
    
    # Track documents to keep
    keep_docs = set(range(len(input_files)))

    # Find candidate duplicate pairs
    candidate_pairs = lsh_candidate_pairs(signatures, num_bands)

    # Verify candidate paris with true Jaccard similarity
    while candidate_pairs:
        processed_pairs = set()
        for doc1_idx, doc2_idx in candidate_pairs:
            if doc1_idx in keep_docs and doc2_idx in keep_docs:
                jac_sim = ngram_jaccard_similarity(
                    normalized_docs[doc1_idx],
                    normalized_docs[doc2_idx],
                    ngrams
                )

                # If similarity exceeds threshold, remove one document
                if jac_sim >= jaccard_threshold:
                    input_files_names = [os.path.basename(f) for f in input_files]
                    to_remove = (
                        doc2_idx if input_files_names[doc1_idx] <= input_files_names[doc2_idx]
                        else doc1_idx
                    )

                    # Ensure the document is still in keep_docs
                    if to_remove in keep_docs:
                        keep_docs.remove(to_remove)

                processed_pairs.add((doc1_idx, doc2_idx))

        # Remove processed pairs from candidate pairs
        candidate_pairs -= processed_pairs

    # Copy files to output directory, keeping only selected documents
    for i, input_file in enumerate(input_files):
        if i in keep_docs:
            output_file = os.path.join(output_directory, os.path.basename(input_file))
            shutil.copy2(input_file, output_file)
