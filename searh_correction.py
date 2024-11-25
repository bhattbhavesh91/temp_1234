# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1yxC-WTruB35GKq4ZRjvVSIxjmwvzYeEI
"""

import json
import re
from typing import Dict, List, Tuple

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.replacement = None

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, phrase: str, replacement: str):
        words = phrase.lower().split()
        for i in range(len(words)):
            node = self.root
            for word in words[i:]:
                for char in word:
                    if char not in node.children:
                        node.children[char] = TrieNode()
                    node = node.children[char]
                if word != words[-1]:
                    if ' ' not in node.children:
                        node.children[' '] = TrieNode()
                    node = node.children[' ']
            node.is_end = True
            node.replacement = ' '.join(replacement.split()[-(len(words)-i):])

    def search_and_replace(self, query: str) -> str:
        # Split the query into words and punctuation
        tokens = re.findall(r'\w+|[^\w\s]', query.lower())
        result = []
        i = 0
        while i < len(tokens):
            if tokens[i].isalnum():  # Only process alphanumeric tokens
                matched, replacement, length = self.find_longest_match(tokens[i:])
                if matched:
                    result.extend(replacement.split())
                    i += length
                else:
                    result.append(tokens[i])
                    i += 1
            else:  # Preserve punctuation
                result.append(tokens[i])
                i += 1
        return ' '.join(result)

    def find_longest_match(self, words: List[str]) -> Tuple[bool, str, int]:
        longest_match = (False, '', 0)
        node = self.root
        current_phrase = []
        for i, word in enumerate(words):
            if not word.isalnum():  # Skip punctuation
                break
            if i > 0 and ' ' in node.children:
                node = node.children[' ']
            for char in word:
                if char not in node.children:
                    return longest_match
                node = node.children[char]
            current_phrase.append(word)
            if node.is_end:
                longest_match = (True, node.replacement, len(current_phrase))
        return longest_match

def load_dictionary(file_path: str) -> Dict[str, str]:
    with open(file_path, 'r') as f:
        return json.load(f)

def create_trie_from_dict(dictionary: Dict[str, str]) -> Trie:
    trie = Trie()
    for incorrect, correct in dictionary.items():
        trie.insert(incorrect, correct)
    return trie

def correct_query(trie: Trie, query: str) -> str:
    return trie.search_and_replace(query)

# Example usage
if __name__ == "__main__":
    # Load the dictionary from a JSON file
    dictionary = load_dictionary('banking-terms-dictionary-claude.json')

    # Create a trie from the dictionary
    trie = create_trie_from_dict(dictionary)

    # Test the correction
    test_queries = [
        "What would be my car lona emi for a 5-year tenore?",
        "What's the current interst rate for home lona?",
        "What is the tenor for home lone?",
        "I need a home lona with a good interst rate.",
        "Can you explain the tenur for homeloan process?",
        "How long is the lona tenure for home?",
        "Can you check my home lona eligibilty?",
        "Where can I home lona apply online?",
        "Do you have a loan tenure calclator?",
        "How does car lona interest work with a baloon payment?",
        "What's the typical personal lona tenure?",
        "Can you explain the difference between fixed and floating interst rates?",
        "How does my credit scor affect my lona eligibilty?",
        "What happens if I want to make a pre paymet on my homeloan?",
        "Is fore closure allowed on a car lona?",
        "How is the principle amount calculated in a lona amortizaton schedule?"
    ]

    for query in test_queries:
        corrected = correct_query(trie, query)
        print(f"Original: {query}")
        print(f"Corrected: {corrected}")
        print()

