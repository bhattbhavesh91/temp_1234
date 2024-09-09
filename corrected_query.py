import json

# Load correction data from a JSON file
with open('home_loan_corrections.json', 'r') as f:
    corrections = json.load(f)

def correct_spelling(query):
    words = query.split()  # Split the sentence into individual words
    corrected_words = []

    # Loop through each word and check for corrections
    for i in range(len(words)):
        # Check single word corrections
        corrected_word = corrections.get(words[i], words[i])
        
        # Check multi-word corrections by combining words (for bigrams/trigrams)
        for j in range(2, 4):  # Handling bigrams and trigrams
            if i + j <= len(words):
                multi_word_phrase = ' '.join(words[i:i+j])
                corrected_multi_word = corrections.get(multi_word_phrase, None)
                if corrected_multi_word:
                    corrected_word = corrected_multi_word
                    i += j - 1  # Skip the next words that are part of the multi-word correction
                    break
        
        corrected_words.append(corrected_word)

    # Reassemble the corrected words into a sentence
    return ' '.join(corrected_words)

# Example usage
search_query = "What is the tenor for home lona with lowest intrest"
corrected_query = correct_spelling(search_query)
print(f"Corrected Query: {corrected_query}")
