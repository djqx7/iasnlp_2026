import pandas as pd
import re
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Common English subjects
SUBJECTS = {
    "i", "you", "he", "she", "it",
    "we", "they", "this", "that",
    "these", "those"
}

def is_imperative(sentence):
    sentence = str(sentence).strip()

    if not sentence:
        return False

    # Rule 3: Exclude questions
    if sentence.endswith("?"):
        return False

    # Rule 1: Starts with "Please"
    if re.match(r'^\s*please\b', sentence, re.IGNORECASE):
        return True

    # Rule 4: Starts with "Don't" or "Do not"
    if re.match(r"^\s*(?:don't|do\s+not)\b", sentence, re.IGNORECASE):
        return True

    doc = nlp(sentence)

    if len(doc) == 0:
        return False

    first = doc[0]

    # Rule 2: First word is a verb and sentence length < 5 words
    if first.tag_ == "VB" and len(doc) < 5:
        return True

    # Rule 5: Starts with a verb and not a subject
    if first.tag_ == "VB" and first.lower_ not in SUBJECTS:
        return True

    return False

df = pd.read_csv("dataset.csv")

# Extract imperative sentences
imperative_df = df[df["transcript"].apply(is_imperative)]

# Save to a new CSV
imperative_df.to_csv(
    "imperative_sentences4.csv",
    index=False,
    encoding="utf-8"
)

print(f"Found {len(imperative_df)} imperative sentences.")
print("Saved to imperative_sentences.csv")

