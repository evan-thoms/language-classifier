import wikipedia
import re
from datasets import load_dataset
import os
import random
import time
import json

from feature_extraction import create_ngram_vocab
BASE_DIR = os.path.dirname(os.path.abspath(__file__))        
PROJECT_DIR = os.path.dirname(BASE_DIR)                      
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "best_model.pth")
VOCAB_PATH = os.path.join(PROJECT_DIR, "models", "vocab.json")


def clean_sentences(sentence):
    sentence = re.sub(r'\[.*?\]', '', sentence)
    sentence = re.sub(r'\[.*?\]', '', sentence)  
    sentence = re.sub(r'==.*?==', '', sentence)  
    sentence = re.sub(r'^\d+\W*$', '', sentence)
    return sentence.strip()

def get_wikipedia_sentences(language, topics, max_sentences=200):
    print("Retrieving for language ", language)
    wikipedia.set_lang(language)
    all_sentences = []
    for topic in topics:
        for attempt in range(3):
            try:
                content = wikipedia.page(topic, auto_suggest=False).content
                raw_sentences = sent_split(content)
                cleaned = [clean_sentences(i) for i in raw_sentences]
                all_sentences.extend(cleaned[:max_sentences])
                break 
            except Exception as e:
                print(f"Attempt {attempt+1} failed for topic '{topic}': {e}")
                time.sleep(2)
        else:
            print(f"Skipping topic '{topic}' after {3} failed attempts.")
    return all_sentences

def sent_split(text):
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if len(s) > 3]

def retrieve_parallel_talks(lang):
    if lang == "en":
        spec = "en-pt"
        talks = load_dataset("sentence-transformers/parallel-sentences-talks", spec, split="train[:1000]")
        target_sentences = talks['english']
    else:
        spec = f"en-{lang}"
        talks = load_dataset("sentence-transformers/parallel-sentences-talks", spec, split="train[:1000]")
        target_sentences = talks['non_english']
    return target_sentences
    

def load_parallel_sentences(path, samp_size=300):
    with open(path, "r", encoding='utf-8') as f:
        all_data = json.load(f)
    
    sample = {}

    for lang, sentences in all_data.items():
        if len(sentences) >= samp_size:
            sample[lang] = random.sample(sentences, samp_size)
        else:
            sample[lang] = sentences
    return sample

def split_sentences_each_lang(sents, label):
    n = len(sents)
    indices = list(range(n))
    random.shuffle(indices)
    train_idx = int(.7*n)
    val_idx = int(.85*n)

    train_indices = indices[:train_idx]
    val_indices = indices[train_idx:val_idx]
    test_indices = indices[val_idx:]
    return (
        [sents[i] for i in train_indices],
        [label] * len(train_indices),
        [sents[i] for i in val_indices],
        [label] * len(val_indices),
        [sents[i] for i in test_indices],
        [label] * len(test_indices))


def load_data():
    print("Loading Data...")
    languages = ["en", "es", "fr", "pt", "it", "ro"]
    lang_labels = {lang:i for i,lang in enumerate(languages)}
    with open("../data/unknown_sentences.json", "r", encoding="utf-8") as f:
        unknown_sentences = json.load(f)

    wiki_topics = {
        "en": ["History", "Computer Science", "Earth"],
        "es": ["Historia", "Ciencia de la computación", "Tierra"],
        "fr": ["Histoire", "Informatique", "Terre"],
        "pt": ["História", "Ciência da computação", "Terra"],
        "it": ["Storia", "Informatica", "Terra"],
        "ro": ["Istorie", "Informatică", "Pământ"]
    }

    parallel_path = "../data/parallel_sentences.json"
    if not os.path.exists(parallel_path):
        print("Retrieving parallel talks sentences")
        parallel_data = {}
        for lang in languages:
            parallel_data[lang] = retrieve_parallel_talks(lang)
        with open(parallel_path, "w", encoding="utf-8") as f:
            json.dump(parallel_data, f, ensure_ascii=False, indent=2)
    print("Loading parallel talks sentences from file")
    parallel_sentences = load_parallel_sentences(parallel_path)
    

    wiki_path = "../data/wiki_sentences.json"
    if os.path.exists(wiki_path):
        print("Loading wiki data from file")
        with open(wiki_path, "r", encoding="utf-8") as f:
            wiki_data = json.load(f)
    else:
        print("Retrieving and writing wiki sentences")
        wiki_data = {}
        for lang in languages:
            topic = wiki_topics[lang]
            wiki_data[lang] = get_wikipedia_sentences(lang, topic)
        with open(wiki_path, "w", encoding="utf-8") as f:
            json.dump(wiki_data, f, ensure_ascii=False, indent=2)
    
    print("Creating training splits")
    train_sents,train_labels = [], []
    val_sents,val_labels = [],[]
    test_sents,test_labels = [], []

    

    for lang in languages:
        combined = wiki_data[lang] + parallel_sentences[lang]
        label = lang_labels[lang]
        tr_s, tr_l, vl_s, vl_l, ts_s, ts_l = split_sentences_each_lang(combined, label)
        train_sents += tr_s; 
        train_labels += tr_l
        val_sents += vl_s; 
        val_labels += vl_l
        test_sents += ts_s; 
        test_labels += ts_l

    tr_s, tr_l, vl_s, vl_l, ts_s, ts_l = split_sentences_each_lang(unknown_sentences, 6)

    train_sents += tr_s; 
    train_labels += tr_l
    val_sents += vl_s; 
    val_labels += vl_l
    test_sents += ts_s; 
    test_labels += ts_l
    
    return train_sents, train_labels, val_sents, val_labels, test_sents, test_labels

def create_vocab_dict(sentences, path=VOCAB_PATH):
    if os.path.exists(path):
        print("Vocab Dict already existst")
        return
    vocab = create_ngram_vocab(sentences)
    vocab_dict = {ngram: i for i, ngram in enumerate(vocab)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
    return vocab_dict

def load_vocab_dict(path=VOCAB_PATH):
    if not os.path.exists(path):
        print("Vocab Dict doesn't exist yet")
        return
    with open(path, "r", encoding="utf-8") as f:
        vocab_dict = json.load(f)
    return vocab_dict

# def get_vocab_dict(sentences):
#     if not os.path.exists("../data/vocab.json"):
#         vocab = create_ngram_vocab(sentences)
#         vocab_dict = {ngram: i for i, ngram in enumerate(vocab)}
#         with open("../data/vocab.json", "w", encoding="utf-8") as f:
#             json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
#     else:
#         with open("../data/vocab.json", "r", encoding="utf-8") as f:
#             vocab = json.load(f)
#         vocab_dict = {ngram: i for i, ngram in enumerate(vocab)}
#     return vocab_dict

if __name__ == "__main__":
    load_data()