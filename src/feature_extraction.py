def create_ngram_vocab(words, n=4):
    all_ngrams=set()
    for word in words:
        marked_word = f"^{word.lower()}$"
        for i in range(len(marked_word)-n+1):
            all_ngrams.add(marked_word[i:i+n])
    return list(sorted(all_ngrams))

def word_to_ngram_features(word, vocab_dict, n=4):
    features = [0]*len(vocab_dict)
    marked_word = f"^{word.lower()}$"
    for i in range(len(marked_word)-n+1):
        ngram = marked_word[i:i+n]
        idx = vocab_dict.get(ngram)
        if idx is not None:
            features[idx] += 1
    return features