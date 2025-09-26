# P0 - Estadísticas del corpus

import matplotlib.pyplot as plt
from collections import Counter
import re

# TODO 1: Obtener los tokens del corpus del archivo "tiny_cc_news.txt", en el orden original
with open("tiny_cc_news.txt", "r", encoding="utf-8") as f:
    documentos = f.readlines()

tokens_por_doc = [re.findall(r"\w+", doc.lower()) for doc in documentos]
tokens = [tok for doc in tokens_por_doc for tok in doc]

# TODO 2: Leer los tokens correspondientes a stopwords desde el archivo "stopwords.txt"
with open("stopwords.txt", "r", encoding="utf-8") as f:
    stopwords = set(f.read().split())

# TODO 3: Obtener diccionarios de frecuencias para tokens y tokens que no son stopwords
freq_tokens = Counter(tokens)
tokens_no_stop = [t for t in tokens if t not in stopwords]
freq_tokens_no_stop = Counter(tokens_no_stop)

# TODO 4: Obtener estadísticas básicas del corpus
num_docs = len(documentos)
num_tokens = len(tokens)
num_stopwords = sum(1 for t in tokens if t in stopwords)
num_tokens_no_stop = len(tokens_no_stop)
prop_stopwords = num_stopwords / num_tokens if num_tokens > 0 else 0

vocab = set(tokens)
vocab_no_stop = set(tokens_no_stop)

doc_lengths = [len(doc) for doc in tokens_por_doc]
token_lengths = [len(t) for t in tokens]
token_lengths_no_stop = [len(t) for t in tokens_no_stop]

print("=== Estadísticas del corpus ===")
print("Número de documentos:", num_docs)
print("Número de tokens:", num_tokens)
print("Número de stopwords:", num_stopwords)
print("Número de tokens sin stopwords:", num_tokens_no_stop)
print("Proporción de stopwords:", round(prop_stopwords, 3))
print("Tamaño del vocabulario:", len(vocab))
print("Tamaño del vocabulario sin stopwords:", len(vocab_no_stop))
print("Longitud media de documento:", sum(doc_lengths)/len(doc_lengths))
print("Longitud mínima/máxima de documento:", min(doc_lengths), "/", max(doc_lengths))
print("Longitud media de token:", sum(token_lengths)/len(token_lengths))
print("Longitud mínima/máxima de token:", min(token_lengths), "/", max(token_lengths))
print("Longitud media de tokens sin stopwords:", sum(token_lengths_no_stop)/len(token_lengths_no_stop))
print("Longitud mínima/máxima de tokens sin stopwords:", min(token_lengths_no_stop), "/", max(token_lengths_no_stop))

# TODO 5: Obtener métricas de riqueza léxica
ttr = len(vocab) / num_tokens
hapax = sum(1 for t, c in freq_tokens.items() if c == 1)
print("\n=== Riqueza léxica ===")
print("Type-Token Ratio (TTR):", round(ttr, 3))
print("Número de hapax legomena:", hapax)

# TODO 6: Obtener los 10 tokens más frecuentes y los 10 tokens más frecuentes sin stopwords
print("\nTop 10 tokens más frecuentes:")
print(freq_tokens.most_common(10))

print("\nTop 10 tokens más frecuentes (sin stopwords):")
print(freq_tokens_no_stop.most_common(10))

# TODO 7: Obtener bigramas y trigramas de tokens y de tokens sin stopwords
def ngrams(lista, n=2):
    return [" ".join(lista[i:i+n]) for i in range(len(lista)-n+1)]

bigrams = Counter(ngrams(tokens, 2))
trigrams = Counter(ngrams(tokens, 3))

bigrams_no_stop = Counter(ngrams(tokens_no_stop, 2))
trigrams_no_stop = Counter(ngrams(tokens_no_stop, 3))

# TODO 8: Obtener los 10 bigramas y trigramas más frecuentes
print("\nTop 10 bigramas:")
print(bigrams.most_common(10))

print("\nTop 10 trigramas:")
print(trigrams.most_common(10))

print("\nTop 10 bigramas (sin stopwords):")
print(bigrams_no_stop.most_common(10))

print("\nTop 10 trigramas (sin stopwords):")
print(trigrams_no_stop.most_common(10))

# TODO 9: Mostrar gráfica de la Ley de Zipf
sorted_freqs = sorted(freq_tokens.values(), reverse=True)
ranks = range(1, len(sorted_freqs) + 1)

def plot_zipfs_law(sorted_freqs, ranks):
    plt.figure(figsize=(6, 4))
    plt.plot(ranks, sorted_freqs, marker=".", linestyle="solid")
    plt.title("Zipf's Law")
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()

plot_zipfs_law(sorted_freqs, ranks)
