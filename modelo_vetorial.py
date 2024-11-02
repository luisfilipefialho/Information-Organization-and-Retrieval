import spacy
import sys
import math
import string

# Carregar modelo SpaCy para Português
nlp = spacy.load("pt_core_news_sm")
stopwords = nlp.Defaults.stop_words
pontuacao = list(string.punctuation) + ['...', "''", '\x97', '..', '....', "!...", ",."]

def preprocessar(texto):
    """Função para lematizar e remover stopwords e pontuação."""
    doc = nlp(texto)
    tokens = [
        token.lemma_ for token in doc 
        if token.lemma_ not in stopwords and token.lemma_ not in pontuacao and not token.is_space
    ]
    return tokens

def calcular_tfidf(base_arquivo):
    """Função para calcular TF-IDF e salvar em arquivos de saída `indice.txt` e `pesos.txt`."""
    with open(base_arquivo, 'r', encoding='utf-8') as arquivos_de_base:
        base = [linha.strip() for linha in arquivos_de_base.readlines()]

    numero_de_documentos = len(base)
    documentos_tfidf = {documento: {} for documento in base}
    documentos_com_token = {}
    all_tokens = set()

    # Indexação dos documentos e cálculo de TF
    for documento in base:
        with open(documento, 'r', encoding='utf-8') as caminho_arquivo:
            texto = caminho_arquivo.read()
            tokens = preprocessar(texto)
            all_tokens.update(tokens)

            for token in set(tokens):
                documentos_com_token[token] = documentos_com_token.get(token, 0) + 1
                tf = 1 + math.log10(tokens.count(token))
                documentos_tfidf[documento][token] = tf

    # Cálculo de IDF e TF-IDF
    for documento in base:
        for token in all_tokens:
            if token in documentos_tfidf[documento]:
                idf = math.log10(numero_de_documentos / documentos_com_token[token])
                documentos_tfidf[documento][token] *= idf

# Escrevendo o arquivo pesos.txt de forma padronizada
    with open("pesos.txt", "w", encoding='utf-8') as arquivo_de_pesos:
        for documento, termos in documentos_tfidf.items():
            linha_termos = ' '.join([f"{termo},{peso:.4f}" for termo, peso in termos.items() if peso > 0])
            if linha_termos:
                arquivo_de_pesos.write(f"{documento}: {linha_termos}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso incorreto! Execute com: > python tfidf.py base.txt")
        sys.exit(1)

    arquivo_base = sys.argv[1]
    calcular_tfidf(arquivo_base)
