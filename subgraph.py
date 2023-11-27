import pickle
# import graphvite as gv
import dataset
import re
import spacy
import tqdm
from tqdm.contrib.concurrent import process_map
import multiprocessing as mp
import time
import json
import os
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

RELEVANT_KEYS = ['knowledge', 'question']

file = open("simple_wikidata5m.pkl", "rb")
model = pickle.load(file)
entity2id = model.graph.entity2id
relation2id = model.graph.relation2id
entity_embeddings = model.solver.entity_embeddings
relation_embeddings = model.solver.relation_embeddings


alias2entity = dataset.wikidata5m.load_alias("./entity.txt.gz")

nlp = spacy.load('en_core_web_sm') 

def clean_text(text):
    # Extract named entities
    nlp = spacy.load('en_core_web_sm') 
    named_entities = nlp(text).ents

    # Tokenize the text
    words = word_tokenize(text)

    # Remove punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words]

    # Part-of-speech tagging
    tagged_words = pos_tag(words)

    cleaned_text = ' '.join(words).lower()
    
    return cleaned_text, named_entities

def extract_continuous_word_sets(text, window_size):
    words = text.split()
    word_sets = []
    
    for i in range(len(words) - window_size + 1):
        word_set = ' '.join(words[i:i + window_size])
        word_sets.append(word_set)

    return word_sets

def query_to_kg_embeddings(query):
    wiki_embeddings = {}
    cleaned_text, named_entities = clean_text(query)
    for ne in named_entities: 
        try:
            ne = ne.orth_.lower()
            wiki_embeddings[ne] = entity_embeddings[entity2id[alias2entity[ne]]]
            cleaned_text = cleaned_text.replace(ne, "")
        except Exception as e:
            print("KeyError: ", e)
    num_of_words = len(word_tokenize(cleaned_text))
    for i in range(num_of_words, 0, -1):
        word_sets = extract_continuous_word_sets(cleaned_text, i)
        for word in word_sets:
            try:
                wiki_embeddings[word] = entity_embeddings[entity2id[alias2entity[word]]]
                cleaned_text = cleaned_text.replace(word, "")
            except Exception as e:
                print("KeyError: ", e)

    return wiki_embeddings

def query_to_node_id(query):
    node_ids = {}
    cleaned_text, named_entities = clean_text(query)
    for ne in named_entities: 
        try:
            ne = ne.orth_.lower()
            node_ids[ne] = alias2entity[ne]
            cleaned_text = cleaned_text.replace(ne, "")
        except Exception as e:
            pass
            #print("KeyError: ", e)
    num_of_words = len(word_tokenize(cleaned_text))
    for i in range(num_of_words, 0, -1):
        word_sets = extract_continuous_word_sets(cleaned_text, i)
        for word in word_sets:
            try:
                node_ids[word] = alias2entity[word]
                cleaned_text = cleaned_text.replace(word, "")
            except Exception as e:
                pass
                #print("KeyError: ", e)
    return node_ids

def process_data(data):
    result = {}
    for k in RELEVANT_KEYS:
        node_vals = query_to_node_id(data[k])
        result.update(node_vals)
        
    return result

def main():
    halueval_path = os.path.join(os.getcwd(), "submodules", "HaluEval", "data", "qa_data.json")
    with(open(halueval_path, "r")) as f:
        data_lst = f.readlines()

    data_lst = [json.loads(data) for data in data_lst]

    from tqdm.contrib.concurrent import thread_map

    start = time.time()
    result = thread_map(process_data, data_lst, max_workers=2, chunksize=25)
    end = time.time()
    print(end-start)

    pickle.dump(result, open("processed_node_ids.pkl", "wb"))

if __name__ == "__main__":
    main()