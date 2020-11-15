#!/usr/bin/env python3

import pandas as pd
pd.set_option('display.max_colwidth', 0)
df = pd.read_csv('data.csv')

import spacy
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")
tok_text=[] # for our tokenised corpus
#Tokenising using SpaCy:
for doc in tqdm(nlp.pipe(df.text.str.lower().values, disable=["tagger", "parser","ner"])):
    tok = [t.text for t in doc if t.is_alpha]
    tok_text.append(tok)


from rank_bm25 import BM25Okapi
bm25 = BM25Okapi(tok_text)


query = "flood defence"
tokenized_query = query.lower().split(" ")
import time

t0 = time.time()
results = bm25.get_top_n(tokenized_query, df.text.values, n=3)
t1 = time.time()
print(f'Searched 50,000 records in {round(t1-t0,3) } seconds \n')
for i in results:
  print(i)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#output --
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
​
#This programs give 3 below search results of json files which has "flood defence" text caption of 50,000 json files in data.csv (data-set)

#1. Forge Island "Flood Defence" and Public Realm Works Award of Flood defence and public realm works along the canal embankment at Forge Island, Market Street, Rotherham as part of the Rotherham Renaissance Flood Alleviation Scheme.
​
#2. "Flood defence" maintenance works for Lewisham and Southwark College **AWARD** Following RfQ NCG contracted with T Gunning for Flood defence maintenance works for Lewisham and Southwark College
​
#3. Freckleton St Byrom Street River Walls Freckleton St Byrom Street River Walls, Strengthening of existing river wall parapets to provide "flood defence" measures
