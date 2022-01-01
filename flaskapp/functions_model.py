# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 15:28:02 2021

@author: celine
"""

import re
import gensim, spacy
import gensim.corpora as corpora
from nltk.corpus import stopwords

# Load NLTK stopwords
stop_words = stopwords.words('english')

# add stopwords
stop_words.extend(['like', 'thank', 'use', 'quot', 'try', 'know', 'find', 'way',\
                   'change', 'want', 'add', 'follow', 'look', 'problem', 'need',\
                   'solution', 'sorry', 'funny', 'exist', 'current', 'help', 'view', 'new',
                   'thread', 'void', 'true', 'false', 'private', 'state', 'null',\
                   'code', 'element', 'question', 'self', 'intent', 'build', \
                   'context', 'foo', 'click','check', 'div', 'abc', 'xyz','enum',\
                   'yyyy','understand','let','think','simple','thing', 'instead',\
                   'right', 'wrong', 'instance', 'good','instance', 'idea', 'able',\
                   'obtain', '+', '-','.+','├'])

    
def clean_text(text):    
    '''first cleaning of text'''
    #Put text in lowercase
    text = text.lower()
    
    #Remove tags, back to line, digits, urls, urls, double spaces, tabs    
    text = re.sub(r'<[^<>]+>|\n|\d+|\t|\s{2,}|=|&[gl][te]|\r|\f|http.+?(?="|<)',
                  ' ',
                  text)
    #Remove punctuation    
    text = re.sub(r'[\'\.\-!"#$%&\\*,:;<=>?@^`()|~={}\/\[\]\|]',
                  ' ',
                  text)
    
    #Remove isolated digits
    text = re.sub(r'\b[0-9]+\b',
                  ' ',
                  text)
    #remove words preceded by a dot
    text = re.sub(r'\.\w+',
                  ' ',
                  text)
    
    #Remove isolated letters except "r" and "c" for programming languages
    text = re.sub( r'\b[abdefghijklmnopqstuvwxyz]\b',
                  ' ',
                  text)
    
    # Remove two letter words
    text = re.sub( r'\b[a-z][a-z]\b',
                  ' ',
                  text)
    
    #Remove double spaces and tabs
    text = re.sub(r'\s{2,}|\t',
                  ' ',
                  text)
    
    
    text = text.strip()
    
    return text
    
def clean_tags(x) :
    x = x.lower()
    x =  re.sub('><', ',', x)
    x = re.sub(r'<','', x)
    x = re.sub(r'>', '', x)
    x = x.strip()
    x = x.split(',')
    return x


nlp = spacy.load("en_core_web_sm")

def remove_stop_words(text):
    '''remove stopwords in text data'''
    lis=[]
    for token in text.split():
        if (token not in stop_words) and (token not in gensim.parsing.preprocessing.STOPWORDS):
            lis.append(token)
    return(' '.join(lis))

def lemmatize(text):
    '''lemmatize text data'''
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not (token.is_stop or token.is_punct)]
    return ' '.join(tokens)

    