#!/usr/bin/env python
# coding: utf-8
import numpy as np
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from textblob import TextBlob
import string
import math
from scipy import stats
import pandas
import itertools
from collections import namedtuple
import file_len

SeedsAux = namedtuple('Seeds',['neg','pos'])
class Seeds(SeedsAux):
    def neg_words(self): # return only the words portion of this tuple
        return list(set(list(zip(*self.neg))[0]))
    def pos_words(self):
        return list(set(list(zip(*self.pos))[0]))
def get_docs(folder):
    docs = {}
    for doc in os.listdir(folder):
        with open(folder + doc,mode='r') as file:
            docs[doc] = file.read()
    return docs 

def get_expert_docs(folder):
    docs = {}
    expert_docs = file_len.file_lens(folder)
    get_these_tups = filter(lambda tup : tup[1] > 4, expert_docs)
    get_these = list(list(zip(*get_these_tups))[0])
    for doc in get_these:
        with open(folder + doc,mode='r') as file:
            docs[doc] = file.read()
    return docs

def rm_header(doc): # remove the ticker and hashtags from the top of the document
    return doc.split('\n',6)[6]

def rm_all_headers(docs):
    ret = {}
    for k,v in docs.items():
        ret[k] = rm_header(v)
    return ret

def docs_list(docs_dict):
    docs = []
    names = []
    for k,v in docs_dict.items():
        docs.append(v) # concat docs into a single list
        names.append(k)
    return docs, names

def process_doc(doc):
    return rm_header(doc).replace('<br />',' ')
def matcher(val1,*val2s): # return true if val1 is equal to any of the following arguments
    for val2 in val2s:
        if val1 == val2:
            return True
    return False

def pos_pattern_filter(trigrams): # each trigram is tagged with pos
    filtered = []
    for t in trigrams:
        t0, t1, t2 = t
        t0_pos, t1_pos, t2_pos = t0[1], t1[1], t2[1]
        if t0_pos == 'JJ' and matcher(t1_pos,'NN','NNS'):
            filtered.append(t)
            continue
        if matcher(t0_pos,'RB','RBR','RBS') and t1_pos == 'JJ' and not matcher(t2_pos,'NN','NNS'):
            filtered.append(t)
            continue
        if t0_pos == 'JJ' and t1_pos == 'JJ' and not matcher(t2_pos,'NN','NNS'):
            filtered.append(t)
            continue
        if matcher(t0_pos,'NN','NNS') and t1_pos == 'JJ' and not matcher(t2_pos,'NN','NNS'):
            filtered.append(t)
            continue
        if matcher(t0_pos,'RB','RBR','RBS') and matcher(t1_pos,'VB','VBD','VBN','VBG'):
            filtered.append(t)
    return filtered

def gen_pos_trigrams(doc):
    toks = nltk.word_tokenize(doc)
    pos_trigrams = list(nltk.trigrams(nltk.pos_tag(toks)))
    #trigrams = list(nltk.trigrams(toks))
    return pos_trigrams

def in_pos(word):
    return word.lower() in ['positive','better','cultivated','leader','revenue-generating',
                    'engaging','loyalty','enhance','strong']

def in_neg(word):
    return word.lower() in ['negative','sub-prime','weak','leaderless','penalized','susceptible',
                   'adverse','dismal','lost','unfavorable','failed']

def in_pos2(word):
    return word.lower() in ['leader','revenue-generating','engaging','strong']

def in_neg2(word):
    return word.lower() in ['sub-prime','weak','leaderless','penalized','adverse']

def get_in_list(lis):
    def in_list(word):
        return word.lower() in lis
    return in_list

# returns a dictionary where the keys are trigrams and the values are the number of times the positive or negative word
# occurs within that trigram
def co_occurences(trigrams,func_dict):
    count_dict = {}
    for tup in trigrams:
        for check_word, weight in func_dict.items():
            if True in [check_word(w) for w in list(zip(*tup))[0]]:
                if tup not in count_dict:
                    count_dict[tup] = weight
                else:
                    count_dict[tup] += weight
    return count_dict
class WeightGen:
    def __init__(self,check_word,weight):
        self.lifetimes = {}
        self.check_word = check_word
        self.weight = weight
    def cur_weight(self,tup):
        if True in [self.check_word(w) for w in list(zip(*tup))[0]]:
            self.lifetimes[tup] = 10
        for k in self.lifetimes.keys():
            self.lifetimes[k] -= 1
        temp = {k : v for k, v in self.lifetimes.items() if v > 0}
        self.lifetimes = temp
        return len(self.lifetimes) * self.weight
def within_10(trigrams,func_dict):
    count_dict = {}
    for check_word, weight in func_dict.items():
        w = WeightGen(check_word,weight)
        for tup in trigrams:
            if tup not in count_dict:
                count_dict[tup] = w.cur_weight(tup)
    return count_dict
def pos_trigrams_to_trigrams(pos_trigrams):
    return [(tup[0][0],tup[1][0],tup[2][0]) for tup in pos_trigrams]
def get_doc_counts(doc):
    pos_count = 0
    neg_count = 0
    for word in nltk.word_tokenize(doc):
        if in_pos(word):
            pos_count += 1
        if in_neg(word):
            neg_count += 1
    return neg_count, pos_count

def get_counts(doc_list):
    pos_count = 0
    neg_count = 0
    for doc in doc_list:
        temp_neg_count, temp_pos_count = get_doc_counts(doc)
        neg_count += temp_neg_count
        pos_count += temp_pos_count
    return neg_count, pos_count

def get_count_near(count_dict):
    def count_near(phrase):
        try:
            ret = count_dict[phrase]
        except:
            ret = 0
        return ret + .001
    return count_near

def get_PMI(neg_count,pos_count,count_near_neg,count_near_pos): # int, int, func, func
    def PMI(phrase):
        #print(phrase,count_near_pos(phrase),neg_count,count_near_neg(phrase),pos_count)
        return math.log(count_near_pos(phrase) * neg_count / count_near_neg(phrase) * pos_count,2)
    return PMI

def SO_of_doc(PMI,filtered_trigrams): # semantic orientation of doc
    SO = 0
    for phrase in filtered_trigrams:
        SO += PMI(phrase)
    return SO

def lists_to_list(lists):
    return list(itertools.chain.from_iterable(lists))

def adj_filter(tup): # arg is a trigram. returns list of words
    def is_adj(word_pos_pair):
        return word_pos_pair[1] in ['JJ','RBR','RBS']
    return list(filter(is_adj,tup))

def filter_list(pos_trigrams): 
    # takes a list of trigrams with associated pos
    # returns a list of WORDS
    filtered_list = []
    for trigram in pos_trigrams:
        filtered_list += adj_filter(trigram)
    return filtered_list

def min_max_tups(PMI,pos_trigrams): # returns the trigrams that are the most positive or the most negative
    pmi_tups = [(PMI(trigram),trigram) for trigram in pos_trigrams]
    pmi_list = list(zip(*pmi_tups))[0]
    max_pmi = max(pmi_list)
    min_pmi = min(pmi_list)
    neg_tups = list(list(zip(*filter(lambda pmi : pmi[0] <= min_pmi + 1,pmi_tups)))[1])
    pos_tups = list(list(zip(*filter(lambda pmi : pmi[0] >= max_pmi - 1,pmi_tups)))[1])
    return neg_tups,pos_tups

def gen_new_seeds(PMI,pos_trigrams):
    neg_tups, pos_tups = min_max_tups(PMI,pos_trigrams)
    return Seeds(neg=filter_list(neg_tups),pos=filter_list(pos_tups)) # returns a Seeds tuple

def my_write(data,name):
    print('writing ' + name + ' to file')
    fd = open(name,'w+')
    data = sorted(data,key=lambda tup : tup[1])
    for val, ticker in data:
        fd.write(ticker.replace('.expert','') + ' ' + str(val) + '\n')
    fd.close()

def first_gen(pos_filtered_trigrams,list_pos_filtered_trigrams):
    #     pos_dict = co_occurences(list_pos_filtered_trigrams,{in_pos : 1 })
    #     neg_dict = co_occurences(list_pos_filtered_trigrams,{in_neg : 1 })
    pos_dict = within_10(list_pos_filtered_trigrams,{in_pos : 1})
    neg_dict = within_10(list_pos_filtered_trigrams,{in_neg: 1})
    count_near_pos = get_count_near(pos_dict)
    count_near_neg = get_count_near(neg_dict)
    neg_count, pos_count = get_counts(all_docs)
    PMI = get_PMI(neg_count,pos_count,count_near_neg,count_near_pos)
    so_list = [(SO_of_doc(PMI,doc),name) for doc,name in zip(pos_filtered_trigrams,names)]
    new_seeds = gen_new_seeds(PMI,list_pos_filtered_trigrams)
    return so_list, new_seeds

def next_gen(pos_filtered_trigrams,list_pos_filtered_trigrams,seeds):
    new_in_pos = get_in_list(seeds.pos_words())
    new_in_neg = get_in_list(seeds.neg_words())
    pos_dict = within_10(list_pos_filtered_trigrams,{in_pos : 1, new_in_pos : .1 })
    neg_dict = within_10(list_pos_filtered_trigrams,{in_neg : 1, new_in_neg : .1 })
    count_near_pos = get_count_near(pos_dict)
    count_near_neg = get_count_near(neg_dict)
    neg_count, pos_count = get_counts(all_docs)
    PMI = get_PMI(neg_count,pos_count,count_near_neg,count_near_pos)
    so_list = [(SO_of_doc(PMI,doc),name) for doc,name in zip(pos_filtered_trigrams,names)]
    temp_seeds = gen_new_seeds(PMI,list_pos_filtered_trigrams)
    new_seeds = Seeds(pos=temp_seeds.pos + seeds.pos, neg=temp_seeds.neg + seeds.neg)
    return so_list, new_seeds

def output_to_file(so_list,names):
    so_dict = dict([tup[::-1] for tup in so_list])
#     print(len(so_dict.keys()))
#     print(so_dict)
    f = open('tickers.txt','r')
    out = open('new_tickers.txt','w+')
    f_text = f.read()
    actual_names = []
    for line in f_text.splitlines():
        name = line.split('/')[2].split()[0]
        #print(name)
        for our_name in names :
            print(our_name)
            if our_name in line: 
                line += ' so: ' + str(so_dict[our_name])
                actual_names.append(line)
    for output_line in actual_names:
        out.write(output_line)
    out.close()
    return actual_names


# neg_count, pos_count = get_counts(all_docs)
# print(neg_count, pos_count)


# In[ ]:


# so_list = sorted(so_list,key=lambda SO_score : SO_score[0])
# list(zip(*so_list))[1]
# so_list


# In[ ]:


# vader = SentimentIntensityAnalyzer()
# vader_scores = [(vader.polarity_scores(doc)['compound'],name) for doc, name in zip(all_docs,names)]


# In[ ]:


# vader_scores = sorted(vader_scores,key=lambda score : score[0]) # from greatest to least


# In[ ]:


# dead simple metric
# [vader_stock for vader_stock, so_stock in zip(vader_scores,so_list) if vader_stock[1] == so_stock[1]]


# In[ ]:


# https://chrisalbon.com/statistics/frequentist/spearmans_rank_correlation/
# def spearman_correlation(my_scores,other_scores):
#     other_ranks = pandas.Series(list(zip(*other_scores))[1]).rank()
#     my_ranks = pandas.Series(list(zip(*my_scores))[1]).rank()
#     return stats.spearmanr(other_ranks,my_ranks)
# spearman_correlation(so_list,vader_scores)


# In[ ]:


# TextBlob analysis
# blob = TextBlob(review)
# blob.sentiment
if __name__ == '__main__':
    all_docs, names = docs_list(get_expert_docs('experts/'))
    pos_trigrams = [gen_pos_trigrams(doc) for doc in all_docs]
    filtered_pos_trigrams = list(map(pos_pattern_filter,pos_trigrams))


    # put all the lists together
    list_pos_filtered_trigrams = lists_to_list(filtered_pos_trigrams)

    so_list,first_gen_seeds = first_gen(filtered_pos_trigrams,list_pos_filtered_trigrams)
    
    so_list_2,second_gen_seeds = next_gen(filtered_pos_trigrams,list_pos_filtered_trigrams,first_gen_seeds)

    so_list_3,third_gen_seeds = next_gen(filtered_pos_trigrams,list_pos_filtered_trigrams,second_gen_seeds)

    so_list_4,fourth_gen_seeds=next_gen(filtered_pos_trigrams,list_pos_filtered_trigrams,third_gen_seeds
    my_write(so_list_4,'so_list_4.txt')
