import os
import pickle
import json
from sklearn.utils import shuffle
import random
from tqdm import tqdm
#################
# parse json file
#################
def parse(path):
    g = open(path, 'r')
    for l in g:
        yield json.loads(l.strip())


##################
# Class for corpus
##################
class Corpus(object):

    def __init__(self):

        # dictionary with all documents
        self.dict_author_type_doc = {}
        self.dict_author_type_doc_test = {}



        # dictionaries for the splits
        self.dict_author_doc_train = {}
        self.dict_author_doc_train_L = {}


    def shuffledict(self,big_dict,labels):
        keys = list(big_dict.keys())
        random.shuffle(keys)
        Shuffled = dict()
        LShuffled = dict()
        for key in keys:
            if key not in Shuffled.keys():
                Shuffled[key] = {}
            if key not in LShuffled.keys():
                LShuffled[key] = {}
            Shuffled[key]= big_dict[key]
            LShuffled[key]= labels[key]

        return Shuffled,LShuffled
    def train_val_split_perID(self,shuffledDict,shuffledDictL):
        tmp_train = {}
        tmp_train_L = {}
        tmp_val = {}
        tmp_val_L = {}
        n = int(0.85 * len(list(shuffledDict.keys())))
        keys = list(shuffledDict.keys())
        train_keys = keys[:n]
        val_keys = keys[n:]
        for key in train_keys:
            if key not in shuffledDict.keys():
                tmp_train[key] = {}
                tmp_train_L[key] = {}
            tmp_train[key]= shuffledDict[key]
            tmp_train_L[key]= shuffledDictL[key]
        for key in val_keys:
            if key not in shuffledDict.keys():
                tmp_val[key] = {}
                tmp_val_L[key] = {}
            tmp_val[key]= shuffledDict[key]
            tmp_val_L[key]= shuffledDictL[key]

        return tmp_train,tmp_train_L,tmp_val,tmp_val_L

    def parseAll(self,dir_pairs, dir_labels):
        # open json files
        with open(dir_pairs, 'r', encoding='utf-8') as f:
            lines_pairs = f.readlines()
        with open(dir_labels, 'r', encoding='utf-8') as f:
            lines_labels = f.readlines()
        for n in range(len(lines_pairs)):
            pair, label = json.loads(lines_pairs[n].strip()), json.loads(lines_labels[n].strip())
            if pair['id'] == label['id']:
                if pair['id'] not in self.dict_author_doc_train.keys():
                    self.dict_author_doc_train[pair['id']] = {}
                self.dict_author_doc_train[pair['id']] = pair
                if label['id'] not in self.dict_author_doc_train_L.keys():
                    self.dict_author_doc_train_L[label['id']] = {}
                self.dict_author_doc_train_L[label['id']] = label
        Shuffled,LShuffled = self.shuffledict(self.dict_author_doc_train,self.dict_author_doc_train_L)
        train,train_L,test,test_L = self.train_val_split_perID(Shuffled,LShuffled)

        return train,train_L,test,test_L
    def parse_raw_data_v2(self,dir_pairs,dir_labels):
        # open json files
        with open(dir_pairs, 'r', encoding='utf-8') as f:
            lines_pairs = f.readlines()
        with open(dir_labels, 'r', encoding='utf-8') as f:
            lines_labels = f.readlines()

        for n in range(len(lines_pairs)):
            pair, label = json.loads(lines_pairs[n].strip()), json.loads(lines_labels[n].strip())
            author1 = label['authors'][0]
            author2 = label['authors'][1]
            type1 = pair['discourse_types'][0] 
            type2 = pair['discourse_types'][1]
            doc1 = pair['pair'][0]
            doc2 = pair['pair'][1]
            if author1 == author2:
                if author1 not in self.dict_author_type_doc.keys():
                    self.dict_author_type_doc[author1] = {}
                if type1 not in self.dict_author_type_doc[author1].keys():
                    self.dict_author_type_doc[author1][type1] = []
                if type2 not in self.dict_author_type_doc[author1].keys():
                    self.dict_author_type_doc[author1][type2] = []
                self.dict_author_type_doc[author1][type1].append(doc1)
                self.dict_author_type_doc[author1][type2].append(doc2)
            else:
                if author1 not in self.dict_author_type_doc.keys():
                    self.dict_author_type_doc[author1] = {}
                if author2 not in self.dict_author_type_doc.keys():
                    self.dict_author_type_doc[author2] = {}
                if type1 not in self.dict_author_type_doc[author1].keys():
                    self.dict_author_type_doc[author1][type1] = []
                if type2 not in self.dict_author_type_doc[author2].keys():
                    self.dict_author_type_doc[author2][type2] = []
                self.dict_author_type_doc[author1][type1].append(doc1)
                self.dict_author_type_doc[author2][type2].append(doc2)

    def train_test_split(self):
        tmp_train = {}
        tmp_val = {}
        n = int(0.80 * len(list(self.dict_author_type_doc.keys())))
        keys = list(self.dict_author_type_doc.keys())
        train_keys = keys[:n]
        val_keys = keys[n:]
        for key in train_keys:
            if key not in self.dict_author_type_doc.keys():
                tmp_train[key] = {}
            tmp_train[key]= self.dict_author_type_doc[key]
        for key in val_keys:
            if key not in self.dict_author_type_doc.keys():
                tmp_val[key] = {}
            tmp_val[key]= self.dict_author_type_doc[key]
    
        return tmp_train,tmp_val

    def constructTest(self,test):
        final_test={}
        pairId = 0
        pos = 0
        neg = 0
        for auth1 in tqdm(test.keys()):
            for auth2 in test.keys():
                if auth1 == auth2:
                    for type1 in test[auth1].keys():
                        for type2 in test[auth2].keys():
                            if type1 != type2:
                                doc1 = random.choice(test[auth1][type1])
                                doc2 = random.choice(test[auth2][type2])
                                label = 1
                                if pairId not in final_test.keys():
                                    final_test[pairId] = []
                                final_test[pairId].append((doc1,doc2,label,type1,type2))
                                pairId+=1
                                pos+=1
        for auth1 in tqdm(test.keys()):
            for auth2 in test.keys():
                if auth1 != auth2:
                    for type1 in test[auth1].keys():
                        for type2 in test[auth2].keys():
                            if type1 != type2:

                                if neg > pos:
                                   break
                                doc1 = random.choice(test[auth1][type1])
                                doc2 = random.choice(test[auth2][type2])
                                label = 0

                                if pairId not in final_test.keys():
                                    final_test[pairId] = []
                                final_test[pairId].append((doc1,doc2,label,type1,type2))
                                pairId+=1
                                neg+=1
        print(pos)
        print(neg)
        return final_test


    def parse_raw_data(self,train,train_L,test,test_L):
        # open json files
        # with open(dir_pairs, 'r', encoding='utf-8') as f:
        #     lines_pairs = f.readlines()
        # with open(dir_labels, 'r', encoding='utf-8') as f:
        #     lines_labels = f.readlines()

        for k in train.keys():
            pair, label = train[k], train_L[k]
            print(label)


            # get author-ID
            pairId = pair['id']
            labelId = label['id']
            author1 = label['authors'][0]
            author2 = label['authors'][1]
            type1 = pair['discourse_types'][0]
            type2 = pair['discourse_types'][1]
            doc1 = pair['pair'][0]
            doc2 = pair['pair'][1]

            if pairId == labelId:
                if author1 == author2:
                    if author1 not in self.dict_author_type_doc.keys():
                        self.dict_author_type_doc[author1] = {}
                    if type1 not in self.dict_author_type_doc[author1].keys():
                        self.dict_author_type_doc[author1][type1] = []
                    if type2 not in self.dict_author_type_doc[author1].keys():
                        self.dict_author_type_doc[author1][type2] = []
                    self.dict_author_type_doc[author1][type1].append(doc1)
                    self.dict_author_type_doc[author1][type2].append(doc2)
                else:
                    if author1 not in self.dict_author_type_doc.keys():
                        self.dict_author_type_doc[author1] = {}
                    if author2 not in self.dict_author_type_doc.keys():
                        self.dict_author_type_doc[author2] = {}
                    if type1 not in self.dict_author_type_doc[author1].keys():
                        self.dict_author_type_doc[author1][type1] = []
                    if type2 not in self.dict_author_type_doc[author2].keys():
                        self.dict_author_type_doc[author2][type2] = []
                    self.dict_author_type_doc[author1][type1].append(doc1)
                    self.dict_author_type_doc[author2][type2].append(doc2)
        for k in test.keys():
            pair, label = test[k], test_L[k]
            # get author-ID
            pairId = pair['id']
            labelId = label['id']

            if pairId == labelId:
                type1 = pair['discourse_types'][0]
                type2 = pair['discourse_types'][1]
                doc1 = pair['pair'][0]
                doc2 = pair['pair'][1]
                L = 0
                if label['same'] == True:
                    L =1
                elif label['same'] == False:
                    L =0
                if pairId not in self.dict_author_type_doc_test.keys():
                    self.dict_author_type_doc_test[pairId] = []

                self.dict_author_type_doc_test[pairId].append((doc1,doc2,L,type1,type2))





def shuffledict(big_dict):
    keys = list(big_dict.keys())
    random.shuffle(keys)
    Shuffled = dict()
    for key in keys:
        if key not in Shuffled.keys():
            Shuffled[key] = {}
        Shuffled[key]= big_dict[key]

    return Shuffled


def train_val_split_perAuthor(shuffledDict):
    tmp_train = {}
    tmp_val = {}
    n = int(0.85 * len(list(shuffledDict.keys())))
    keys = list(shuffledDict.keys())
    train_keys = keys[:n]
    val_keys = keys[n:]
    for key in train_keys:
        if key not in shuffledDict.keys():
            tmp_train[key] = {}
        tmp_train[key]= shuffledDict[key]
    for key in val_keys:
        if key not in shuffledDict.keys():
            tmp_val[key] = {}
        tmp_val[key]= shuffledDict[key]

    return tmp_train,tmp_val







corpus = Corpus()
filename = '../pairs.jsonl'
labels = '../truth.jsonl'
print("Parse ", filename)
# train,train_L,test,test_L = corpus.parseAll(filename, labels)
# print("Split Dataset...")
# corpus.split_data()

corpus.parse_raw_data_v2(filename,labels)
train,test = corpus.train_test_split()
test_new = corpus.constructTest(test)


##############################
# store results (binary files)
##############################
with open('dict_auth_type_pan23_train', 'wb') as f:
    pickle.dump(train, f)

with open('dict_auth_type_pan23_test', 'wb') as f:
    pickle.dump(test_new, f)