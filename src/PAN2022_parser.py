import os
import spacy
import pickle
import re
import torch
from torch.nn import functional as F
from transformers import BertTokenizerFast, BertTokenizer
import emojis
import demoji
import random
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
##################
# Class for corpus
##################
class Corpus(object):

    def __init__(self, dict_dataset):

        # define tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # raw dataset
        self.dict_dataset_raw = dict_dataset

        self.dict_dataset_per_auth_ids= {}
        self.dict_dataset_per_auth_masks= {}
        self.dict_positives_pairs = {}
        self.dict_negatives_pairs = {}
        self.dict_All_pairs = {}
        self.dict_All_pairs_shuffle = {}
        self.usedAuthors = {}
        self.list_of_non_freq_words = []
    ####################
    # doc pre-processing
    ####################
    def removeHTMLtags(self,text):


        x = re.sub('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', '',text)
        x2 = re.sub('\s\s', '',x)
        x3 = re.sub('\t', ' ',x2)
        return x3.strip()
    def maskNumbers(self,text, symbol='1'):
        x = re.sub('[0-9]', symbol,text)
        return x

    def maskEmoticons(self,text, symbol='0'):
        new_list1 = emojis.get(text)
        xx = ''
        for x in new_list1:

            xx = text.replace(x,'0 ')

        dem = demoji.findall(xx)
        for item in dem.keys():
            xx = xx.replace(item, symbol)

        if xx=='':
            return text
        else:
            return xx

    def paddingChunksToMaxlen(self,IdsChunks,isSecond = False,maxLen = 126):#listMasksChunksmasksChunks=None,
        listIdsChunks = list()
        listMasksChunks = list()
        listmasks = []
        for i in range(len(IdsChunks)):
            tmplistmasks = []
            for j in IdsChunks[i]:
                tmplistmasks.append(1)
            listmasks.append(torch.LongTensor(tmplistmasks))

        for i in range(len(IdsChunks)):

            pad_len = maxLen  - IdsChunks[i].shape[0]

            tmplistids = IdsChunks[i].tolist()
            # if isSecond == False:
            tmplistids.insert(0,101)
            tmplistids.insert(len(tmplistids),102)

            tmplistmasks = listmasks[i].tolist()#listmasks[i].tolist()
            # if isSecond == False:
            tmplistmasks.insert(0,1)
            tmplistmasks.insert(len(tmplistmasks),1)
            # print(tmplistids)
            listIdsChunks.append(torch.LongTensor(tmplistids))
            listMasksChunks.append(torch.LongTensor(tmplistmasks))
            del tmplistmasks, tmplistids

            if pad_len > 0:

                listIdsChunks[i] =  F.pad(listIdsChunks[i], (0,pad_len), "constant", 0)
                listMasksChunks[i] =  F.pad(listMasksChunks[i], (0,pad_len), "constant", 0)
        del IdsChunks, pad_len
        # gc.collect()
        return listIdsChunks ,listMasksChunks

    def createRandomSetence(self,listOfSetences):
        tmpList = []
        for x in listOfSetences:
            tmpValue = random.choice(listOfSetences)
            tmpList.append(tmpValue)
        sete = ' '.join(tmpList)
        return sete
    def keepLastSubWordOfBert(self,text):
        tokens = self.tokenizer.tokenize(text)
        construct_sentence = ''
        toremove = []
        # print(tokens)
        for idx in range(len(tokens)):
            if tokens[idx].startswith('##'):
                i = idx-1
                tokens[i] = '<DEL>'
                # toremove.append([tokens[i],i])
                # last = ''
                # while  tokens[i].startswith('##'):
                #     last = tokens[i]
                #     toremove.append(tokens[idx-1])
                #     if tokens[i].startswith('##')==False: break
                #     i+=1
        # print("after")
        # print(tokens)
        # for element in tokens:
        #     if element == '<DEL>':
        tokens = [x for x in tokens if x != '<DEL>']
        # for element in tokens:
        #     if element == '<DEL>':
        #         tokens.remove(element)
        # print(tokens)


        # for i,j in toremove:
        #     if j < len(tokens):
        #         tokens.pop(j)
        # print("after")
        # print(tokens)
        # tokens2 = self.tokenizer.prepare_for_tokenization(tokens,is_split_into_words=True)
        # # tokens2 = self.tokenizer.tokenize(tokens)
        # print()
        # print(tokens2)
        return tokens



    def replaceTokensWith(self,text,symbol='2'):
        for token in self.list_of_non_freq_words:
            text = text.replace(token,symbol)
        print(text)
        return text

    def chunkingTextsBasedOnBert(self,text_tokens):
        input_ids=[]
        attention_masks=[]


        # setences1 = nltk.sent_tokenize(row['Text1'])
        # setences2 = nltk.sent_tokenize(row['Text2'])
        # # print(setences1)
        # set1 = createRandomSetence(setences1)
        # # print(set1)
        # set2 = createRandomSetence(setences2)

        encoded_dict = self.tokenizer.encode_plus(
            text_tokens ,                    # Sentence to encode.
            add_special_tokens = False, # Add '[CLS]' and '[SEP]'
            # max_length = 512,           # Pad & truncate all sentences.
            # pad_to_max_length = True,
            # truncation = False,
            return_attention_mask = True,   # Construct attn. masks.
            return_tensors = 'pt',
            is_split_into_words=True
        )
        # print(encoded_dict['input_ids'])
        # tokens = self.tokenizer.convert_tokens_to_ids(text_tokens)
        # print(tokens)
        # tokens_tensor = torch.LongTensor(tokens)
        # print(tokens_tensor)
        # print(self.tokenizer.convert_ids_to_tokens(tokens_tensor))
        tensorsIdList1,tensorsMaskList1 = self.paddingChunksToMaxlen(encoded_dict['input_ids'][0].split(64),encoded_dict['attention_mask'][0].split(64),False,64)
        # tensorsIdList1,tensorsMaskList1 = self.paddingChunksToMaxlen(tokens_tensor.split(254),False,64)
        # print(tensorsIdList1)


        return tensorsIdList1, tensorsMaskList1


    def preprocess_doc(self, doc,isPAN20=False):
        doc = self.maskNumbers(doc)
        if isPAN20 == False:
            doc = self.removeHTMLtags(doc)
            doc = self.maskEmoticons(doc)

        return doc.strip()
    def getNonFrequentWordsPerDoc(self,doc,threshold=5):
        words = doc.split('')
        fdist = FreqDist(words)

        for word,freq in fdist.items():
            if freq <= 2:
                self.list_of_non_freq_words.append(word)




    ################
    # split data set
    ################
    def parse_dictionary(self):

        # authors
        for a in tqdm(self.dict_dataset_raw.keys()):
            # for f in self.dict_dataset_raw[a].keys():
            # fandom categories
            listDocsPerAuthor = []
            listMasksPerAuthor = []
            # print(a)
            # for f in self.dict_dataset_raw[a].keys():
            # documents
            # print(f)

            for i, docs in enumerate(self.dict_dataset_raw[a]):


                processed_doc1 = self.preprocess_doc(docs,False)

                # self.getNonFrequentWordsPerDoc(processed_doc1)
                # processed_doc1 = self.replaceTokensWith(processed_doc1)
                # processed_doc2 = self.preprocess_doc(docs[1])
                # tokens = self.keepLastSubWordOfBert(processed_doc1)
                inputIDs1,attn_masks1 = self.chunkingTextsBasedOnBert(processed_doc1)
                # inputIDs2,attn_masks2 = self.chunkingTextsBasedOnBert(processed_doc2)
                flat_IDs = []
                flat_Masks = []
                # flat_IDs2 = []
                # flat_Masks2 = []
                for item in inputIDs1:
                    flat_IDs.append(item)
                for item in attn_masks1:
                    flat_Masks.append(item)
                # for item in inputIDs2:
                #     flat_IDs2.append(item)
                # for item in attn_masks2:
                #     flat_Masks2.append(item)


                listDocsPerAuthor.append(flat_IDs)
                listMasksPerAuthor.append(flat_Masks)
            if a not in self.dict_dataset_per_auth_ids.keys():
                self.dict_dataset_per_auth_ids[a] = {}
            if a not in self.dict_dataset_per_auth_masks.keys():
                self.dict_dataset_per_auth_masks[a] = {}
            # if f not in self.dict_dataset_per_auth_ids[a].keys():
            #     self.dict_dataset_per_auth_ids[a][f] = {}
            # if f not in self.dict_dataset_per_auth_masks[a].keys():
            #     self.dict_dataset_per_auth_masks[a][f] = {}
            # print(len(listDocsPerAuthor))
            self.dict_dataset_per_auth_ids[a] = listDocsPerAuthor  # authID:{[x1,x2,x3],[y1,y2...yn],...}
            self.dict_dataset_per_auth_masks[a] = listMasksPerAuthor
            # print(self.dict_dataset_per_auth_ids[a])
        # print(self.dict_dataset_per_auth_ids[a])


    def generatePairs(self,isval = False):
        if isval == False:
            pairIDpos = 0
            pairIDneg = 0

            for a in tqdm(self.dict_dataset_per_auth_ids.keys()):
                auth1 = a
                # print(a)
                # print(self.dict_dataset_per_auth_ids[auth1])
                for a2 in self.dict_dataset_per_auth_ids.keys():

                    auth2 = a2

                    # auth2 = random.choice(list(self.dict_dataset_per_auth_ids.keys()))
                    #check if the same auth
                    #if the same check if exact same texts
                    if auth1==auth2: #positive pair
                        # for f1 in self.dict_dataset_per_auth_ids[auth1].keys():
                        for i in range(0,len(self.dict_dataset_per_auth_ids[auth1])):
                            for j in range(0,len(self.dict_dataset_per_auth_ids[auth1])):
                                # ii = random.choice(range(0,len(self.dict_dataset_per_auth_ids[auth1][i])))
                                if j > i: #not chunks from same Text

                                    for k in range(0,len(self.dict_dataset_per_auth_ids[auth1][i])):
                                        text1 = self.dict_dataset_per_auth_ids[auth1][i][k]
                                        mask1 =  self.dict_dataset_per_auth_masks[auth1][i][k]
                                        for z in range(0,len(self.dict_dataset_per_auth_ids[auth1][j])):
                                            # if z > k:
                                            pairIDpos += 1
                                            text2 = self.dict_dataset_per_auth_ids[auth1][j][z]
                                            mask2 = self.dict_dataset_per_auth_masks[auth1][j][z]
                                            cnt = 0
                                            while torch.equal(text1, text2):
                                                # cnt+=1
                                                # print("SOS")
                                                # if cnt >= 200:
                                                #     print("stuck")
                                                nn = random.choice(range(0,len(self.dict_dataset_per_auth_ids[auth1][j])))
                                                text2 = self.dict_dataset_per_auth_ids[auth1][j][nn]
                                                mask2 = self.dict_dataset_per_auth_masks[auth1][j][nn]
                                            label = 1

                                            if pairIDpos not in self.dict_positives_pairs.keys():
                                                self.dict_positives_pairs[pairIDpos] = []
                                            self.dict_positives_pairs[pairIDpos].append([text1,mask1,text2,mask2,label])


                    elif auth1!=auth2: #diff authors
                        # for f1 in self.dict_dataset_per_auth_ids[auth1].keys():
                        #     for f2 in self.dict_dataset_per_auth_ids[auth2].keys():
                        # if f1 == f2:
                        if auth1 not in self.usedAuthors.keys():
                            if auth2 not in self.usedAuthors.keys():

                                for i in range(0,len(self.dict_dataset_per_auth_ids[auth1])):
                                    for ii in range(0,len(self.dict_dataset_per_auth_ids[auth1][i])):
                                    # ii = random.choice(range(0,len(self.dict_dataset_per_auth_ids[auth1][i])))
                                        text1 = self.dict_dataset_per_auth_ids[auth1][i][ii]
                                        mask1 =  self.dict_dataset_per_auth_masks[auth1][i][ii]
                                        for j in range(0,len(self.dict_dataset_per_auth_ids[auth2])):
                                            for jj in range(0,len(self.dict_dataset_per_auth_ids[auth2][j])):
                                                pairIDneg +=1
                                            # j = random.choice(range(0,len(self.dict_dataset_per_auth_ids[auth2])))
                                            # jj = random.choice(range(0,len(self.dict_dataset_per_auth_ids[auth2][j])))

                                                text2 = self.dict_dataset_per_auth_ids[auth2][j][jj]
                                                mask2 = self.dict_dataset_per_auth_masks[auth2][j][jj]
                                                label = 0
                                                # print(torch.equal(text1,text2))
                                                if pairIDneg not in self.dict_negatives_pairs.keys():
                                                    self.dict_negatives_pairs[pairIDneg] = []
                                                self.dict_negatives_pairs[pairIDneg].append([text1,mask1,text2,mask2,label])
                                self.usedAuthors[auth2] = []
                                self.usedAuthors[auth1] = []
                    # else:
                    #     if auth1 not in self.usedAuthors.keys():
                    #         if auth2 not in self.usedAuthors.keys():
                    #
                    #             for i in range(0,len(self.dict_dataset_per_auth_ids[auth1][f1])):
                    #
                    #                 ii = random.choice(range(0,len(self.dict_dataset_per_auth_ids[auth1][f1][i])))
                    #                 text1 = self.dict_dataset_per_auth_ids[auth1][f1][i][ii]
                    #                 mask1 =  self.dict_dataset_per_auth_masks[auth1][f1][i][ii]
                    #                 for j in range(0,len(self.dict_dataset_per_auth_ids[auth2][f2])):
                    #                     pairIDneg +=1
                    #                     # j = random.choice(range(0,len(self.dict_dataset_per_auth_ids[auth2])))
                    #                     jj = random.choice(range(0,len(self.dict_dataset_per_auth_ids[auth2][j])))
                    #
                    #                     text2 = self.dict_dataset_per_auth_ids[auth2][j][jj]
                    #                     mask2 = self.dict_dataset_per_auth_masks[auth2][j][jj]
                    #                     label = 0
                    #                     if pairIDneg not in self.dict_negatives_pairs.keys():
                    #                         self.dict_negatives_pairs[pairIDneg] = []
                    #                     self.dict_negatives_pairs[pairIDneg].append([text1,mask1,text2,mask2,label])
                    #                     # print(auth1)
                    #                     # print(auth2)
                    #                     # print(f1)
                    #                     # print(f2)
                    #                     # print(text1)
                    #                     # print(text2)
                    #             self.usedAuthors[auth2] = []




            print(pairIDneg)
            print(pairIDpos)



    def shuffleDict(self):
        keys = list(self.dict_All_pairs.keys())
        random.shuffle(keys)

        Shuffled = dict()

        for key in keys:
            if key not in Shuffled.keys():
                Shuffled[key] = []
            Shuffled[key].append(self.dict_All_pairs[key])

        return Shuffled
    def generateLastDict(self):
        generalID = 0
        for posid in tqdm(self.dict_positives_pairs.keys()):
            generalID+=1
            if generalID not in self.dict_All_pairs.keys():
                self.dict_All_pairs[generalID] = []
            self.dict_All_pairs[generalID].append(self.dict_positives_pairs[posid])
        for negid in tqdm(self.dict_negatives_pairs.keys()):
            generalID+=1
            if generalID not in self.dict_All_pairs.keys():
                self.dict_All_pairs[generalID] = []
            self.dict_All_pairs[generalID].append(self.dict_negatives_pairs[negid])

        # self.dict_All_pairs_shuffle = self.shuffleFinalDict()




def shuffledict(big_dict):
    keys = list(big_dict.keys())
    random.shuffle(keys)

    Shuffled = dict()

    for key in keys:
        if key not in Shuffled.keys():
            Shuffled[key] = {}
        Shuffled[key]= big_dict[key]

    return Shuffled

def train_val_split(shuffledDict):


    tmp_train = {}
    tmp_val = {}
    for k,v in shuffledDict.items():
        for k2,v2 in v.items():
            if len(v2) > 2:
                n = int(0.80 * len(v2))
                if k not in tmp_train.keys():
                    tmp_train[k] = {}
                if k not in tmp_val.keys():
                    tmp_val[k] = {}
                if k2 not in tmp_train[k].keys():
                    tmp_train[k][k2] = []
                if k2 not in tmp_val[k].keys():
                    tmp_val[k][k2] = []

                tmp_train[k][k2] = v2[:n]
                tmp_val[k][k2]=v2[n:]
            elif len(v2) == 1:
                if k not in tmp_train.keys():
                    tmp_train[k] = {}
                if k2 not in tmp_train[k].keys():
                    tmp_train[k][k2] = []
                tmp_train[k][k2] = v2



    return tmp_train,tmp_val
####################
# data folders/files
####################


datasets = ['dict_per_auth_type_docs_pan22'
            # 'dict_author_fandom_doc_val'
            ]

for dataset in datasets:


    ######
    # load
    ######
    with open( dataset, 'rb') as f:
        dict_dataset = pickle.load(f)

    ############
    # preprocess
    ############
    shuffled_data = shuffledict(dict_dataset)
    train,val = train_val_split(shuffled_data)
    corpus = Corpus(dict_dataset=train)
    corpus.parse_dictionary()
    corpus.generatePairs()
    corpus.generateLastDict()


    #######
    # store
    #######
    with open( dataset + "_PAN22_train_pos_pairs_64_keepLast", 'wb') as f:
        pickle.dump(corpus.dict_positives_pairs, f)
    with open( dataset + "_PAN22_train_neg_pairs_64_keepLast", 'wb') as f:
        pickle.dump(corpus.dict_negatives_pairs, f)
    del corpus
    corpus_val = Corpus(dict_dataset=val)
    corpus_val.parse_dictionary()
    corpus_val.generatePairs()
    corpus_val.generateLastDict()

    with open( dataset + "_PAN22_val_pos_pairs_64_keepLast", 'wb') as f:
        pickle.dump(corpus_val.dict_positives_pairs, f)
    with open( dataset + "_PAN22_val_neg_pairs_64_keepLast", 'wb') as f:
        pickle.dump(corpus_val.dict_negatives_pairs, f)
    # with open( dataset + "_test", 'wb') as f:
    #     pickle.dump(corpus.dict_dataset_per_auth_ids, f)

    del corpus_val