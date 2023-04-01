import pickle
import re
import os
import nltk
import torch
from torch.nn import functional as F
from transformers import BertTokenizer
# import emojis
# import demoji
import random
from tqdm import tqdm
import pandas as pd
class Corpus(object):

    def __init__(self, dict_dataset):

        # define tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # raw dataset
        self.dict_dataset_raw = dict_dataset

        self.dict_dataset_per_auth_ids= {}
        self.dict_dataset_per_auth_masks= {}

    def paddingChunksToMaxlen(self,IdsChunks,masksChunks,isSecond = False,maxLen = 126):#listMasksChunks
            listIdsChunks = list()
            listMasksChunks = list()

            for i in range(len(IdsChunks)):

                pad_len = maxLen  - IdsChunks[i].shape[0]

                tmplistids = IdsChunks[i].tolist()
                if len(tmplistids) < maxLen:
                    if i > 0: #from second element
                        temp = IdsChunks[i-1].tolist()
                        lenToadd = maxLen - len(tmplistids)
                        # print("=================================")
                        # print("prev = ",str(len(temp)))
                        # print("curr = ", str(len(tmplistids)))

                        if len(temp) >= lenToadd:

                            tmplistids = temp[len(temp)-lenToadd:len(temp)] + tmplistids
                            pad_len = maxLen  - len(tmplistids)
                        else:
                            tmplistids = temp + tmplistids
                            pad_len = maxLen  - len(tmplistids)
                    else:
                        tmplistids = tmplistids
                        pad_len = maxLen  - len(tmplistids)
                elif len(tmplistids) == maxLen:
                    tmplistids = tmplistids
                    pad_len = maxLen  - len(tmplistids)
                # if isSecond == False:
                tmplistids.insert(0,101) #101 for BERT
                tmplistids.insert(len(tmplistids),102) #102 for Bert
                # print("after = ",str(len(tmplistids)))
                # print("=================================")
                tmplistmasks = masksChunks[i].tolist()#listmasks[i].tolist()
                if len(tmplistmasks) < maxLen:
                    if i > 0: #from second element
                        temp = masksChunks[i-1].tolist()
                        lenToadd = maxLen - len(tmplistmasks)
                        if len(temp) >= lenToadd:
                            tmplistmasks = temp[len(temp)-lenToadd:len(temp)] + tmplistmasks
                        else:
                            tmplistmasks = temp + tmplistmasks
                    else:
                        tmplistmasks = tmplistmasks

                elif len(tmplistmasks) == maxLen:
                    tmplistmasks = tmplistmasks
                # if isSecond == False:
                tmplistmasks.insert(0,1)
                tmplistmasks.insert(len(tmplistmasks),1)
                # print(tmplistids)
                # if len(tmplistids) > 128:
                listIdsChunks.append(torch.LongTensor(tmplistids))
                listMasksChunks.append(torch.LongTensor(tmplistmasks))
                del tmplistmasks, tmplistids

                if pad_len > 0:

                    listIdsChunks[i] =  F.pad(listIdsChunks[i], (0,pad_len), "constant", 0) # 0 for bert
                    listMasksChunks[i] =  F.pad(listMasksChunks[i], (0,pad_len), "constant", 0)
            del IdsChunks, masksChunks, pad_len
            # gc.collect()
            return listIdsChunks ,listMasksChunks
    def chunkingTextsBasedOnBert(self,text):
            input_ids=[]
            attention_masks=[]


            # setences1 = nltk.sent_tokenize(row['Text1'])
            # setences2 = nltk.sent_tokenize(row['Text2'])
            # # print(setences1)
            # set1 = createRandomSetence(setences1)
            # # print(set1)
            # set2 = createRandomSetence(setences2)

            encoded_dict = self.tokenizer.encode_plus(
                text ,                    # Sentence to encode.
                add_special_tokens = False, # Add '[CLS]' and '[SEP]'
                # max_length = 512,           # Pad & truncate all sentences.
                # pad_to_max_length = True,
                # truncation = True,
                return_attention_mask = True,   # Construct attn. masks.
                return_tensors = 'pt',     # Return pytorch tensors.
            )

            # print(encoded_dict['input_ids'][0])
            tensorsIdList1,tensorsMaskList1 = self.paddingChunksToMaxlen(encoded_dict['input_ids'][0].split(126),encoded_dict['attention_mask'][0].split(126),False,126)
            # tensorsIdList1,tensorsMaskList1 = encoded_dict['input_ids'][0],encoded_dict['attention_mask'][0]
            # del encoded_dict

            return tensorsIdList1, tensorsMaskList1
    def maskNumbers(self,text, symbol='1'):
        x = re.sub('[0-9]', symbol,text)
        return x
    def preprocess_doc(self, doc):
            doc = self.maskNumbers(doc)
            return doc.strip()

    def parse_dictionary_tst(self):
        cnt = 0
        # authors
        print("Parse per author...for ",str(len(list(self.dict_dataset_raw.keys()))))
        for a in tqdm(self.dict_dataset_raw.keys()):
            # fandom categories
            listDocsPerAuthor = []
            listMasksPerAuthor = []
            # print(a)
            # for f in self.dict_dataset_raw[a].keys():
            # documents
            # print(f)
            # print(self.dict_dataset_raw[a])

            # for docs in self.dict_dataset_raw[a]:

                #
            docs = self.dict_dataset_raw[a]
            print(len(docs))
            processed_doc1 = self.preprocess_doc(docs[0])
            processed_doc2 = self.preprocess_doc(docs[1])
            label = docs[2]
                # type1 = docs[3]
                # type2 = docs[4]

                # processed_doc2 = self.preprocess_doc(docs)
            inputIDs1,attn_masks1 = self.chunkingTextsBasedOnBert(processed_doc1)
            inputIDs2,attn_masks2 = self.chunkingTextsBasedOnBert(processed_doc2)
                # print(inputIDs1)
                # i = random.choice(range(0,len(inputIDs1)))
                # ii = random.choice(range(0,len(inputIDs1)))
                # j = random.choice(range(0,len(inputIDs2)))
                # jj = random.choice(range(0,len(inputIDs2)))

            if len(inputIDs1) == len(inputIDs2):
                    for i in range(0,len(inputIDs1)):

                        text1 = inputIDs1[i]
                        mask1 = attn_masks1[i]
                        text2 = inputIDs2[i]
                        mask2 = attn_masks2[i]
                        if a not in self.dict_dataset_per_auth_ids.keys():
                            self.dict_dataset_per_auth_ids[a] = []
                        self.dict_dataset_per_auth_ids[a].append([text1,mask1,text2,mask2,label])
            elif len(inputIDs1) > len(inputIDs2):
                    for i in range(0,len(inputIDs1)):
                        if i < len(inputIDs2):
                            text1 = inputIDs1[i]
                            mask1 = attn_masks1[i]
                            text2 = inputIDs2[i]
                            mask2 = attn_masks2[i]
                            if a not in self.dict_dataset_per_auth_ids.keys():
                                self.dict_dataset_per_auth_ids[a] = []
                            self.dict_dataset_per_auth_ids[a].append([text1,mask1,text2,mask2,label])
                        else:
                            text1 = inputIDs1[i]
                            mask1 = attn_masks1[i]
                            j = random.choice(range(0,len(inputIDs2)))
                            text2 = inputIDs2[j]
                            mask2 = attn_masks2[j]
                            if a not in self.dict_dataset_per_auth_ids.keys():
                                self.dict_dataset_per_auth_ids[a] = []
                            self.dict_dataset_per_auth_ids[a].append([text1,mask1,text2,mask2,label])
            elif len(inputIDs1) < len(inputIDs2):
                    for j in range(0,len(inputIDs2)):
                        if j < len(inputIDs1):
                            text1 = inputIDs1[j]
                            mask1 = attn_masks1[j]
                            text2 = inputIDs2[j]
                            mask2 = attn_masks2[j]
                            if a not in self.dict_dataset_per_auth_ids.keys():
                                self.dict_dataset_per_auth_ids[a] = []
                            self.dict_dataset_per_auth_ids[a].append([text1,mask1,text2,mask2,label])
                        else:
                            text2 = inputIDs2[j]
                            mask2 = attn_masks2[j]
                            i = random.choice(range(0,len(inputIDs1)))
                            text1 = inputIDs1[i]
                            mask1 = attn_masks1[i]
                            if a not in self.dict_dataset_per_auth_ids.keys():
                                self.dict_dataset_per_auth_ids[a] = []
                            self.dict_dataset_per_auth_ids[a].append([text1,mask1,text2,mask2,label])


    def parse_dictionary(self):
            cnt = 0
            # authors
            print("Parse per author...for ",str(len(list(self.dict_dataset_raw.keys()))))
            cnt = 0
            # authors
            print("Parse per author...for ",str(len(list(self.dict_dataset_raw.keys()))))
            for a in tqdm(self.dict_dataset_raw.keys()):
                # fandom categories
                listDocsPerAuthor = []
                listMasksPerAuthor = []
                # print(a)
                # for f in self.dict_dataset_raw[a].keys():
                # documents
                # print(f)
                # print(self.dict_dataset_raw[a])
                itemid = 0
                for i, docs in enumerate(self.dict_dataset_raw[a]):
                    # processed_doc1 = self.preprocess_doc(docs,True)
                    # processed_doc2 = self.preprocess_doc(docs)
                    processed_doc2 = self.preprocess_doc(docs)

                    inputIDs1,attn_masks1 = self.chunkingTextsBasedOnBert(processed_doc2)
                    # flat_IDs = []
                    # flat_Masks = []
                    # print(inputIDs1)
                    # for item in inputIDs1:
                    #     flat_IDs.append(item)
                    # for item in attn_masks1:
                    #     flat_Masks.append(item)

                    # listDocsPerAuthor.append(flat_IDs)
                    # listMasksPerAuthor.append(flat_Masks)
                    if a not in self.dict_dataset_per_auth_ids.keys():
                        self.dict_dataset_per_auth_ids[a] = {}
                    if itemid not in self.dict_dataset_per_auth_ids[a].keys():
                        self.dict_dataset_per_auth_ids[a][itemid] = {}


                    if a not in self.dict_dataset_per_auth_masks.keys():
                        self.dict_dataset_per_auth_masks[a] = {}
                    if itemid not in self.dict_dataset_per_auth_masks[a].keys():
                        self.dict_dataset_per_auth_masks[a][itemid] = {}



                    self.dict_dataset_per_auth_ids[a][itemid]= inputIDs1

                    self.dict_dataset_per_auth_masks[a][itemid]=attn_masks1

                    itemid+=1

# path = 'C:/Users/ppetropo/Desktop/Thesis/pan15-authorship-verification-training-dataset-2015-04-19/pan15-authorship-verification-training-dataset-english-2015-04-19'
path = 'C:/Users/ppetropo/Desktop/Thesis/pan15-authorship-verification-test-dataset2-2015-04-19/pan15-authorship-verification-test-dataset2-english-2015-04-19'
def getGroundTruth(gtFile):

    gt = pd.read_csv(gtFile,sep=' ',names=["DirName", "Label"])
    return gt

def readFilesAndGetDict(path):
    gt = path+'/'+'truth.txt'
    dir_list = os.listdir(path)
    groundTruth = getGroundTruth(gt)
    dict_per_auth_doc = {}
    for dir in dir_list:
        if dir != 'truth.txt' and dir != 'contents.json':
               tmp_list_files = os.listdir(path+'/'+dir)
               row = groundTruth[groundTruth['DirName']==dir]
               if row.Label.item() == 'Y':
                  fileKnown = tmp_list_files[0]
                  fileUnKnown = tmp_list_files[1]
                  with open(path+'/'+dir+'/'+fileKnown,'r') as f:
                       content = f.read()

                  with open(path+'/'+dir+'/'+fileUnKnown,'r') as f:
                      content2 = f.read()
                  tmpList = []
                  if dir not in dict_per_auth_doc.keys():
                      dict_per_auth_doc[dir] = []
                  tmpList.append(content)
                  tmpList.append(content2)
                  # tmpList.append(1)

                  dict_per_auth_doc[dir] = tmpList
               elif row.Label.item() == 'N':
                   fileKnown = tmp_list_files[0]
                   fileUnKnown = tmp_list_files[1]
                   tmpList = []
                   with open(path+'/'+dir+'/'+fileKnown,'r') as f:
                       content = f.read()
                   with open(path+'/'+dir+'/'+fileUnKnown,'r') as f:
                       content2 = f.read()
                   if dir not in dict_per_auth_doc.keys():
                       dict_per_auth_doc[dir] = []
                   tmpList.append(content)
                   tmpList.append(content2)
                   # tmpList.append(0)

                   dict_per_auth_doc[dir] = tmpList
    with open("PAN15_dict_per_auth_doc_test", 'wb') as f:
        pickle.dump(dict_per_auth_doc, f)
    return dict_per_auth_doc


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
    n = int(0.90 * len(list(shuffledDict.keys())))
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
dict_per_auth_doc = readFilesAndGetDict(path)
shuffledDict = shuffledict(dict_per_auth_doc)
train,val = train_val_split_perAuthor(shuffledDict)
print(len(list(train.keys())))
print(len(list(val.keys())))

corpus = Corpus(dict_dataset=train)
corpus.parse_dictionary()
with open( "PAN15_128_train_uncased_ids", 'wb') as f:
    pickle.dump(corpus.dict_dataset_per_auth_ids, f)
with open( "PAN15_128_train_uncased_masks", 'wb') as f:
    pickle.dump(corpus.dict_dataset_per_auth_masks, f)
#
corpusval = Corpus(dict_dataset=val)
corpusval.parse_dictionary()
with open( "PAN15_128_val_uncased_ids", 'wb') as f:
    pickle.dump(corpusval.dict_dataset_per_auth_ids, f)
with open( "PAN15_128_val_uncased_masks", 'wb') as f:
    pickle.dump(corpusval.dict_dataset_per_auth_masks, f)