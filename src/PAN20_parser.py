import pickle
import re

import nltk
import torch
from torch.nn import functional as F
from transformers import RobertaTokenizer, BertTokenizer
# import emojis
# import demoji
import random
from tqdm import tqdm
##################
# Class for corpus
##################
class Corpus(object):

    def __init__(self, dict_dataset):

        # define tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        # raw dataset
        self.dict_dataset_raw = dict_dataset
        self.dict_dataset_raw2 = {}
        self.dict_dataset_per_auth_ids= {}
        self.dict_dataset_per_auth_masks= {}
        self.dict_positives_pairs = {}
        self.dict_negatives_pairs = {}
        self.dict_All_pairs = {}
        self.dict_All_pairs_shuffle = {}
        self.usedAuthors = {}

        self.dict_pairs_per_author = {}
        self.per_author_batch_ids = {}
        self.per_author_batch_masks = {}
        self.per_author_batch_ids2 = {}
        self.per_author_batch_masks2 = {}
        self.per_author_batch_lbl = {}

    def getNonFrequentWordsPerDoc(self,doc,threshold=5):
        words = nltk.word_tokenize(doc)

        fdist = nltk.FreqDist(words)
        list_of_non_freq_words = []
        for word,freq in fdist.items():
            if freq < threshold:
                # if len(word) > 4:
                list_of_non_freq_words.append(word)
        for token in list_of_non_freq_words:
            if len(token) > 4:
                doc = doc.replace(token,'2')

        return doc

    # def replaceTokensWith(self,text,symbol='2'):
    #     for token in self.list_of_non_freq_words:
    #         text = text.replace(token,symbol)
    #     return text

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

    def createRandomSetence(self,listOfSetences):
        tmpList = []
        for x in listOfSetences:
            tmpValue = random.choice(listOfSetences)
            tmpList.append(tmpValue)
        sete = ' '.join(tmpList)
        return sete

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
        tensorsIdList1,tensorsMaskList1 = self.paddingChunksToMaxlen(encoded_dict['input_ids'][0].split(510),encoded_dict['attention_mask'][0].split(510),False,510)
        # tensorsIdList1,tensorsMaskList1 = encoded_dict['input_ids'][0],encoded_dict['attention_mask'][0]
        # del encoded_dict

        return tensorsIdList1, tensorsMaskList1
    def chunkingTextsBasedOnBert_singleBert(self,text):
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
        tensorsIdList1,tensorsMaskList1 = self.paddingChunksToMaxlen(encoded_dict['input_ids'][0].split(510),encoded_dict['attention_mask'][0].split(510),False,510)
        # tensorsIdList1,tensorsMaskList1,typeIds = encoded_dict['input_ids'][0],encoded_dict['attention_mask'][0]
        # del encoded_dict

        return tensorsIdList1, tensorsMaskList1
    def maskNumbers(self,text, symbol='1'):
        x = re.sub('[0-9]', symbol,text)
        return x
    def preprocess_doc(self, doc):
        # print(len(doc.strip()))
        # doc = doc.replace('``','')
        # doc = doc.replace('\'\'','')
        doc = self.maskNumbers(doc)
        # print(doc)
        # doc = self.getNonFrequentWordsPerDoc(doc,threshold=10)
        # print("==============================================")
        # print(doc)
        return doc.strip()
    # def convert_to_csv(self):
    #     print("Parse per author...for ",str(len(list(self.dict_dataset_raw.keys()))))
    #     for a in tqdm(self.dict_dataset_raw.keys()):
    #         for i, docs in enumerate(self.dict_dataset_raw[a]):
    #             processed_doc1 = self.preprocess_doc(docs)
    #             inputIDs1,attn_masks1 = self.chunkingTextsBasedOnBert(processed_doc1)

    def parse_dictionary_v2(self):
        cnt = 0
        # authors
        print("Parse per author...for ",str(len(list(self.dict_dataset_raw.keys()))))
        for a in tqdm(self.dict_dataset_raw.keys()):
            # fandom categories
            listDocsPerAuthor = []
            listMasksPerAuthor = []
            for i, docs in enumerate(self.dict_dataset_raw[a]):
                inputIDs1,attn_masks1 = self.chunkingTextsBasedOnBert(docs)
                flat_IDs = []
                flat_Masks = []
                    # flat_IDs2 = []
                    # flat_Masks2 = []
                for item in inputIDs1:
                    flat_IDs.append(item)
                for item in attn_masks1:
                    flat_Masks.append(item)

                listDocsPerAuthor.append(flat_IDs)
                listMasksPerAuthor.append(flat_Masks)
                if a not in self.dict_dataset_per_auth_ids.keys():
                    self.dict_dataset_per_auth_ids[a] = []
                if a not in self.dict_dataset_per_auth_masks.keys():
                    self.dict_dataset_per_auth_masks[a] = []

            self.dict_dataset_per_auth_ids[a] = listDocsPerAuthor  # authID:{[x1,x2,x3],[y1,y2...yn],...}
            self.dict_dataset_per_auth_masks[a] = listMasksPerAuthor

    # def createPairsWithSEPToken(ids1,masks1,ids2,masks2,label,combinations='one-by-one'):
    #
    #     rIds = []
    #     rMasks = []
    #     if combinations=='one-by-one':
    #         for idx in range(len(ids1)):
    #             if idx <= len(ids2)-1:
    #                 rIds.append(torch.cat((ids1[idx],ids2[idx]),0))
    #                 rMasks.append(torch.cat((masks1[idx],masks2[idx]),0))
    #
    #             elif idx >= len(ids2):
    #                 i = random.randint(0, len(ids2)-1)
    #                 rIds.append(torch.cat((ids1[idx],ids2[i]),0))
    #                 rMasks.append(torch.cat((masks1[idx],masks2[i]),0))
    #     return rIds, rMasks
    def parse_dictionary(self):
        cnt = 0
        # authors
        print("Parse per author...for ",str(len(list(self.dict_dataset_raw.keys()))))
        for a in tqdm(self.dict_dataset_raw.keys()):
            # fandom categories
            #     listDocsPerAuthor = []
            #     listMasksPerAuthor = []
            # print(a)
            # for f in self.dict_dataset_raw[a].keys():
            # documents
            # print(f)
            # print(self.dict_dataset_raw[a])

                # for i, docs in enumerate(self.dict_dataset_raw[a]):
                    print(len(self.dict_dataset_raw[a]))
                    processed_doc1 = self.preprocess_doc(self.dict_dataset_raw[a][0][0])
                    processed_doc2 = self.preprocess_doc(self.dict_dataset_raw[a][0][1])
                    auth1 = self.dict_dataset_raw[a][1]
                    auth2 = self.dict_dataset_raw[a][2]
                    #label = docs[2]
                    # type1 = docs[3]
                    # type2 = docs[4]
                    # print(docs)
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
                            self.dict_dataset_per_auth_ids[a].append([text1,mask1,text2,mask2,auth1,auth2])
                    elif len(inputIDs1) > len(inputIDs2):
                        for i in range(0,len(inputIDs1)):
                            if i < len(inputIDs2):
                                text1 = inputIDs1[i]
                                mask1 = attn_masks1[i]
                                text2 = inputIDs2[i]
                                mask2 = attn_masks2[i]
                                if a not in self.dict_dataset_per_auth_ids.keys():
                                    self.dict_dataset_per_auth_ids[a] = []
                                self.dict_dataset_per_auth_ids[a].append([text1,mask1,text2,mask2,auth1,auth2])
                            else:
                                text1 = inputIDs1[i]
                                mask1 = attn_masks1[i]
                                j = random.choice(range(0,len(inputIDs2)))
                                text2 = inputIDs2[j]
                                mask2 = attn_masks2[j]
                                if a not in self.dict_dataset_per_auth_ids.keys():
                                    self.dict_dataset_per_auth_ids[a] = []
                                self.dict_dataset_per_auth_ids[a].append([text1,mask1,text2,mask2,auth1,auth2])
                    elif len(inputIDs1) < len(inputIDs2):
                        for j in range(0,len(inputIDs2)):
                            if j < len(inputIDs1):
                                text1 = inputIDs1[j]
                                mask1 = attn_masks1[j]
                                text2 = inputIDs2[j]
                                mask2 = attn_masks2[j]
                                if a not in self.dict_dataset_per_auth_ids.keys():
                                    self.dict_dataset_per_auth_ids[a] = []
                                self.dict_dataset_per_auth_ids[a].append([text1,mask1,text2,mask2,auth1,auth2])
                            else:
                                text2 = inputIDs2[j]
                                mask2 = attn_masks2[j]
                                i = random.choice(range(0,len(inputIDs1)))
                                text1 = inputIDs1[i]
                                mask1 = attn_masks1[i]
                                if a not in self.dict_dataset_per_auth_ids.keys():
                                    self.dict_dataset_per_auth_ids[a] = []
                                self.dict_dataset_per_auth_ids[a].append([text1,mask1,text2,mask2,auth1,auth2])
                # cnt+=1
                # if cnt == 201:
                #     break



                    # flat_IDs = []
                    # flat_Masks = []
                    # flat_IDs2 = []
                    # flat_Masks2 = []
                    # for item in inputIDs1:
                    #     flat_IDs.append(item)
                    # for item in attn_masks1:
                    #     flat_Masks.append(item)
                    # for item in inputIDs2:
                    #     flat_IDs2.append(item)
                    # for item in attn_masks2:
                    #     flat_Masks2.append(item)
                    # listDocsPerAuthor.append([flat_IDs,flat_IDs2,label])
                    # listMasksPerAuthor.append([flat_Masks,flat_Masks2])
                    # listDocsPerAuthor.append(flat_IDs)
                    # listMasksPerAuthor.append(flat_Masks)
                    # if a not in self.dict_dataset_per_auth_ids.keys():
                    #     self.dict_dataset_per_auth_ids[a] = []
                    # self.dict_dataset_per_auth_ids[a].append([text1,mask1,text2,mask2,label])
                    # if a not in self.dict_dataset_per_auth_masks.keys():
                    #     self.dict_dataset_per_auth_masks[a] = []
                    # for item in listDocsPerAuthor:
                    # # print(item)
                    # for x in item:




                # for item in listMasksPerAuthor:
                #     for x in item:



    # def create_pairs_per_author_v2(self):
    #     for a in tqdm(self.dict_dataset_per_auth_ids.keys()):
    #         used = {}
    #         for i in range(0,len(self.dict_dataset_per_auth_ids[a])):


    def create_pairs_per_author(self):
        pairid = 0
        neg = 0
        pos = 0
        used = {}

        for a in tqdm(self.dict_dataset_per_auth_ids.keys()):
            noOfTextsPos = 0
            noOfTextsNeg = 0
            for i in range(0,len(self.dict_dataset_per_auth_ids[a])):
                ii = 0
                text1 = self.dict_dataset_per_auth_ids[a][i][ii]
                mask1 =  self.dict_dataset_per_auth_masks[a][i][ii]
                for j in range(0,len(self.dict_dataset_per_auth_ids[a])):
                    if j>i:
                        jj = 0
                        text2 = self.dict_dataset_per_auth_ids[a][j][jj]
                        mask2 =  self.dict_dataset_per_auth_masks[a][j][jj]
                        label = 1
                        if pairid not in self.dict_pairs_per_author.keys():
                            self.dict_pairs_per_author[pairid] = []
                        self.dict_pairs_per_author[pairid].append([text1,mask1,text2,mask2,label])
                        pairid+=1
                        pos+=1
                        a2 = random.choice(list(self.dict_dataset_per_auth_ids.keys()))
                        while a==a2 and a2 in used.keys():
                            a2 = random.choice(list(self.dict_dataset_per_auth_ids.keys()))
                        i2 = random.choice(range(0,len(self.dict_dataset_per_auth_ids[a])))
                        ii2 = 0
                        text2 = self.dict_dataset_per_auth_ids[a2][i2][ii2]
                        mask2 =  self.dict_dataset_per_auth_masks[a2][i2][ii2]
                        label = 0
                        if pairid not in self.dict_pairs_per_author.keys():
                            self.dict_pairs_per_author[pairid] = []
                        self.dict_pairs_per_author[pairid].append([text1,mask1,text2,mask2,label])
                        pairid+=1
                        neg+=1
                        if a2 not in used.keys():
                            used[a2] = []

            used = {}
            while noOfTextsPos < 1:
                    i = random.choice(range(0,len(self.dict_dataset_per_auth_ids[a])))
                    idx = random.choice(range(0,len(self.dict_dataset_per_auth_ids[a][i])))
                    while idx==0:
                          idx = random.choice(range(0,len(self.dict_dataset_per_auth_ids[a][i])))
                    text1 = self.dict_dataset_per_auth_ids[a][i][idx]
                    mask1 =  self.dict_dataset_per_auth_masks[a][i][idx]
                    j = random.choice(range(0,len(self.dict_dataset_per_auth_ids[a])))
                    counter = 0
                    while i==j:
                        counter+=1
                        if counter == 5:
                            break
                        j = random.choice(range(0,len(self.dict_dataset_per_auth_ids[a])))
                    idx2 = random.choice(range(0,len(self.dict_dataset_per_auth_ids[a][j])))
                    while idx2==0:
                        idx2 = random.choice(range(0,len(self.dict_dataset_per_auth_ids[a][j])))

                    text2 = self.dict_dataset_per_auth_ids[a][j][idx2]
                    mask2 =  self.dict_dataset_per_auth_masks[a][j][idx2]
                    label = 1
                    if pairid not in self.dict_pairs_per_author.keys():
                        self.dict_pairs_per_author[pairid] = []
                    self.dict_pairs_per_author[pairid].append([text1,mask1,text2,mask2,label])
                    pairid+=1
                    pos+=1
                    noOfTextsPos += 1
            while noOfTextsNeg < 1:
                i = random.choice(range(0,len(self.dict_dataset_per_auth_ids[a])))
                idx = random.choice(range(0,len(self.dict_dataset_per_auth_ids[a][i])))
                while idx==0:
                    idx = random.choice(range(0,len(self.dict_dataset_per_auth_ids[a][i])))
                text1 = self.dict_dataset_per_auth_ids[a][i][idx]
                mask1 =  self.dict_dataset_per_auth_masks[a][i][idx]

                a2 = random.choice(list(self.dict_dataset_per_auth_ids.keys()))
                while a==a2 and a2 in used.keys():
                     a2 = random.choice(list(self.dict_dataset_per_auth_ids.keys()))
                i2 = random.choice(range(0,len(self.dict_dataset_per_auth_ids[a2])))
                ii2 = random.choice(range(0,len(self.dict_dataset_per_auth_ids[a2][i2])))
                while ii2==0:
                    ii2 = random.choice(range(0,len(self.dict_dataset_per_auth_ids[a2][i2])))
                text2 = self.dict_dataset_per_auth_ids[a2][i2][ii2]
                mask2 =  self.dict_dataset_per_auth_masks[a2][i2][ii2]
                label = 0
                if pairid not in self.dict_pairs_per_author.keys():
                    self.dict_pairs_per_author[pairid] = []
                self.dict_pairs_per_author[pairid].append([text1,mask1,text2,mask2,label])
                pairid+=1
                neg+=1
                if a2 not in used.keys():
                    used[a2] = []
                noOfTextsNeg+=1

        print(pos)
        print(neg)





    def create_anchor_batches(self):
        pairIDpos = 0
        pairIDneg = 0
        pairid = 0
        print("Create pos... ")
        for a in tqdm(self.dict_dataset_per_auth_ids.keys()):

            # for a2 in self.dict_dataset_per_auth_ids.keys():
            #     if a == a2:
                    cntf = 0
                    cntnf = 0
                    for f1 in self.dict_dataset_per_auth_ids[a].keys():
                        for i in range(0,len(self.dict_dataset_per_auth_ids[a][f1])):

                            for j in range(0,len(self.dict_dataset_per_auth_ids[a][f1])):
                                if j > i:

                                    for ii in range(0,len(self.dict_dataset_per_auth_ids[a][f1][i])):

                                        text1 = self.dict_dataset_per_auth_ids[a][f1][i][ii]
                                        mask1 =  self.dict_dataset_per_auth_masks[a][f1][i][ii]

                                        for jj in range(0,len(self.dict_dataset_per_auth_ids[a][f1][j])):

                                            if ii==jj:
                                                text2 = self.dict_dataset_per_auth_ids[a][f1][j][jj]
                                                mask2 =  self.dict_dataset_per_auth_masks[a][f1][j][jj]
                                                label = 1

                                                if cntf >= 5 and cntnf>=5:
                                                    break

                                                if pairid not in self.dict_pairs_per_author.keys():
                                                    self.dict_pairs_per_author[pairid] = []
                                                self.dict_pairs_per_author[pairid].append([text1,mask1,text2,mask2,label,f1,f1])
                                                cntf+=1
                                                pairid += 1
                                                pairIDpos+=1
                                                f2 = random.choice(list(self.dict_dataset_per_auth_ids[a].keys()))
                                                counter = 0
                                                while f1 == f2:
                                                    counter+=1
                                                    if counter == 100:
                                                        break
                                                    # print("stuck")
                                                    f2 = random.choice(list(self.dict_dataset_per_auth_ids[a].keys()))
                                                if f1!=f2:
                                                    j2 = random.choice(range(0,len(self.dict_dataset_per_auth_ids[a][f2])))
                                                    jj2 = random.choice(range(0,len(self.dict_dataset_per_auth_ids[a][f2][j2])))
                                                    text2 = self.dict_dataset_per_auth_ids[a][f2][j2][jj2]
                                                    mask2 =  self.dict_dataset_per_auth_masks[a][f2][j2][jj2]
                                                    label = 1
                                                    # if cntnf >= 3:
                                                    #     break
                                                    if pairid not in self.dict_pairs_per_author.keys():
                                                        self.dict_pairs_per_author[pairid] = []
                                                    self.dict_pairs_per_author[pairid].append([text1,mask1,text2,mask2,label,f1,f2])
                                                    cntnf+=1
                                                    pairid += 1
                                                    pairIDpos+=1

        print(pairIDpos)
        print("Create neg...")
        for a in tqdm(self.dict_dataset_per_auth_ids.keys()):
            cntf = 0
            cntnf = 0
            for a2 in self.dict_dataset_per_auth_ids.keys():
                if a != a2:

                    # for f1 in self.dict_dataset_per_auth_ids[a].keys():
                        # for f2 in self.dict_dataset_per_auth_ids[a2].keys():
                        # if f1==f2:
                        if cntf >= 4:
                            break
                        if pairIDneg>pairIDpos:
                            break
                        f1 = random.choice(list(self.dict_dataset_per_auth_ids[a].keys()))
                        i2= random.choice(range(0,len(self.dict_dataset_per_auth_ids[a][f1])))
                        ii2 = random.choice(range(0,len(self.dict_dataset_per_auth_ids[a][f1][i2])))
                        text1 = self.dict_dataset_per_auth_ids[a][f1][i2][ii2]
                        mask1 =  self.dict_dataset_per_auth_masks[a][f1][i2][ii2]
                        # if f1 in self.dict_dataset_per_auth_ids[a2].keys():
                        f2 = random.choice(list(self.dict_dataset_per_auth_ids[a2].keys()))
                        j2 = random.choice(range(0,len(self.dict_dataset_per_auth_ids[a2][f2])))
                        jj2 = random.choice(range(0,len(self.dict_dataset_per_auth_ids[a2][f2][j2])))
                        text2 = self.dict_dataset_per_auth_ids[a2][f2][j2][jj2]
                        mask2 =  self.dict_dataset_per_auth_masks[a2][f2][j2][jj2]
                        label = 0



                        if pairid not in self.dict_pairs_per_author.keys():
                                    self.dict_pairs_per_author[pairid] = []

                        self.dict_pairs_per_author[pairid].append([text1,mask1,text2,mask2,label,f1,f2])
                                # self.dict_pairs_per_author[pairid].append([text1,mask1,text2,mask2,label])
                        pairid += 1
                        pairIDneg +=1
                        cntf+=1
                        # else:
                        if cntnf >= 4:
                            break
                        f2 = random.choice(list(self.dict_dataset_per_auth_ids[a2].keys()))
                        counter = 0
                        while f1 != f2:
                            counter+=1
                            if counter == 100:
                                break
                            f2 = random.choice(list(self.dict_dataset_per_auth_ids[a2].keys()))
                        if f1==f2:
                                j2 = random.choice(range(0,len(self.dict_dataset_per_auth_ids[a2][f2])))
                                jj2 = random.choice(range(0,len(self.dict_dataset_per_auth_ids[a2][f2][j2])))
                                text2 = self.dict_dataset_per_auth_ids[a2][f2][j2][jj2]
                                mask2 =  self.dict_dataset_per_auth_masks[a2][f2][j2][jj2]
                                label = 0

                                if pairid not in self.dict_pairs_per_author.keys():
                                    self.dict_pairs_per_author[pairid] = []

                                self.dict_pairs_per_author[pairid].append([text1,mask1,text2,mask2,label,f1,f2])
                                # self.dict_pairs_per_author[pairid].append([text1,mask1,text2,mask2,label])
                                pairid += 1
                                pairIDneg +=1
                                cntnf+=1





datasets = ['PAN21_smallTest_for_demo'
    #'dict_perPairid_pan21_test'
            # 'dict_author_fandom_doc_val'
            ]


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

            if len(v2) >= 2:
                n = int(0.85 * len(v2))
                if k not in tmp_train.keys():
                    tmp_train[k] = {}
                if k not in tmp_val.keys():
                    tmp_val[k] = {}
                if k2 not in tmp_train[k].keys():
                    tmp_train[k][k2] = []
                if k2 not in tmp_val[k].keys():
                    tmp_val[k][k2] = []
                # print(v2)
                # print(len(v2[:n]))
                # print("=============================================")
                tmp_train[k][k2] = v2[:n]
                tmp_val[k][k2]=v2[n:]
            elif len(v2) == 1:
                if k not in tmp_train.keys():
                    tmp_train[k] = {}
                if k2 not in tmp_train[k].keys():
                    tmp_train[k][k2] = []
                tmp_train[k][k2]=v2



    return tmp_train,tmp_val
# def train_val_split(shuffledDict):
#
#
#     tmp_train = {}
#     tmp_val = {}
#     for k,v in shuffledDict.items():
#         print(v)
#         if len(v) >= 2:
#             n = int(0.85 * len(v))
#             if k not in tmp_train.keys():
#                 tmp_train[k] = {}
#                 if k not in tmp_val.keys():
#                     tmp_val[k] = {}
#
#                 tmp_train[k] = v[:n]
#                 tmp_val[k]=v[n:]
#         elif len(v) == 1:
#             if k not in tmp_train.keys():
#                 tmp_train[k] = {}
#             tmp_train[k] = v
#
#     return tmp_train,tmp_val



for dataset in datasets:


    ######
    # load
    ######
    with open( dataset, 'rb') as f:
        dict_dataset = pickle.load(f)
    print(len(list(dict_dataset.keys())))
    # shuffled_data = shuffledict(dict_dataset)
    # train,val = train_val_split(shuffled_data)
    ############
    # preprocess
    ############
    corpus = Corpus(dict_dataset=dict_dataset)
    corpus.parse_dictionary()
    # corpus.create_pairs_per_author_v2()#create_anchor_batches()#generatePairs()
    # corpus.generateLastDict()
    #######
    # store
    #######
    with open("PAN21_uncased_test_forDemo", 'wb') as f:
        pickle.dump(corpus.dict_dataset_per_auth_ids, f)
    # with open("PAN20_512_Notrunc_perPairid_onlyNums_cased_test_masks", 'wb') as f:
    #     pickle.dump(corpus.dict_dataset_per_auth_masks, f)
    # with open( "PAN20_neg_pairs_512_train_overlapping", 'wb') as f:
    #     pickle.dump(corpus.dict_negatives_pairs, f)
    # del corpus
    # corpusval = Corpus(dict_dataset=val)
    # corpusval.parse_dictionary()
    # # corpusval.create_pairs_per_author_v2()#generatePairs()
    # # # corpusval.generateLastDict()
    # #
    # with open( "PAN20_512_trunc_perAuth_onlyNums_cased_val_ids", 'wb') as f:
    #     pickle.dump(corpusval.dict_dataset_per_auth_ids, f)
    # with open( "PAN20_512_trunc_perAuth_onlyNums_cased_val_masks", 'wb') as f:
    #     pickle.dump(corpusval.dict_dataset_per_auth_masks, f)
    # # with open( "PAN20_neg_pairs_512_val_overlapping", 'wb') as f:
    # #     pickle.dump(corpusval.dict_negatives_pairs, f)
    # # with open( dataset + "_processed_texts_test", 'wb') as f:
    # #     pickle.dump(corpus.dict_dataset_per_auth_ids, f)
    # del corpusval

# import numpy as np
# class AuthorshipDataset(torch.utils.data.Dataset):
#     """Dataset for Author Verification on the IMDB62 Dataset."""
#
#     def __init__(self,
#                  dict_per_auth,
#                  base_rate: float = 0.5
#                  ):
#         """
#         Args:
#             data_file (string): the path to the IMDB62 Dataset txt file
#         """
#         # get the dataset, then break it up into dict key'd on authors with values a list of texts.
#         self.per_author_dataset = dict_per_auth
#         self.base_rate = base_rate
#
#     def __len__(self):
#         return sum([len(x) for x in self.per_author_dataset.values()])
#
#     def __getitem__(self, idx):
#         # we want this to work with contrastive, so sample on the author level
#         auth1 = random.choice(list(self.per_author_dataset.keys()))
#
#         if np.random.uniform() < self.base_rate:
#             # this is a same_author sample
#             # make sure the auth has multiple samples
#             while len(self.per_author_dataset[auth1]) < 2:
#                 auth1 = random.choice(list(self.per_author_dataset.keys()))
#
#             text1 = text2 = random.choice(self.per_author_dataset[auth1])[0]
#
#             mask1 = mask2 = random.choice(self.per_author_dataset[auth1])[1]
#             # make sure the texts are different
#             counter = 0
#             im_confused_counter = 0
#             auths_tried = 0
#             while torch.equal(text1,text2):
#                 text2 = random.choice(self.per_author_dataset[auth1])[0]
#                 mask2 = random.choice(self.per_author_dataset[auth1])[1]
#                 counter += 1
#                 if counter > 100:
#                     # these texts are the same, get a different author
#                     while len(self.per_author_dataset[auth1]) < 2:
#                         auth1 = random.choice(list(self.per_author_dataset.keys()))
#                     auths_tried += 1
#                     text1 = text2 = random.choice(self.per_author_dataset[auth1])[0]
#                     mask1 = mask2 = random.choice(self.per_author_dataset[auth1])[1]
#                     if auths_tried > 50:
#                         assert False, "we've got problems, can't find a different text from same author"
#                     counter = 0
#                 if im_confused_counter > 10000:
#                     print(auth1)
#                     print(text1)
#                     assert False, "we've got problems, stuck in this same-author loop again."
#                 im_confused_counter += 1
#             auth2 = auth1
#             label = 1
#
#         else:
#             # this is a different author sample
#             auth2 = auth1
#             while auth1 == auth2:
#                 auth2 = random.choice(list(self.per_author_dataset.keys()))
#             # now get a text from both authors
#             text1 = random.choice(self.per_author_dataset[auth1])[0]
#             text2 = random.choice(self.per_author_dataset[auth2])[0]
#             mask1 = random.choice(self.per_author_dataset[auth1])[1]
#             mask2 = random.choice(self.per_author_dataset[auth2])[1]
#             label = 0
#         return text1,mask1,text2,mask2,label,auth1,auth2
#
#
#
# dataset_train = AuthorshipDataset(corpus.dict_pairs_per_author)
# train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=32,shuffle=False)
#
# for text1,mask1, text2,mask2,label,a1,a2 in train_dataloader:
#     print(a1)
#     print(text1.shape)
#     print("==============")
#     print(a2)
#     print(text2.shape)
#     print(label.shape)
#     break
# class MyDatasetTest(torch.utils.data.Dataset):
#     def __init__(self,
#                  data_pos,
#
#                  base_rate: float = 0.5
#                  ):
#
#         # get the dataset, then break it up into dict key'd on authors with values a list of chunks.
#         self.per_pair_dataset = data_pos
#
#
#
#         # self.per_author_dataset_masks = data_masks
#         self.base_rate = base_rate
#
#         #self.per_author_dataset_ids,self.per_author_dataset_masks = shuffleAll(self.tmp_per_author_dataset_ids,self.tmp_per_author_dataset_masks)
#         #del self.tmp_per_author_dataset_ids, self.tmp_per_author_dataset_masks
#         # for x in self.dict_All_pairs:
#         #     print(x)
#
#
#
#     def __len__(self):
#
#         # return sum([len(self.per_pair_dataset[x]) for x in self.per_pair_dataset.keys()])
#         return len(list(self.per_pair_dataset.keys()))#+len(list(self.per_pair_dataset2.keys()))
#
#     def __getitem__(self, idx):
#         id = random.choice(list(self.per_pair_dataset.keys()))
#         x = self.per_pair_dataset[id]
#
#         batchText1 = []
#         batchText2 = []
#         batchMask1 = []
#         batchMask2 = []
#         labels = []
#         print(len(x))
#         for item in x:
#             # print(item)
#             batchText1.append(item[0])
#             batchMask1.append(item[1])
#             batchText2.append(item[2])
#             batchMask2.append(item[3])
#
#             labels.append(item[4])
#
#         return torch.stack(batchText1), torch.stack(batchMask1), torch.stack(batchText2), torch.stack(batchMask2), torch.LongTensor(labels)
# dataset_test = MyDatasetTest(dict_dataset,0.5)
# test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=1,shuffle=False)
# print(len(test_dataloader))
#
# for text1,mask1, text2,mask2, label in test_dataloader:
#     print(text1)
#     print("====")
#     print(text2)
#     print(label)
    # break