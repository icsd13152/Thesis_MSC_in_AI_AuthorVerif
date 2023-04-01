import os
import nltk
import spacy
import pickle
import re
import torch
from torch.nn import functional as F
from transformers import BertTokenizerFast, BertTokenizer,RobertaTokenizer,T5TokenizerFast
import emojis
import demoji
import random
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import unicodedata
from textblob import TextBlob


class Corpus(object):

    def __init__(self, dict_dataset):

        # define tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.tokenizer = T5TokenizerFast.from_pretrained('t5-base')
        # self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.dict_dataset_raw = dict_dataset
        self.maxSeq = set()
        self.dict_dataset_per_auth_test= {}

    ####################
    # doc pre-processing
    ####################
    def removeHTMLtags(self,text):


        x = re.sub('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', ' ',text)
        x2 = re.sub('\s\s', ' ',x)
        x3 = re.sub('\t', ' ',x2)
        return x3.strip()
    def maskNumbers(self,text, symbol='1'):
        x = re.sub('[0-9]', symbol,text)
        return x

    def maskEmoticons(self,text, symbol='0 '):
        new_list1 = emojis.get(text)
        xx = ''
        for x in new_list1:

            xx = text.replace(x,' 0')

        dem = demoji.findall(xx)
        for item in dem.keys():
            xx = xx.replace(item, ' 0')

        if xx=='':
            return text
        else:
            return xx

    def paddingChunksToMaxlen(self,IdsChunks,masksChunks,txtlen,maxLen = 254):#listMasksChunksmasksChunks=None,
        listIdsChunks = list()
        listMasksChunks = list()
        # listmasks = []
        # for i in range(len(IdsChunks)):
        #     tmplistmasks = []
        #     for j in IdsChunks[i]:
        #         tmplistmasks.append(1)
        #     listmasks.append(torch.LongTensor(tmplistmasks))
        totalLen = 375
        for i in range(len(IdsChunks)):

            pad_len = maxLen  - IdsChunks[i].shape[0]

            tmplistids = IdsChunks[i].tolist()
            if len(tmplistids) < maxLen:
                if i > 0: #from second element
                    temp = IdsChunks[i-1].tolist()
                    lenToadd = maxLen - len(tmplistids)

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
            ids_with_pos = self.addPOSTAGS(tmplistids,txtlen)
            tmplistmasks = [1]*len(ids_with_pos)
            pad_len_final = totalLen  - len(ids_with_pos)
            self.maxSeq.add(len(ids_with_pos))
            # tmplistmasks.insert(0,1)
            # tmplistmasks.insert(len(tmplistmasks),1)
            # print(tmplistids)
            # if len(tmplistids) > 128:
            # listIdsChunks.append(torch.LongTensor(tmplistids))
            listIdsChunks.append(torch.LongTensor(ids_with_pos))
            listMasksChunks.append(torch.LongTensor(tmplistmasks))
            del tmplistmasks, tmplistids
            if pad_len_final > 0:

                listIdsChunks[i] =  F.pad(listIdsChunks[i], (0,pad_len_final), "constant", 0) # 0 for bert
                listMasksChunks[i] =  F.pad(listMasksChunks[i], (0,pad_len_final), "constant", 0)
                # print(listIdsChunks[i].shape[0])
            # if pad_len > 0:
            #
            #     listIdsChunks[i] =  F.pad(listIdsChunks[i], (0,pad_len), "constant", 0) # 0 for bert
            #     listMasksChunks[i] =  F.pad(listMasksChunks[i], (0,pad_len), "constant", 0)

        del IdsChunks, pad_len
        # gc.collect()
        return listIdsChunks ,listMasksChunks


    def preprocess_doc(self, doc):
        doc = self.maskNumbers(doc)
        doc = doc.replace('0',' 1 ')
        doc = doc.replace('<nl>',' ') #3
        # doc = re.sub('<.*?>',' ',doc)
        ptrn = re.compile(r'<pers[1-9]*_[A-W]+>|<pers[1-9]*_[A-W]+_[A-W]+>')
        doc = re.sub(ptrn, 'David', doc)
        ptrn2 = re.compile(r'<addr[1-9]*_[A-Z]*>')
        doc = re.sub(ptrn2, 'Mexico', doc)
        ptrn3 = re.compile(r'<city[1-9]*>')
        doc = re.sub(ptrn3, 'Mexico', doc)

        ptrn4 = re.compile(r'<country[1-9]*>')
        doc = re.sub(ptrn4, 'Mexico', doc)

        ptrn5 = re.compile(r'<job_title[1-9]*>')
        doc = re.sub(ptrn5, ' ', doc)

        ptrn6 = re.compile(r'<organisation[1-9]*>')
        doc = re.sub(ptrn6, ' ', doc)

        ptrn7 = re.compile(r'<university[1-9]*>')
        doc = re.sub(ptrn7, ' ', doc)

        doc = doc.replace('<new>',' ')

        doc = doc.replace('*',' ')

        # doc = re.sub('<.*?>',' ',doc)
        doc = re.sub('<','',doc)
        doc = self.removeHTMLtags(doc)
        doc = self.maskEmoticons(doc,' 0 ')
        pater = re.compile(r'[lL][oO]+[lL]')
        doc = re.sub(pater, 'lol', doc)
        doc = unicodedata.normalize('NFKD', doc).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        doc = doc.replace('dr1_FN>',' ')
        doc = doc.replace('part_FN_MN','')
        doc = doc.replace('dr1_NN>',' ')
        pat = re.compile(r'addr[1]{2}_FN')
        doc = re.sub(pat, '', doc)
        # doc = doc.replace('_FN',' ')
        doc = doc.replace('new>',' ')
        doc = doc.replace('dule_code>',' ')
        doc = doc.replace('stem>',' ')
        # pattern = re.compile(r'<\w+>')
        # doc = re.sub(pattern, ' ', doc)
        doc = doc.replace('Â','')
        doc = doc.replace('CafÃ©:','')
        doc = doc.replace('^','')
        doc = doc.replace('Re :','')
        doc = doc.replace('@','')
        doc = doc.replace('()','')
        doc = doc.replace(')))))',' 0 ')
        doc = doc.replace(') ) ) ) )',' ')
        doc = doc.replace(':)',' 0 ')
        doc = doc.replace(': )',' 0 ')
        doc = doc.replace(':((',' 0 ')
        doc = doc.replace(':(',' 0 ')
        doc = doc.replace(': (',' 0 ')
        doc = doc.replace('; )',' 0 ')
        doc = doc.replace(';)',' 0 ')
        doc = doc.replace(': )',' 0 ')
        doc = doc.replace('�',' ')
        doc = doc.replace('( )','')
        doc = doc.replace('\' \'','')
        doc = doc.replace('. .','.')
        doc = doc.replace(', , , ,',',')
        doc = doc.replace('-------------------------------------------------------------------------------','')
        doc = doc.replace('-----------------------------------------------------------------------------------------','')
        doc = doc.replace('Full Name: DOB:DDMMYYYYEmail:','')
        doc = doc.replace('Full Name: DOB: &it;DDMMYYYY&it; Email:','')
        doc = doc.replace('111111 , 11:11:11 ] Raveen :','')
        doc = doc.replace('[ 11111111 , 11:11:11 ] Raveen :','')
        doc = doc.replace('[ 11111111 , 11:11:11 ]','')
        doc = doc.replace('Full Name : DOB :  lt ; DDMMYYYY  gt ; Email :','')
        doc = doc.replace('lt ; DDMMYYYY  gt ;','')
        doc = doc.replace('Lmfaoooo','2')
        doc = doc.replace('Full Name : DOB :  lt ; DDMMYYYY  gt ; Email :','')
        doc = doc.replace('DOB :  lt ; DDMMYYYY  gt ;','')
        doc = doc.replace('Raveen :','')

        doc = doc.replace('lol0','2 0')
        doc = doc.replace('Age :','')
        doc = doc.replace('DOB :','')
        doc = doc.replace('RE :','')
        # doc = doc.replace('ffs. rah .','')
        # doc = doc.replace('‘’ ’’','')
        pat = re.compile(r'Full Name:\s+DOB:\s+DDMMYYYY\s+Email:')
        doc = re.sub(pat, ' ', doc)
        pattern = re.compile(r'\s+\s+')

        doc = re.sub(pattern, ' ', doc)
        doc = doc.replace('. .','')
        doc = doc.replace('>','')
        doc = doc.replace('dr1_NN','')
        doc = doc.replace('Idio111_1','')
        doc = doc.replace('&','')
        doc = doc.replace("lt;DDMMYYYYgt;",'Date')
        doc = doc.replace("lt ; DDMMYYYY  gt ;",'Date')
        doc = doc.replace("lt ; DDMMYY  gt",'Date')
        doc = doc.replace("lt ; DDMMYYYY  gt",'Date')
        doc = doc.replace("lt ; DDMMYYYY  gt ;",'Date')
        doc = doc.replace("lt ; DDMM.YY  gt",'Date')

        # print("="*10)
        # print(doc.strip())
        # print("="*10)

        return doc.strip()


    def chunkingTextsBasedOnBert(self,text_tokens,txtlen):

        encoded_dict = self.tokenizer.encode_plus(
            text_tokens ,                    # Sentence to encode.
            add_special_tokens = False, # Add '[CLS]' and '[SEP]'
            # pad_token = None,
            # add_prefix_space = False,
            # max_length = 512,           # Pad & truncate all sentences.
            # pad_to_max_length = True,
            # truncation = False,
            return_attention_mask = True,   # Construct attn. masks.
            return_tensors = 'pt'
            # is_split_into_words=True
        )

        tensorsIdList1,tensorsMaskList1 = self.paddingChunksToMaxlen(encoded_dict['input_ids'][0].split(126),encoded_dict['attention_mask'][0].split(126),txtlen,126)

        return tensorsIdList1, tensorsMaskList1

    def calcLenOfText(self,text):
        return len(text)

    def addPOSTAGS(self,text,txtlen):

        tokens = self.tokenizer.convert_ids_to_tokens(text)
        original_tokens = []
        for i, token in enumerate(tokens):
            if token.startswith("##"):
                original_tokens[-1] += token[2:]
            else:
                original_tokens.append(token)
        # tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        pos = []
        for i in pos_tags:
            pos.append(i[1])

        sequence = original_tokens + pos + ['[SEP]'] + [str(txtlen)] + ['[SEP]']
        sent = ' '.join(t for t in sequence)

        input_ids = self.tokenizer.encode_plus(
            sent ,                    # Sentence to encode.
            add_special_tokens = False, # Add '[CLS]' and '[SEP]'
            # pad_token = None,
            # add_prefix_space = False,
            # max_length = 512,           # Pad & truncate all sentences.
            # pad_to_max_length = True,
            # truncation = False,
            return_attention_mask = True   # Construct attn. masks.
            # return_tensors = 'pt'
            # is_split_into_words=True
        )
        # print(input_ids['input_ids'])
        return input_ids['input_ids']
    def parseDict_per_pairid_test(self):

        for a in tqdm(self.dict_dataset_raw.keys()):
            listDocsPerAuthor = []
            listMasksPerAuthor = []
            for i, docs in enumerate(self.dict_dataset_raw[a]):
                txtlen1 = self.calcLenOfText(docs[0])
                txtlen2 = self.calcLenOfText(docs[1])
                processed_doc1 = self.preprocess_doc(docs[0])
                    # processed_doc1 = self.addPOSTAGS(processed_doc1)
                processed_doc2 = self.preprocess_doc(docs[1])
                    # processed_doc2 = self.addPOSTAGS(processed_doc2)
                inputIDs1,attn_masks1 = self.chunkingTextsBasedOnBert(processed_doc1,txtlen1)
                inputIDs2,attn_masks2 = self.chunkingTextsBasedOnBert(processed_doc2,txtlen2)
                label = docs[2]
                type1 = docs[3]
                type2 = docs[4]
                # print(type1)
                # print(type2)
                # if (type1 == 'essay' and type2 == 'essay') or (type2 == 'essay' and type1 == 'essay'):
                if (type1 == 'email' and type2 == 'essay') or (type1 == 'essay' and type2 == 'email') or (type1 == 'essay' and type2 == 'essay')  or (type1 == 'email' and type2 == 'email'):

                    if len(inputIDs1) == len(inputIDs2):
                        for i in range(0,len(inputIDs1)):

                            text1 = inputIDs1[i]
                            mask1 = attn_masks1[i]
                            text2 = inputIDs2[i]
                            mask2 = attn_masks2[i]
                            if a not in self.dict_dataset_per_auth_test.keys():
                                self.dict_dataset_per_auth_test[a] = []
                            self.dict_dataset_per_auth_test[a].append([text1,mask1,text2,mask2,label,type1,type2])
                    elif len(inputIDs1) > len(inputIDs2):
                        for i in range(0,len(inputIDs1)):
                            if i < len(inputIDs2):
                                text1 = inputIDs1[i]
                                mask1 = attn_masks1[i]
                                text2 = inputIDs2[i]
                                mask2 = attn_masks2[i]
                                if a not in self.dict_dataset_per_auth_test.keys():
                                    self.dict_dataset_per_auth_test[a] = []
                                self.dict_dataset_per_auth_test[a].append([text1,mask1,text2,mask2,label,type1,type2])
                            else:
                                text1 = inputIDs1[i]
                                mask1 = attn_masks1[i]
                                j = random.choice(range(0,len(inputIDs2)))
                                text2 = inputIDs2[j]
                                mask2 = attn_masks2[j]
                                if a not in self.dict_dataset_per_auth_test.keys():
                                    self.dict_dataset_per_auth_test[a] = []
                                self.dict_dataset_per_auth_test[a].append([text1,mask1,text2,mask2,label,type1,type2])
                    elif len(inputIDs1) < len(inputIDs2):
                        for j in range(0,len(inputIDs2)):
                            if j < len(inputIDs1):
                                text1 = inputIDs1[j]
                                mask1 = attn_masks1[j]
                                text2 = inputIDs2[j]
                                mask2 = attn_masks2[j]
                                if a not in self.dict_dataset_per_auth_test.keys():
                                    self.dict_dataset_per_auth_test[a] = []
                                self.dict_dataset_per_auth_test[a].append([text1,mask1,text2,mask2,label,type1,type2])
                            else:
                                text2 = inputIDs2[j]
                                mask2 = attn_masks2[j]
                                i = random.choice(range(0,len(inputIDs1)))
                                text1 = inputIDs1[i]
                                mask1 = attn_masks1[i]
                                if a not in self.dict_dataset_per_auth_test.keys():
                                    self.dict_dataset_per_auth_test[a] = []
                                self.dict_dataset_per_auth_test[a].append([text1,mask1,text2,mask2,label,type1,type2])



def shuffledict(big_dict):
    keys = list(big_dict.keys())
    random.shuffle(keys)
    Shuffled = dict()
    for key in keys:
        if key not in Shuffled.keys():
            Shuffled[key] = {}
        Shuffled[key]= big_dict[key]

    return Shuffled





dataset = 'dict_auth_type_pan23_test'
#
with open( dataset, 'rb') as f:
    dict_dataset = pickle.load(f)
#
# for k in dict_dataset.keys():
#     lab = dict_dataset[k]
#     print(lab)

shuffled_data = shuffledict(dict_dataset)
corpus = Corpus(dict_dataset=shuffled_data)
corpus.parseDict_per_pairid_test()
def MAX(sets):
    return (max(sets))

# Driver Code

# Python code to get the minimum element from a set
def MIN(sets):
    return (min(sets))

# Driver Code
print(MAX(corpus.maxSeq))
print(MIN(corpus.maxSeq))
print(corpus.maxSeq)
with open("PAN23_370_overlap_uncased_test_openset-posLen_concat-email-essay", 'wb') as f:
    pickle.dump(corpus.dict_dataset_per_auth_test, f)
# del corpus