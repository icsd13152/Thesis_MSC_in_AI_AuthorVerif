import helpers
import random
from transformers import BertTokenizer

class CorpusAV(object):

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.proc_dict_per_pairId = {}
    
    def read_and_process(self,dict_dataset):
        for id in dict_dataset.keys():
            for i, docs in enumerate(dict_dataset[id]):
                txtlen1 = len(docs[0])
                txtlen2 = len(docs[1])
                processed_doc1 = helpers.preprocess_doc(docs[0])
                processed_doc2 = helpers.preprocess_doc(docs[1])
                inputIDs1,attn_masks1 = helpers.chunkingTextsBasedOnBert(self.tokenizer,processed_doc1,txtlen1)
                inputIDs2,attn_masks2 = helpers.chunkingTextsBasedOnBert(self.tokenizer,processed_doc2,txtlen2)
                # label = docs[2]
                type1 = docs[2]
                type2 = docs[3]

                if len(inputIDs1) == len(inputIDs2):
                    for i in range(0,len(inputIDs1)):

                        text1 = inputIDs1[i]
                        mask1 = attn_masks1[i]
                        text2 = inputIDs2[i]
                        mask2 = attn_masks2[i]
                        if id not in self.proc_dict_per_pairId.keys():
                            self.proc_dict_per_pairId[id] = []
                        self.proc_dict_per_pairId[id].append([text1,mask1,text2,mask2,type1,type2])
                elif len(inputIDs1) > len(inputIDs2):
                    for i in range(0,len(inputIDs1)):
                        if i < len(inputIDs2):
                            text1 = inputIDs1[i]
                            mask1 = attn_masks1[i]
                            text2 = inputIDs2[i]
                            mask2 = attn_masks2[i]
                            if id not in self.proc_dict_per_pairId.keys():
                                self.proc_dict_per_pairId[id] = []
                            self.proc_dict_per_pairId[id].append([text1,mask1,text2,mask2,type1,type2])
                        else:
                            text1 = inputIDs1[i]
                            mask1 = attn_masks1[i]
                            j = random.choice(range(0,len(inputIDs2)))
                            text2 = inputIDs2[j]
                            mask2 = attn_masks2[j]
                            if id not in self.proc_dict_per_pairId.keys():
                                self.proc_dict_per_pairId[id] = []
                            self.proc_dict_per_pairId[id].append([text1,mask1,text2,mask2,type1,type2])
                elif len(inputIDs1) < len(inputIDs2):
                    for j in range(0,len(inputIDs2)):
                        if j < len(inputIDs1):
                            text1 = inputIDs1[j]
                            mask1 = attn_masks1[j]
                            text2 = inputIDs2[j]
                            mask2 = attn_masks2[j]
                            if id not in self.proc_dict_per_pairId.keys():
                                self.proc_dict_per_pairId[id] = []
                            self.proc_dict_per_pairId[id].append([text1,mask1,text2,mask2,type1,type2])
                        else:
                            text2 = inputIDs2[j]
                            mask2 = attn_masks2[j]
                            i = random.choice(range(0,len(inputIDs1)))
                            text1 = inputIDs1[i]
                            mask1 = attn_masks1[i]
                            if id not in self.proc_dict_per_pairId.keys():
                                self.proc_dict_per_pairId[id] = []
                            self.proc_dict_per_pairId[id].append([text1,mask1,text2,mask2,type1,type2])


