import json



##################
# Class for corpus
##################
class Corpus(object):

    def __init__(self):

        # dictionary with all documents
        self.dict_per_pairID = {}

    def parse_raw_data(self,dir_pairs,dir_labels=None):
        # open json files
        with open(dir_pairs, 'r', encoding='utf-8') as f:
            lines_pairs = f.readlines()
        # with open(dir_labels, 'r', encoding='utf-8') as f:
        #     lines_labels = f.readlines()

        for n in range(len(lines_pairs)):
            pair, label = json.loads(lines_pairs[n].strip())#, json.loads(lines_labels[n].strip())
            pairId = pair['id']
            # labelId = label['id']
            # author1 = label['authors'][0]
            # author2 = label['authors'][1]
            type1 = pair['discourse_types'][0]
            type2 = pair['discourse_types'][1]
            doc1 = pair['pair'][0]
            doc2 = pair['pair'][1]
            # L = int(label['same'])
            # if pairId == labelId:
            if pairId not in self.dict_per_pairID.keys():
                  self.dict_per_pairID[pairId] = []
            self.dict_per_pairID[pairId].append((doc1,doc2,type1,type2))
