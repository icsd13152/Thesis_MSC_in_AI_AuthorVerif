import random
import torch
import data_processing

class MyDatasetTest(torch.utils.data.Dataset):
    def __init__(self,
                 data_dict
                 ):
        self.per_pair_dataset = data_dict
        self.used = {}
    def __len__(self):
        return len(list(self.per_pair_dataset.keys()))

    def __getitem__(self, idx):
        id = random.choice(list(self.per_pair_dataset.keys()))
        x = self.per_pair_dataset[id]
        batchText1 = []
        batchText2 = []
        batchMask1 = []
        batchMask2 = []
        labels = []
        type1 = None
        type2 = None

        for item in x:

            batchText1.append(item[0])
            batchMask1.append(item[1])
            batchText2.append(item[2])
            batchMask2.append(item[3])
            # labels.append(item[4])
            type1 = item[4]
            type2 = item[5]

        return torch.stack(batchText1), torch.stack(batchMask1), torch.stack(batchText2), torch.stack(batchMask2),id,type1,type2

def getDataLoader(dict_orig_data):
    corpus = data_processing.CorpusAV()
    dict_per_pair = corpus.read_and_process(dict_orig_data)
    dataset = MyDatasetTest(dict_per_pair)
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=False)
    del corpus
    return test_dataloader

