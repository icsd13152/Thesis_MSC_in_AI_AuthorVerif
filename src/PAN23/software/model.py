import torch
import torch.nn as nn
from transformers import BertModel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class myModelEmbeddings(nn.Module):
    def __init__(self,bert_emb_layer,startLayer,endLayer,groupLayersMode = (False,False)):#(True,True)-> Grouping and Summing | #(True,False)-> Grouping and Concat
        super(myModelEmbeddings, self).__init__()
        self.bert_emb_layer = bert_emb_layer
        self.startLayer = startLayer
        self.endLayer = endLayer
        self.groupLayersMode = groupLayersMode
        self.bertModel = BertModel.from_pretrained('bert-base-uncased',output_hidden_states = True)


        inputFeatures = 0
        if self.groupLayersMode == (True,False):
            inputFeatures = (endLayer - startLayer)*768
        elif self.groupLayersMode == (True,True):
            inputFeatures = 768
        else:
            inputFeatures = 768
        self.bilstm = nn.LSTM(input_size=768, hidden_size=768,batch_first=True,bidirectional=True)#num_layers=3,dropout=0.2,

    def getSpecificLayerOfBERT(self,bertOutputs):
        hidden_states = bertOutputs[2][1:]
        layerOutput = hidden_states[self.bert_emb_layer] # get specific Layer (from 0 to 11) for all tuples (batch_size, sequence_length, hidden_size)

        return  layerOutput

    def concatSpecificLayersOfBERT(self,bertOutputs):
        hidden_states = bertOutputs[2][0:]
        concatEmbeddingLayers = torch.cat([hidden_states[i] for i in range(self.startLayer,self.endLayer)], dim=-1)

        return concatEmbeddingLayers
    def getCLSEmbeddings(self,bertOutputs ):
        embeddings = bertOutputs[0] #last hidden states
        #embeddings = bertOutputs[1] # pooler
        return embeddings
    def getCLSEmbeddingsFromLayers(self,bertOutputs ):
        hidden_states = bertOutputs[2][0:]


        # Extract the hidden state for the [CLS] token from last four encode layers
        last_layer_hidden_states = hidden_states[2:13]
        cls = []
        for layer in last_layer_hidden_states:
            cls.append(layer[:,0,:])
        cls_embeddings = torch.stack(cls, dim=1)
        del cls

        return cls_embeddings
    def sumSpecificLayersOfBERT(self,bertOutputs):
        #Number of layers: 13   (initial embeddings + 12 BERT layers) - So we need [2][1:] 1 and onwards
        hidden_states = bertOutputs[2][0:]
        # `hidden_states` is a Python list.

        # sumEmbeddingLayers = torch.stack(hidden_states[self.startLayer:self.endLayer]).sum(0)
        sumEmbeddingLayers = torch.stack(hidden_states[-4:]).sum(0)
        # sumEmbeddingLayers = torch.stack(hidden_states[-4:]).mean(dim=0)
        del hidden_states

        return sumEmbeddingLayers
    def pooling(self,token_embeddings, mask, strategy='avg'):
        if strategy == 'max':
            #  avg_setence_embeddings = torch.mean(token_embeddings,dim=1)
            #  print(avg_setence_embeddings.shape)
            input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_setence_embeddings = torch.max(token_embeddings, 1)[0]
            return max_setence_embeddings
        elif strategy == 'avg':
            in_mask = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            # perform mean-pooling but exclude padding tokens (specified by in_mask)
            avg_setence_embeddings = torch.sum(token_embeddings * in_mask, 1) / torch.clamp(in_mask.sum(1), min=1e-9)
            return avg_setence_embeddings
        elif strategy == 'sum':
            sum_setence_embeddings = torch.sum(token_embeddings[0:len(token_embeddings)],1)
            return sum_setence_embeddings

    def forwardOnce(self, sent_id, mask):
        outputs =  self.bertModel(input_ids=sent_id, attention_mask=mask)#,decoder_input_ids=sent_id

        if self.groupLayersMode == (True,False):
            embeddings = self.concatSpecificLayersOfBERT(outputs)
            return  embeddings
        elif self.groupLayersMode == (True,True):
            embeddings = self.sumSpecificLayersOfBERT(outputs)
            # embeddings = self.getCLSEmbeddingsFromLayers(outputs)
            return embeddings
        else:
            # embeddings = self.getSpecificLayerOfBERT(outputs)
            # embeddings = self.getCLSEmbeddings(outputs )
            embeddings = self.getCLSEmbeddingsFromLayers(outputs)
            return embeddings

    def init_hidden(self, batch_size):
        #Initialization of the LSTM hidden and cell states
        h0 = torch.zeros((2*1, batch_size, 768)).detach().to(device)
        c0 = torch.zeros((2*1, batch_size, 768)).detach().to(device)
        hidden = (h0, c0)
        return hidden
    def forward(self, sent_id1, mask1,hidden):

        # forward pass of input 1
        output1 = self.forwardOnce(sent_id1, mask1)
        out1, (hidden1,cell1) = self.bilstm(output1,hidden)
        out_split1 = out1.view(sent_id1.shape[0], 11, 2, 768)
        out_forward1 = out_split1[:, :, 0, :]
        out_backward1 = out_split1[:, :, 1, :]
        batch_indices = torch.arange(0, sent_id1.shape[0], device=device)
        seq_indices = 11 - 1
        direction_full1 = torch.cat([out_split1[batch_indices, seq_indices, 0], out_split1[batch_indices, 0, 1]], dim=-1)
        return direction_full1#F.normalize(direction_full1)

def getModels(models_path):
    modelEmbEE = myModelEmbeddings(bert_emb_layer=10,startLayer=6,endLayer=10)
    modelEmbEE.load_state_dict(torch.load(models_path+'/'+'checkpointEmbUncased_PAN23_essaiEmail_v2.pt'))

    modelEmbSI = myModelEmbeddings(bert_emb_layer=10,startLayer=6,endLayer=10)
    modelEmbSI.load_state_dict(torch.load(models_path+'/'+'checkpointEmbUncased_PAN23_speech-interview.pt'))

    modelEmbGen = myModelEmbeddings(bert_emb_layer=10,startLayer=6,endLayer=10)
    modelEmbGen.load_state_dict(torch.load(models_path+'/'+'checkpointEmbUncased_PAN23_gen.pt'))

    return modelEmbEE, modelEmbSI, modelEmbGen