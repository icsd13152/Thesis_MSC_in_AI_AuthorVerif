import helpers
import torch
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ContrastiveChunkerEvaluator:

    def __init__(self, test_dataset,
                 thresholdEE: float = 0.5, thresholdSI: float = 0.5,
                 thresholdEE2: float = 0.5, thresholdSI2: float = 0.5,
                 thresholdGen: float = 0.5, outputPath = "answers.jsonl",find_threshold: bool = False):
        self.test_dataset = test_dataset
        self.distance_metric = torch.nn.CosineSimilarity()

        self.thresholdEE = thresholdEE
        self.thresholdSI = thresholdSI
        self.thresholdEE2 = thresholdEE2
        self.thresholdSI2 = thresholdSI2
        self.thresholdGen = thresholdGen
        self.find_threshold = find_threshold
        self.output = outputPath

    def ensemble(self,predEE2,predSI2,predGen):
        ensemble_preds = np.column_stack((predEE2.reshape(-1,1), predSI2.reshape(-1,1), predGen.reshape(-1,1)))
        final_pred = np.round(np.mean(ensemble_preds, axis=1))
        return final_pred
    def ensembleProbas(self,simEE2,simSI2,simGen):
        ensemble_preds = np.column_stack((simEE2.reshape(-1,1), simSI2.reshape(-1,1), simGen.reshape(-1,1)))
        final_pred_probas = np.mean(ensemble_preds, axis=1)
        return final_pred_probas
    def getProbasFromSimilarity(self,similarities,threshold):
        similarities = (1 + similarities) / 2
        fixed_sims = similarities.copy()
        # need to make sure this is < 0.5 - just normalize to [0, 0.5)

        fixed_sims[np.logical_and(similarities < threshold, similarities > 0.5)] = 1-fixed_sims[np.logical_and(similarities < threshold, similarities > 0.5)]

        # need to make sure samples above high thresh is normalized to be in (0.5, 1]

        fixed_sims[similarities > threshold+0.05] = fixed_sims[similarities > threshold+0.05]
        fixed_sims[np.logical_and(similarities >= threshold, similarities <= threshold+0.05)] = 0.5
        return (1 + similarities) / 2

    def call(self, modelEssay_Email, modelSpeech_interv,modelGeneral):
        modelEssay_Email.to(device)
        modelSpeech_interv.to(device)
        modelGeneral.to(device)
        modelEssay_Email.eval()
        modelSpeech_interv.eval()
        modelGeneral.eval()
        print('evaluating...')

        # truthEE,truthSI,truthGen = [],[],[]
        embEE1, embEE2 = [], []
        embSI1, embSI2 = [], []
        embGen1, embGen2 = [], []
        embEE12, embEE22 = [], []
        embSI12, embSI22 = [], []
        ids1,ids2,ids3 = {},{},{}
        with torch.no_grad():
            for input1, mask1, input2, mask2,target1,id,type1,type2 in self.test_dataset:
                b_input_ids1 = input1[0].to(device)
                b_input_mask1 = mask1[0].to(device)
                label = target1[0].type(torch.LongTensor)
                b_input_ids2 = input2[0].to(device)
                b_input_mask2 = mask2[0].to(device)
                b_labels = label.to(device)
                if (type1[0] == 'essay' and type2[0] == 'email') or (type1[0] == 'email' and type2[0] == 'essay') or (type1[0] == 'essay' and type2[0] == 'essay')  or (type1[0] == 'email' and type2[0] == 'email'):
                    h = modelEssay_Email.init_hidden(b_input_ids1.size(0))
                    FC11 = modelEssay_Email(b_input_ids1, b_input_mask1,h)
                    FC21 = modelEssay_Email(b_input_ids2, b_input_mask2,h)
                    FC11=FC11.mean(0)
                    FC21=FC21.mean(0)
                    embEE1.append(FC11)
                    embEE2.append(FC21)
                    # truthEE.append(b_labels[0].cpu().data.numpy())
                    if id[0] not in ids1.keys():
                       ids1[id[0]] = []
                    ids1[id[0]].append(id[0])
                elif (type1[0] == 'interview' and type2[0] == 'speech_transcription') or (type1[0] == 'speech_transcription' and type2[0] == 'interview') or (type1[0] == 'speech_transcription' and type2[0] == 'speech_transcription') or (type1[0] == 'interview' and type2[0] == 'interview'):
                    h = modelSpeech_interv.init_hidden(b_input_ids1.size(0))
                    FC12 = modelSpeech_interv(b_input_ids1, b_input_mask1,h)
                    FC22 = modelSpeech_interv(b_input_ids2, b_input_mask2,h)
                    FC12=FC12.mean(0)
                    FC22=FC22.mean(0)
                    embSI1.append(FC12)
                    embSI2.append(FC22)
                    # truthSI.append(b_labels[0].cpu().data.numpy())
                    if id[0] not in ids2.keys():
                        ids2[id[0]] = []
                    ids2[id[0]].append(id[0])
                else:
                    h = modelGeneral.init_hidden(b_input_ids1.size(0))
                    FC13 = modelGeneral(b_input_ids1, b_input_mask1,h)
                    FC23 = modelGeneral(b_input_ids2, b_input_mask2,h)
                    FC13=FC13.mean(0)
                    FC23=FC23.mean(0)
                    embGen1.append(FC13)
                    embGen2.append(FC23)

                    h = modelEssay_Email.init_hidden(b_input_ids1.size(0))
                    EE1 = modelEssay_Email(b_input_ids1, b_input_mask1,h)
                    EE2 = modelEssay_Email(b_input_ids2, b_input_mask2,h)
                    EE1 = EE1.mean(0)
                    EE2 = EE2.mean(0)
                    embEE12.append(EE1)
                    embEE22.append(EE2)

                    h = modelSpeech_interv.init_hidden(b_input_ids1.size(0))
                    EE12 = modelSpeech_interv(b_input_ids1, b_input_mask1,h)
                    EE22 = modelSpeech_interv(b_input_ids2, b_input_mask2,h)
                    EE12 = EE12.mean(0)
                    EE22 = EE22.mean(0)
                    embSI12.append(EE12)
                    embSI22.append(EE22)

                    if id[0] not in ids3.keys():
                        ids3[id[0]] = []
                    ids3[id[0]].append(id[0])
                    # truthGen.append(b_labels[0].cpu().data.numpy())


                # ids.append(id)

            predictionsEE = self.distance_metric(torch.stack(embEE1), torch.stack(embEE2))
            predictionsSI = self.distance_metric(torch.stack(embSI1), torch.stack(embSI2))

            predictionsGen = self.distance_metric(torch.stack(embGen1), torch.stack(embGen2))
            predictionsSI2 = self.distance_metric(torch.stack(embSI12), torch.stack(embSI22))
            predictionsEE2 = self.distance_metric(torch.stack(embEE12), torch.stack(embEE22))

        similaritiesEE =  predictionsEE #torch.max(predictions) -
        normalized_similaritiesEE = similaritiesEE/torch.max(similaritiesEE)
        normalized_similaritiesEE = normalized_similaritiesEE.cpu().data.numpy()

        similaritiesSI =  predictionsSI #torch.max(predictions) -
        normalized_similaritiesSI = similaritiesSI/torch.max(similaritiesSI)
        normalized_similaritiesSI = normalized_similaritiesSI.cpu().data.numpy()

        similaritiesGen =  predictionsGen #torch.max(predictions) -
        normalized_similaritiesGen = similaritiesGen/torch.max(similaritiesGen)
        normalized_similaritiesGen = normalized_similaritiesGen.cpu().data.numpy()

        similaritiesEE2 =  predictionsEE2 #torch.max(predictions) -
        normalized_similaritiesEE2 = similaritiesEE2/torch.max(similaritiesEE2)
        normalized_similaritiesEE2 = normalized_similaritiesEE2.cpu().data.numpy()

        similaritiesSI2 =  predictionsSI2 #torch.max(predictions) -
        normalized_similaritiesSI2 = similaritiesSI2/torch.max(similaritiesSI2)
        normalized_similaritiesSI2 = normalized_similaritiesSI2.cpu().data.numpy()

        # now select best threshold for final reporting?

        binarized_predictionsEE = helpers.binarize(normalized_similaritiesEE, self.thresholdEE)
        binarized_predictionsEE2 = helpers.binarize(normalized_similaritiesEE2, self.thresholdEE2)
        binarized_predictionsSI = helpers.binarize(normalized_similaritiesSI, self.thresholdSI)
        binarized_predictionsSI2 = helpers.binarize(normalized_similaritiesSI2, self.thresholdSI2)
        binarized_predictionsGen = helpers.binarize(normalized_similaritiesGen, self.thresholdGen)

        predsGen = self.ensemble(binarized_predictionsEE2,binarized_predictionsSI2,binarized_predictionsGen)
        predsGenProbas = self.ensembleProbas(normalized_similaritiesEE2,normalized_similaritiesSI2,normalized_similaritiesGen)

        probasEE = self.getProbasFromSimilarity(normalized_similaritiesEE,self.thresholdEE)
        probasSI = self.getProbasFromSimilarity(normalized_similaritiesSI,self.thresholdSI)
        probasGen = self.getProbasFromSimilarity(predsGenProbas,self.thresholdGen)

        helpers.writeOutput(self.output,probasEE,ids1)
        helpers.writeOutput(self.output,probasSI,ids2)
        helpers.writeOutput(self.output,probasGen,ids3)
