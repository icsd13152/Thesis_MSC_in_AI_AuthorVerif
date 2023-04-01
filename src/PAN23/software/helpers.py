import nltk
import re
import torch
from torch.nn import functional as F
import emojis
import demoji
import unicodedata
import numpy as np
import json
from sklearn.metrics import f1_score, roc_auc_score

def writeOutput(outputFile,probas,ids):
    with open(outputFile, 'a') as f:
        for i in range(len(probas)):
            d = {
                'id': ids[i],
                'value': probas[i]
            }
            json.dump(d, f)
            f.write('\n')
        f.close()


def removeHTMLtags(text):
    x = re.sub('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', ' ',text)
    x2 = re.sub('\s\s', ' ',x)
    x3 = re.sub('\t', ' ',x2)
    return x3.strip()

def maskNumbers(text, symbol='1'):
    x = re.sub('[0-9]', symbol,text)
    return x

def maskEmoticons(text):
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

def addPOSTAGS(tokenizer,text,txtLen = 500):

    tokens = tokenizer.convert_ids_to_tokens(text)
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

    sequence = original_tokens + pos + ['[SEP]'] + [str(txtLen)] + ['[SEP]']
    sent = ' '.join(t for t in sequence)
    # print(sequence)
    input_ids = tokenizer.encode_plus(
        sent ,                    # Sentence to encode.
        add_special_tokens = False, # Add '[CLS]' and '[SEP]'
        return_attention_mask = False   # Construct attn. masks.
    )

    return input_ids['input_ids']

def paddingChunksToMaxlen(tokenizer,IdsChunks,masksChunks,txtlen,maxLen = 254):#listMasksChunksmasksChunks=None,
    listIdsChunks = list()
    listMasksChunks = list()
    # listmasks = []
    # for i in range(len(IdsChunks)):
    #     tmplistmasks = []
    #     for j in IdsChunks[i]:
    #         tmplistmasks.append(1)
    #     listmasks.append(torch.LongTensor(tmplistmasks))
    totalLen = 370
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

        tmplistids.insert(0,101) #101 for BERT
        tmplistids.insert(len(tmplistids),102) #102 for Bert

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

        ids_with_pos = addPOSTAGS(tmplistids,txtlen)
        tmplistmasks = [1]*len(ids_with_pos)
        pad_len_final = totalLen  - len(ids_with_pos)

        listIdsChunks.append(torch.LongTensor(ids_with_pos))
        listMasksChunks.append(torch.LongTensor(tmplistmasks))
        del tmplistmasks, tmplistids
        if pad_len_final > 0:

            listIdsChunks[i] =  F.pad(listIdsChunks[i], (0,pad_len_final), "constant", 0) # 0 for bert
            listMasksChunks[i] =  F.pad(listMasksChunks[i], (0,pad_len_final), "constant", 0)

        # if pad_len > 0:
        #
        #     listIdsChunks[i] =  F.pad(listIdsChunks[i], (0,pad_len), "constant", 0) # 0 for bert
        #     listMasksChunks[i] =  F.pad(listMasksChunks[i], (0,pad_len), "constant", 0)

    del IdsChunks, pad_len
    # gc.collect()
    return listIdsChunks ,listMasksChunks


def preprocess_doc(doc):
    doc = maskNumbers(doc)
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
    doc = removeHTMLtags(doc)
    doc = maskEmoticons(doc,' 0 ')
    pater = re.compile(r'[lL][oO]+[lL]')
    doc = re.sub(pater, 'lol', doc)
    doc = unicodedata.normalize('NFKD', doc).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    doc = doc.replace('dr1_FN>',' ')
    doc = doc.replace('part_FN_MN','')
    doc = doc.replace('part_FN_SN','')
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


def chunkingTextsBasedOnBert(tokenizer,text,txtlen):
    # text_with_pos = addPOSTAGS(text)
    encoded_dict = tokenizer.encode_plus(
        text ,                    # Sentence to encode.
        add_special_tokens = False,
        return_attention_mask = True,
        return_tensors = 'pt'
    )

    tensorsIdList1,tensorsMaskList1 = paddingChunksToMaxlen(tokenizer,encoded_dict['input_ids'][0].split(126),encoded_dict['attention_mask'][0].split(126),txtlen,126)

    return tensorsIdList1, tensorsMaskList1

def binarize(y, threshold=0.5):
    y = np.array(y)
    y = np.ma.fix_invalid(y, fill_value=threshold)
    y[y >= threshold] = 1
    y[y < threshold] = 0

    return y


def auc(true_y, pred_y):
    """
    Calculates the AUC score (Area Under the Curve), a well-known
    scalar evaluation score for binary classifiers. This score
    also considers "unanswered" problem, where score = 0.5.
    Parameters
    ----------
    prediction_scores : array [n_problems]
        The predictions outputted by a verification system.
        Assumes `0 >= prediction <=1`.
    ground_truth_scores : array [n_problems]
        The gold annotations provided for each problem.
        Will typically be `0` or `1`.
    Returns
    ----------
    auc = the Area Under the Curve.
    References
    ----------
        E. Stamatatos, et al. Overview of the Author Identification
        Task at PAN 2014. CLEF (Working Notes) 2014: 877-897.
    """
    try:
        return roc_auc_score(true_y, pred_y)
    except ValueError:
        return 0.0


def c_at_1(true_y, pred_y, threshold=0.5):
    """
    Calculates the c@1 score, an evaluation method specific to the
    PAN competition. This method rewards predictions which leave
    some problems unanswered (score = 0.5). See:
        A. Peñas and A. Rodrigo. A Simple Measure to Assess Nonresponse.
        In Proc. of the 49th Annual Meeting of the Association for
        Computational Linguistics, Vol. 1, pages 1415-1424, 2011.
    Parameters
    ----------
    prediction_scores : array [n_problems]
        The predictions outputted by a verification system.
        Assumes `0 >= prediction <=1`.
    ground_truth_scores : array [n_problems]
        The gold annotations provided for each problem.
        Will always be `0` or `1`.
    Returns
    ----------
    c@1 = the c@1 measure (which accounts for unanswered
        problems.)
    References
    ----------
        - E. Stamatatos, et al. Overview of the Author Identification
        Task at PAN 2014. CLEF (Working Notes) 2014: 877-897.
        - A. Peñas and A. Rodrigo. A Simple Measure to Assess Nonresponse.
        In Proc. of the 49th Annual Meeting of the Association for
        Computational Linguistics, Vol. 1, pages 1415-1424, 2011.
    """

    n = float(len(pred_y))
    nc, nu = 0.0, 0.0

    for gt_score, pred_score in zip(true_y, pred_y):
        if pred_score == 0.5 or (pred_score >= 0.50 and pred_score <= 0.51):
            nu += 1
        elif (pred_score > 0.5) == (gt_score > 0.5):
            nc += 1.0

    return (1 / n) * (nc + (nu * nc / n))


def f1(true_y, pred_y):
    """
    Assesses verification performance, assuming that every
    `score > 0.5` represents a same-author pair decision.
    Note that all non-decisions (scores == 0.5) are ignored
    by this metric.
    Parameters
    ----------
    prediction_scores : array [n_problems]
        The predictions outputted by a verification system.
        Assumes `0 >= prediction <=1`.
    ground_truth_scores : array [n_problems]
        The gold annotations provided for each problem.
        Will typically be `0` or `1`.
    Returns
    ----------
    acc = The number of correct attributions.
    References
    ----------
        E. Stamatatos, et al. Overview of the Author Identification
        Task at PAN 2014. CLEF (Working Notes) 2014: 877-897.
    """
    true_y_filtered, pred_y_filtered = [], []

    for true, pred in zip(true_y, pred_y):
        if pred != 0.5:
            true_y_filtered.append(true)
            pred_y_filtered.append(pred)

    pred_y_filtered = binarize(pred_y_filtered)

    return f1_score(true_y_filtered, pred_y_filtered)


def f_05_u_score(true_y, pred_y, pos_label=1, threshold=0.5):
    """
    Return F0.5u score of prediction.
    :param true_y: true labels
    :param pred_y: predicted labels
    :param threshold: indication for non-decisions (default = 0.5)
    :param pos_label: positive class label (default = 1)
    :return: F0.5u score
    """

    pred_y = binarize(pred_y)

    n_tp = 0
    n_fn = 0
    n_fp = 0
    n_u = 0

    for i, pred in enumerate(pred_y):
        if pred == threshold:
            n_u += 1
        elif pred == pos_label and pred == true_y[i]:
            n_tp += 1
        elif pred == pos_label and pred != true_y[i]:
            n_fp += 1
        elif true_y[i] == pos_label and pred != true_y[i]:
            n_fn += 1

    return (1.25 * n_tp) / (1.25 * n_tp + 0.25 * (n_fn + n_u) + n_fp)

def rescale(value, orig_min, orig_max, new_min, new_max):
    """
    Rescales a `value` in the old range defined by
    `orig_min` and `orig_max`, to the new range
    `new_min` and `new_max`. Assumes that
    `orig_min` <= value <= `orig_max`.
    Parameters
    ----------
    value: float, default=None
        The value to be rescaled.
    orig_min: float, default=None
        The minimum of the original range.
    orig_max: float, default=None
        The minimum of the original range.
    new_min: float, default=None
        The minimum of the new range.
    new_max: float, default=None
        The minimum of the new range.
    Returns
    ----------
    new_value: float
        The rescaled value.
    """

    orig_span = orig_max - orig_min
    new_span = new_max - new_min

    try:
        scaled_value = float(value - orig_min) / float(orig_span)
    except ZeroDivisionError:
        orig_span += 1e-6
        scaled_value = float(value - orig_min) / float(orig_span)

    return new_min + (scaled_value * new_span)


def correct_scores(scores, p1, p2):
    new_scores = []
    for sc in scores:
        if sc <= p1:
            sc = rescale(sc, 0, p1, 0, 0.49)
            new_scores.append(sc)
        elif sc > p1 and sc < p2:
            new_scores.append(0.5)
        else:
            sc = rescale(sc, p2, 1, 0.51, 1)
            new_scores.append(sc)
    return np.array(new_scores)


