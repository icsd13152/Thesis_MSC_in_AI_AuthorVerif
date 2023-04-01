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
##################
# Class for corpus
##################
lookupDict = {"4ao" : "for adults only",
                  "a3" : "anytime anywhere anyplace",
                  "aamof" : "as a matter of fact",
                  "adih" : "another day in hell",
                  "afaic" : "as far as i am concerned",
                  "afaict" : "as far as i can tell",
                  "afaik" : "as far as i know",
                  "afair" : "as far as i remember",
                  "afk" : "away from keyboard",
                  "app" : "application",
                  "approx" : "approximately", "dmx":"",
                  "apps" : "applications","dw":"", "skl":"",
                  "asap" : "as soon as possible", "CEO":"",
                  "asl" : "age, sex, location", "Ey":"",
                  "atk" : "at the keyboard", "yh":"", "nahhh":"",
                  "ave." : "avenue", "FYP":"", "TV":"",
                  "IDK": "I do not know","Ewwwwww":"","OMG":"Oh My God",
                  "aymm" : "are you my mother",
                  "ayor" : "at your own risk", "lmaooo":"", "OMDS":"","ahha":"",
                  "b&b" : "bed and breakfast", "ik":"",
                  "b+b" : "bed and breakfast", "Woohoo":"",
                  "b.c" : "before christ", "Dr":"", "EU":"",
                  "b2b" : "business to business",
                  "b2c" : "business to customer", "Img":"",
                  "b4" : "before", "Huhhhh":"",
                  "b4n" : "bye for now",
                  "b@u" : "back at you",
                  "bae" : "before anyone else",
                  "bak" : "back at keyboard",
                  "bbbg" : "bye bye be good",
                  "bbc" : "british broadcasting corporation",
                  "bbias" : "be back in a second",
                  "bbl" : "be back later",
                  "bbs" : "be back soon",
                  "be4" : "before",
                  "bfn" : "bye for now",
                  "blvd" : "boulevard",
                  "bout" : "about",
                  "brb" : "be right back",
                  "bros" : "brothers",
                  "brt" : "be right there",
                  "bsaaw" : "big smile and a wink",
                  "btw" : "by the way",
                  "bwl" : "bursting with laughter",
                  "c/o" : "care of",
                  "cet" : "central european time",
                  "cf" : "compare",
                  "cia" : "central intelligence agency",
                  "csl" : "can not stop laughing",
                  "cu" : "see you",
                  "cul8r" : "see you later",
                  "cv" : "curriculum vitae",
                  "cwot" : "complete waste of time",
                  "cya" : "see you",
                  "cyt" : "see you tomorrow",
                  "dbmib" : "do not bother me i am busy",
                  "diy" : "do it yourself",
                  "e123" : "easy as one two three",
                  "embm" : "early morning business meeting",
                  "encl" : "enclosed","etc":"",
                  "encl." : "enclosed", "ASAP":"",
                  "faq" : "frequently asked questions",
                  "fawc" : "for anyone who cares",
                  "fb" : "facebook",
                  "fc" : "fingers crossed",
                  "fimh" : "forever in my heart",
                  "ft" : "featuring",
                  "ftl" : "for the loss",
                  "ftw" : "for the win",
                  "fwiw" : "for what it is worth",
                  "fyi" : "for your information",
                  "g9" : "genius",
                  "gahoy" : "get a hold of yourself",
                  "gal" : "get a life",
                  "gcse" : "general certificate of secondary education",
                  "gfn" : "gone for now",
                  "gl" : "good luck",
                  "glhf" : "good luck have fun",
                  "gmt" : "greenwich mean time",
                  "gmta" : "great minds think alike",
                  "gn" : "good night",
                  "g.o.a.t" : "greatest of all time",
                  "goat" : "greatest of all time",
                  "gps" : "global positioning system",
                  "gr8" : "great",
                  "gratz" : "congratulations",
                  "gyal" : "girl",
                  "h&c" : "hot and cold",
                  "ibrb" : "i will be right back",
                  "ic" : "i see",
                  "icq" : "i seek you",
                  "ill":"i will", "Ikr":"",
                  "icymi" : "in case you missed it",
                  "idc" : "i do not care", "OMGGGGG":"","OMG":"",
                  "idgadf" : "i do not give a damn fuck",
                  "idgaf" : "i do not give a fuck",
                  "idk" : "i do not know", "LMAO":"",
                  "IG" : "instagram",
                  "iirc" : "if i remember correctly",
                  "ilu" : "i love you",
                  "ily" : "i love you",
                  "imho" : "in my humble opinion",
                  "imo" : "in my opinion",
                  "imu" : "i miss you",
                  "iow" : "in other words", "Ill":"I will","Ill":"I will",
                  "irl" : "in real life",
                  "j4f" : "just for fun",
                  "jic" : "just in case",
                  "jk" : "just kidding",
                  "jsyk" : "just so you know",
                  "l8r" : "later",
                  "lb" : "pound",
                  "lbs" : "pounds",
                  "ldr" : "long distance relationship",
                  "lmao" : "laugh my ass off",
                  "lmfao" : "laugh my fucking ass off", "LMFAO":"",
                  "lol" : "laughing out loud",
                  "Loooool": "laughing out loud",
                  "lol." : "laughing out loud.",
                  "ltd" : "limited",
                  "ltns" : "long time no see",
                  "m8" : "mate",
                  "mf" : "motherfucker",
                  "mfs" : "motherfuckers",
                  "mfw" : "my face when",
                  "mofo" : "motherfucker",
                  "mph" : "miles per hour",
                  "mr" : "mister",
                  "mrw" : "my reaction when",
                  "ms" : "miss",
                  "mte" : "my thoughts exactly",
                  "nagi" : "not a good idea",
                  "nbc" : "national broadcasting company",
                  "nbd" : "not big deal",
                  "nfs" : "not for sale",
                  "ngl" : "not going to lie", "amp":"",
                  "nhs" : "national health service",
                  "nrn" : "no reply necessary",
                  "nsfl" : "not safe for life",
                  "nsfw" : "not safe for work",
                  "nth" : "nice to have",
                  "nvr" : "never",
                  "nyc" : "new york city",
                  "ohp" : "overhead projector",
                  "oic" : "oh i see",
                  "omdb" : "over my dead body",
                  "omg" : "oh my god",
                  "omw" : "on my way",
                  "prw" : "parents are watching",
                  "ps" : "postscript",
                  "pt" : "point", "bby":"","Yuppp":"",
                  "ptb" : "please text back",
                  "pto" : "please turn over",
                  "qpsa" : "what happens",
                  "ratchet" : "rude", "BW":"",
                  "rbtl" : "read between the lines",
                  "rlrt" : "real life retweet",
                  "rofl" : "rolling on the floor laughing",
                  "roflol" : "rolling on the floor laughing out loud",
                  "rotflmao" : "rolling on the floor laughing my ass off",
                  "ruok" : "are you ok",
                  "sfw" : "safe for work",
                  "smh" : "shake my head", ".ppt":"",
                  "srsly" : "seriously",
                  "ssdd" : "same stuff different day",
                  "tbh" : "to be honest",
                  "tfw" : "that feeling when", "pffff":"","Hmmm":"",
                  "thks" : "thank you", "bro":"",
                  "thx" : "thank you",
                  "tl;dr" : "too long i did not read",
                  "tldr" : "too long i did not read",
                  "tmb" : "tweet me back",
                  "tntl" : "trying not to laugh",
                  "ttyl" : "talk to you later",
                  "u2" : "you too",
                  "u4e" : "yours for ever", "obvs":"","HR":"",
                  "w/" : "with",
                  "w/o" : "without", "BRO":"",
                  "w8" : "wait", "Omg":"",
                  "wassup" : "what is up",
                  "wb" : "welcome back",
                  "wtf" : "what the fuck","etc":"","etc.":"",
                  "wtg" : "way to go", "uo":"",
                  "wtpa" : "where the party at", "hmrc":"","CV":"","VAT":"",
                  "wuf" : "where are you from", "umm":"","ugh":"",
                  "wuzup" : "what is up", "ahhhh":"","wow":"", "Wyd":"",
                  "wywh" : "wish you were here", "lmaoo":"2","lmao":"","Lmao":"","Ahhh":"",
                  "yd" : "yard", "u":"you","abit":"a bit","Iv":"I have","cuz":"because", "rah":"2", "Naaahhh":"2","nah":"2","Nah":"2",
                  "ygtr" : "you got that right","Ill":"I will","Im":"I am", "Wya":"Where you at", "ffs.":"for fuck's sake", "ffs":"for fuck's sake",
                  "ynk" : "you never know","nah":"no","im":"i am","youll":"you will","y're":"you are","tmrw":"tomorrow",
                  "zzz" : "sleeping bored and tired","ur":"your","Idk":"I do not know","lmk":"let me know","x":"",
                  "ain't": "am not", "Ur":"Your","U":"You","yy":"yes","yeap":"yes","xx":"","xx.":"","Xx":"","xxx":"","x.":"","x":"","xxxx.":"","xxxx":"","xxxxx":""}
class Corpus(object):

    def __init__(self, dict_dataset):

        # define tokenizer
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # raw dataset
        self.tokenizer = T5TokenizerFast.from_pretrained('t5-base')
        # self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.dict_dataset_raw = dict_dataset

        self.dict_dataset_per_auth_ids= {}
        self.dict_dataset_per_auth_ids2 = {}
        self.dict_dataset_per_auth_masks= {}
        self.dict_positives_pairs = {}
        self.dict_negatives_pairs = {}
        self.dict_All_pairs = {}
        self.dict_All_pairs_shuffle = {}
        self.usedAuthors = {}
        self.list_of_non_freq_words = []
        self.labels = {}
        self.posTags = {}
        self.dict_pairs_per_author = {}
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
    def replaceabbrev(self,doc,typ):
        # if typ == "text_message" or typ == "email" :
            if isinstance(doc, float) == False and doc is not None:
                tokens = nltk.word_tokenize(doc)
                for idx in range(len(tokens)):
                    if tokens[idx] in lookupDict.keys():
                        # if word in doc.split():
                        tokens[idx] = '2'
            doc = ' '.join(tok for tok in tokens)
        # else:
        #     doc = doc
            return doc
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

    def paddingChunksToMaxlen(self,IdsChunks,masksChunks,isSecond = False,maxLen = 254):#listMasksChunksmasksChunks=None,
        listIdsChunks = list()
        listMasksChunks = list()
        # listmasks = []
        # for i in range(len(IdsChunks)):
        #     tmplistmasks = []
        #     for j in IdsChunks[i]:
        #         tmplistmasks.append(1)
        #     listmasks.append(torch.LongTensor(tmplistmasks))

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
            tmplistids.insert(0,1) #101 for BERT
            tmplistids.insert(len(tmplistids),2) #102 for Bert
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
    def addPOSTAGS(self,text):

        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        toks = []
        tags = set()
        print(tokens)
        for i in pos_tags:

            toks.append(i[0]+'-'+i[1])
        print(toks)
            # if i[1] in self.posTags.keys():
            #     toks.append(self.posTags[i[1]])
            # tags.add(i[1])
        # print(tags)
        # spec_tokens = {'new_tokens':list(tags)}
        # print(spec_tokens)
        # print(tags)
        # self.tokenizer.add_tokens(list(tags),special_tokens=False)
        sent = ' '.join(t for t in toks)
        print(sent)
        return sent
    def keepLastSubWordOfBert(self,text):
        tokens = self.tokenizer.tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        print(pos_tags)
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
            print(token)
            text = text.replace(token,symbol)
            print(text)
        self.list_of_non_freq_words=[]
        return text
    def BertEncodeTest(self,text):
        encoded_dict = self.tokenizer.encode_plus(
            text ,                    # Sentence to encode.
            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
            # pad_token = None,
            # add_prefix_space = False,
            max_length = 350,           # Pad & truncate all sentences.
            pad_to_max_length = True,
            truncation = True,
            return_attention_mask = True,   # Construct attn. masks.
            return_tensors = 'pt'
            # is_split_into_words=True
        )

        return encoded_dict['input_ids'],encoded_dict['attention_mask']
    def chunkingTextsBasedOnBert(self,text_tokens):
        input_ids=[]
        attention_masks=[]


        # setences1 = nltk.sent_tokenize(text_tokens)
        # setences2 = nltk.sent_tokenize(row['Text2'])
        # # print(setences1)
        # set1 = self.createRandomSetence(setences1)
        # # print(set1)
        # set2 = createRandomSetence(setences2)

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
        # print(encoded_dict['input_ids'])
        # tokens = self.tokenizer.convert_tokens_to_ids(text_tokens)
        # print(tokens)
        # tokens_tensor = torch.LongTensor(tokens)

        # print(self.tokenizer.convert_ids_to_tokens(tokens_tensor))

        tensorsIdList1,tensorsMaskList1 = self.paddingChunksToMaxlen(encoded_dict['input_ids'][0].split(254),encoded_dict['attention_mask'][0].split(254),False,254)
        # tensorsIdList1,tensorsMaskList1 = self.paddingChunksToMaxlen(tokens_tensor.split(254),False,254)
        # print(tensorsIdList1)


        return tensorsIdList1, tensorsMaskList1

    def correctMissSpelling(self,doc,typeOftext):
            tokens = nltk.word_tokenize(doc)
        # if typeOftext=='text_message' or typeOftext=='email':
            b = TextBlob(tokens[0])
            # print("======================original==========================")
            # print(doc)
            d = b.correct()
            tokens[0] = str(d)

                # print(tokens)
            return ' '.join(tok for tok in tokens)
        # else:
        #     return doc

    def preprocess_doc(self, doc,typeOftext=None,isPAN20=False):
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

        doc = re.sub('<.*?>',' ',doc)
        if isPAN20 == False:
            # doc = self.removeHTMLtags(doc)
            doc = self.maskEmoticons(doc,' 0 ')
            pater = re.compile(r'[lL][oO]+[lL]')
            doc = re.sub(pater, 'lol', doc)
            doc = self.replaceabbrev(doc,typeOftext)


        doc = unicodedata.normalize('NFKD', doc).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        doc = doc.replace('dr1_FN>',' ')
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
        # doc = doc.replace(': )',' 0 ')
        doc = doc.replace('�',' ')
        doc = doc.replace('( )','')
        doc = doc.replace('\' \'','')
        # doc = doc.replace('`` ``','')
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
        doc = doc.replace('LOOOOOOOOOOOOOOOOOL','2')
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
        if doc.strip().startswith('nk you'):
           doc = 'Thank you '+doc[6:]
        if doc.strip().startswith('nks '):
           doc = 'Thanks '+doc[3:]
        doc = doc.strip()
        # if doc.startswith(','):
        #     doc = doc[1:]
        # elif doc.startswith('oool'):
        #     doc = doc[4:]
        # elif doc.startswith('ooool'):
        #     doc = doc[5:]
        # elif doc.startswith('r ,'):
        #     doc = doc[3:]
        # elif doc.startswith('r '):
        #     doc = doc[1:]
        # elif doc.startswith('l '):
        #     doc = doc[1:]
        # elif doc.startswith('ll'):
        #     doc = doc[2:]
        # elif doc.startswith('lo ,'):
        #     doc = doc[4:]
        # elif doc.startswith('lo, '):
        #     doc = doc[3:]
        # elif doc.startswith('lo. '):
        #     doc = doc[3:]
        # elif doc.startswith('lo '):
        #     doc = doc[2:]
        # elif doc.startswith('h. '):
        #     doc = doc[2:]
        # elif doc.startswith('OOL '):
        #     doc = doc[3:]
        # elif doc.startswith('y!.'):
        #     doc = doc[3:]
        # elif doc.startswith('?.'):
        #     doc = doc[2:]
        # elif re.search("^[a-zA-Z]{1}\s", doc) is not None:
        #     doc = doc[1:]
        # elif re.search("^[a-zA-Z][\.,\!]\s", doc) is not None:
        #     doc = doc[2:]
        # elif re.search("^[.,!]\s", doc) is not None:
        #     doc = doc[1:]
        # elif doc.startswith('??.'):
        #     doc = doc[3:]
        # elif doc.startswith(' ?. '):
        #     doc = doc[3:]
        # elif doc.startswith('?. '):
        #     doc = doc[2:]
        # elif doc.startswith('nth - .'):
        #     doc = doc[7:]
        # doc = self.correctMissSpelling(doc,typeOftext)


        return doc.strip()
    def getNonFrequentWordsPerDoc(self,doc,threshold=2):
        words = nltk.word_tokenize(doc)
        fdist = FreqDist(words)

        for word,freq in fdist.items():
            if freq <= threshold:
                if len(word) > 4:
                    self.list_of_non_freq_words.append(word)




    def parseDict_per_pairid(self):
        posidx = 0
        negidx = 0
        for a in tqdm(self.dict_dataset_raw.keys()):
            listDocsPerAuthor = []
            listMasksPerAuthor = []
            for i, docs in enumerate(self.dict_dataset_raw[a]):

                processed_doc1 = self.preprocess_doc(docs[0],False)
                processed_doc1 = self.addPOSTAGS(processed_doc1)
                processed_doc2 = self.preprocess_doc(docs[1],False)
                processed_doc2 = self.addPOSTAGS(processed_doc2)
                inputIDs1,attn_masks1 = self.chunkingTextsBasedOnBert(processed_doc1)
                inputIDs2,attn_masks2 = self.chunkingTextsBasedOnBert(processed_doc2)
                label = docs[2]
                type1 = docs[3]
                type2 = docs[4]
                # print(type1)
                # print(type2)
                # if (type1 == 'essay' and type2 == 'essay') or (type2 == 'essay' and type1 == 'essay'):
                if (type1 == 'memo' and type2 == 'essay') or (type1 == 'essay' and type2 == 'memo') or (type1 == 'essay' and type2 == 'essay')  or (type1 == 'memo' and type2 == 'memo'):

                    if len(inputIDs1) == len(inputIDs2):
                        for i in range(0,len(inputIDs1)):

                            text1 = inputIDs1[i]
                            mask1 = attn_masks1[i]
                            text2 = inputIDs2[i]
                            mask2 = attn_masks2[i]
                            if a not in self.dict_dataset_per_auth_ids.keys():
                                self.dict_dataset_per_auth_ids[a] = []
                            self.dict_dataset_per_auth_ids[a].append([text1,mask1,text2,mask2,label,type1,type2])
                    elif len(inputIDs1) > len(inputIDs2):
                        for i in range(0,len(inputIDs1)):
                            if i < len(inputIDs2):
                                text1 = inputIDs1[i]
                                mask1 = attn_masks1[i]
                                text2 = inputIDs2[i]
                                mask2 = attn_masks2[i]
                                if a not in self.dict_dataset_per_auth_ids.keys():
                                    self.dict_dataset_per_auth_ids[a] = []
                                self.dict_dataset_per_auth_ids[a].append([text1,mask1,text2,mask2,label,type1,type2])
                            else:
                                text1 = inputIDs1[i]
                                mask1 = attn_masks1[i]
                                j = random.choice(range(0,len(inputIDs2)))
                                text2 = inputIDs2[j]
                                mask2 = attn_masks2[j]
                                if a not in self.dict_dataset_per_auth_ids.keys():
                                    self.dict_dataset_per_auth_ids[a] = []
                                self.dict_dataset_per_auth_ids[a].append([text1,mask1,text2,mask2,label,type1,type2])
                    elif len(inputIDs1) < len(inputIDs2):
                        for j in range(0,len(inputIDs2)):
                            if j < len(inputIDs1):
                                text1 = inputIDs1[j]
                                mask1 = attn_masks1[j]
                                text2 = inputIDs2[j]
                                mask2 = attn_masks2[j]
                                if a not in self.dict_dataset_per_auth_ids.keys():
                                    self.dict_dataset_per_auth_ids[a] = []
                                self.dict_dataset_per_auth_ids[a].append([text1,mask1,text2,mask2,label,type1,type2])
                            else:
                                text2 = inputIDs2[j]
                                mask2 = attn_masks2[j]
                                i = random.choice(range(0,len(inputIDs1)))
                                text1 = inputIDs1[i]
                                mask1 = attn_masks1[i]
                                if a not in self.dict_dataset_per_auth_ids.keys():
                                    self.dict_dataset_per_auth_ids[a] = []
                                self.dict_dataset_per_auth_ids[a].append([text1,mask1,text2,mask2,label,type1,type2])

    def buildPOS(self):
        id = 2
        for a in tqdm(self.dict_dataset_raw.keys()):
            for f in self.dict_dataset_raw[a].keys():
                for i, docs in enumerate(self.dict_dataset_raw[a][f]):

                    processed_doc1 = self.preprocess_doc(docs,f,False)
                    tokens = self.tokenizer.tokenize(processed_doc1)
                    pos_tags = nltk.pos_tag(tokens)
                    toks = []
                    tags = set()
                    for j in pos_tags:

                        if j[1] not in self.posTags.keys():
                            if j[1] != '.' and j[1] != ',' and j[1]!='(' and j[1]!=')' and j[1]!='$' and j[1]!=':' and j[1]!='#':
                                id+=1
                                self.posTags[j[1]] = str(id)


    def parse_dictionary(self):

        # authors
        for a in tqdm(self.dict_dataset_raw.keys()):
            for f in self.dict_dataset_raw[a].keys():
            # fandom categories
                listDocsPerAuthor = []
                listMasksPerAuthor = []

                # for f in self.dict_dataset_raw[a].keys():
                # documents

                if f=='memo' or f=='essay':
                    itemid = 0
                    # print("len per auth",str(len(self.dict_dataset_raw[a])))
                    # print(len(self.dict_dataset_raw[a][f]))
                    for i, docs in enumerate(self.dict_dataset_raw[a][f]):


                        processed_doc1 = self.preprocess_doc(docs,f,False)
                        processed_doc1 = self.addPOSTAGS(processed_doc1)
                        # print(processed_doc1)
                        # if f == 'essay':
                        #     self.getNonFrequentWordsPerDoc(processed_doc1,5)
                        # elif f =='memo':
                        #     self.getNonFrequentWordsPerDoc(processed_doc1,5)
                        # elif f =='email':
                        #     self.getNonFrequentWordsPerDoc(processed_doc1,2)
                        # elif f =='text_message':
                        #     self.getNonFrequentWordsPerDoc(processed_doc1,2)
                        # processed_doc1 = self.replaceTokensWith(processed_doc1)
                        # processed_doc2 = self.preprocess_doc(docs[1])
                        # tokens = self.keepLastSubWordOfBert(processed_doc1)
                        # tokens = self.addPOSTAGS(processed_doc1)
                        # print(tokens)
                        # tokens2 = self.tokenizer.convert_tokens_to_ids(tokens)
                        # print(tokens2)
                        # break
                        inputIDs1,attn_masks1 = self.chunkingTextsBasedOnBert(processed_doc1)
                        # inputIDs2,attn_masks2 = self.chunkingTextsBasedOnBert(processed_doc2)
                        # flat_IDs = []
                        # flat_Masks = []
                        # # flat_IDs2 = []
                        # # flat_Masks2 = []
                        # for item in inputIDs1:
                        #     flat_IDs.append(item)
                        # for item in attn_masks1:
                        #     flat_Masks.append(item)
                        # for item in inputIDs2:
                        #     flat_IDs2.append(item)
                        # for item in attn_masks2:
                        #     flat_Masks2.append(item)


                        if a not in self.dict_dataset_per_auth_ids.keys():
                            self.dict_dataset_per_auth_ids[a] = {}
                        if itemid not in self.dict_dataset_per_auth_ids[a].keys():
                            self.dict_dataset_per_auth_ids[a][itemid] = {}
                        if f not in  self.dict_dataset_per_auth_ids[a][itemid].keys():
                            self.dict_dataset_per_auth_ids[a][itemid][f] = {}
                        if a not in self.dict_dataset_per_auth_masks.keys():
                            self.dict_dataset_per_auth_masks[a] = {}
                        if itemid not in self.dict_dataset_per_auth_masks[a].keys():
                            self.dict_dataset_per_auth_masks[a][itemid] = {}
                        if f not in  self.dict_dataset_per_auth_masks[a][itemid].keys():
                            self.dict_dataset_per_auth_masks[a][itemid][f] = {}
                        self.dict_dataset_per_auth_ids[a][itemid][f]= inputIDs1
                        self.dict_dataset_per_auth_masks[a][itemid][f]=attn_masks1
                        itemid+=1

                    # print(len(listDocsPerAuthor))
                    # self.dict_dataset_per_auth_ids[a][f] = listDocsPerAuthor  # authID:{[x1,x2,x3],[y1,y2...yn],...}
                    # self.dict_dataset_per_auth_masks[a][f] = listMasksPerAuthor


















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


def train_val_split_per_pair(shuffledDict):



    n = int(0.80 * len(list(shuffledDict.keys())))


    data_train = dict(list(shuffledDict.items())[:n])
    data_val = dict(list(shuffledDict.items())[n:])
    return data_train,data_val

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

def train_val_split(shuffledDict):


    tmp_train = {}
    tmp_val = {}
    counter = 0
    for k,v in shuffledDict.items():
        counter += 1
        for k2,v2 in v.items():
            if v2 =='memo' or v2=='essay':
                if len(v2) > 2:
                    n = int(0.90 * len(v2))
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
                    if counter % 2 == 0:
                        if k not in tmp_train.keys():
                            tmp_train[k] = {}
                        if k2 not in tmp_train[k].keys():
                            tmp_train[k][k2] = []
                        tmp_train[k][k2]=v2
                    else:
                        if k not in tmp_val.keys():
                            tmp_val[k] = {}
                        if k2 not in tmp_val[k].keys():
                            tmp_val[k][k2] = []
                        tmp_val[k][k2]=v2



    return tmp_train,tmp_val
####################
# data folders/files
####################


datasets = [#'dict_auth_type_pan22_train'
            'dict_perPairid_pan22_test'
            #'dict_auth_type_pan22_train'
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
    train,val = train_val_split_perAuthor(shuffled_data)
    corpus = Corpus(dict_dataset=dict_dataset)
    # corpus.parse_dictionary()
    corpus.parseDict_per_pairid()
    # corpus.create_pairs_per_author_v2()#create_anchor_batches()
    # corpus.generatePairs_withPairsID()#create_anchor_batches()
    # corpus.buildPOS()
    # corpus.parse_dictionary()
    # print(corpus.posTags)
    # corpus.generatePairs_withPairsID()#parseDict_per_pairid()
    # corpus.create_batches_per_author()
    # corpus.generatePairs_withPairsID()
    # corpus.generateLastDict()


    #######
    # store
    #######
    with open( dataset + "_PAN22_256_overlap_T5_memo-essays_POS_test", 'wb') as f:
        pickle.dump(corpus.dict_dataset_per_auth_ids, f)
    # with open( dataset + "_PAN22_256_train_overlap_T5_memo-essays_POS_masks", 'wb') as f:
    #     pickle.dump(corpus.dict_dataset_per_auth_masks, f)
    # with open( dataset + "_PAN22_350_test_clean_NER_neg", 'wb') as f:
    #     pickle.dump(corpus.dict_dataset_per_auth_ids2, f)


    # for x,v in corpus.dict_dataset_per_auth_ids.items():
    #     print(x)
    #     print(v)

    # del corpus
    # corpus_val = Corpus(dict_dataset=val)
    # # # # # # # # # # # #
    # corpus_val.parse_dictionary()
    # # corpus_val.create_pairs_per_author_v2()
    # # # # # # # corpus_val.buildPOS()
    # # # # # corpus_val.parse_dictionary()#parseDict_per_pairid()
    # # # # # corpus_val.create_batches_per_author()
    # # # # # # # corpus_val.generatePairs_withPairsID()
    # # # # # # # corpus_val.generateLastDict()
    # # # # # #
    # with open( dataset + "_PAN22_256_val_overlap_T5_memo-essays_POS_ids", 'wb') as f:
    #     pickle.dump(corpus_val.dict_dataset_per_auth_ids, f)
    # with open( dataset + "_PAN22_256_val_overlap_T5_memo-essays_POS_masks", 'wb') as f:
    #     pickle.dump(corpus_val.dict_dataset_per_auth_masks, f)
    # with open( dataset + "_PAN22_256_val_clean_overlap_NER_essay-memo_cased", 'wb') as f:
    #     pickle.dump(corpus_val.dict_positives_pairs, f)
    # with open( dataset + "_PAN22_val_neg_pairs_256_auth_type_balance_clean_Nooverlapping", 'wb') as f:
    #     pickle.dump(corpus_val.dict_negatives_pairs, f)
    # # with open( dataset + "_test", 'wb') as f:
    # #     pickle.dump(corpus.dict_dataset_per_auth_ids, f)
    #
    #
    # del corpus_val



import numpy as np
# class MyDatasetTest(torch.utils.data.Dataset):
#     def __init__(self,
#                  data,
#                  data2,
#                  base_rate: float = 0.5
#                  ):
#
#         # get the dataset, then break it up into dict key'd on authors with values a list of chunks.
#         self.per_pair_dataset = data
#         self.per_pair_dataset2 = data2
#
#         # self.per_author_dataset_masks = data_masks
#         self.base_rate = base_rate
#
#         #self.per_author_dataset_ids,self.per_author_dataset_masks = shuffleAll(self.tmp_per_author_dataset_ids,self.tmp_per_author_dataset_masks)
#         #del self.tmp_per_author_dataset_ids, self.tmp_per_author_dataset_masks
#         # for x in self.dict_All_pairs:
#         #     print(x)
#     def __len__(self):
#
#         # return sum([len(self.per_pair_dataset[x]) for x in self.per_pair_dataset.keys()])
#         return len(list(self.per_pair_dataset.keys()))
#
#     def __getitem__(self, idx):
#         # i = random.choice(list(self.per_pair_dataset.keys()))
#         # i = random.choice(range(0,len(self.per_pair_dataset[author])))
#         if np.random.uniform() >= 0.5:
#             x = self.per_pair_dataset[idx]
#             print(x[0].shape)
#             text1 = x[0].squeeze(0)
#             mask1 = x[1].squeeze(0)
#             text2 = x[2].squeeze(0)
#             mask2 = x[3].squeeze(0)
#             label = x[4]
#         else:
#             x = self.per_pair_dataset2[idx]
#             print(x[0].shape)
#             text1 = x[0].squeeze(0)
#             mask1 = x[1].squeeze(0)
#             text2 = x[2].squeeze(0)
#             mask2 = x[3].squeeze(0)
#             label = x[4]
#
#         return text1, mask1, text2, mask2, label
#
#
#
# dataset_test = MyDatasetTest(corpus.dict_dataset_per_auth_ids,corpus.dict_dataset_per_auth_ids2,0.5)
# test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=2,shuffle=True)
# i=0
# for text1,mask1, text2,mask2, label in test_dataloader:
#     # print(text1)
#     # print(mask1)
#     i+=1
#     print(corpus.tokenizer.convert_ids_to_tokens(text1[0], skip_special_tokens=False))
#     print(corpus.tokenizer.convert_ids_to_tokens(text2[0], skip_special_tokens=False))
#     print(corpus.tokenizer.convert_ids_to_tokens(text1[1], skip_special_tokens=False))
#     print(corpus.tokenizer.convert_ids_to_tokens(text2[1], skip_special_tokens=False))
#     print(label)
#
#     print("=====================================================")
#     if i == 10:
#         break

# class AuthorshipDataset(torch.utils.data.Dataset):
#     def __init__(self, text_pairs):
#         self.text_pairs = text_pairs
#
#     def __len__(self):
#         return len(self.text_pairs)
#
#     def __getitem__(self, idx):
#         text_pair = self.text_pairs[idx]
#         print(text_pair)
#         text1 = text_pair[0][0]
#         mask1 = text_pair[0][1]
#         text2 = text_pair[0][2]
#         mask2 = text_pair[0][3]
#         label = text_pair[0][4]
#
#         return text1,mask1, text2,mask2, label
#
#
# dataset_train = AuthorshipDataset(corpus.dict_pairs_per_author)
# train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=2,shuffle=True)
# #
# i = 0
# for text1,mask1, text2,mask2, label in train_dataloader:
#     # print(text1[0])
#
#     # print(corpus.tokenizer.convert_ids_to_tokens(text1[0], skip_special_tokens=False))
#     # print(corpus.tokenizer.convert_ids_to_tokens(text2[0], skip_special_tokens=False))
#     # print(corpus.tokenizer.convert_ids_to_tokens(text1[1], skip_special_tokens=False))
#     # print(corpus.tokenizer.convert_ids_to_tokens(text2[1], skip_special_tokens=False))
#     # print(label)
#
#     print("=====================================================")
#     if i == 5:
#         break

# class AuthorshipDataset(torch.utils.data.Dataset):
#     def __init__(self, text_pairs):
#         self.text_pairs = text_pairs
#
#     def __len__(self):
#         return len(list(self.text_pairs.keys()))
#         # return sum(len(self.text_pairs[x]) for x in self.text_pairs.keys())
#         # counter = 0
#         # for a in self.text_pairs.keys():
#         #     for idx in self.text_pairs[a]:
#         #         counter += 1
#         # return counter
#
#     def __getitem__(self, idx):
#         # a = random.choice(list(self.text_pairs[idx].keys()))
#         # text_pair = self.text_pairs[idx]
#         # print(text_pair)
#         # if np.random.uniform() >= 0.5:
#         #     a = random.choice(list(self.text_pairs.keys()))
#         #     idx = random.choice(list(self.text_pairs.keys()))
#         text_pair = self.text_pairs[idx]
#         text1 = text_pair[0][0]
#         mask1 = text_pair[0][1]
#         text2 = text_pair[0][2]
#         mask2 = text_pair[0][3]
#         label = text_pair[0][4]
#         f1,f2 = text_pair[0][5],text_pair[0][6]
#         a1,a2 = text_pair[0][7],text_pair[0][8]
#         # else:
#         #     a = random.choice(list(self.text_pairs.keys()))
#         #     text_pair = self.text_pairs[a][idx]
#         #     text1 = text_pair[0][0]
#         #     mask1 = text_pair[0][1]
#         #     text2 = text_pair[0][2]
#         #     mask2 = text_pair[0][3]
#         #     label = text_pair[0][4]
#         #     f1,f2 = text_pair[0][5],text_pair[0][6]
#         #     a1,a2 = text_pair[0][7],text_pair[0][8]
#
#         return text1,mask1, text2,mask2, label,f1,f2,a1,a2
#
#
# dataset_train = AuthorshipDataset(corpus.dict_pairs_per_author)
# train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=8,shuffle=False)
#
# for text1,mask1, text2,mask2, label,f1,f2,a1,a2 in train_dataloader:
#     print(a1)
#     print(text1)
#     print("==============")
#     print(a2)
#     print(text2)
#     print(label)
#     break