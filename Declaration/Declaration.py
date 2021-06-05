import numpy as np
import pandas as pd
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler,TensorDataset
from transformers import BertForTokenClassification, BertTokenizer, BertConfig, BertModel
from sklearn.model_selection import train_test_split
from pathlib import Path
from transformers import BertTokenizerFast
import tqdm
# from torch.utils.data
import re
from transformers import AutoTokenizer
# from transformers.models.bert.modeling_bert
import transformers
import My_NER_Data
import json
from transformers import BertForTokenClassification
import Declaration.Config

# global EPOCHS
# global LEARNING_RATE
# global MAX_LEN
# global TRAIN_BATCH_SIZE
# global VALID_BATCH_SIZE
# global optimizer
# global dev
# global id2tag
# global tag2id
# def declare_variable():
#     EPOCHS = 5
#     LEARNING_RATE = 2e-05
#     MAX_LEN = 200
#     TRAIN_BATCH_SIZE = 32
#     VALID_BATCH_SIZE = 16
#     dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     return EPOCHS,LEARNING_RATE,MAX_LEN,TRAIN_BATCH_SIZE,VALID_BATCH_SIZE,dev


def train(epoch,training_loader,model,optimizer):

    model.train()
    for _, data in enumerate(training_loader, 0):
        # for
        ids = data['ids'].to(Declaration.Config.dev, dtype=torch.long)
        mask = data['mask'].to(Declaration.Config.dev, dtype=torch.long)
        targets = data['tags'].to(Declaration.Config.dev, dtype=torch.long)
        # print('ID={} target={}'.format(_,targets))
        # sub_token = data['sub_token'].to(dev,dtype= torch.long)
        loss = model(ids, mask, labels=targets)[0]

        ## optimizer.zero_grad()
        if _ % 500 == 0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()



class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertForTokenClassification.from_pretrained('bert-base-cased', len(Declaration.Config.id2tag)) #num_labels= 18)#

    def forward(self, ids, mask, labels=None):
        output_1 = self.l1(ids, mask, labels=labels)

        return output_1

def dict_write(dictionary,filename,location):
    # BERTNER / My_NER_Data
    loc = location + '/' + filename
    try:
        file = open(loc, 'wt')
        file.write(str(dictionary))
        file.close()
    except:
        print("Unable to write to file")
def dict_read(file_location):
    with open(file_location) as json_file:
        s = json_file.read()
        data = eval(s)
        # data = s #dict(s)
        # print('a')

    return data

def get_output(text,prediction,prediction_decode):
    output = [ '{}:{}'.format(t,p) for t,p in zip(text,prediction_decode)]
    return output
def verify_accuracy(text,label,prediction,prediction_decode):
    total = len(prediction)
    right = 0
    miss = 0
    output = [ '{}:{}'.format(t,p) for t,p in zip(text,prediction_decode)]
    for l,p in zip(label,prediction):
        if(l==p):
            right += 1
        else:
            miss += 1
    print('Accuracy = {}'.format(str((right/total)*100)))
    return output
def read_wnut(file_path):
    file_path = Path(file_path)
    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split('\n'):
            token, tag = line.split('\t')
            tokens.append(token)
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)

    return token_docs, tag_docs

def get_input_maskT(text, tokenizer, max_len):
    ntokens1, input_mask1,  sub_token,maximum_token = get_input_maskT_(text, tokenizer, max_len)
    sub_token1 =[]
    for sent in sub_token:
        list_sent=[]
        for word in sent:
            list_word =[]
            len_word = len(word)
            if maximum_token == 1:
                diff =1
            else:
                diff = maximum_token - len_word
            tmp_list = [0] * diff
            list_word.append(len_word)
            list_word.extend(word)
            list_word.extend(tmp_list)
            list_sent.append(list_word)
        sub_token1.append(list_sent)
    # print(sub_token1)
    return ntokens1, input_mask1,  sub_token1

def get_input_maskT_(text, tokenizera, max_len):
    tokens1 = []
    # labels1 = []
    ntokens1 = []
    # labelid1 = []
    input_mask1 = []
    sub_token1 = []
    sub_token1_max1 = []
    for j, wordss in enumerate(text):
        tokens = []
        # labels = []
        sub_token = []
        ntokens = []
        # labelid = []
        sub_token1_max = []
        sub_token.append([tokenizera.convert_tokens_to_ids('[CLS]')])
        # label_temp = label[j]
        for i, word in enumerate(wordss):

            token = tokenizera.tokenize(word)
            sub_token1_max1.append(len(token))

            # sub_token.append(tokenizer.convert_tokens_to_ids(token))
            if(len(tokens)< (max_len - 2)):
                sub_token.append(tokenizera.convert_tokens_to_ids(token))
                tokens.extend(token)
                # for m in range(len(token)):
                #     if m == 0:
                #         labels.append(tag2id[label_temp[i]])
                #     else:
                #         labels.append(14)

        if len(tokens) >= max_len - 1:
            tokens = tokens[0:(max_len - 2)]
            # labels = labels[0:(max_len - 2)]

        ntokens.append("[CLS]")
        # labelid.append(tag2id["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            # labelid.append(labels[i])
        sub_token.append([tokenizera.convert_tokens_to_ids("[SEP]")])
        ntokens.append("[SEP]")
        # labelid.append(tag2id["[SEP]"])
        input_mask = [1] * len(ntokens)
        while len(sub_token) < max_len:
            sub_token.append([tokenizera.convert_tokens_to_ids("[PAD]")])
            sub_token1_max1.append(1)
        while len(ntokens) < max_len:
            ntokens.append('[PAD]')
            # labelid.append(tag2id["[PAD]"])
            input_mask.append(0)
        ntokens = tokenizera.convert_tokens_to_ids(ntokens)
        ntokens1.append(ntokens)
        sub_token1.append(sub_token)
        # labelid1.append(labelid)
        input_mask1.append(input_mask)

    maximum_token = max(sub_token1_max1)
    return ntokens1, input_mask1,sub_token1,maximum_token# , labelid1

def get_input_mask(text, tokenizer, max_len, label):
    ntokens1, input_mask1, labelid1, sub_token,maximum_token = get_input_mask_(text, tokenizer, max_len, label)
    sub_token1 =[]
    for sent in sub_token:
        list_sent=[]
        for word in sent:
            list_word =[]
            len_word = len(word)
            diff = maximum_token - len_word
            tmp_list = [0] * diff
            list_word.append(len_word)
            list_word.extend(word)
            list_word.extend(tmp_list)
            list_sent.append(list_word)
        sub_token1.append(list_sent)
    # print(sub_token1)
    return ntokens1, input_mask1, labelid1, sub_token1

def get_input_mask_(text, tokenizera, max_len, label):
    tokens1 = []
    labels1 = []
    ntokens1 = []
    labelid1 = []
    input_mask1 = []
    sub_token1 = []
    sub_token1_max1 = []
    for j, wordss in enumerate(text):
        tokens = []
        labels = []
        sub_token = []
        ntokens = []
        labelid = []
        sub_token1_max = []
        sub_token.append([tokenizera.convert_tokens_to_ids('[CLS]')])
        label_temp = label[j]
        for i, word in enumerate(wordss):

            token = tokenizera.tokenize(word)
            sub_token1_max1.append(len(token))

            # sub_token.append(tokenizer.convert_tokens_to_ids(token))
            if(len(tokens)< (max_len - 2)):
                sub_token.append(tokenizera.convert_tokens_to_ids(token))
                tokens.extend(token)
                for m in range(len(token)):
                    if m == 0:
                        labels.append(Declaration.Config.tag2id[label_temp[i]])
                    else:
                        labels.append(14)

        if len(tokens) >= max_len - 1:
            tokens = tokens[0:(max_len - 2)]
            labels = labels[0:(max_len - 2)]

        ntokens.append("[CLS]")
        labelid.append(Declaration.Config.tag2id["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            labelid.append(labels[i])
        sub_token.append([tokenizera.convert_tokens_to_ids("[SEP]")])
        ntokens.append("[SEP]")
        labelid.append(Declaration.Config.tag2id["[SEP]"])
        input_mask = [1] * len(ntokens)
        while len(sub_token) < max_len:
            sub_token.append([tokenizera.convert_tokens_to_ids("[PAD]")])
            sub_token1_max.append(1)
        while len(ntokens) < max_len:
            ntokens.append('[PAD]')
            labelid.append(Declaration.Config.tag2id["[PAD]"])
            input_mask.append(0)
        ntokens = tokenizera.convert_tokens_to_ids(ntokens)
        ntokens1.append(ntokens)
        sub_token1.append(sub_token)
        labelid1.append(labelid)
        input_mask1.append(input_mask)

    maximum_token = max(sub_token1_max1)
    return ntokens1, input_mask1, labelid1,sub_token1,maximum_token

def get_correct_data(tokens,sub_tokens,tokenizer,labels=None,predictions=None):
    lb = []
    lb_decoded = []
    tk = []
    lp = []
    lp_decoded = []
    for i_st,SubToken in enumerate(sub_tokens):
        token = tokens[i_st]
        label = None
        prediction = None
        if labels is not None:
            label = labels[i_st]
        if predictions is not None:
            prediction =  predictions[i_st]
        index = 0

        for st in SubToken:
            No_Token = st[0]
            if No_Token == 1:
                st1 = [st[1]]
            else:
                st1 = st[1:1 + No_Token]
            if((len(st1)==1) & (st1[0] not in [0,100,101,102])):
                if label is not None:
                    lb.append(label[index])
                    lb_decoded.append(Declaration.Config.id2tag[label[index]])
                if prediction is not None:
                    lp.append(prediction[index])
                    lp_decoded.append(Declaration.Config.id2tag[prediction[index]])
                # tk.append(tokenizer.convert_ids_to_tokens(token[index]) )
                tk.extend(tokenizer.convert_ids_to_tokens(st1))
                index = index + len(st1)
            elif((len(st1)>1) & (st1[0] not in [0,100,101,102])):
                if label is not None:
                    lb.append(label[index])
                    lb_decoded.append(Declaration.Config.id2tag[label[index]])
                if prediction is not None:
                    lp.append(prediction[index])
                    lp_decoded.append(Declaration.Config.id2tag[prediction[index]])
                tk.append(tokenizer.decode(st1))
                index = index + len(st1)
            else:
                index += 1
    return tk,lb,lb_decoded,lp,lp_decoded


class CustomDataset(Dataset):
    # def __init__(self, tokenizer, sentences, labels, max_len):
    def __init__(self, token, mask, label, sub_token):
        self.len = len(token)
        self.token = token
        self.mask = mask
        self.label = label
        self.sub_token = sub_token

    def __getitem__(self, index):

        ids = self.token[index]
        mask = self.mask[index]
        label = self.label[index]
        # subtoken = self.sub_token[index]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'tags': torch.tensor(label, dtype=torch.long),
            # 'sub_token': torch.tensor(subtoken, dtype=torch.long)
        }

    def __len__(self):
        return self.len
class testDataset(Dataset):
    def __init__(self, token, mask, sub_token):
        self.len = len(token)
        self.token = token
        self.mask = mask
        self.sub_token = sub_token

    def __getitem__(self, index):
        ids = self.token[index]
        mask = self.mask[index]
        # subtoken = self.sub_token[index]
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            # 'sub_token': torch.tensor(subtoken, dtype=torch.long)
        }

    def __len__(self):
        return self.len


def valid(model, testing_loader):
    tt = []
    model.eval()
    predictions = []
    # dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(Declaration.Config.dev, dtype=torch.long)
            mask = data['mask'].to(Declaration.Config.dev, dtype=torch.long)
            targets = data['tags'].to(Declaration.Config.dev, dtype=torch.long)
            output = model(ids, mask, labels=targets)
            loss, logits = output[:2]
            logits = logits.detach().cpu().numpy()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])

    return predictions

def get_test_prediction(model, testing_loader):
    tt = []
    model.eval()
    predictions = []
    # dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(Declaration.Config.dev, dtype=torch.long)
            mask = data['mask'].to(Declaration.Config.dev, dtype=torch.long)
            # targets = data['tags'].to(dev, dtype=torch.long)
            output = model(ids, mask )
            logits = output[0]
            logits = logits.detach().cpu().numpy()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])

    return predictions


def split_data(texts,tags,size = 0.2):
    train_texts, val_texts, train_tags, val_tags = train_test_split(texts, tags, test_size=size, random_state=43)
    unique_tags = set(tag for doc in tags for tag in doc)
    unique_tags = sorted(unique_tags)
    tag2id = {tag: id for id, tag in enumerate(unique_tags,1)}
    tag2id['[SEP]']= 17 #102
    tag2id['[CLS]']= 16 #101
    tag2id['[PAD]']= 18
    tag2id['[UNK]']= 15 #100
    tag2id['Sub_Token']= 14
    id2tag = {id: tag for tag, id in tag2id.items()}

    dict_write(tag2id,'tag2id.json','../My_NER_Data')
    dict_write(id2tag,'id2tag.json','../My_NER_Data')
    return train_texts, val_texts, train_tags, val_tags#,id2tag,tag2id


def read_pretrained_model_tokenizer():
    # declare_variable()
    # id2tag = dict_read('./My_NER_Data/id2tag.json')
    # tag2id = dict_read('./My_NER_Data/tag2id.json')
    # BERTClass()
    tokenizera = transformers.BertTokenizer.from_pretrained('./My_NER_Data')
    model = torch.load('./My_NER_Data/NER.pt')
    return model,tokenizera

def read_training_data():
    texts, tags = read_wnut('../My_NER_Data/wnut17train.txt')  # ('./data/wnut17train.txt')
    train_texts, val_texts, train_tags, val_tags = split_data(texts, tags, size=0.2)  # ,id2tag,tag2id
    return train_texts, val_texts, train_tags, val_tags

# BERTNER/My_NER_Data/NER.pt

def Perform_Training(train_texts, train_tags):
    ## code to

    tokenizera = transformers.BertTokenizer.from_pretrained('bert-base-cased')
    training_text, training_mask, training_label, trainin_sub_token = get_input_mask(train_texts, tokenizera, Declaration.Config.MAX_LEN,
                                                                                     train_tags)
    training_set = CustomDataset(training_text, training_mask, training_label, trainin_sub_token)
    train_params = {'batch_size': Declaration.Config.TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }
    training_loader = DataLoader(training_set, **train_params)
    model = BERTClass()
    model.to(Declaration.Config.dev)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=Declaration.Config.LEARNING_RATE)

    for epoch in range(Declaration.Config.EPOCHS):
        train(epoch,training_loader,model,optimizer)
    torch.save(model, '../My_NER_Data/NER.pt')
    tokenizera.save_pretrained('../My_NER_Data')
    print('Saved Tokenizer and Model')
    return model,tokenizera

def perform_validation(val_texts,val_tags,model,tokenizera):
    validation_text, validation_mask, validation_label, validation_sub_token = get_input_mask(val_texts, tokenizera,
                                                                                              Declaration.Config.MAX_LEN, val_tags)

    testing_set = CustomDataset(validation_text, validation_mask, validation_label, validation_sub_token)

    test_params = {'batch_size': Declaration.Config.VALID_BATCH_SIZE,
                   'shuffle': False,
                   'num_workers': 0
                   }

    testing_loader = DataLoader(testing_set, **test_params)

    predictions = valid(model, testing_loader)

    tk, lb, lb_decoded, lp, lp_decoded = get_correct_data(validation_text, validation_sub_token, tokenizera,
                                                          validation_label, predictions)
    output = verify_accuracy(tk, lb, lp, lp_decoded)
    print('Output = {}'.format(output))

def perform_test(text,model,tokenizera):
    # EPOCHS,LEARNING_RATE,MAX_LEN,TRAIN_BATCH_SIZE,VALID_BATCH_SIZE,dev = declare_variable()
    lst_word = [text.split()]
    ntokens1, input_mask1,  sub_token1 = get_input_maskT(lst_word, tokenizera, Declaration.Config.MAX_LEN)

    testing_set = testDataset(ntokens1, input_mask1, sub_token1)

    test_params = {'batch_size': Declaration.Config.VALID_BATCH_SIZE,
                   'shuffle': False,
                   'num_workers': 0
                   }

    testing_loader = DataLoader(testing_set, **test_params)
    predictions = get_test_prediction(model, testing_loader)

    tk, lb, lb_decoded, lp, lp_decoded = get_correct_data(ntokens1, sub_token1, tokenizera, predictions=predictions)
    output =get_output(tk, lp, lp_decoded)
    return output
     # verify_accuracy
    # print('Output = {}'.format(output))



