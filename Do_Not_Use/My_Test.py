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

def get_input_mask_(text, tokenizer, max_len, label):
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
        sub_token.append([tokenizer.convert_tokens_to_ids('[CLS]')])
        label_temp = label[j]
        for i, word in enumerate(wordss):

            token = tokenizera.tokenize(word)
            sub_token1_max1.append(len(token))

            # sub_token.append(tokenizer.convert_tokens_to_ids(token))
            if(len(tokens)< (max_len - 2)):
                sub_token.append(tokenizer.convert_tokens_to_ids(token))
                tokens.extend(token)
                for m in range(len(token)):
                    if m == 0:
                        labels.append(tag2id[label_temp[i]])
                    else:
                        labels.append(14)

        if len(tokens) >= max_len - 1:
            tokens = tokens[0:(max_len - 2)]
            labels = labels[0:(max_len - 2)]

        ntokens.append("[CLS]")
        labelid.append(tag2id["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            labelid.append(labels[i])
        sub_token.append([tokenizer.convert_tokens_to_ids("[SEP]")])
        ntokens.append("[SEP]")
        labelid.append(tag2id["[SEP]"])
        input_mask = [1] * len(ntokens)
        while len(sub_token) < max_len:
            sub_token.append([tokenizer.convert_tokens_to_ids("[PAD]")])
            sub_token1_max.append(1)
        while len(ntokens) < max_len:
            ntokens.append('[PAD]')
            labelid.append(tag2id["[PAD]"])
            input_mask.append(0)
        ntokens = tokenizer.convert_tokens_to_ids(ntokens)
        ntokens1.append(ntokens)
        sub_token1.append(sub_token)
        labelid1.append(labelid)
        input_mask1.append(input_mask)

    maximum_token = max(sub_token1_max1)
    return ntokens1, input_mask1, labelid1,sub_token1,maximum_token

def get_correct_data(tokens,sub_tokens,labels,tokenizer):
    lb = []
    lb_decoded = []
    tk = []
    for i_st,SubToken in enumerate(sub_tokens):
        token = tokens[i_st]
        label = labels[i_st]
        index = 0
        for st in SubToken:
            No_Token = st[0]
            if No_Token == 1:
                st1 = [st[1]]
            else:
                st1 = st[1:1 + No_Token]
            if((len(st1)==1) & (st1[0] not in [0,100,101,102])):
                lb.append(label[index])
                lb_decoded.append(id2tag[label[index]])
                # tk.append(tokenizer.convert_ids_to_tokens(token[index]) )
                tk.append(tokenizer.convert_ids_to_tokens(st1))
                index = index + len(st1)
            elif((len(st1)>1) & (st1[0] not in [0,100,101,102])):
                lb.append(label[index])
                # tmp1 =
                # tmp =
                lb_decoded.append(id2tag[label[index]])
                # dc = tokenizer.decode(st)
                tk.append(tokenizer.decode(st1))
                index = index + len(st1)
            else:
                index += 1
    return tk,lb,lb_decoded


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


class CustomDataset(Dataset):
    # def __init__(self, tokenizer, sentences, labels, max_len):
    def __init__(self, token, mask, label, sub_token):
        self.len = len(token)
        self.token = token
        self.mask = mask
        self.label = label
        self.sub_token = sub_token
        # self.max_len = max_len

    def __getitem__(self, index):

        ids = self.token[index]
        mask = self.mask[index]
        label = self.label[index]
        subtoken = self.sub_token[index]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'tags': torch.tensor(label, dtype=torch.long),
            'sub_token': torch.tensor(subtoken, dtype=torch.long)
        }

    def __len__(self):
        return self.len

MAX_LEN = 200
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-05
# tokenizera = transformers.BertTokenizer.from_pretrained('bert-base-cased')
# tokenizera= transformers.PreTrainedTokenizerFast.from_pretrained('bert-base-cased')
tokenizera = transformers.BertTokenizerFast.from_pretrained('bert-base-cased')
texts, tags = read_wnut('D:\\Virtual_Env\\GitHub_projects\\NER_USING_BERT\\BERTNER\\data\\wnut17train.txt')
train_texts, val_texts, train_tags, val_tags = train_test_split(texts, tags, test_size=.2,random_state=43)
unique_tags = set(tag for doc in tags for tag in doc)
unique_tags = sorted(unique_tags)
# tag2id = {tag: id for id, tag in enumerate(unique_tags,1)}
# tag2id['[SEP]']= 102
# tag2id['[CLS]']= 101
# tag2id['[PAD]']= 0
# tag2id['[UNK]']= 100
# tag2id['Sub_Token']=14
global tag2id
tag2id = {tag: id for id, tag in enumerate(unique_tags,1)}
tag2id['[SEP]']= 17 #102
tag2id['[CLS]']= 16 #101
tag2id['[PAD]']= 18
tag2id['[UNK]']= 15 #100
tag2id['Sub_Token']=14
global id2tag
id2tag = {id: tag for tag, id in tag2id.items()}
t,m,l,s= get_input_mask(val_texts,tokenizera,200,val_tags)
training_set = CustomDataset(t, m, l,s )
train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
tk,lb,lb_decoded =get_correct_data(t,s,l,tokenizera)
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# for _,data in enumerate(training_loader, 0):
for data in training_loader:
      # for
        ids = data['ids'].to(dev, dtype = torch.long)
print(tag2id)

