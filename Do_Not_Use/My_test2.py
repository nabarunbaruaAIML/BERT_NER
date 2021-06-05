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
import json
from transformers import BertForTokenClassification

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

texts, tags = read_wnut('./data/wnut17train.txt')
train_texts, val_texts, train_tags, val_tags = train_test_split(texts, tags, test_size=.2,random_state=43)
unique_tags = set(tag for doc in tags for tag in doc)
unique_tags = sorted(unique_tags)
tag2id = {tag: id for id, tag in enumerate(unique_tags,1)}
tag2id['[SEP]']= 17 #102
tag2id['[CLS]']= 16 #101
tag2id['[PAD]']= 18
tag2id['[UNK]']= 15 #100
tag2id['Sub_Token']= 14
id2tag = {id: tag for tag, id in tag2id.items()}
# dict_write(tag2id,'tag2id.json','./My_NER_Data')
# dict_write(id2tag,'id2tag.json','./My_NER_Data')
id2tag = dict_read('./My_NER_Data/id2tag.json')
tag2id = dict_read('./My_NER_Data/tag2id.json')
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

def get_correct_data(tokens,sub_tokens,tokenizer,labels=None,predictions=None):
    lb = []
    lb_decoded = []
    tk = []
    lp = []
    lp_decoded = []
    for i_st,SubToken in enumerate(sub_tokens):
        token = tokens[i_st]
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
                lb.append(label[index])
                lb_decoded.append(id2tag[label[index]])
                lp.append(prediction[index])
                lp_decoded.append(id2tag[prediction[index]])
                # tk.append(tokenizer.convert_ids_to_tokens(token[index]) )
                tk.extend(tokenizer.convert_ids_to_tokens(st1))
                index = index + len(st1)
            elif((len(st1)>1) & (st1[0] not in [0,100,101,102])):
                lb.append(label[index])
                lb_decoded.append(id2tag[label[index]])
                lp.append(prediction[index])
                lp_decoded.append(id2tag[prediction[index]])
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
    # eval_loss = 0;
    # eval_accuracy = 0
    # n_correct = 0;
    # n_wrong = 0;
    # total = 0
    # Subtoken_l1 = []
    # t2 = []
    # , true_labels, pred_label = [], [], []
    # nb_eval_steps, nb_eval_examples = 0, 0

    predictions = []

    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(dev, dtype=torch.long)
            mask = data['mask'].to(dev, dtype=torch.long)
            targets = data['tags'].to(dev, dtype=torch.long)
            # sub_token = data['sub_token'].to(dev, dtype=torch.long)
            output = model(ids, mask, labels=targets)
            loss, logits = output[:2]
            logits = logits.detach().cpu().numpy()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])

            # label_ids = targets.to('cpu').numpy()
            # sub = sub_token.to('cpu').numpy()
            # id = ids.to('cpu').numpy()
            # tt.extend([list(p) for p in id])

            # pred_label.append([list(p) for p in np.argmax(logits, axis=2)])
            # t1.append([list(p) for p in np.argmax(logits, axis=2)])
            # t2.append([p for p in np.argmax(logits, axis=2)])
            # print('Prediction={}'.format(predictions))
            # true_labels.extend([list(p) for p in label_ids])
            # for p in sub:
            #     tp1 = []
            #     for pi in p:
            #         pii = list(pi)
            #         # print(pii)
            #         tp1.append(pii)
            #     Subtoken_l1.append(tp1)
            # t2.append(t1)
            # for p in sub:
            #     pi = list(p)
            #     t
                # print(pi)
            # print('T1')
            # t1.append([list(t2.append(pi)) for p in sub for pi in p ])
            # accuracy = flat_accuracy(logits, label_ids)
            # eval_loss += loss.mean().item()
            # # eval_accuracy += accuracy
            # nb_eval_examples += ids.size(0)
            # nb_eval_steps += 1
        # eval_loss = eval_loss / nb_eval_steps
        # print('T1 ={}'.format(t1))
        # print("Validation loss: {}".format(eval_loss))
        # print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
        # # pred_tags = [id2tag[p_i] for p in predictions for p_i in p]
        # # print(pred_label)
        # valid_tags = [id2tag[l_i] for l in true_labels for l_i in l ]
    return predictions


MAX_LEN = 200
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16



class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertForTokenClassification.from_pretrained('bert-base-cased', num_labels= 18)# len(id2tag))

    def forward(self, ids, mask, labels=None):
        output_1 = self.l1(ids, mask, labels=labels)

        return output_1
# tokenizera = transformers.BertTokenizer.from_pretrained('D:\Virtual_Env\GitHub_projects\NER_USING_BERT\BERTNER\My_NER_Data')
tokenizera = transformers.BertTokenizer.from_pretrained('./My_NER_Data')
validation_text,validation_mask,validation_label,validation_sub_token = get_input_mask(val_texts,tokenizera,MAX_LEN,val_tags)
testing_set = CustomDataset(validation_text, validation_mask,validation_label,validation_sub_token)

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }

testing_loader = DataLoader(testing_set, **test_params)

# BERTNER/My_NER_Data/NER.pt
model = torch.load('./My_NER_Data/NER.pt')

predictions = valid(model, testing_loader)

tk,lb,lb_decoded,lp,lp_decoded = get_correct_data(validation_text,validation_sub_token,tokenizera,validation_label,predictions)
output = verify_accuracy(tk,lb,lp,lp_decoded)
print('Output = {}'.format(output))