import torch
import transformers
# from Declaration.Declaration import BERTClass

def dict_read(file_location):
    with open(file_location) as json_file:
        s = json_file.read()
        data = eval(s)
        # data = s #dict(s)
        # print('a')

    return data
# class BERTClass(torch.nn.Module):
#     def __init__(self):
#         super(BERTClass, self).__init__()
#         self.l1 = transformers.BertForTokenClassification.from_pretrained('bert-base-cased', len(id2tag)) #num_labels= 18)#
#
#     def forward(self, ids, mask, labels=None):
#         output_1 = self.l1(ids, mask, labels=labels)
#
#         return output_1
global tokenizera
global model
EPOCHS = 5
LEARNING_RATE = 2e-05
MAX_LEN = 200
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
id2tag = dict_read('./My_NER_Data/id2tag.json')
tag2id = dict_read('./My_NER_Data/tag2id.json')
# tokenizera = transformers.BertTokenizer.from_pretrained('./My_NER_Data')
# model = torch.load('./My_NER_Data/NER.pt')
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")