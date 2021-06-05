from Declaration.Declaration import perform_test,BERTClass,read_pretrained_model_tokenizer,read_training_data,Perform_Training,perform_validation
import Declaration.Declaration
from Declaration import Config

train_texts, val_texts, train_tags, val_tags = read_training_data()

model,tokenizer = Perform_Training(train_texts, train_tags)

perform_validation(val_texts,val_tags,model,tokenizer)

