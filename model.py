import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForQuestionAnswering

print("#### Downloading DistilBERT tokenizer ####")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
tokenizer.save_pretrained("./models/tokenizer")
print("#### Downloading DistilBertForQuestionAnswering model ####")
model = TFDistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')
model.save_pretrained("./models/")

print('#### Done ####')

