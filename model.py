import tensorflow as tf

from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("Aaroosh/bert-finetuned-squad")

model = AutoModelForQuestionAnswering.from_pretrained("Aaroosh/bert-finetuned-squad")

model.save_pretrained("./models/")

print('#### Done ####')

