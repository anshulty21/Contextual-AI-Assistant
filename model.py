from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("Aaroosh/bert-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("Aaroosh/bert-finetuned-squad")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print("Model loaded successfully!")