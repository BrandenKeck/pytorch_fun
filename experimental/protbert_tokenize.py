import os, torch
os.environ['CURL_CA_BUNDLE'] = ""
os.environ['REQUESTS_CA_BUNDLE'] = ""

# Tokenize
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
tokens = tokenizer([
    " ".join("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    " [CLS] [UNK] [SEP] [MASK] [PAD] [PAD]"
])['input_ids']
tokenizer.pad_token
tokenizer("[PAD] [PAD]")

{
    "[PAD]": 0, "[UNK]": 1, "[CLS]": 2,
    "[SEP]": 3, "[MASK]": 4,
    "A":6, "B":27, "C":23, "D":14, "E":9, "F":19,
    "G":7, "H":22, "I":11, "J":1, "K":12, "L":5,
    "M":21, "N":17, "O":29, "P":16, "Q":18, "R":13,
    "S":10, "T":15, "U":26, "V":8, "W":24, "Y":20, "Z":28
}

# Embed
model.embeddings.word_embeddings(torch.tensor(tokens))
