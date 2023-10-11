from datasets import load_dataset

from my_tokenizer.tokenizer import REMITokenizer

data_train = load_dataset("roszcz/maestro-v1-sustain", split="train")
record = data_train[0]["notes"]

tokenizer = REMITokenizer()

# Encode
tokenized_record = tokenizer.encode(record, segments=1)

decoded_record = tokenizer.decode(tokenized_record)

for token in tokenized_record:
    if token["name"] == "Bar":
        print(token)
