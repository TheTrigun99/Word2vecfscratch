from datasets import load_dataset
data = load_dataset("wikitext", "wikitext-2-raw-v1")
train = data["train"]["text"]
valid = data["validation"]["text"]
test = data["test"]["text"]
print(test[0])