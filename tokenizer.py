import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def tokenize_text(text_input):

    # why to add [CLS] and {SEP] to the text ?
    marked_text = "[CLS] " + text_input + " [SEP]"

    # marked_text = text_input

    # Tokenize our sentence with the BERT tokenizer.
    text_tokens = tokenizer.tokenize(marked_text)

    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(text_tokens)


    segments_ids = [1] * len(text_tokens)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    segments_tensors = torch.tensor([segments_ids]).to(device)

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased').to(device)

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)

    token_embeddings = torch.stack(encoded_layers, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1,0,2)

    # Stores the token vectors, with shape [22 x 3,072]
    token_vecs_cat = []

    # `token_embeddings` is a [22 x 12 x 768] tensor.

    # For each token in the sentence...
    for token in token_embeddings:
        
        # `token` is a [12 x 768] tensor

        # Concatenate the vectors (that is, append them together) from the last 
        # four layers.
        # Each layer vector is 768 values, so `cat_vec` is length 3,072.
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
        
        # Use `cat_vec` to represent `token`.
        token_vecs_cat.append(cat_vec)

    print(token_vecs_cat)
    # return the tokens.
    return text_tokens


def read_dataset_lines(file_path):
    with open(file_path, encoding="utf8") as fp:
        lines = fp.readlines()
        lines = lines [1:len(lines)]
        for i, line in enumerate(lines):
            line = line.split("\t")
            sentence = line[-1]
            irony = line[1]
            #print(line)
            print(i)
            print(sentence)
            print("irony : ", int(irony))
            print(tokenize_text(sentence))
            print("-------------------------------------------")
            if i > 1:
                break


if __name__== "__main__":
    read_dataset_lines("data/datasets/train/SemEval2018-T3-train-taskA.txt")
    text = "Here is the sentence I want embeddings for."
    output = tokenize_text(text)
    print(output)
