import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def remove_links(text_input):
    words = text_input.split()
    new_words = []
    for word in words:
        if 'http' not in word:
            new_words.append(word)
        else:
            new_words.append('<link>')
    return ' '.join(new_words)


def pre_process(text_input):
    text_input = text_input.lower()
    text_input = remove_links(text_input)
    return text_input


def get_bert_embedding(text_input):
    marked_text = "[CLS] " + text_input + " [SEP]"
    text_tokens = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(text_tokens)
    segments_ids = [1] * len(text_tokens)

    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    segments_tensors = torch.tensor([segments_ids]).to(device)

    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    model.eval()

    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)

    token_embeddings = torch.stack(encoded_layers, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1,0,2)

    token_vecs_cat = []

    for token in token_embeddings:
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
        token_vecs_cat.append(cat_vec)

    return token_vecs_cat


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
            sentence = pre_process(sentence)
            print(sentence)
            sent_embed = get_bert_embedding(sentence)
            print(sent_embed)
            print("-------------------------------------------")
            # break


if __name__== "__main__":
    read_dataset_lines("data/datasets/train/SemEval2018-T3-train-taskA.txt")
    # text = "Here is the sentence I want embeddings for."
    # output = tokenize_text(text)
    # print(output)
