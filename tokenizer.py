import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def tokenize_text(text_input):

    # why to add [CLS] and {SEP] to the text ?
    #marked_text = "[CLS] " + text_input + " [SEP]"

    marked_text = text_input

    # Tokenize our sentence with the BERT tokenizer.
    text_tokens = tokenizer.tokenize(marked_text)


    # return the tokens.
    return text_tokens


def read_dataset_lines(file_path):
    with open(file_path, encoding="utf8") as fp:
        lines = fp.readlines()
        lines = lines [1:len(lines)]
        for line in lines:
            line = line.split("\t")
            sentence = line[-1]
            irony = line[1]
            #print(line)
            print(sentence)
            print("irony : ", int(irony))
            print(tokenize_text(sentence))
            print("-------------------------------------------")


if __name__== "__main__":
    read_dataset_lines("SemEval2018-Task3-master/datasets/train/SemEval2018-T3-train-taskA.txt")
    text = "Here is the sentence I want embeddings for."
    output = tokenize_text(text)
    print(output)
