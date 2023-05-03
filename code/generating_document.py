import json
import re
import random
import nltk

nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import pandas as pd


def extract_gpt_fragments():
    # x_valid = pd.read_json('small-117M-k40.valid.jsonl', lines=True)
    # print(x_valid.head())
    nice_fragments = []
    with open('fragments.txt', 'w') as f:
        for i in range(len(x_valid)):
            a = x_valid.iloc[i]
            if 200 <= a['length'] <= 300:
                text = a['text']
                processed_text = re.sub("\n", " ", text)
                f.write(processed_text)
                f.write("\n")
                nice_fragments.append(a['text'])
        return nice_fragments


def extract_human_fragments():
    x_valid = pd.read_csv('human_validation.csv')
    texts = x_valid['highlights'].tolist()
    return texts


def sentenize(cur_doc):
    sentences = sent_tokenize(cur_doc)
    return sentences


def generate_document_v1(num_of_doc: int, gpt: list, human: list):
    text = ""
    len_of_text = 0
    position_of_human = []
    position_of_gpt = []
    sentences = []
    changes = []
    splitted = []
    with open(f'generated_documents_1/valid/problem-{num_of_doc}.txt', 'w') as f:
        for j in range(3):
            human_fragment = random.choice(human)
            text += human_fragment
            position_of_human.append({"from": len_of_text, "to": len_of_text + len(human_fragment) - 1})
            len_of_text += len(human_fragment)
            sentences_of_fragment = sentenize(human_fragment)
            num_of_human_sent = len(sentences_of_fragment)
            sentences.extend([0] * num_of_human_sent)
            if j == 0:
                changes.extend([0] * num_of_human_sent)
            else:
                changes.append(1)
                changes.extend([0] * (num_of_human_sent - 1))
            splitted.extend(sentences_of_fragment)
            gpt_fragment = random.choice(gpt)
            text += gpt_fragment
            position_of_gpt.append({"from": len_of_text, "to": len_of_text + len(gpt_fragment) - 1})
            len_of_text += len(gpt_fragment)
            sentences_of_fragment = sentenize(gpt_fragment)
            num_of_gpt_sent = len(sentences_of_fragment)
            sentences.extend([1] * num_of_gpt_sent)
            changes.append(1)
            changes.extend([0] * (num_of_gpt_sent - 1))
            splitted.extend(sentences_of_fragment)
        f.write(text)
    output = {"authors": [[], []]}
    for i in position_of_human:
        output['authors'][0].append(i)
    for j in position_of_gpt:
        output['authors'][1].append(j)
    with open(f'generated_documents_1/valid/problem-{num_of_doc}.truth', 'w') as f2:
        json_object = json.dumps(output, indent=4)
        f2.write(json_object)

    with open(f'generated_documents_1/valid/problem-{num_of_doc}.meta', 'w') as f3:
        meta = {
            "language": "en",
            "type": "diarization",
            "numAuthors": 2
        }
        json_object = json.dumps(meta, indent=4)
        f3.write(json_object)
    return text, sentences, changes, splitted


def generate_document_v2(num_of_doc: int, gpt: list, human: list):
    text = ""
    len_of_text = 0
    position_of_human = []
    position_of_gpt = []
    sentences = []
    changes = []
    splitted = []
    with open(f'generated_documents_1/valid/problem-{num_of_doc}.txt', 'w') as f:
        for j in range(3):
            gpt_fragment = random.choice(gpt)
            text += gpt_fragment
            position_of_gpt.append({"from": len_of_text, "to": len_of_text + len(gpt_fragment) - 1})
            len_of_text += len(gpt_fragment)

            # additional data for BERT fine-tuning
            # sentences_of_fragment = sentenize(gpt_fragment)
            # num_of_gpt_sent = len(sentences_of_fragment)
            # sentences.extend([1] * num_of_gpt_sent)
            # changes.append(1)
            # changes.extend([0] * (num_of_gpt_sent - 1))
            # splitted.extend(sentences_of_fragment)

            human_fragment = random.choice(human)
            text += human_fragment
            position_of_human.append({"from": len_of_text, "to": len_of_text + len(human_fragment) - 1})
            len_of_text += len(human_fragment)

            # additional data for BERT fine-tuning
            # sentences_of_fragment = sentenize(human_fragment)
            # num_of_human_sent = len(sentences_of_fragment)
            # sentences.extend([0] * num_of_human_sent)
            # if j == 0:
            #     changes.extend([0] * num_of_human_sent)
            # else:
            #     changes.append(1)
            #     changes.extend([0] * (num_of_human_sent - 1))
            # splitted.extend(sentences_of_fragment)
        f.write(text)
    output = {"authors": [[], []]}
    for i in position_of_human:
        output['authors'][0].append(i)
    for j in position_of_gpt:
        output['authors'][1].append(j)
    with open(f'generated_documents_1/valid/problem-{num_of_doc}.truth', 'w') as f2:
        json_object = json.dumps(output, indent=4)
        f2.write(json_object)

    with open(f'generated_documents_1/valid/problem-{num_of_doc}.meta', 'w') as f3:
        meta = {
            "language": "en",
            "type": "diarization",
            "numAuthors": 2
        }
        json_object = json.dumps(meta, indent=4)
        f3.write(json_object)
    return text, sentences, changes, splitted


# picking only fragments of specific length
x = pd.read_csv('cnn_dailymail/validation.csv')
x['len_of_text'] = x['highlights'].str.len()
short_texts = x.loc[((x['len_of_text'] >= 300) & (x['len_of_text'] <= 500))]
column_st = short_texts['highlights']
column_st.to_csv('human_validation.csv')

# randomly extracting fragments
gpt_fragments = extract_gpt_fragments()
human_fragments = extract_human_fragments()
# generating a dataset for BERT fine-tuning
df = pd.DataFrame(columns=['text', 'sentence-level label', 'changes', 'splitted'])
for i in range(2000):
    text, sentence_labels, changes, splitted = generate_document_v1(i, gpt_fragments, human_fragments)
    # filling a dataset for BERT fine-tuning
    # df2 = pd.DataFrame([[text, sentence_labels, changes, splitted]], columns = ['text', 'sentence-level label', 'changes', 'splitted'], index=[i])
    # df = pd.concat([df, df2], ignore_index=False)
for i in range(2000):
    text, sentence_labels, changes, splitted = generate_document_v2(5000 + i, gpt_fragments, human_fragments)
    # filling a dataset for BERT fine-tuning
    # df2 = pd.DataFrame([[text, sentence_labels, changes, splitted]], columns = ['text', 'sentence-level label', 'changes', 'splitted'], index=[i])
    # df = pd.concat([df, df2], ignore_index=False)
# df.to_csv('dataset_valid.csv', index=False)
