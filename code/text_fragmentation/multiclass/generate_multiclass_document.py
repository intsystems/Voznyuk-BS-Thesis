import json
import string

import pandas as pd
import random
from nltk import sent_tokenize
import re

#'new_xlnet.csv', 'new_gpt2.csv', 'new_human.csv', 'new_pplm.csv',
#'new_gpt.csv', 'new_grover.csv', 'new_fair.csv', 'new_xlm.csv',
#'new_ctrl.csv']

ctrl = pd.read_csv('data/multiclass_dataset/new_ctrl.csv')
gpt = pd.read_csv('data/multiclass_dataset/new_gpt.csv')
gpt2 = pd.read_csv('data/multiclass_dataset/new_gpt2.csv')
grover = pd.read_csv('data/multiclass_dataset/new_grover.csv')
xlm = pd.read_csv('data/multiclass_dataset/new_xlm.csv')
xlnet = pd.read_csv('data/multiclass_dataset/new_xlnet.csv')
pplm = pd.read_csv('data/multiclass_dataset/new_pplm.csv')
fair = pd.read_csv('data/multiclass_dataset/new_fair.csv')
human = pd.read_csv('data/multiclass_dataset/new_human.csv')

def sample_from_text(cur_model):
    obj = cur_model.sample(1)
    text = obj['Generation'].values
    text = text[0]
    return text

def remove_non_ascii(a_str):
    ascii_chars = set(string.printable)

    return ''.join(
        filter(lambda x: x in ascii_chars, a_str)
    )


def if_sample_appropriate(text):
    sentences = sent_tokenize(text)
    concatendated_sent = ""
    for sent in sentences:
        result = re.sub(r"\S*https?:\S*", "", sent)
        result = re.sub(r'[\?\!\:\;\"]', '', result)
        result = result.replace('\\', '')
        result = re.sub(r"[\*\%\&\$\+\^\@\|]", "", result)
        result = remove_non_ascii(result)
        if len(concatendated_sent + result) > 500:
            concatendated_sent += result
            concatendated_sent += '\n'
            break
        concatendated_sent += result
        concatendated_sent += '\n'
    return concatendated_sent


def extract_fragments(num_of_doc, df):
    models = [xlnet, gpt2, human, pplm, gpt, grover, fair, xlm, ctrl]
    labels = ['new_xlnet.csv', 'new_gpt2.csv', 'new_human.csv', 'new_pplm.csv',
                'new_gpt.csv', 'new_grover.csv', 'new_fair.csv', 'new_xlm.csv',
                'new_ctrl.csv']
    model_to_label = {}
    cnt = 0
    for label in labels:
        model_to_label.update({label: cnt})
        cnt += 1
    # num_authors = 3
    # num_authors = 5
    num_authors = 4
    picked_labels = random.sample(labels, num_authors)
    picked_models = [models[idx] for idx in range(len(models)) if labels[idx] in picked_labels]
    final_text = ""
    # positions = [[], [], [], [], []]
    # positions = [[], [], []]
    positions = [[], [], [], []]
    cur_labels = []
    cur_label = random.randint(0, len(picked_labels) - 1)
    previous_label = (cur_label + 1) % num_authors
    for i in range(7):
        while cur_label == previous_label:
            cur_label = random.randint(0, len(picked_labels) - 1)
        if cur_label not in cur_labels:
            cur_labels.append(cur_label)
        name_of_model = picked_labels[cur_label]
        cur_model = picked_models[cur_label]
        text = sample_from_text(cur_model)
        position_from = len(final_text)

        fragment_from_text = if_sample_appropriate(text)
        while len(fragment_from_text) > 800:
            text = sample_from_text(cur_model)
            fragment_from_text = if_sample_appropriate(text)

        positions[cur_label].append({"from": position_from, "to": position_from + len(fragment_from_text) - 1})
        final_text += fragment_from_text
        df["text"].append(fragment_from_text)
        df["author"].append(name_of_model)
        df["id"].append(model_to_label[name_of_model])
        previous_label = cur_label
        cur_label = random.randint(0, len(picked_labels) - 1)

    output = {"authors": [[], [], [], []]}
    for idx in range(len(positions)):
        for elem in positions[idx]:
            output['authors'][idx].append(elem)
    with open(f'generated_documents_multiclass_preprocessed_{num_authors}/problem-{num_of_doc}.txt', 'w') as f1:
        f1.write(final_text)
    with open(f'generated_documents_multiclass_preprocessed_{num_authors}/problem-{num_of_doc}.truth', 'w') as f2:
        json_object = json.dumps(output, indent=4)
        f2.write(json_object)
    with open(f'generated_documents_multiclass_preprocessed_{num_authors}/problem-{num_of_doc}.meta', 'w') as f3:
        meta = {
            "language": "en",
            "type": "diarization",
            "numAuthors": len(cur_labels)
        }
        json_object = json.dumps(meta, indent=4)
        f3.write(json_object)
    with open(f'generated_documents_multiclass_preprocessed_{num_authors}/problem-{num_of_doc}.authors', 'w') as f4:
        meta = {}
        for i in range(len(cur_labels)):
            meta.update({f"author{i}": f"{picked_labels[cur_labels[i]]}"})
        json_object = json.dumps(meta, indent=4)
        f4.write(json_object)

df = {"text": [], "author": [], "id": []}
for i in range(2000):
    extract_fragments(i, df)
dataframe = pd.DataFrame(data=df)
dataframe.to_csv("multiclass_dataframe.csv")
print(dataframe.head())