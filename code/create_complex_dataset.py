import os
import re
import json
import nltk
import pandas as pd

dataset = pd.read_csv('/Users/anastasia/intrincsicplagiarism/data/new_version/medium.csv')
def preprocess(article):
    # delete urls
    url_pattern = re.compile(
        r"(https?:\/\/)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*")
    article = re.sub(url_pattern, '', str(article))

    # reorder paragraphs
    paragraphs = article.split('\n')
    processed_paragraphs = []
    temp_paragraph = ""
    for number, paragraph in enumerate(paragraphs):
        # remove heading
        if number == 0:
            if len(nltk.tokenize.sent_tokenize(paragraph)) == 1:
                continue

        if paragraph.lower().startswith('image source') \
                or paragraph.lower().startswith('source') \
                or paragraph.lower().startswith('photo by'):
            continue
        elif paragraph == '':
            continue

        if len(paragraph) < 55:  # probabliy it is a subheading
            continue

        if len(paragraph) < 450 and temp_paragraph == "":  # the paragraph is too small
            temp_paragraph = paragraph
        elif temp_paragraph != "":
            temp_paragraph += paragraph
            temp_paragraph += " "
            if len(temp_paragraph) >= 500:
                processed_paragraphs.append(temp_paragraph)
                temp_paragraph = ""
        else:
            processed_paragraphs.append(paragraph)

    return processed_paragraphs


def get_positions(paragraphs, picks, number):
    true_paragraphs = preprocess(dataset['0'][number])
    human_pos = []
    machine_pos = []
    prev_pos = 0
    cur_pos = 0
    text = "".join(paragraphs)
    # if len(paragraphs) == len(true_paragraphs):
    #     for i in range(len(paragraphs)):
    #         prev_pos = cur_pos
    #         cur_pos = cur_pos + len(paragraphs[i]) + 1
    #         if i not in picks:
    #             human_pos.append({"from": prev_pos, "to": cur_pos})
    #         else:
    #             machine_pos.append({"from": prev_pos, "to": cur_pos})
    # else:
    paragraphs_mapping = {}
    for i in range(len(true_paragraphs)):
        if i not in picks:
            # human
            paragraphs_mapping[i] = "human"
    for j in range(len(paragraphs)):
        if paragraphs_mapping.get(j) is None:
            paragraphs_mapping[j] = "machine"
    k = 0
    pos_start = 0
    while k < len(paragraphs):
        if paragraphs_mapping[k] == "human":
            cur_len = len(paragraphs[k])
            while k < len(paragraphs) - 1 and paragraphs_mapping[k + 1] == "human":
                k += 1
                cur_len += len(paragraphs[k])
            human_pos.append({"from": pos_start, "to": pos_start + cur_len - 1})
            pos_start = pos_start + cur_len
        elif paragraphs_mapping[k] == "machine":
            cur_len = len(paragraphs[k])
            while k < len(paragraphs) - 1 and paragraphs_mapping[k + 1] == "machine":
                k += 1
                cur_len += len(paragraphs[k]) + 1
            machine_pos.append({"from": pos_start, "to": pos_start + cur_len - 1})
            pos_start = pos_start + cur_len
        k += 1
    return human_pos, machine_pos

path = "/Users/anastasia/intrincsicplagiarism/data/new_version/documents_newest"
with open(r'/Users/anastasia/intrincsicplagiarism/data/new_version/ailed.txt', 'r') as failed_docs:
    failed_documents = failed_docs.read().split("\n")
    docs = {}
    for file in os.listdir("/Users/anastasia/intrincsicplagiarism/data/new_version/documents_newest"):
        filename = os.fsdecode(file)
        number = int(filename.split("_", maxsplit=1)[1].split("_")[0])
        if number in failed_docs:
            continue
        if filename.startswith("output"):
            with open(f"{path}/{filename}", "r") as text_file:
                paragraphs = text_file.readlines()
                # paragraphs.remove("\n")
            with open(f"/Users/anastasia/intrincsicplagiarism/data/new_version/documents_newest/picks_{number}_1.txt", "r") as picks_file:
                picks = picks_file.read()
                picks = [int(i) for i in picks.split(" ")[:-1]]
                human, machine = get_positions(paragraphs, picks, number)
            # generate files
            prefix_path = "/Users/anastasia/intrincsicplagiarism/data/gen_doc_new_version"
            with open(f'{prefix_path}/problem-{number}.txt', 'w') as f1:
                f1.write("".join(paragraphs))
            with open(f'{prefix_path}/problem-{number}.truth', 'w') as f2:
                output = {"authors": [human, machine]}
                json_object = json.dumps(output, indent=4)
                f2.write(json_object)
            with open(f'{prefix_path}//problem-{number}.meta','w') as f3:
                meta = {
                    "language": "en",
                    "type": "diarization",
                    "numAuthors": 2
                }
                json_object = json.dumps(meta, indent=4)
                f3.write(json_object)
        else:
            continue