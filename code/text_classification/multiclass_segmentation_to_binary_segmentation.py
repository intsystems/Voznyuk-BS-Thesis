import os
import json

def check_if_human_present(num_of_doc):
    with open(f"generated_documents_multiclass_preprocessed_5_FOR_BINARY/problem-{num_of_doc}.authors", "r") as f:
        content = json.load(f)
        idx = 0
        for author in content.values():
            if author == "new_human.csv":
                return True, idx
            idx += 1
    return False, -1

def process():
    documents = []
    for file in os.listdir("generated_documents_multiclass_preprocessed_5_FOR_BINARY"):
        filename = os.fsdecode(file)
        number = int(filename.split("-")[1].split(".")[0])
        lm_positions = []
        if filename.endswith(".truth"):
            with open(f"generated_documents_multiclass_preprocessed_5_FOR_BINARY/{filename}", 'r') as f:
                content = json.load(f)
                cur_author = 0
                human_present, pos_human = check_if_human_present(number)
                if not human_present:
                    continue
                for author in content['authors']:
                    if len(author) == 0:
                        continue
                    if human_present and pos_human == cur_author:
                        cur_author += 1
                        continue
                    for elem in author:
                        lm_positions.append(elem)
                    cur_author += 1
            documents.append({number:lm_positions})
        else:
            continue
    return documents

a = process()
with open("documents_multi_to_binary", "w") as f:
    content = json.dumps(a)
    f.write(content)