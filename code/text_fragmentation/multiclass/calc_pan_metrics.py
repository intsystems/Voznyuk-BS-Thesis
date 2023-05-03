import json
import os
from panmetric import IntrinsicCase, PanIntrinsicMetricGlobal

def process_predictions(test_size=10000):
    cnt = 1400
    list_of_intrinsic_cases = []
    with open("multiclass_predictions/file_test_5_authors.txt", "r") as f:
        line = f.readline()
        line = f.readline()
        while len(line) > 1:
            positions = [[], [], [], [], []]
            positions_for_pan_metric = []
            while not line.startswith("document number") and len(line) > 1:
                splitted = line.split(":")[1]
                start_end = splitted.split("+")
                start = int(start_end[0].split(" ")[2])
                end = int(start_end[1].split(",")[0].split(' ')[2])
                author = int(splitted.split(', author = ')[1][0])

                position = {
                    "from": start,
                    "to": end - 1,
                    "author": author
                }
                positions[author].append(position)
                positions_for_pan_metric.append([start, end, author])
                line = f.readline()
            output = {"authors": [[], [], [], [], []]}
            for idx in range(len(positions)):
                for elem in positions[idx]:
                    output['authors'][idx].append(elem)
            with open(f"5_authors/predicted_document_{cnt}", "w") as f1:
                json_object = json.dumps(output, indent=4)
                f1.write(json_object)
            list_of_intrinsic_cases.append(IntrinsicCase(positions_for_pan_metric, f"problem-{cnt}.truth"))
            cnt += 1
            if cnt > 1400 + test_size:
                return list_of_intrinsic_cases
            line = f.readline()
    return list_of_intrinsic_cases

def process_truth(test_size = 10000):
    dir = "generated_documents_multiclass_preprocessed_5"
    positions = []
    initial_position = 1400
    for file in os.listdir("generated_documents_multiclass_preprocessed_5"):
        filename = os.fsdecode(file)
        number = int(filename.split("-")[1].split(".")[0])
        if filename.endswith(".truth") and initial_position <= number <= initial_position + test_size:
            with open(f"{dir}/{filename}", 'r') as f:
                content = json.load(f)
                cur_author = 0
                for author in content['authors']:
                    cur_pos = []
                    for elem in author:
                        cur_pos.append([int(elem['from']), int(elem['to']) - 1, cur_author])
                    cur_author += 1
                positions.append(IntrinsicCase(cur_pos, filename))
        else:
            continue
    return positions


precision = []
recall = []
granularity = []

for test_size in range(100, 600):
    truth = process_truth(test_size)
    preds = process_predictions(test_size)
    pan_metric = PanIntrinsicMetricGlobal()
    metrics = pan_metric.calculate(truth, preds)
    precision.append(metrics['precision'])
    recall.append(metrics['recall'])
    granularity.append(metrics['granularity'])
    print(test_size)
print(precision)
print(recall)
print(granularity)

