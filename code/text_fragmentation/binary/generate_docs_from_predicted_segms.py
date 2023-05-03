import json
import os
from panmetric import IntrinsicCase, PanIntrinsicMetricGlobal


def process_predictions():
    cnt = 0
    list_of_intrinsic_cases = []
    with open("binary_predictions/file.txt", "r") as f:
        line = f.readline()
        line = f.readline()
        while len(line) > 1:
            human = True
            positions_of_human = []
            positions_of_machine = []
            positions_for_pan_metric = []
            while not line.startswith("document number") and len(line) > 1:
                splitted = line.split(":")[1]
                start_end = splitted.split("+")
                start = int(start_end[0].split(" ")[2])
                end = int(start_end[1].split(",")[0].split(' ')[2])
                #author = splitted.split(', author = ')[1][0]

                position = {
                    "from": start,
                    "to": end - 1
                }
                if human:
                    positions_of_human.append(position)
                    human = False
                else:
                    positions_for_pan_metric.append([start, end])
                    positions_of_machine.append(position)
                    human = True
                line = f.readline()
            output = {"authors": [[], []]}
            for i in positions_of_human:
                output['authors'][0].append(i)
            for j in positions_of_machine:
                output['authors'][1].append(j)
            with open(f"predictions_binary/predicted_document_{cnt}", "w") as f1:
                json_object = json.dumps(output, indent=4)
                f1.write(json_object)
            list_of_intrinsic_cases.append(IntrinsicCase(positions_for_pan_metric, f"problem-{cnt}.truth"))
            cnt += 1
            line = f.readline()
    return list_of_intrinsic_cases

def process_truth():
    dir = "generated_documents_1/train"
    positions = []
    for file in os.listdir("generated_documents_1/train"):
        filename = os.fsdecode(file)
        number = int(filename.split("-")[1].split(".")[0])
        if filename.endswith(".truth") and number >= 7000:
            with open(f"{dir}/{filename}", 'r') as f:
                content = json.load(f)
                machine_pos = content['authors'][1]
                cur_pos = []
                for elem in machine_pos:
                    cur_pos.append([int(elem['from']), int(elem['to']) - 1])
                positions.append(IntrinsicCase(cur_pos, filename))
        else:
            continue
    return positions

truth = process_truth()
preds = process_predictions()
pan_metric = PanIntrinsicMetricGlobal()
metrics = pan_metric.calculate(truth, preds)
print(metrics)

