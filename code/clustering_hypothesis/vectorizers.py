from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

embedder = SentenceTransformer('all-mpnet-base-v2')

def get_predicted_clusters():
    path = "/Users/anastasia/Downloads/documents"
    for file in os.listdir(path):
        filename = os.fsdecode(file)
        if filename.startswith("output"):
            number = int(filename.split("_", maxsplit=1)[1].split("_")[0])
            if number > 1000:
                continue
            filename = os.fsdecode(file)
            with open(f"{path}/{filename}", 'r') as read_file:
                paragraphs = read_file.readlines()
            corpus_embeddings = embedder.encode(paragraphs)
            num_clusters = 2
            clustering_model = KMeans(n_clusters=num_clusters)
            try:
                clustering_model.fit(corpus_embeddings)
                cluster_assignment = clustering_model.labels_
                with open(f'clustering_with_sbert/clusters_{number}.txt', 'w') as write_file:
                    write_file.write(str(cluster_assignment))
            except ValueError:
                print(number)

def get_true_clusters():
    path = "/Users/anastasia/Downloads/documents"
    for file in os.listdir(path):
        filename = os.fsdecode(file)
        if filename.startswith("output"):
            number = int(filename.split("_", maxsplit=1)[1].split("_")[0])
            if number > 1000:
                continue
            with open(f"{path}/{filename}", 'r') as read_file:
                paragraphs = read_file.readlines()
            with open(f"{path}/picks_{number}_1.txt", 'r') as read_file:
                machine_generated_paragraphs = read_file.read().split(' ')
            machine_generated_paragraphs = [int(i) for i in machine_generated_paragraphs[:-1]]
            clusters = []
            for i in range(len(paragraphs)):
                if i in machine_generated_paragraphs:
                    clusters.append(1)
                else:
                    clusters.append(0)
            clusters_str = str(clusters)
            clusters_str = clusters_str.replace(',', '')
            with open(f'clustering_with_sbert/true/clusters_{number}.txt', 'w') as write_file:
                write_file.write(clusters_str)

def compare_clusters():
    path = "clustering_with_sbert"
    path2 = "clustering_with_sbert/true"
    metrics = []
    for file in os.listdir(path):
        filename = os.fsdecode(file)
        try:
            number = int(filename.split("_", maxsplit=1)[1].split(".")[0])
        except IndexError as e:
            print(filename)
        with open(f'{path2}/clusters_{number}.txt', 'r') as true_file:
            true_cluster = true_file.read()
        with open(f'{path}/clusters_{number}.txt', 'r') as pred_file:
            pred_cluster = pred_file.read()
        true_cluster = true_cluster[1:-1].split(' ')
        pred_cluster = pred_cluster[1:-1].split(' ')
        if len(true_cluster) != len(pred_cluster):
            print(f"Lengths are not compatible for number {number}")
        else:
            correct = 0
            for i in range(len(true_cluster)):
                if true_cluster[i] == pred_cluster[i]:
                    correct += 1
            total_error = correct / len(true_cluster)
            metrics.append(total_error)
            # print(f"File number {number}: {total_error}")
    return pd.Series(metrics)

metrics = compare_clusters()
metrics.to_csv('naive_clustering.csv')


# metrics = pd.read_csv('naive_clustering.csv')
# metrics = metrics['0']
# mean_quality = metrics.mean()
# print(mean_quality)

''' Plotting '''
# sns.histplot(metrics['0'], bins=10)
# plt.xlabel('Доля корректных ответов')
# plt.ylabel('Количество документов')
# plt.show()