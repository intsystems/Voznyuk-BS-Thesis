from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import os

embedder = SentenceTransformer('all-mpnet-base-v2')

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




