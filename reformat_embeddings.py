import os
import sys
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import color_scheme
from sklearn.decomposition import PCA
from testing_grounding_strategies import load_array, save_array

def reformat_openai_response(output_dir, response_file, basename):
    """
    Code takes in an openai api embedding vector file and reformats it to npy.
    """
    output_filename = os.path.join(output_dir, "openai_%s_embeddings.npy"%basename)
    if os.path.isfile(output_filename):
        return

    all_vectors = []
    with open(response_file, "r") as rfile:
        for line in rfile:
            data = json.loads(line)
            vector = data['response']['body']['data'][0]['embedding']
            all_vectors.append(vector)

    
    #stack the vectors to a certain dimension
    array = np.array(all_vectors)
    save_array(array, output_filename)

def plot_embedding_spaces(filenames):
    figure_dir = "/home/caceves/su_openai_dev/su_figures"
    X = np.empty((0, 1536))
    labels = []
    #make a PCA plot of each database
    for filename in filenames:
        tmp_label = os.path.basename(filename).replace("_embeddings.npy", "").replace("openai_","")
        tmp_array = load_array(filename) 
        X = np.vstack((tmp_array, X))
        labels.extend([tmp_label] * tmp_array.shape[0])
       
    pca = PCA(n_components=10)
    x_pca = pca.fit_transform(X)
    print(x_pca.shape)

    #convert to df for plotting
    df_pca = pd.DataFrame(x_pca[:, :3], columns=['PC1', 'PC2', 'PC3'])
    df_pca['database'] = labels

    print("Variance Captured per Component:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {var:.4f} ({var*100:.2f}%)")
        print(df_pca)
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='database', alpha=0.3)
    plt.savefig(os.path.join(figure_dir, "embedding_scatterplot.png"), dpi=300)


    plt.clf()
    plt.close()
    import umap

    X_umap = umap.UMAP(n_components=2, random_state=42).fit_transform(X)

    sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=y)
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.title("UMAP Visualization")
    plt.savefig(os.path.join(figure_dir, "embedding_umap_scatterplot.png"), dpi=300)

def main():
    openai_output_dir = "/home/caceves/su_openai_dev/batch_request_outputs"
    embeddings_output_dir = "/home/caceves/su_openai_dev/embeddings"

    #find all embedding responses in the output dir
    all_embedding_responses = [os.path.join(openai_output_dir, x) for x in os.listdir(openai_output_dir) if x.endswith("_vectors_results.json")]

    for response in all_embedding_responses:
        basename = os.path.basename(response).replace("_vectors_results.json","")
        reformat_openai_response(embeddings_output_dir, response, basename)
    
    all_finished_embeddings = [os.path.join(embeddings_output_dir, x) for x in os.listdir(embeddings_output_dir) if x.endswith(".npy")]

    plot_embedding_spaces(all_finished_embeddings)

if __name__ == "__main__":
    main()
