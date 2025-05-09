import json
import matplotlib.pyplot as plt
from mmagent.utils.general import load_video_graph
from mmagent.utils.chat_api import parallel_get_embedding
from mmagent.retrieve import translate
import mmagent.videograph
import sys
import os
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
sys.modules["videograph"] = mmagent.videograph

def get_data(file_path):
    all_mems = []
    all_mem_embs = []
    all_queries = []
    mem_paths = []
    sample_num = 0
    with open(file_path, 'r') as f:
        for line in f:
            sample_num += 1

    with open(file_path, 'r') as f:
        for line in tqdm(f, total=sample_num, desc="Loading data"):
            data = json.loads(line)
            if data['mem_path'] not in mem_paths:
                mem = load_video_graph(data['mem_path'])
                mem.refresh_equivalences()
                mems = [mem.nodes[node].metadata['contents'][0] for node in mem.text_nodes]
                mems = translate(mem, mems)
                all_mems.extend(mems)
                
                mem_embs = [mem.nodes[node].embeddings[0] for node in mem.text_nodes if not mem.nodes[node].metadata['contents'][0].lower().startswith("equivalence: ")]
                all_mem_embs.extend(mem_embs)

                mem_paths.append(data['mem_path'])

            # print(len(all_mems), len(all_mem_embs))
            
            for turn in data['session']:
                if turn['role'] == 'user':
                    continue
                try:
                    action = turn['content'].split("</think>")[-1].strip()
                    type, content = action.split("Content:")
                    type, content = type.strip(), content.strip()
                    if "[Search]" in type:
                        all_queries.append(content)
                except:
                    continue
    
    print(f"Found {len(all_mems)} memories and {len(all_queries)} queries")
    
    return all_mems, all_queries, all_mem_embs

def plot_distribution(file_path, embs_path):
    if not os.path.exists(embs_path):
        mems, queries, mem_embs = get_data(file_path)
        
        assert len(mem_embs) == len(mems)
        
        query_embs = parallel_get_embedding("text-embedding-3-large", queries)[0]
        
        mems_embs, query_embs = np.array(mem_embs), np.array(query_embs)
        
        np.save(embs_path, {"mems_embs": mems_embs, "query_embs": query_embs})
    else:
        data = np.load(embs_path, allow_pickle=True)
        mems_embs, query_embs = data["mems_embs"], data["query_embs"]
    
    print(mems_embs.shape, query_embs.shape)

    
    # Perform dimensionality reduction using PCA
    # Initialize PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    
    # Fit PCA on memory embeddings and transform both memory and query embeddings
    mem_embs_2d = pca.fit_transform(mem_embs)
    query_embs_2d = pca.transform(query_embs)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(mem_embs_2d[:, 0], mem_embs_2d[:, 1], c='blue', alpha=0.5, label='Memories')
    plt.scatter(query_embs_2d[:, 0], query_embs_2d[:, 1], c='red', alpha=0.5, label='Queries')
    
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component') 
    plt.title('Distribution of Memory and Query Embeddings')
    plt.legend()
    plt.grid(True)
    
    save_path = f"data/analysis/distribution_comparison_{file_path.split('/')[-1].split('.')[0]}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path)
    plt.close()
    

if __name__ == "__main__":
    plot_distribution("/mnt/hdfs/foundation/agent/heyc/ckpts/Qwen3-8B/output/3.jsonl", "data/analysis/Qwen3-8B_3.npy")
    