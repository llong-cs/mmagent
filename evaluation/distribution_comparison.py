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
        for line in tqdm(f, total=sample_num, desc="Loading ours data"):
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

def get_baseline_data(file_path):
    all_queries = []
    
    sample_num = 0
    with open(file_path, 'r') as f:
        for line in f:
            sample_num += 1
    
    with open(file_path, 'r') as f:
        for line in tqdm(f, total=sample_num, desc="Loading baseline data"):
            data = json.loads(line)
            try:
                for action in data['session'][1]:
                    if action['action_type'] == 'search':
                        all_queries.append(action['action_content'])
            except:
                continue
    
    return all_queries

def plot_distribution(file_path, baseline_path, embs_path):
    if not os.path.exists(embs_path):
        print("Embeddings not found, generating...")
        mems, ours_queries, mem_embs = get_data(file_path)
        
        ours_query_embs = parallel_get_embedding("text-embedding-3-large", ours_queries)[0]
        
        baseline_queries = get_baseline_data(baseline_path)
        baseline_query_embs = parallel_get_embedding("text-embedding-3-large", baseline_queries)[0]
        
        mems_embs, ours_query_embs, baseline_query_embs = np.array(mem_embs), np.array(ours_query_embs), np.array(baseline_query_embs)
        
        np.savez(embs_path, mems_embs=mems_embs, ours_query_embs=ours_query_embs, baseline_query_embs=baseline_query_embs)
    else:
        print("Embeddings found, loading...")
        mems, ours_queries, mem_embs = get_data(file_path)
        data = np.load(embs_path)
        mems_embs, ours_query_embs, baseline_query_embs = data['mems_embs'], data['ours_query_embs'], data['baseline_query_embs']
    
    assert len(mem_embs) == len(mems)
    
    print(mems_embs.shape, ours_query_embs.shape, baseline_query_embs.shape)

    
    # Perform dimensionality reduction using PCA
    # Initialize PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    
    # Fit PCA on memory embeddings and transform both memory and query embeddings
    mem_embs_2d = pca.fit_transform(mems_embs)
    ours_query_embs_2d = pca.transform(ours_query_embs)
    baseline_query_embs_2d = pca.transform(baseline_query_embs)
    
    # Create scatter plot
    plt.figure(figsize=(12, 10))
    plt.scatter(mem_embs_2d[:, 0], mem_embs_2d[:, 1], c='lightblue', alpha=0.3, label='Memories')
    plt.scatter(ours_query_embs_2d[:, 0], ours_query_embs_2d[:, 1], c='lightpink', alpha=0.8, label='Ours Queries')
    plt.scatter(baseline_query_embs_2d[:, 0], baseline_query_embs_2d[:, 1], c='lightgreen', alpha=0.8, label='Baseline Queries')
    
    # Randomly sample and annotate some memory points
    num_samples = min(10, len(mems))  # Adjust the number of samples as needed
    sampled_indices = np.random.choice(len(mems), num_samples, replace=False)
    
    for idx in sampled_indices:
        x, y = mem_embs_2d[idx]
        plt.annotate(mems[idx][:30] + "...", (x, y), xytext=(5, 5), textcoords='offset points', fontsize=4, alpha=0.7)
        plt.plot(x, y, 'ro', markersize=5)  # Highlight the sampled points
    
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component') 
    plt.title('Distribution of Memory and Query Embeddings')
    plt.legend()
    plt.grid(True)
    
    save_path = f"data/analysis/distribution_comparison_{file_path.split('/')[-1].split('.')[0]}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    

if __name__ == "__main__":
    plot_distribution("/mnt/hdfs/foundation/agent/heyc/ckpts/Qwen3-8B/output/3.jsonl", "data/annotations/results/5_rounds_threshold_0_3_no_planning/small_test_with_agent_answer_0.jsonl", "data/analysis/Qwen3-8B_3.npz")
    