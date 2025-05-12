import json
import matplotlib.pyplot as plt
from mmagent.utils.general import load_video_graph, plot_cosine_similarity_distribution
from mmagent.utils.chat_api import parallel_get_embedding
from mmagent.retrieve import translate
import mmagent.videograph
import sys
import os
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
import argparse
from sklearn.cluster import KMeans
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

def plot_distribution(mems, mems_embs, ours_query_embs, baseline_query_embs, save_path=None, num_samples=20):
    # Perform dimensionality reduction using PCA
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
    
    # Use K-means to sample evenly distributed points
    num_samples = min(num_samples, len(mems))
    kmeans = KMeans(n_clusters=num_samples, random_state=42)
    kmeans.fit(mem_embs_2d)
    
    # Find the closest point to each cluster center
    sampled_indices = []
    for center in kmeans.cluster_centers_:
        distances = np.linalg.norm(mem_embs_2d - center, axis=1)
        closest_idx = np.argmin(distances)
        sampled_indices.append(closest_idx)
    
    # Annotate the sampled points
    for idx in sampled_indices:
        x, y = mem_embs_2d[idx]
        text = mems[idx][:50] + "..."
        # Add white background and black border
        plt.annotate(text, (x, y), 
                    xytext=(5, 5), 
                    textcoords='offset points', 
                    fontsize=4, 
                    alpha=0.7,
                    bbox=dict(facecolor='white', 
                             edgecolor='black',
                             linewidth=0.2,
                             alpha=0.7,
                             boxstyle='round,pad=0.3'))
        plt.plot(x, y, 'ro', markersize=3)  # Highlight the sampled points
    
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component') 
    plt.title('Distribution of Memory and Query Embeddings')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        # Save plot in output directory
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
def main():
    args = parse_args()
    file_path=args.ours_file
    baseline_path=args.baseline_file
    output_dir=args.output_dir
    num_samples = args.num_samples
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set embeddings path within output directory
    embs_path = os.path.join(output_dir, "embeddings.npz")
    
    if not os.path.exists(embs_path):
        print("Embeddings not found, generating...")
        mems, ours_queries, mem_embs = get_data(file_path)
        
        ours_query_embs = parallel_get_embedding("text-embedding-3-large", ours_queries)[0]
        
        baseline_queries = get_baseline_data(baseline_path)
        baseline_query_embs = parallel_get_embedding("text-embedding-3-large", baseline_queries)[0]
        
        mems_embs, ours_query_embs, baseline_query_embs = np.array(mem_embs), np.array(ours_query_embs), np.array(baseline_query_embs)
        
        np.savez(embs_path, mems=mems, mems_embs=mems_embs, ours_query_embs=ours_query_embs, baseline_query_embs=baseline_query_embs)
    else:
        print("Embeddings found, loading...")
        data = np.load(embs_path)
        mems, mems_embs, ours_query_embs, baseline_query_embs = data['mems'], data['mems_embs'], data['ours_query_embs'], data['baseline_query_embs']
    
    assert len(mems_embs) == len(mems)
    
    print(mems_embs.shape, ours_query_embs.shape, baseline_query_embs.shape)
    
    plot_cosine_similarity_distribution(mems_embs, ours_query_embs, os.path.join(output_dir, f"ours_cosine_similarity_distribution.png"))
    plot_cosine_similarity_distribution(mems_embs, baseline_query_embs, os.path.join(output_dir, f"baseline_cosine_similarity_distribution.png"))
    
    plot_distribution(mems, mems_embs, ours_query_embs, baseline_query_embs, os.path.join(output_dir, f"distribution_comparison.png"), num_samples)

def parse_args():
    parser = argparse.ArgumentParser(description='Plot distribution of memory and query embeddings')
    parser.add_argument('--ours_file', type=str, help='Path to the ours JSONL file', default="/mnt/hdfs/foundation/agent/heyc/ckpts/Qwen3-8B/output/3.jsonl")
    parser.add_argument('--baseline_file', type=str, help='Path to the baseline JSONL file', default="data/annotations/results/5_rounds_threshold_0_3_no_planning/small_test_with_agent_answer_0.jsonl")
    parser.add_argument('--output_dir', type=str, help='Directory to save the output files', default="/mnt/hdfs/foundation/longlin.kylin/mmagent/analysis/distribution")
    parser.add_argument('--num_samples', type=int, default=20, help='Number of samples to annotate (default: 20)')
    return parser.parse_args()

if __name__ == "__main__":
    main()
    