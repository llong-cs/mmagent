import json
import re
import logging
import random
from .utils.chat_api import (
    generate_messages,
    get_response_with_retry,
    parallel_get_embedding,
    get_embedding_with_retry,
)
from .utils.general import validate_and_fix_python_list
from .prompts import *
from .memory_processing import parse_video_caption
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

processing_config = json.load(open("configs/processing_config.json"))
MAX_RETRIES = processing_config["max_retries"]
# Configure logging
logger = logging.getLogger(__name__)

def translate(video_graph, memories):
    new_memories = []
    for memory in memories:
        if memory.lower().startswith("equivalence: "):
            continue
        new_memory = memory
        entities = parse_video_caption(video_graph, memory)
        entities = list(set(entities))
        for entity in entities:
            entity_str = f"{entity[0]}_{entity[1]}"
            if entity_str in video_graph.reverse_character_mappings.keys():
                new_memory = new_memory.replace(entity_str, video_graph.reverse_character_mappings[entity_str])
        new_memories.append(new_memory)
    return new_memories

def back_translate(video_graph, queries):
    translated_queries = []
    for query in queries:
        entities = parse_video_caption(video_graph, query)
        entities = list(set(entities))
        to_be_translated = [query]
        for entity in entities:
            entity_str = f"{entity[0]}_{entity[1]}"
            if entity_str in video_graph.character_mappings.keys():
                mappings = video_graph.character_mappings[entity_str]
                
                # Create new queries for each mapping
                new_queries = []
                for mapping in mappings:
                    for partially_translated in to_be_translated:
                        new_query = partially_translated.replace(entity_str, mapping)
                        new_queries.append(new_query)
                
                # Update translated_query with all variants
                to_be_translated = new_queries
                
        # Add all variants of the translated query
        translated_queries.extend(to_be_translated)
    return translated_queries

# retrieve by clip
def retrieve_from_videograph(video_graph, query, topk=5, mode='max', threshold=0):
    top_clips = []
    # find all CLIP_x in query
    pattern = r"CLIP_(\d+)"
    matches = re.finditer(pattern, query)
    top_clips = []
    for match in matches:
        try:
            clip_id = int(match.group(1))
            top_clips.append(clip_id)
        except ValueError:
            continue
    
    queries = back_translate(video_graph, [query])
    if len(queries) > 100:
        logger.error(f"Anomaly detected from query: {query}, randomly sample 100 translatedqueries")
        queries = random.sample(queries, 100)
    
    related_nodes = get_related_nodes(video_graph, query)

    model = "text-embedding-3-large"
    query_embeddings = parallel_get_embedding(model, queries)[0]

    full_clip_scores = {}
    clip_scores = {}

    if mode not in ['sum', 'max', 'mean']:
        raise ValueError(f"Unknown mode: {mode}")

    # calculate scores for each node
    nodes = video_graph.search_text_nodes(query_embeddings, related_nodes, mode='max')
    
    
    # collect node scores for each clip
    for node_id, node_score in nodes:
        clip_id = video_graph.nodes[node_id].metadata['timestamp']
        if clip_id not in full_clip_scores:
            full_clip_scores[clip_id] = []
        full_clip_scores[clip_id].append(node_score)

    # calculate scores for each clip
    for clip_id, scores in full_clip_scores.items():
        if mode == 'sum':
            clip_score = sum(scores)
        elif mode == 'max':
            clip_score = max(scores)
        elif mode == 'mean':
            clip_score = np.mean(scores)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        clip_scores[clip_id] = clip_score

    # sort clips by score
    sorted_clips = sorted(clip_scores.items(), key=lambda x: x[1], reverse=True)
    # filter out clips that have 0 score and get top k clips
    top_clips = [clip_id for clip_id, score in sorted_clips if score >= threshold][:topk]
    
    return top_clips, clip_scores, nodes

def get_related_nodes(video_graph, query):
    related_nodes = []
    entities = parse_video_caption(video_graph, query)
    for entity in entities:
        type = entity[0]
        node_id = entity[1]
        if not (f"{type}_{node_id}" in video_graph.character_mappings.keys() or f"{type}_{node_id}" in video_graph.reverse_character_mappings.keys()):
            continue
        if type == "character":
            related_nodes.extend([int(node.split("_")[1]) for node in video_graph.character_mappings[f"{type}_{node_id}"]])
        else:
            related_nodes.append(node_id)
    return list(set(related_nodes))

def generate_action(question, knowledge, retrieval_plan=None, multiple_queries=False, responses=[], switch=False, model="gpt-4o-2024-11-20"):
    # select prompt
    if not switch:
        if multiple_queries:
            prompt = prompt_generate_action_with_plan_multiple_queries
        else:
            prompt = prompt_generate_action_with_plan
            # prompt = prompt_generate_action_with_plan_multiple_queries
    else:
        logger.info(f"Route switch triggered.")
        if multiple_queries:
            prompt = prompt_generate_action_with_plan_multiple_queries_new_direction
        else:
            prompt = prompt_generate_action_with_plan_new_direction
            # prompt = prompt_generate_action_with_plan_multiple_queries_new_direction
    
    input = [
        {
            "type": "text",
            "content": prompt.format(
                question=question,
                knowledge=knowledge,
                retrieval_plan=retrieval_plan,
            )
        }
    ]
    messages = generate_messages(input)
    action_type = None
    action_content = None
    for i in range(MAX_RETRIES):
        action = get_response_with_retry(model, messages)[0]
        if "[ANSWER]" in action:
            action_type = "answer"
            reasoning = action.split("[ANSWER]")[0].strip()
            action_content = action.split("[ANSWER]")[1].strip()
        elif "[SEARCH]" in action:
            if not multiple_queries:
                action_type = "search"
                reasoning = action.split("[SEARCH]")[0].strip()
                action_content = action.split("[SEARCH]")[1].strip() 
            else:
                action_type = "search"
                reasoning = action.split("[SEARCH]")[0].strip()
                action_content = select_queries(validate_and_fix_python_list(action.split("[SEARCH]")[1].strip()), responses)
        else:
            raise ValueError(f"Unknown action type: {action}")
        if action_content is not None:
            break
    if action_content is None:
        raise Exception("Failed to generate action")
    return reasoning, action_type, action_content

def select_queries(action_content, responses):
    if not action_content:
        return None
    
    history_queries = [response["action_content"] for response in responses]
    history_embeddings = parallel_get_embedding("text-embedding-3-large", history_queries)[0]
    
    queries = action_content
    embeddings = parallel_get_embedding("text-embedding-3-large", queries)[0]
    
    # If there are no history queries, return the first query
    if not history_queries:
        return queries[0]
    
    # Calculate cosine similarity between each query and all history queries
    avg_similarities = []
    for query_embedding in embeddings:
        similarities = []
        for history_embedding in history_embeddings:
            # Compute cosine similarity
            dot_product = sum(a*b for a,b in zip(query_embedding, history_embedding))
            query_norm = sum(a*a for a in query_embedding) ** 0.5
            history_norm = sum(b*b for b in history_embedding) ** 0.5
            cos_sim = dot_product / (query_norm * history_norm)
            similarities.append(cos_sim)
        # Calculate average similarity for this query
        avg_similarity = sum(similarities) / len(similarities)
        avg_similarities.append(avg_similarity)
    
    # Return query with lowest average similarity
    min_similarity_idx = avg_similarities.index(min(avg_similarities))
    return queries[min_similarity_idx]

def search(video_graph, query, current_clips, topk=5, mode='max', threshold=0, mem_wise=False):
    top_clips, clip_scores, nodes = retrieve_from_videograph(video_graph, query, topk, mode, threshold)
    
    if mem_wise:
        new_memories = {}
        top_nodes_num = 0
        # fetch top nodes
        for top_node, _ in nodes:
            clip_id = video_graph.nodes[top_node].metadata['timestamp']
            if clip_id not in new_memories:
                new_memories[clip_id] = []
            new_ = translate(video_graph, video_graph.nodes[top_node].metadata['contents'])
            new_memories[clip_id].extend(new_)
            top_nodes_num += len(new_)
            if top_nodes_num >= topk:
                break
        # sort related_memories by timestamp
        new_memories = dict(sorted(new_memories.items(), key=lambda x: x[0]))
        new_memories = {f"CLIP_{k}": v for k, v in new_memories.items() if len(v) > 0}
        return new_memories, current_clips, clip_scores
    
    new_clips = [top_clip for top_clip in top_clips if top_clip not in current_clips]
    new_memories = {}
    current_clips.extend(new_clips)
    
    for new_clip in new_clips:
        if new_clip not in video_graph.text_nodes_by_clip:
            new_memories[new_clip] = [f"CLIP_{new_clip} not found in memory bank, please search for other information"]
        else:
            related_nodes = video_graph.text_nodes_by_clip[new_clip]
            new_memories[new_clip] = translate(video_graph, [video_graph.nodes[node_id].metadata['contents'][0] for node_id in related_nodes])
                        
    # sort related_memories by timestamp
    new_memories = dict(sorted(new_memories.items(), key=lambda x: x[0]))
    new_memories = {f"CLIP_{k}": v for k, v in new_memories.items()}
    
    return new_memories, current_clips, clip_scores

def answer_with_retrieval(video_graph, question, video_clip_base64=None, topk=5, auto_refresh=False, mode='max', multiple_queries=False, max_retrieval_steps=10, route_switch=True, threshold=0, model="gpt-4o-2024-11-20"):
    if auto_refresh:
        video_graph.refresh_equivalences()
        
    related_clips = []
    context = []

    final_answer = None
    
    memories = [[]]
    responses = []
    
    if video_clip_base64 is not None:
        input = [
            {
                "type": "video_base64/mp4",
                "content": video_clip_base64,
            },
            {
                "type": "text",
                "content": prompt_generate_plan.format(question=question),
            }
        ]

        messages = generate_messages(input)
        plan_model = "gemini-1.5-pro-002"
        retrieval_plan = get_response_with_retry(plan_model, messages)[0]
        logger.info(f"Retrieval plan: {retrieval_plan}")
    else:
        retrieval_plan = None
        
    switch = False
    for i in range(max_retrieval_steps):
        # reasoning, action_type, action_content = generate_action(question, context, retrieval_plan)
        reasoning, action_type, action_content = generate_action(question, context, retrieval_plan, multiple_queries=multiple_queries, responses=responses, switch=switch, model=model)
        reasoning = reasoning.strip("### Reasoning:").strip("### Answer or Search:").strip("Reasoning:").strip()
        if action_type == "answer":
            final_answer = action_content
            responses.append({
                "reasoning": reasoning,
                "action_type": action_type,
                "action_content": action_content
            })
            logger.info(f"Answer: {final_answer}")
            break
        elif action_type == "search":
            if i == max_retrieval_steps - 1:
                input = [
                    {
                        "type": "text",
                        "content": prompt_answer_with_retrieval_final.format(
                            question=question,
                            information=context,
                        ),
                    }
                ]
                messages = generate_messages(input)
                resp = get_response_with_retry(model, messages)[0]
                reasoning = resp.split("[ANSWER]")[0].strip()
                final_answer = resp.split("[ANSWER]")[1].strip()
                responses.append({
                    "reasoning": reasoning,
                    "action_type": "answer",
                    "action_content": final_answer
                })
                logger.info(f"Forced answer: {final_answer}")
                break
            
            new_memories, related_clips, _ = search(video_graph, action_content, related_clips, topk, mode, threshold=threshold)
            
            if len(new_memories.items()) == 0 and route_switch:
                switch = True
            else:
                switch = False
            
            context.append({
                "reasoning": reasoning,
                "query": action_content,
                "retrieved memories": new_memories
            })
            
            new_response_item = {
                "reasoning": reasoning,
                "action_type": action_type,
                "action_content": action_content
            }
            responses.append(new_response_item)
            
            new_memory_items = [{
                "clip_id": k,
                "memory": v
            } for k, v in new_memories.items()]
            memories.append(new_memory_items)
            
            if processing_config["logging"] == "DETAIL":
                logger.debug("=" * 10 + "Retrieval Step " + str(i+1) + "=" * 10)
                logger.debug(new_response_item)
                logger.debug(new_memory_items)
            
    return final_answer, (memories, responses)

def verify_qa(question, gt, pred, model="gpt-4o-2024-11-20"):
    try:
        input = [
            {
                "type": "text",
                "content": prompt_agent_verify_answer_referencing.format(
                    question=question,
                    ground_truth_answer=gt,
                    agent_answer=pred,
                ),
            }   
        ]
        messages = generate_messages(input)
        response = get_response_with_retry(model, messages)
        result = response[0]
    except Exception as e:
        logger.error(f"Error verifying qa: {question}")
        logger.error(str(e))
        return None
    return result

def calculate_similarity(mem, query, related_nodes):
    related_nodes_embeddings = np.array([mem.nodes[node_id].embeddings[0] for node_id in related_nodes])
    query_embedding = np.array(get_embedding_with_retry("text-embedding-3-large", query)[0]).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, related_nodes_embeddings)[0]
    return similarities.tolist()

if __name__ == "__main__":
    from utils.general import load_video_graph
    import base64
    processing_config["logging"] = "DETAIL"
    processing_config["topk"] = 30

    def video_to_base64(video_path):
        with open(video_path, 'rb') as video_file:
            video_bytes = video_file.read()
            base64_encoded = base64.b64encode(video_bytes).decode('utf-8')
            return base64_encoded

    video_graph_path = "/mnt/hdfs/foundation/longlin.kylin/mmagent/data/mems/CZ_1/Efk3K4epEzg_30_5_-1_10_20_0.3_0.6.pkl"
    video_graph = load_video_graph(video_graph_path)

    question = "Which collection has the highest starting price?"
    answer = answer_with_retrieval(video_graph, question, video_to_base64("/mnt/hdfs/foundation/longlin.kylin/mmagent/data/video_clips/CZ_1/Efk3K4epEzg/39.mp4"), topk=processing_config["topk"], multiple_queries=processing_config["multiple_queries"], max_retrieval_steps=processing_config["max_retrieval_steps"])
