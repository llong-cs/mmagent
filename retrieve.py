import json
from videograph import VideoGraph
from utils.chat_api import (
    generate_messages,
    get_response_with_retry,
    parallel_get_embedding,
)
from utils.general import validate_and_fix_python_list
from prompts import prompt_memory_retrieval, prompt_answer_with_retrieval_clipwise, prompt_answer_with_retrieval
from memory_processing import parse_video_caption

MAX_RETRIES = 3

def translate(video_graph, memories):
    for i, memory in enumerate(memories):
        entities = parse_video_caption(memory)
        for entity in entities:
            entity_str = f"{entity[0]}_{entity[1]}"
            if entity_str in video_graph.reverse_character_mappings.keys():
                memories[i] = memory.replace(entity_str, video_graph.reverse_character_mappings[entity_str])
    return memories

def back_translate(video_graph, queries):
    translated_queries = []
    for i, query in enumerate(queries):
        entities = parse_video_caption(query)
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

def generate_queries(question, related_memories, query_num=5):
    input = [
        {
            "type": "text",
            "content": prompt_memory_retrieval.format(
                question=question,
                query_num=query_num,
                knowledge=related_memories,
            ),
        }
    ]
    messages = generate_messages(input)
    model = "gpt-4o-2024-11-20"
    # model = "gemini-1.5-pro-002"
    queries = None
    for i in range(MAX_RETRIES):
        print(f"Generating queries {i} times")
        queries = get_response_with_retry(model, messages)[0]
        queries = validate_and_fix_python_list(queries)
        if queries is not None:
            break
    if queries is None:
        raise Exception("Failed to generate queries")
    return queries

# retrieve by entry
# def retrieve_from_videograph(video_graph, question, related_memories, query_num=5, topk=5):
#     queries = generate_queries(question, related_memories, query_num)
#     queries = back_translate(video_graph, queries)
#     print(f"Queries: {queries}")

#     model = "text-embedding-3-large"
#     query_embeddings = parallel_get_embedding(model, queries)[0]

#     related_nodes = []

#     for query_embedding in query_embeddings:
#         nodes = video_graph.search_text_nodes([query_embedding])
#         related_nodes.extend(nodes[:topk])

#     related_nodes = list(set(related_nodes))
#     return related_nodes

# def answer_with_retrieval(video_graph, question, query_num=5, topk=5, auto_refresh=False):
#     if auto_refresh:
#         video_graph.refresh_equivalences()
        
#     related_nodes = []
#     related_memories = []
    
#     continue_retrieving = True
#     while continue_retrieving:
#         new_nodes = retrieve_from_videograph(video_graph, question, related_memories, query_num, topk)
#         new_nodes = [new_node for new_node in new_nodes if new_node not in related_nodes]
#         related_nodes.extend(new_nodes)
            
#         new_memories = [video_graph.nodes[node_id].metadata['contents'][0] for node_id in new_nodes]
#         new_memories = translate(video_graph, new_memories)
#         print(new_memories)
#         related_memories.extend(new_memories)
#         # replace the entities in the memories with the character mappings
#         input = [
#             {
#                 "type": "text",
#                 "content": prompt_answer_with_retrieval.format(
#                     question=question,
#                     related_memories=related_memories,
#                 ),
#             }
#         ]
#         messages = generate_messages(input)
#         model = "gpt-4o-2024-11-20"
#         # model = "gemini-1.5-pro-002"
#         answer = get_response_with_retry(model, messages)[0]
#         print(answer)
        
#         answer_type = answer[answer.find("[")+1:answer.find("]")] if "[" in answer and "]" in answer else ""
#         if answer_type.lower() == "intermediate":
#             continue_retrieving = True
#             question += answer[answer.find("]")+1:]
#         elif answer_type.lower() == "final":
#             continue_retrieving = False
#         else:
#             raise ValueError(f"Unknown answer type: {answer_type}")
#     return answer

# retrieve by clip
def retrieve_from_videograph(video_graph, question, related_memories, query_num=5, topk=5):
    queries = generate_queries(question, related_memories, query_num)
    queries = back_translate(video_graph, queries)
    print(f"Queries: {queries}")

    model = "text-embedding-3-large"
    query_embeddings = parallel_get_embedding(model, queries)[0]

    clip_scores = {}

    for query_embedding in query_embeddings:
        nodes = video_graph.search_text_nodes([query_embedding])
        for node in nodes:
            node_id = node[0]
            node_score = node[1]
            clip_id = video_graph.nodes[node_id].metadata['timestamp']
            if clip_id not in clip_scores:
                clip_scores[clip_id] = 0
            clip_scores[clip_id] += node_score
            
    # Sort clips by score and get top k clips
    sorted_clips = sorted(clip_scores.items(), key=lambda x: x[1], reverse=True)[:topk]
    top_clips = [clip_id for clip_id, _ in sorted_clips]

    return top_clips

def answer_with_retrieval(video_graph, question, query_num=5, topk=5, auto_refresh=False):
    if auto_refresh:
        video_graph.refresh_equivalences()
        
    related_clips = []
    related_memories = {}
    
    continue_retrieving = True
    while continue_retrieving:
        new_clips = retrieve_from_videograph(video_graph, question, related_memories, query_num, topk)
        new_clips = [new_clip for new_clip in new_clips if new_clip not in related_clips]
        related_clips.extend(new_clips)
        
        for new_clip in new_clips:
            related_nodes = video_graph.search_text_nodes_by_clip(new_clip)
            related_memories[new_clip] = translate(video_graph, [video_graph.nodes[node_id].metadata['contents'][0] for node_id in related_nodes])
            
            print(f"New memories from clip {new_clip}: {related_memories[new_clip]}")        
            
        # sort related_memories by timestamp
        related_memories = dict(sorted(related_memories.items(), key=lambda x: x[0]))        

        # replace the entities in the memories with the character mappings
        input = [
            {
                "type": "text",
                "content": prompt_answer_with_retrieval_clipwise.format(
                    question=question,
                    related_memories=json.dumps({f"clip_{k}": v for k, v in related_memories.items()}),
                ),
            }
        ]
        messages = generate_messages(input)
        model = "gpt-4o-2024-11-20"
        # model = "gemini-1.5-pro-002"
        answer = get_response_with_retry(model, messages)[0]
        print(answer)
        
        answer_type = answer[answer.find("[")+1:answer.find("]")] if "[" in answer and "]" in answer else ""
        if answer_type.lower() == "intermediate":
            continue_retrieving = True
            question += answer[answer.find("]")+1:]
        elif answer_type.lower() == "final":
            continue_retrieving = False
        else:
            raise ValueError(f"Unknown answer type: {answer_type}")
    return answer

if __name__ == "__main__":
    video_graph = VideoGraph()
    question = "What is the main character's name?"
    answer = answer_with_retrieval(video_graph, question)
