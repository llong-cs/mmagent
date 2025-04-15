import json
import re
from videograph import VideoGraph
from utils.chat_api import (
    generate_messages,
    get_response_with_retry,
    parallel_get_embedding,
)
from utils.general import validate_and_fix_python_list
from prompts import prompt_memory_retrieval, prompt_answer_with_retrieval_clipwise, prompt_answer_with_retrieval_clipwise_final, prompt_generate_action
from memory_processing import parse_video_caption

processing_config = json.load(open("configs/processing_config.json"))
MAX_RETRIES = processing_config["max_retries"]
max_retrieval_steps = processing_config["max_retrieval_steps"]

def translate(video_graph, memories):
    new_memories = []
    for i, memory in enumerate(memories):
        if memory.lower().startswith("equivalence: "):
            continue
        new_memory = memory
        entities = parse_video_caption(video_graph, memory)
        for entity in entities:
            entity_str = f"{entity[0]}_{entity[1]}"
            if entity_str in video_graph.reverse_character_mappings.keys():
                new_memory = new_memory.replace(entity_str, video_graph.reverse_character_mappings[entity_str])
        new_memories.append(new_memory)
    return new_memories

def back_translate(video_graph, queries):
    translated_queries = []
    for i, query in enumerate(queries):
        entities = parse_video_caption(video_graph, query)
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

# def generate_queries(question, related_memories, query_num=5):
#     input = [
#         {
#             "type": "text",
#             "content": prompt_memory_retrieval.format(
#                 question=question,
#                 query_num=query_num,
#                 knowledge=related_memories,
#             ),
#         }
#     ]
#     messages = generate_messages(input)
#     model = "gpt-4o-2024-11-20"
#     # model = "gemini-1.5-pro-002"
#     queries = None
#     for i in range(MAX_RETRIES):
#         print(f"Generating queries {i} times")
#         queries = get_response_with_retry(model, messages)[0]
#         queries = validate_and_fix_python_list(queries)
#         if queries is not None:
#             break
#     if queries is None:
#         raise Exception("Failed to generate queries")
#     return queries

# retrieve by entry
# def retrieve_from_videograph(video_graph, question, related_memories, query_num=5, topk=5):
#     queries = generate_queries(question, related_memories, query_num)
#     queries = back_translate(video_graph, queries)
#     print(f"Queries: {queries}")

#     model = "text-embedding-3-large"
#     query_embeddings = parallel_get_embedding(model, queries)[0]

#     related_nodes = []

#     for query_embedding in query_embeddings:
#         nodes = video_graph.search_text_nodes([query_embedding], threshold=0.5)
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
def retrieve_from_videograph(video_graph, query, topk=5, mode='argmax'):
    if "CLIP_" in query:
        # find CLIP_x in query
        pattern = r"CLIP_(\d+)"
        match = re.search(pattern, query)
        if match:
            clip_id = int(match.group(1))
            if clip_id in video_graph.text_nodes_by_clip:
                top_clips = [clip_id]
            else:
                raise ValueError(f"Clip {clip_id} not found in video graph")
        else:
            raise ValueError(f"Invalid query: {query}")
    else:
        queries = back_translate(video_graph, [query])

        model = "text-embedding-3-large"
        query_embeddings = parallel_get_embedding(model, queries)[0]

        clip_scores = {}

        if mode == 'argmax':
            threshold = 0
        elif mode == 'accumulate':
            threshold = 0.2
        else:
            raise ValueError(f"Unknown mode: {mode}")

        for query_embedding in query_embeddings:
            nodes = video_graph.search_text_nodes([query_embedding], threshold=threshold)
            for node in nodes:
                node_id = node[0]
                node_score = node[1]
                clip_id = video_graph.nodes[node_id].metadata['timestamp']
                if mode == 'accumulate':
                    if clip_id not in clip_scores:
                        clip_scores[clip_id] = 0
                    clip_scores[clip_id] += node_score
                elif mode == 'argmax':
                    if clip_id not in clip_scores:
                        clip_scores[clip_id] = node_score
                    elif node_score > clip_scores[clip_id]:
                        clip_scores[clip_id] = node_score
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                
        # Sort clips by score and get top k clips
        sorted_clips = sorted(clip_scores.items(), key=lambda x: x[1], reverse=True)[:topk]
        top_clips = [clip_id for clip_id, _ in sorted_clips]

    return top_clips

# def answer_with_retrieval(video_graph, question, query_num=5, topk=5, auto_refresh=False, mode='argmax'):
#     if auto_refresh:
#         video_graph.refresh_equivalences()
        
#     related_clips = []
#     related_memories = {}
    
#     context = []
    
#     for i in range(max_retrieval_steps):
#         new_clips, queries = retrieve_from_videograph(video_graph, question, related_memories, query_num, topk, mode)
#         new_clips = [new_clip for new_clip in new_clips if new_clip not in related_clips]
#         new_memories = {}
#         related_clips.extend(new_clips)
        
#         for new_clip in new_clips:
#             related_nodes = video_graph.text_nodes_by_clip[new_clip]
#             related_memories[new_clip] = translate(video_graph, [video_graph.nodes[node_id].metadata['contents'][0] for node_id in related_nodes])
#             new_memories[new_clip] = related_memories[new_clip]
            
#             print(f"New memories from clip {new_clip}: {related_memories[new_clip]}")        
            
#         # sort related_memories by timestamp
#         related_memories = dict(sorted(related_memories.items(), key=lambda x: x[0]))
#         related_memories = {f"clip_{k}": v for k, v in related_memories.items()}
#         new_memories = dict(sorted(new_memories.items(), key=lambda x: x[0]))
#         new_memories = {f"clip_{k}": v for k, v in new_memories.items()}
        
#         context.append({
#             "queries": queries,
#             "retrieved memories": new_memories
#         })

#         # replace the entities in the memories with the character mappings
#         input = [
#             {
#                 "type": "text",
#                 "content": prompt_answer_with_retrieval_clipwise.format(
#                     question=question,
#                     related_memories=json.dumps(related_memories),
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
#             question += answer[answer.find("]")+1:]
#         elif answer_type.lower() == "final":
#             break
#         else:
#             raise ValueError(f"Unknown answer type: {answer_type}")
    
#     input = [
#         {
#             "type": "text",
#             "content": prompt_answer_with_retrieval_clipwise_final.format(
#                 question=question,
#                 related_memories=json.dumps(related_memories),
#             ),
#         }
#     ]
#     messages = generate_messages(input)
#     model = "gpt-4o-2024-11-20"
#     final_answer = get_response_with_retry(model, messages)[0]
#     print(f"Final answer: {final_answer}")
    
#     return final_answer

def generate_action(question, knowledge):
    print(knowledge)
    input = [
        {
            "type": "text",
            "content": prompt_generate_action.format(
                question=question,
                knowledge=knowledge,
            ),
        }
    ]
    messages = generate_messages(input)
    model = "gpt-4o-2024-11-20"
    # model = "gemini-1.5-pro-002"
    action_type = None
    action_content = None
    for i in range(MAX_RETRIES):
        action = get_response_with_retry(model, messages)[0]
        if "[ANSWER]" in action:
            action_type = "answer"
            reasoning = action.split("[ANSWER]")[0].strip()
            action_content = action.split("[ANSWER]")[1].strip()
        elif "[SEARCH]" in action:
            action_type = "search"
            reasoning = action.split("[SEARCH]")[0].strip()
            action_content = action.split("[SEARCH]")[1].strip()
        else:
            raise ValueError(f"Unknown action type: {action}")
        if action_content is not None:
            break
    if action_content is None:
        raise Exception("Failed to generate action")
    print(action)
    return reasoning, action_type, action_content

def answer_with_retrieval(video_graph, question, topk=5, auto_refresh=False, mode='argmax'):
    if auto_refresh:
        video_graph.refresh_equivalences()
        
    related_clips = []
    context = []

    final_answer = None
    
    memories = [[]]
    responses = []
    
    for i in range(max_retrieval_steps):
        reasoning, action_type, action_content = generate_action(question, context)
        reasoning = reasoning.strip("### Reasoning:").strip("### Answer or Search:").strip("Reasoning:").strip()
        if action_type == "answer":
            final_answer = action_content
            responses.append({
                "reasoning": reasoning,
                "action_type": action_type,
                "action_content": action_content
            })
            print(f"Answer: {final_answer}")
            break
        elif action_type == "search":
            new_clips = retrieve_from_videograph(video_graph, action_content, topk, mode)
            new_clips = [new_clip for new_clip in new_clips if new_clip not in related_clips]
            new_memories = {}
            related_clips.extend(new_clips)
            
            for new_clip in new_clips:
                related_nodes = video_graph.text_nodes_by_clip[new_clip]
                new_memories[new_clip] = translate(video_graph, [video_graph.nodes[node_id].metadata['contents'][0] for node_id in related_nodes])
                                
            # sort related_memories by timestamp
            new_memories = dict(sorted(new_memories.items(), key=lambda x: x[0]))
            new_memories = {f"clip_{k}": v for k, v in new_memories.items()}
            
            context.append({
                "query": action_content,
                "retrieved memories": new_memories
            })
            
            memories.append([{
                "clip_id": k,
                "memory": v
            } for k, v in new_memories.items()])
            responses.append({
                "reasoning": reasoning,
                "action_type": action_type,
                "action_content": action_content
            })
    
    if not final_answer:
        input = [
            {
                "type": "text",
                "content": prompt_answer_with_retrieval_clipwise_final.format(
                    question=question,
                    related_memories=json.dumps(context),
                ),
            }
        ]
        messages = generate_messages(input)
        model = "gpt-4o-2024-11-20"
        final_answer = get_response_with_retry(model, messages)[0]
        print(f"Answer: {final_answer}")
    
    return final_answer, (memories, responses)

if __name__ == "__main__":
    video_graph = VideoGraph()
    question = "What is the main character's name?"
    answer = answer_with_retrieval(video_graph, question)
