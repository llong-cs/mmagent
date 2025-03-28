from videograph import Videograph
from utils.chat_api import (
    generate_messages,
    get_response_with_retry,
    parallel_get_embedding,
)
from utils.general import validate_and_fix_python_list
from prompts import prompt_memory_retrieval, prompt_answer_with_retrieval
from memory_processing import parse_video_caption

MAX_RETRIES = 3


def generate_queries(question, existing_knowledge=None, query_num=2):
    input = [
        {
            "type": "text",
            "content": prompt_memory_retrieval.format(
                question=question,
                query_num=query_num,
                existing_knowledge=existing_knowledge,
            ),
        }
    ]
    messages = generate_messages(input)
    model = "gpt-4o-2024-11-20"
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


def retrieve_from_videograph(video_graph, question, topk=5):
    queries = generate_queries(question)
    print(f"Queries: {queries}")

    model = "text-embedding-3-large"
    query_embeddings = parallel_get_embedding(model, queries)[0]

    related_nodes = []

    for query_embedding in query_embeddings:
        nodes = video_graph.search_text_nodes([query_embedding])
        related_nodes.extend(nodes[:topk])

    related_nodes = list(set(related_nodes))
    return related_nodes

def answer_with_retrieval(video_graph, question):
    related_nodes = retrieve_from_videograph(video_graph, question)
    video_graph.refresh_equivalences()
    related_memories = [video_graph.nodes[node_id].metadata['contents'][0] for node_id in related_nodes]
    # replace the entities in the memories with the character mappings
    for memory in related_memories:
        entities = parse_video_caption(memory)
        for entity in entities:
            entity_str = entity[0]+'_'+entity[1]
            if entity_str in video_graph.reverse_character_mappings:
                memory = memory.replace(entity_str, video_graph.reverse_character_mappings[entity_str])
    print(related_memories)
    input = [
        {
            "type": "text",
            "content": prompt_answer_with_retrieval.format(
                question=question,
                related_memories=related_memories,
            ),
        }
    ]
    messages = generate_messages(input)
    model = "gpt-4o-2024-11-20"
    answer = get_response_with_retry(model, messages)[0]
    print(answer)
    return related_memories, answer

if __name__ == "__main__":
    video_graph = Videograph()
    question = "What is the main character's name?"
    related_memories, answer = answer_with_retrieval(video_graph, question)
