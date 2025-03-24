from videograph import VideoGraph
from utils.chat_api import generate_messages, get_response_with_retry, parallel_get_embedding
from utils.general import validate_and_fix_python_list
from prompts import prompt_memory_retrieval

MAX_RETRIES = 3

def generate_queries(question, existing_knowledge=None, query_num=1):
    input = [
        {
            "type": "text",
            "content": prompt_memory_retrieval.format(question=question, query_num=query_num, existing_knowledge=existing_knowledge),
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

def retrieve_from_videograph(videograph, question, topk=3):
    queries = generate_queries(question)
    print(f"Queries: {queries}")

    model = "text-embedding-3-large"
    query_embeddings = parallel_get_embedding(model, queries)[0]

    related_nodes = []

    for query_embedding in query_embeddings:
        nodes = videograph.search_text_nodes(query_embedding, topk)
        related_nodes.extend(nodes)

    related_nodes = list(set(related_nodes))
    return related_nodes

def answer_with_retrieval(videograph, question):
    pass

if __name__ == "__main__":
    videograph = VideoGraph()
    question = "What is the main character's name?"
    answer_with_retrieval(videograph, question)
