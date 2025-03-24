prompt_select_representative_faces = """You are given a set of facial images of a person. Your task is to select **the most distinctive and recognizable** face from the given set and return its index (e.g., 0, 1, 2, 3, ...). The selected image **should meet all the following criteria**:
	1.	Clear Facial Features: The image should clearly show the person's key facial features (eyes, nose, mouth, and overall facial structure).
	2.	Neutral or Natural Expression: The expression should be neutral or naturally pleasant (e.g., a slight smile), avoiding exaggerated emotions (e.g., extreme happiness, sadness, or surprise).
	3.	High Resolution & Clarity: The image should be sharp, with clear facial details and minimal noise or blur.
	4.	Good Lighting & Contrast: The face should be well-lit, with sufficient contrast to distinguish facial features clearly.
	5.	Minimal Occlusions: The face should be free from significant occlusions such as sunglasses, masks, or hair covering key facial features.
	6.	Representative Appearance: The selected image should best represent the person's typical appearance, avoiding extreme makeup, unusual hairstyles, or accessories that alter recognition.

If there is no qualified face in the given set, return -1.

Return only the index of the most recognizable face, without any additional explanation or formatting."""

prompt_select_representative_faces_forced = """You are given a set of facial images of a person. Your task is to select **the most distinctive and recognizable** face from the given set and return its index (e.g., 0, 1, 2, 3, ...). The selected image **should meet all the following criteria**:
	1.	Clear Facial Features: The image should clearly show the person's key facial features (eyes, nose, mouth, and overall facial structure).
	2.	Neutral or Natural Expression: The expression should be neutral or naturally pleasant (e.g., a slight smile), avoiding exaggerated emotions (e.g., extreme happiness, sadness, or surprise).
	3.	High Resolution & Clarity: The image should be sharp, with clear facial details and minimal noise or blur.
	4.	Good Lighting & Contrast: The face should be well-lit, with sufficient contrast to distinguish facial features clearly.
	5.	Minimal Occlusions: The face should be free from significant occlusions such as sunglasses, masks, or hair covering key facial features.
	6.	Representative Appearance: The selected image should best represent the person's typical appearance, avoiding extreme makeup, unusual hairstyles, or accessories that alter recognition.

Return only the index of the most recognizable face, without any additional explanation or formatting."""

prompt_classify_recognizable_faces = """You are given a single facial image. Your task is to determine whether the face in the image is sufficiently distinctive and recognizable based on the following criteria:
	1.	Clear Facial Features: The image should clearly show the person's key facial features (eyes, nose, mouth, and overall facial structure).
	2.	Neutral or Natural Expression: The expression should be neutral or naturally pleasant (e.g., a slight smile), avoiding exaggerated emotions (e.g., extreme happiness, sadness, or surprise).
	3.	High Resolution & Clarity: The image should be sharp, with clear facial details and minimal noise or blur.
	4.	Good Lighting & Contrast: The face should be well-lit, with sufficient contrast to distinguish facial features clearly.
	5.	Minimal Occlusions: The face should be free from significant occlusions such as sunglasses, masks, or hair covering key facial features.
	6.	Representative Appearance: The image should depict the person's typical appearance, avoiding extreme makeup, unusual hairstyles, or accessories that alter recognition.

If the face meets all the above criteria, return 1. Otherwise, return 0.

Return only 1 or 0, without any additional explanation or formatting."""

prompt_generate_captions_with_ids = """You are given a video and a set of characters and speakers. Each character is represented by an image with a bounding box, and each speaker is represented by several audio clips, each with a start time, an end time, and content. Each character and speaker is identified by a unique ID, which is enclosed in angle brackets (< >) and corresponds to their provided image or audio clip.

Your Task:

Analyze the video and generate a structured list of descriptions that captures all relevant details for each identified character. Each description should focus on a single aspect and include (but not limited to) the following:
	1.	Appearance: Describe one specific aspect of the character's appearance, such as their clothing, facial features, or any distinguishing characteristics. Each description should cover only one aspect (e.g., don't mix facial features with clothing).
	2.	Actions & Movements: Describe one specific gesture, movement, or interaction performed by the character. Do not mix multiple actions or interactions in a single description.
	3.	Spoken Dialogue: Transcribe or summarize a specific instance of speech spoken by the character, correctly associating it with the corresponding ID. Each description should focus on one spoken statement, ensuring that it is clear and accurately attributed.
	4.	Contextual Behavior: Describe one specific aspect of the character's role in the scene or their interaction with another character, focusing on their behavior, emotional state, or relationships. Avoid combining multiple behaviors or emotional states in a single description.

Strict Requirement:
	•	Every reference to a person must use their exact ID enclosed in angle brackets (< >).
	•	Do not use inferred names, pronouns, or generic descriptions (e.g., "the man," "the woman," "he," "they").
	•	Each description should focus on one specific detail and provide sufficient specificity and clarity for the given aspect. Avoid combining unrelated details in a single description.
	•	Ensure all descriptions remain consistent with the provided IDs and do not introduce assumptions beyond the given data.

Example Input:

{
	"video": <input_video>,
	"characters": {
		"<char_1>": <img_1>,
		"<char_2>": <img_2>,
		"<char_3>": <img_3>
	},
	"speakers": [
		{"start_time": "00:05", "end_time": "00:08", "speaker": "<speaker_1>", "asr": "Hello, everyone."},
		{"start_time": "00:09", "end_time": "00:12", "speaker": "<speaker_2>", "asr": "Welcome to the meeting."},
		...
	]
}

Example Output:

[
	"<char_1> wears a black suit with a white shirt and tie.",
    "<char_1> has short black hair and wears glasses.",
	"<char_1> enters the conference room, shakes hands with <char_2>, and takes a seat.",
	"<speaker_1> (represented by <char_1>) says: 'Good afternoon, everyone. Let's begin the meeting.'",
	"<char_2> waves at <char_1> and checks her phone.",
	"<speaker_2> (represented by <char_2>) says: 'Hey! I was waiting for you. How was your day?'",
	"<char_3> wears a white hoodie, has a beard, and wears a baseball cap.",
	"<char_3> runs across the street, looking back over his shoulder.",
	"<char_3> hides behind a car and checks his surroundings.",
	"<char_3> says: 'I think someone is following me.'"
]

Please only return the valid string list, without any additional explanation or formatting."""

prompt_audio_diarization = """You are given a video. Your task is to perform Automatic Speech Recognition (ASR) and audio diarization on the provided video. Extract all speech segments with accurate timestamps and speaker identification.

Output Format

Return a JSON list where each entry represents a speech segment with the following fields:
	•	start_time: Start timestamp in MM:SS format.
	•	end_time: End timestamp in MM:SS format.
	•	speaker: Speaker identifier in the format <speaker_X> (e.g., <speaker_1>, <speaker_2>, etc.).
	•	asr: The transcribed text for that segment.

Example Output

[
    {"start_time": "00:05", "end_time": "00:08", "speaker": "<speaker_1>", "asr": "Hello, everyone."},
    {"start_time": "00:09", "end_time": "00:12", "speaker": "<speaker_2>", "asr": "Welcome to the meeting."}
]

Requirements
	•	Ensure precise speech segmentation with accurate timestamps.
	•	Assign consistent speaker labels, meaning the same speaker should always have the same identifier (e.g., <speaker_1> remains the same throughout the output).
	•	Preserve punctuation and capitalization in the ASR output.
	•	Return only the valid json list without other additional text, explanations, or formatting."""

prompt_generate_thinkings_with_ids = """You are given a video, a set of characters, and speakers. Each character is represented by an image with a bounding box, and each speaker is represented by several audio clips, each with a start time, an end time, and content. Each character and speaker is identified by a unique ID, which is enclosed in angle brackets (< >) and corresponds to their provided image or audio clip.

You are also provided with a detailed description of the video scene, including the events happening in the video, the setting, background actions, and character interactions.

Your Task:

Based on the video content, generate high-level thinking conclusions, including but not limited to:
	1.	The correspondence between characters and speakers based on their appearance, speech, and interactions (e.g., character_id -> speaker_id).
	2.	The relationships between different characters, including their interactions, emotions, and possible connections.
	3.	Inferences about the personality traits, profession, hobbies, or distinguishing features of each character, derived from their actions, speech, and appearance.
	4.	Relevant general knowledge or contextual information that helps to understand the characters or the situation they are in.

Strict Requirement:
	•	Every reference to a person must use their exact ID enclosed in angle brackets (< >).
	•	Do not use generic descriptions, inferred names, or pronouns (e.g., "he," "they," "the man").
	•	Focus solely on high-level conclusions derived from the video content and avoid simply repeating information already present in the descriptions or providing basic visual details.
	•	Provide only the final high-level thinking conclusions, without detailing the reasoning process or restating simple observations from the video.

The input will contain the following:
	1.	Video and character details, including their IDs and relevant descriptions (no need for individual character descriptions).
	2.	A series of detailed descriptions of the video.

Example Input:

{
	"video": <input_video>,
	"characters": [
		"<char_1>": <img_1>,
		"<char_2>": <img_2>,
		"<char_3>": <img_3>
	],
	"speakers": [
		{"start_time": "00:05", "end_time": "00:08", "speaker": "<speaker_1>", "asr": "Hello, everyone."},
		{"start_time": "00:09", "end_time": "00:12", "speaker": "<speaker_2>", "asr": "Welcome to the meeting."},
		...
	],
	"video_descriptions": [
		"<char_1> wears a black suit with a white shirt and tie.", 
        "<char_1> has short black hair and wears glasses.",
		"<char_1> enters the conference room, shakes hands with <char_2>, and takes a seat.",
		"<speaker_1> (represented by <char_1>) says: 'Good afternoon, everyone. Let's begin the meeting.'",
		"<char_2> wears a red dress and has long brown hair.",
		"<char_2> walks into the restaurant, looks around, and sits at a table.",
		"<char_2> waves at <char_1> and checks her phone."
	]
}



Example Output:

[
    "<char_1> is <speaker_1>.",
	"<char_2> is <speaker_2>.",
	"<char_1>'s name is David",
    "<speaker_1>'s name is Alice",
	"<char_1> is likely an executive or a presenter, leading a meeting.",
	"<char_2> seems to be a colleague, possibly engaged in the meeting.",
	"<char_3> appears anxious, possibly involved in a tense situation outside the meeting.",
	"<char_2> may work in a collaborative or supportive role.",
	"<char_3> likes eating at Wendy's restaurant.",
	"The show is held every 2 months.",
	"Santa market is a dog-friendly market."
]

Please only return the valid string list, without any additional explanation or formatting."""

prompt_baseline_answer_clipwise_extract = """You are given a video and a question related to that video. You will be shown a specific clip from the video. Your task is to extract any relevant information from this clip that can help answer the question. If the clip does not contain any relevant or helpful information, simply respond with "none"."""

prompt_baseline_answer_clipwise_summarize = """You have reviewed all segments of a video and extracted relevant information in response to a given question. The extracted information is provided in chronological order, following the sequence of the video.

Your task is to distill the most essential core idea from all extracted information and formulate a final answer that is as concise and to the point as possible, while fully addressing the question.

Only provide the direct answer without any explanation, elaboration, or additional commentary."""

prompt_benchmark_verify_answer = """You are provided with a question, the ground truth answer, and a baseline answer. Your task is to assess whether the baseline answer is semantically consistent with the ground truth answer. If the meaning of the baseline answer aligns with the ground truth answer, regardless of exact wording, return "Yes". If the baseline answer is semantically incorrect, return "No".

Input Example:

{
	"question": "What is the capital of France?",
	"answer": "Paris",
	"baseline_answer": "Paris"
}

Output Example:

Yes

Please only return "Yes" or "No", without any additional explanation or formatting."""

prompt_benchmark_verify_answer_strict = """You are provided with a question, the ground truth answer, and a baseline answer. Your task is to strictly assess whether the baseline answer conveys exactly the same meaning as the ground truth answer, without introducing any additional information.

If the baseline answer is semantically identical to the ground truth answer, return “Yes”. If the baseline answer deviates in meaning, includes incorrect details, or adds information beyond the ground truth answer, return “No”.

Input Example:

{
	"question": "What is the capital of France?",
	"answer": "Paris",
	"baseline_answer": "Paris"
}

Output Example:

Yes

Please only return "Yes" or "No", without any additional explanation or formatting."""

prompt_memory_retrieval = """You will be given a question and some “existing knowledge” relevant to the question. Your task is to generate {query_num} distinct and well-defined queries that will be encoded into embeddings and used to retrieve relevant information from a memory bank via vector similarity search. The goal is to retrieve information that is useful for answering the question, considering both the question and the provided existing knowledge.

For each query:
	1.	Clearly define the specific information need it targets, based on your understanding of the question and the existing knowledge.
	2.	Make the query concise, focused, and semantically rich, to ensure effective encoding and retrieval.
	3.	Remember that the queries will be used for embedding-based retrieval, so avoid vague or overly broad formulations.
	4.	Ensure diversity among the queries, covering different aspects or subtopics of the original question where applicable, and incorporating the existing knowledge into the query generation.

Example Input:
Question: How did the protagonist's relationship with her father influence her decision to leave home in the story?
Existing Knowledge:
	•	The protagonist's father is portrayed as controlling and overprotective.
	•	The protagonist often feels restricted in her actions due to her father's behavior.
	•	The protagonist's decision to leave home is motivated by a desire for independence.

Example Output (as a Python-style string list):

[
	"Conflicts between the protagonist and her father", 
	"Father's actions that discouraged the protagonist's independence", 
	"Reasons the protagonist gave for leaving home " 
]

Please return the output as a valid Python string list, without any additional explanation or formatting.

Input:
{question}

Existing Knowledge:
{existing_knowledge}

Output:"""