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

prompt_generate_captions_with_ids = """You are given a video, a set of characters. Each character (some of them may indicate the same individual) is represented by a face image with a bounding box or several voice clips, each with a start time, an end time (both in MM:SS format), and content. Each face and voice is identified by a unique ID enclosed in angle brackets (< >).

Additionally, you are provided with episodic history, representing events from previous clips.

Your Task:

Based on the video content and episodic history, generate a structured list of detailed descriptions of what's shown in the video clip. Each description should focus on a single specific aspect, and include (but is not limited to) the following categories:

	1.	Characters' Appearance: Describe one specific aspect of the character's appearance, such as their clothing, facial features, or any distinguishing characteristics.
	2.	Characters' Actions & Movements: Describe one specific gesture, movement, or interaction performed by the character.
	3.	Characters' Spoken Dialogue: Transcribe or summarize a specific instance of speech spoken by the character.
	4.	Characters' Contextual Behavior: Describe one specific aspect of the character's role in the scene or their interaction with another character, focusing on their behavior, emotional state, or relationships.

Strict Requirements:
	•	Every reference to a character must use their exact ID enclosed in angle brackets (e.g., <face_1>, <voice_2>).
	•	Do not use generic descriptions, inferred names, or pronouns (e.g., "he," "they," "the man").
	•	Each description must focus on one specific detail and provide sufficient specificity and clarity for the given aspect.
	•	Whenever possible, include natural time expressions and physical location cues in the descriptions to improve contextual understanding. These should be based on inferred situational context (e.g., "in the evening at the dinner table," "early morning outside the building"), not on video clip timestamps.
	•	Ensure all descriptions remain consistent with the provided IDs and do not introduce assumptions beyond what can be inferred from the video and audio content.

Example Input:

{
	"video": <input_video>,
	"characters": {
		"<face_1>": <img_1>,
		"<face_2>": <img_2>,
		"<face_3>": <img_3>
	},
	"speakers": [
		{"start_time": "00:05", "end_time": "00:08", "speaker": "<voice_1>", "asr": "Hello, everyone."},
		{"start_time": "00:09", "end_time": "00:12", "speaker": "<voice_2>", "asr": "Welcome to the meeting."}
	],
	"episodic_history": [
		"<face_1> wears a black suit with a white shirt and tie.",
		"<face_1> has short black hair and wears glasses.",
		"<face_1> enters the conference room and shakes hands with <face_2>.",
		"<face_2> sits down at the table next to <face_1> after briefly greeting <face_1>.",
		"<voice_1> says: 'Good afternoon, everyone. Let's begin the meeting.'",
		"<face_2> wears a red dress and has long brown hair.",
		"<face_2> waves at <face_1> while sitting at the table and checks her phone."
	]
}

Example Output:

[
	"<face_1> adjusts his tie and starts speaking to the group.",
	"<face_2> listens attentively to <face_1>'s speech and nods in agreement.",
	"<face_3> enters the room from the back, looking a bit anxious and unsure."
]

Please only return the valid string list, without any additional explanation or formatting."""

prompt_audio_segmentation = """You are given a video. Your task is to perform Automatic Speech Recognition (ASR) and audio diarization on the provided video. Extract all speech segments with accurate timestamps and segment them by speaker turns (i.e., different speakers should have separate segments), but without assigning speaker identifiers.

Return a JSON list where each entry represents a speech segment with the following fields:
	•	start_time: Start timestamp in MM:SS format.
	•	end_time: End timestamp in MM:SS format.
	•	asr: The transcribed text for that segment.

Example Output:

[
    {"start_time": "00:05", "end_time": "00:08", "asr": "Hello, everyone."},
    {"start_time": "00:09", "end_time": "00:12", "asr": "Welcome to the meeting."}
]

Strict Requirements:

	•	Ensure precise speech segmentation with accurate timestamps.
	•	Segment based on speaker turns (i.e., different speakers' utterances should be separated).
	•	Preserve punctuation and capitalization in the ASR output.
	•	Return only the valid JSON list without additional text, explanations, or formatting."""

prompt_generate_thinkings_with_ids = """You are given a video, a set of characters. Each character (some of them may indicate the same individual) is represented by a face image with a bounding box or several voice clips, each with a start time, an end time (both in MM:SS format), and content. Each face and voice is identified by a unique ID enclosed in angle brackets (< >).

You are also provided with a detailed description of the video scene, including events, setting, background actions, and character interactions, along with episodic history to maintain temporal coherence.

Your Task:

Based on the video content and episodic descriptions, generate high-level thinking conclusions, including but not limited to:

	1.	Identifying the correspondence between faces and voices based on the video context. Specifically, find as many equivalent nodes as possible (e.g., Equivalence: <face_1>, <voice_1>).
	2.	The relationships between different characters, including their interactions, emotions, and possible connections.
	3.	Inferences about the personality traits, profession, hobbies, or distinguishing features of each character, derived from their actions, speech, and appearance.
	4.	Relevant general knowledge or contextual information that helps to understand the characters or the situation they are in.

Strict Requirements:

	•	Every reference to a character must use their exact ID enclosed in angle brackets (e.g., <face_1>, <voice_2>), instead of generic descriptions or pronouns.
	•	Focus solely on high-level conclusions derived from the video content and avoid simply repeating information already present in the descriptions or providing basic visual details.
	•	Provide only the final high-level thinking conclusions, without detailing the reasoning process or restating simple observations from the video.
	•	Always highlight equivalent nodes using the format: “Equivalence: <node_1>, <node_2>”.

Example Input:

{
	"video": <input_video>,
	"characters": {
		"<face_1>": <img_1>,
		"<face_2>": <img_2>,
		"<face_3>": <img_3>
	},
	"speakers": [
		{"start_time": "00:05", "end_time": "00:08", "speaker": "<voice_1>", "asr": "Hello, everyone."},
		{"start_time": "00:09", "end_time": "00:12", "speaker": "<voice_2>", "asr": "Welcome to the meeting."}
	],
	"episodic_history": [
		"<face_1> wears a black suit with a white shirt and tie.",
		"<face_1> has short black hair and wears glasses.",
		"<face_1> enters the conference room and shakes hands with <face_2>.",
		"<face_2> sits down at the table next to <face_1> after briefly greeting <face_1>.",
		"<voice_1> says: 'Good afternoon, everyone. Let's begin the meeting.'",
		"<face_2> wears a red dress and has long brown hair.",
		"<face_2> waves at <face_1> while sitting at the table and checks her phone."
	],
	"video_descriptions": [
		"<face_1> adjusts his tie and starts speaking to the group.",
		"<face_2> listens attentively to <face_1>'s speech and nods in agreement.",
		"<face_3> enters the room from the back, looking a bit anxious and unsure."
	]
}

Example Output:

[
    "Equivalence: <face_1>, <voice_1>.",
	"Equivalence: <face_2>, <voice_2>.",
	"<face_1>'s name is David.",
    "<voice_1>'s name is Alice.",
	"<face_1> is likely an executive or a presenter, leading a meeting.",
	"<face_2> seems to be a colleague, possibly engaged in the meeting.",
	"<face_3> appears anxious, possibly involved in a tense situation outside the meeting.",
	"<face_2> may work in a collaborative or supportive role.",
	"<face_3> likes eating at Wendy's restaurant.",
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

prompt_node_summarization = """You are an expert-level reasoning assistant. Given a specific node ID and a set of existing observations or knowledge points about this node in the format shown below, generate new high-level thinking conclusions. These conclusions should reflect abstract inferences or summarizations that go beyond simple visual facts or surface-level descriptions.

⸻

Input Format:
	•	A target node_id (e.g., <face_1>)
	•	A list of observations or knowledge points, each formatted as a short declarative sentence. For example:
		•	"<face_1> is <voice_1>."
		•	"<face_1>'s name is David."
		•	"<face_1> is likely an executive or a presenter, leading a meeting."
		•	"<face_3> appears anxious, possibly involved in a tense situation outside the meeting."
		•	"<face_2> may work in a collaborative or supportive role."

⸻

Your Task:

Generate new high-level thinking conclusions related to the given node. These should include, but are not limited to:
	1.	Identity correspondences, such as which speaker a character is, or vice versa.
	2.	Relationships between the node and other characters (e.g., social, emotional, professional).
	3.	Personality traits, roles, occupations, or behavioral patterns inferred from observed actions and dialogue.
	4.	Background knowledge or contextual reasoning that helps better understand the node's role or state in the video.

⸻

Strict Constraints:
	•	Every person mentioned must be referenced using their exact ID in angle brackets (e.g., <face_1>, <voice_2>).
	•	Do not use names, pronouns, or vague terms like "the man" or "she."
	•	Do not restate simple input facts or give low-level visual details.
	•	Only return the final high-level conclusions, omitting intermediate reasoning steps.

⸻

Example

Node: <face_1>

History Information:
[
	"<voice_1>'s name is David.",
	"<voice_1> is <face_1>.",
	"<voice_2> is <face_2>.",
	"<face_1> is leading the conversation and assigning tasks.",
	"<face_2> appears to be listening attentively and taking notes."
]

Output:
[
	"<face_1>'s name is David.",
	"<face_1> likely holds a leadership or managerial role.",
	"<face_1> appears to be in a position of authority over <face_2>."
]

⸻

Please only return the valid string list, without any additional explanation or formatting. 

Now, use the same logic and structure to analyze the new input.

Node: {node_id}

History Information:
{history_information}

Output:"""

prompt_extract_entities = """You are given a set of semantic memory, which contains various descriptions of characters, actions, interactions, and events. Each description may refer to characters, speakers, or actions and includes unique IDs enclosed in angle brackets (< >). Your task is to identify equivalent nodes that refer to the same character across different descriptions.

For each group of descriptions that refer to the same character, extract and represent them as equivalence relationships using strings in the following format: "Equivalence: <node_1>, <node_2>".

Strict Requirements:
	•	Identify all equivalent nodes, ensuring they refer to the same character or entity across different descriptions.
	•	Use the exact IDs in angle brackets (e.g., <char_1>, <speaker_2>) in your equivalence statements.
	•	Provide the output as a list of strings, each string in the form of "Equivalence: <node_1>, <node_2>".
	•	Focus on finding relationships that represent the same individual, ignoring irrelevant information or assumptions.

Example Input:

[
	"<char_1> wears a black suit and glasses.",
	"<char_1> shakes hands with <char_2>.",
	"<speaker_1> says: 'Hello, everyone.'",
	"<char_2> wears a red dress and has long brown hair.",
	"<char_2> listens attentively to <char_1>.",
	"<speaker_2> says: 'Welcome to the meeting.'",
	"<char_1> is the host of the meeting.",
	"<char_2> is a colleague of <char_1>."
	"Equivalence: <char_3>, <speaker_3>."
]

Example Output:

[
	"Equivalence: <char_1>, <speaker_1>.",
	"Equivalence: <char_2>, <speaker_2>.",
	"Equivalence: <char_3>, <speaker_3>."
]

Please only return the valid string list, without any additional explanation or formatting.

Input:
{semantic_memory}

Output:"""