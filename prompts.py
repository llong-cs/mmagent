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

prompt_generate_captions_with_ids = """You are given a video, a set of character features. Each feature (some of them may belong to the same character) can be a face image represented by a video frame with a bounding box, or can be a voice feature represented by several speech segments, each with a start time, an end time (both in MM:SS format), and the corresponding content. Each face and voice feature is identified by a unique ID enclosed in angle brackets (< >).

Additionally, you are provided with episodic history, representing events from previous consecutive clips.

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
	"speakers": {
		"<voice_1>": [
			{"start_time": "00:05", "end_time": "00:08", "asr": "Hello, everyone."},
			{"start_time": "00:09", "end_time": "00:12", "asr": "Let's get started with today's agenda."}
		],
		"<voice_2>": [
			{"start_time": "00:15", "end_time": "00:18", "asr": "Thank you for having me here."},
			{"start_time": "00:19", "end_time": "00:22", "asr": "I'm excited to share my presentation."}
		]
	},
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

Please only return the valid string list (which starts with "[" and ends with "]"), without any additional explanation or formatting."""

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
	•	Skip the speeches that can hardly be clearly recognized.
	•	Return only the valid JSON list (which starts with "[" and ends with "]") without additional explanations.
    •	If the video contains no speech, return an empty list ("[]").
	
Now generate the JSON list based on the given video:"""

prompt_generate_captions_with_ids_ = """You are given a video, a set of character features. Each feature (some of them may belong to the same character) can be a face image represented by a video frame with a bounding box, or can be a voice feature represented by several speech segments, each with a start time, an end time (both in MM:SS format), and the corresponding content. Each face and voice feature is identified by a unique ID enclosed in angle brackets (< >).

Your Task:

Using the provided feature IDs, generate a detailed and cohesive description of the current video clip. The description should capture the complete set of observable and inferable events in the clip. Your output should incorporate the following categories (but is not limited to them):

	1.	Characters' Appearance: Describe the characters' appearance, such as their clothing, facial features, or any distinguishing characteristics.
	2.	Characters' Actions & Movements: Describe specific gesture, movement, or interaction performed by the characters.
	3.	Characters' Spoken Dialogue: Transcribe or summarize what are spoken by the characters.
	4.	Characters' Contextual Behavior: Describe the characters' roles in the scene or their interaction with other characters, focusing on their behavior, emotional state, or relationships.

Strict Requirements:
	• If a character has an associated feature ID in the input context (either face or voice), refer to them **only** using that feature ID (e.g., <face_1>, <voice_2>).
	• If a character **does not** have an associated feature ID in the input context, use a short descriptive phrase (e.g., "a man in a blue shirt," "a young woman standing near the door") to refer to them.
	• Ensure accurate and consistent mapping between characters and their corresponding feature IDs when provided.
	• Each description must represent a **single atomic event or detail**. Avoid combining multiple unrelated aspects (e.g., appearance and dialogue) into one line. If a sentence can be split without losing clarity, it must be split.
	• Do not use pronouns (e.g., "he," "she," "they") or inferred names to refer to any character.
	• Include natural time expressions and physical location cues wherever inferable from the context (e.g., "in the evening at the dinner table," "early morning outside the building").
	• The generated descriptions must not invent events or characteristics not grounded in the video.
	• The final output must be a list of strings, with each string representing exactly one atomic event or description.

Example Input:

<input_video>,
"<face_1>": <img>,
"<face_2>": <img>,
"<face_3>": <img>,
"<voice_1>": [
	{"start_time": "00:05", "end_time": "00:08", "asr": "Hello, everyone."},
	{"start_time": "00:09", "end_time": "00:12", "asr": "Let's get started with today's agenda."}
],
"<voice_2>": [
	{"start_time": "00:15", "end_time": "00:18", "asr": "Thank you for having me here."},
	{"start_time": "00:19", "end_time": "00:22", "asr": "I'm excited to share my presentation."}
]

Example Output:

[
	"In the bright conference room, <face_1> enters confidently, giving a professional appearance as he approaches <face_2> to shake hands.",
	"<face_1> wears a black suit with a white shirt and tie. He has short black hair and wears glasses.",
	"<face_2>, dressed in a striking red dress with long brown hair.",
	"<face_2> smiles warmly and greets <face_1>. She then sits down at the table beside him, glancing at her phone briefly while occasionally looking up.",
	"<voice_1> speaks to the group, 'Good afternoon, everyone. Let's begin the meeting.' His voice commands attention as the room quiets, and all eyes turn to him.",
	"<face_2> listens attentively to <voice_1>'s words, nodding in agreement while still occasionally checking her phone. The atmosphere is professional, with the participants settling into their roles for the meeting.",
	"<face_1> adjusts his tie and begins discussing the agenda, engaging the participants in a productive conversation."
]

Please only return the valid string list (which starts with "[" and ends with "]"), without any additional explanation or formatting."""

prompt_generate_thinkings_with_ids = """You are given a video and a set of character features. Each feature is either a face (represented by a video frame with a bounding box) or a voice (represented by speech segments with MM:SS timestamps and transcripts). Each feature has a unique ID in angle brackets (e.g., <face_1>, <voice_2>).

Your Task:

Using the provided feature IDs, generate a list of high-level reasoning-based conclusions across the following five categories, going beyond surface-level observations:

1. Equivalence Identification

Identify which face and voice features refer to the same character.
• Use the exact format: Equivalence: <face_x>, <voice_y>.
• Include as many confident matches as possible.

2. Character-Level Attributes

Infer abstract attributes for each character, such as:
• Name (if explicitly stated),
• Personality (e.g., confident, nervous),
• Role/profession (e.g., host, newcomer),
• Interests or background (when inferable),
• Distinctive behaviors or traits (e.g., speaks formally, fidgets).
Avoid restating visual facts—focus on identity construction.

3. Interpersonal Relationships & Dynamics

Describe the relationships and interactions between characters:
• Roles (e.g., host-guest, leader-subordinate),
• Emotions or tone (e.g., respect, tension),
• Power dynamics (e.g., who leads),
• Evidence of cooperation, exclusion, conflict, etc.

4. Video-Level Plot Understanding

Summarize the scene-level narrative, such as:
• Main event or theme,
• Narrative arc or sequence (e.g., intro → discussion → reaction),
• Overall tone (e.g., formal, tense),
• Cause-effect or group dynamics.

5. Contextual & General Knowledge

Include general knowledge that can be learned from the video, such as:
• Likely setting or genre (e.g., corporate meeting, game show),
• Cultural/procedural norms,
• Real-world knowledge (e.g., "Alice market is pet-friendly"),
• Common-sense or format conventions.

Output Format:

• A Python list of concise English sentences, each expressing one high-level conclusion.
• Do not include reasoning steps or restate input observations. Only output the final conclusions.

Strict Requirements:

	• If a character has an associated feature ID in the input context (either face or voice), refer to them **only** using that feature ID (e.g., <face_1>, <voice_2>).
	• If a character **does not** have an associated feature ID in the input context, use a short descriptive phrase (e.g., "a man in a blue shirt," "a young woman standing near the door") to refer to them.
	• Ensure accurate and consistent mapping between characters and their corresponding feature IDs when provided.	
	• Do not use pronouns (e.g., "he," "she," "they") or inferred names to refer to any character.
	• Provide only the final high-level thinking conclusions, without detailing the reasoning process or restating simple observations from the video.
	• Pay more attention to features that are most likely to be the same person, using the format: "Equivalence: <face_x>, <voice_y>".
	• Your output should be a Python list of well-formed, concise English sentences (one per item).

Example Input:

<input_video>,
"<face_1>": <img>,
"<face_2>": <img>,
"<face_3>": <img>,
"<voice_1>": [
	{"start_time": "00:05", "end_time": "00:08", "asr": "Hello, everyone."},
	{"start_time": "00:09", "end_time": "00:12", "asr": "Let's get started with today's agenda."}
],
"<voice_2>": [
	{"start_time": "00:15", "end_time": "00:18", "asr": "Thank you for having me here."},
	{"start_time": "00:19", "end_time": "00:22", "asr": "I'm excited to share my presentation."}
]
"video descriptions": [
	"<face_1> wears a black suit with a white shirt and tie and has short black hair and wears glasses.",
	"<face_1> enters the conference room and shakes hands with <face_2>.",
	"<face_2> sits down at the table next to <face_1> after briefly greeting <face_1>.",
	"<face_2> waves at <face_1> while sitting at the table and checks her phone.",
	"<face_2> listens attentively to <face_1>'s speech and nods in agreement.",
]

Example Output:

[
    "Equivalence: <face_1>, <voice_1>.",
	"<face_1>'s name is David.",
	"<face_1> holds a position of authority, likely as the meeting's organizer or a senior executive.",
    "<face_2> shows social awareness and diplomacy, possibly indicating experience in public or client-facing roles.",
    "<face_1> demonstrates control and composure, suggesting a high level of professionalism and confidence under pressure.",
    "The interaction between <face_1> and <face_2> suggests a working relationship built on mutual respect.",
    "The overall tone of the meeting is structured and goal-oriented, indicating it is part of a larger organizational workflow."
]

Please only return the valid string list (which starts with "[" and ends with "]"), without any additional explanation or formatting."""

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

If the baseline answer is semantically identical to the ground truth answer, return "Yes". If the baseline answer deviates in meaning, includes incorrect details, or adds information beyond the ground truth answer, return "No".

Input Example:

{
	"question": "What is the capital of France?",
	"answer": "Paris",
	"baseline_answer": "Paris"
}

Output Example:

Yes

Please only return "Yes" or "No", without any additional explanation or formatting."""

prompt_generate_action = """You are given a question and some relevant knowledge. Your task is to reason about whether the provided knowledge is sufficient to answer the question. If it is sufficient, output [ANSWER] followed by the answer. If it is not sufficient, output [SEARCH] and generate a query that will be encoded into embeddings for a vector similarity search. The query will help retrieve additional information from a memory bank, considering both the question and the provided knowledge.

Specifically, your response should contain the following two parts:
	1.	Reasoning: First, consider the question and existing knowledge. Think about whether the current information can answer the question. If not, do some reasoning about what is the exact information that is still missing and the reason why it is important to answer the question.
	2.	Answer or Search:
	•	Answer: If you can answer the question based on the provided knowledge, output [ANSWER] and provide the answer.
	•	Search: If you cannot answer the question based on the provided knowledge, output [SEARCH] and generate a query. For the query:
		•	Identify broad topics or themes that may help answer the question. These themes should cover aspects that provide useful context or background to the question, such as character names, behaviors, relationships, personality traits, actions, and key events.
		•	Make the query concise and focused on a specific piece of information that could help answer the question. 
		•	The query should target information outside of the existing knowledge that might help answer the question.
		•	For time-sensitive or chronological information (e.g., events occurring in sequence, changes over time, or specific moments in a timeline), you can generate clip-based queries that reference specific clips or moments in time. These queries should include a reference to the clip number, indicating the index of the clip in the video (a number from 1 to N, where a smaller number indicates an earlier clip). Format these queries as "CLIP_x", where x is the clip number. Note only generate clip-based queries if the question is about a specific moment in time or a sequence of events.

Example 1:

Input:

Question: How did Sarah's relationship with her father, David, influence her decision to leave home in the story?

Knowledge: []

Output:

The provided knowledge does not contain any specific information about Sarah's relationship with her father, David, or the events leading up to her decision to leave home. To answer this question, more information is needed about their dynamic, David's behavior, and Sarah's motivations. Therefore, I will generate queries to retrieve this missing information.
[SEARCH]
"Sarah and David's father-daughter relationship dynamics."

Example 2:

Input:

Question: Who is the host of the meeting?

Knowledge:
[
    {{
		"query": "What is the name of <character_1>?",
		"retrieved new memories": {{
			"CLIP_1": [
			"<voice_1> introduces the meeting and assigns tasks to the participants.",
			"<face_2> listens attentively to <face_1> and takes notes.",
			"Equivalence: <face_1>, <voice_1>.",
			"<character_1> is the host of the meeting."
			]
		}}
    }}
]

Output:

The retrieved information clearly identifies <character_1> as the host of the meeting, as mentioned in 'clip_1'.[ANSWER] <character_1> is the host of the meeting. Next I need to find the name of <character_1>.
[SEARCH]
"What is the name of <character_1>?"

Example 3:

Input:

Question: What happened in the scene when John and Mary had their argument?

Knowledge: 
[
	{{
		"query": "What happens during the argument between John and Mary?",
		"retrieved new memories": {{
			"CLIP_1": [
			"John and Mary are seen arguing in the living room.",
			"John raises his voice, and Mary looks upset.",
			"John accuses Mary of not understanding him."
			]
		}}
	}}
]

Output:

The existing knowledge provides a general overview of the argument between John and Mary, but it does not give the full context or the resolution of the argument. To gain further insight into the emotional dynamics or possible conclusions of the scene, I will generate a query related to the next part of their interaction.
[SEARCH]
"CLIP_2"

Now, generate your response for the following input:

Question: {question}

Knowledge: {knowledge}

Output:"""

prompt_memory_retrieval = """You are given a question and some relevant knowledge. Your task is to generate a list of distinct and well-defined queries that will be encoded into embeddings and used to retrieve relevant information from a memory bank via vector similarity search. The goal is to retrieve additional information that will help answer the question, considering both the question and the provided knowledge.

For each query:
	1.	Identify broad topics or themes that may help answer the question. These themes should cover aspects that provide useful context or background to the question and go beyond the existing knowledge you have. Think about different angles, including but not limited to character names, behaviors, relationships, personality traits, actions, and key events.
	2.	Make each query concise and focused on a specific piece of information that could help answer the question, based on the broad themes you identified. The query should target information **outside of the existing knowledge** that might help answer the question.
	3.	Ensure diversity in the queries by covering different facets of the question. This includes things like character interactions, emotions, motivations, actions, key dialogue, character appearances, and context that have not yet been provided.
	4.	Avoid vague or overly broad formulations. Focus on generating queries that are actionable and specific, which will provide clear, targeted information for embedding-based retrieval.
	5.	The queries should reflect a wide variety of themes and topics, allowing the system to retrieve information from different angles that may not have been covered by the provided knowledge.

The example memory bank contains descriptions like:
	•	"<voice_0> introduces four individuals named Denny, Herm, Aaron, and JC, along with five other unnamed individuals."
	•	"<face_9> wears a black jacket, a plaid shirt, and jeans."
	•	"<face_4> points at <face_9>."
	•	"<face_1> is likely an executive or a presenter, leading a meeting."
	•	"Equivalence: <face_3>, <voice_2>"

Example 1:

Question: How did Sarah's relationship with her father, David, influence her decision to leave home in the story?

Number of Queries: 6

Knowledge:
[]

Queries:

[
	"Names of the characters.",
    "Sarah and David's father-daughter relationship dynamics.",
    "David's controlling behavior towards Sarah.",
    "How Sarah's desire for independence influenced her decision.",
    "Sarah's feelings of restriction due to David's overprotectiveness.",
    "Character traits of Sarah and David in the story."
]

Example 2:

Question: Who is the host of the meeting?

Number of Queries: 3

Knowledge:
[
	"<voice_1> introduces the meeting and assigns tasks to the participants.",
	"<face_2> listens attentively to <face_1> and takes notes.",
	"Equivalence: <face_1>, <voice_1>.",
	"<character_1> is the host of the meeting."
]

Queries:

[
	"What is the name of <face_1>?",
	"What is the name of <voice_1>?",
	"What is the name of <character_1>?"
]

Now, given the example memory bank, here is how the system will generate queries:
	1.	Identify broad themes such as character relationships, motivations, emotions, actions, and pivotal moments in the story.
	2.	Ensure the queries cover a variety of aspects of the question, such as character dynamics, feelings of restriction, desire for freedom, and key story events that can provide insight into the protagonist's decision.

Please ensure that the output queries are diverse, targeted, and well-aligned with the provided knowledge while covering different angles of the question.

Input:

Question: {question}

Number of Queries: {query_num}

Knowledge: {knowledge}

Queries:"""

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

History Information: {history_information}

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

Please only return the valid string list (which starts with "[" and ends with "]"), without any additional explanation or formatting.

Input:
{semantic_memory}

Output:"""

prompt_answer_with_retrieval = """You are given a question and a list of related memories. Your task is to answer the question based on the provided memories, ensuring that your response is clearly categorized as either an intermediate thought process or a final answer.

For each answer:
	1.	If you have gathered enough information to provide the final answer, start the response with "[FINAL]" and write the answer.
	•	In [FINAL] answers, when you need to refer to a character, use their specific names, instead of id tags like <character_1>, or ambiguous descriptions like "the man in the suit" or "the person speaking.".
	•	If the characters' names are unknown and references are needed, do not fabricate them -- return an [INTERMEDIATE] answer and explain what information is still needed.
	2.	If you have not yet gathered complete information to provide the final answer and need to express an intermediate step, start the response with "[INTERMEDIATE]".
	•	In [INTERMEDIATE] answers, when referencing characters, you can use their feature ID (e.g., <character_1>) from the memories. 
	•	In each [INTERMEDIATE] answer, include the next step or question that needs to be resolved in order to reach the final answer, such as identifying a character's name or confirming a specific event.

Strict Requirements:
	•	Do not use tags or placeholders like <character_1> in [FINAL] answers. 
	•	Do not use ambiguous descriptive references in [FINAL] answers. 
	•	The final answer should be a clear, human-readable piece of information derived from the memories. 
	•	If the characters' specific names are unknown, respond with [INTERMEDIATE] answers and include a next step of identifying the characters' names.
    
Example 1:

Question: Who is the host of the meeting?

Related Memories:

[
    "<character_1> introduces the meeting and assigns tasks to the participants.",
    "<character_2> listens attentively to <character_1> and takes notes."
]

Answer:

[INTERMEDIATE] <character_1> is the host of the meeting. Next, I need to verify the identity the name of <character_1>.

Example 2:

Question: Who is the host of the meeting?

Related Memories:

[
    "<character_1> introduces the meeting and assigns tasks to the participants.",
    "<character_2> listens attentively to <character_1> and takes notes.",
    "<character_1> says: 'My name is David.'"
]

Answer:

[FINAL] David is the host of the meeting.

Your Task:
	•	If you can definitively answer the question based on the provided memories, start your response with "[FINAL]" and avoid using  tags.
	•	If you need more information to make the final decision, provide an intermediate answer with the "[INTERMEDIATE]" tag and include  tags for relationships or inferred information.

Please ensure that the output includes only one type of answer: either "[INTERMEDIATE]" or "[FINAL]".

Question: {question}

Related Memories: {related_memories}

Answer:"""

prompt_answer_with_retrieval_clipwise = """You are given a question and a dictionary of related memories, where each key is a clip_id (a positive integer) representing a video segment in chronological order, and the corresponding value is a list of memory strings from that clip.

Your task is to answer the question based on all provided memories, ensuring that your response is clearly categorized as either an intermediate thought process or a final answer.

For each answer:
	1.	If you have gathered enough information to provide the final answer, start the response with "[FINAL]" and write the answer.
	•	In [FINAL] answers, when you need to refer to a character, use their specific names, instead of id tags like <character_1>, or ambiguous descriptions like "the man in the suit" or "the person speaking.".
	•	If the characters' names are unknown and references are needed, do not fabricate them -- return an [INTERMEDIATE] answer and explain what information is still needed.
	2.	If you have not yet gathered complete information to provide the final answer and need to express an intermediate step, start the response with "[INTERMEDIATE]".
	•	In [INTERMEDIATE] answers, when referencing characters, you can use their feature ID (e.g., <character_1>) from the memories. 
	•	In each [INTERMEDIATE] answer, include the next step or question that needs to be resolved in order to reach the final answer, such as identifying a character's name or confirming a specific event.

Example 1

Question: Who is the host of the meeting?

Related Memories:

{{
    "clip_1": [
        "<character_1> enters the meeting room and walks to the front.",
        "<character_1> introduces the meeting and assigns tasks to the participants.",
        "<character_2> listens attentively to <character_1> and takes notes."
    ]
}}

Answer:

[INTERMEDIATE] <character_1> is the host of the meeting. Next, I need to identify the name of <character_1>.

Example 2

Question: Who is the host of the meeting?

Related Memories:

{{
    "clip_1": [
        "<character_1> enters the meeting room and walks to the front.",
        "<character_1> introduces the meeting and assigns tasks to the participants.",
        "<character_2> listens attentively to <character_1> and takes notes."
    ],
    "clip_2": [
        "<character_1> says: 'My name is David.'"
    ]
}}

Answer:

[FINAL] David is the host of the meeting.


Your Task:
	•	If you can definitively answer the question based on the full memory dict, respond with a [FINAL] answer using specific names only (if references are needed).
	•	If you still need key information (such as a character’s identity), respond with an [INTERMEDIATE] answer using <ID> references, and clearly state what the next step is to reach the final answer.

Only provide one type of answer per response: either [INTERMEDIATE] or [FINAL].

Question: {question}

Related Memories: 
{related_memories}

Answer:"""

prompt_answer_with_retrieval_clipwise_final = """You are given a question and a dictionary of related memories. Each key in the dictionary is a clip_id (a positive integer), representing a video segment in chronological order. The corresponding value is a list of memory strings from that clip.

Your task is to answer the question based on all the provided memories.

Important Instructions:
	•	When referring to a character, always use their specific name if it appears in the memories.
	•	Do not use placeholder IDs like <character_1>, or vague descriptions such as "the man in the suit" or "the person speaking".
	•	Your answer should be short, clear, and directly address the question.
	•	Avoid repeating or summarizing the memories—focus only on delivering the final answer.

Question: {question}

Related Memories: {related_memories}

Answer:"""

prompt_refine_qa_list = """You are given a list of question-answer (QA) pairs derived from a video. Please revise each question and answer to ensure grammatical correctness, clarity, and formal expression. Keep the revisions as concise as possible without changing the original meaning. Return only the revised list in the same format.

Example input:

[
	{{"question": "what's the man doing?", "answer": "he fixing the car."}},
	{{"question": "why she looks angry?", "answer": "because someone take her bag."}}
]

Expected output:

[
	{{"question": "What is the man doing?", "answer": "He is repairing the car."}},
	{{"question": "Why does she appear upset?", "answer": "Because someone took her bag."}}
]

Please only return the valid json list, without any additional explanation or formatting. Now, use the same logic and structure to analyze the new input.

Input:
{qa_list}

Output:"""