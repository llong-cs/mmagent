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

prompt_generate_captions_with_ids = """You are given a video, a set of character features. Each feature (some of them may belong to the same character) can be a face image represented by a video frame with a bounding box, or can be a voice feature represented by several speech segments, each with a start time, an end time (both in MM:SS format), and the corresponding content. Each face and voice feature is identified by a unique ID enclosed in angle brackets (< >).

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

prompt_generate_action = """You are given a question and some relevant knowledge about a specific video. Your task is to reason about whether the provided knowledge is sufficient to answer the question. If it is sufficient, output [ANSWER] followed by the answer. If it is not sufficient, output [SEARCH] and generate a query that will be encoded into embeddings for a vector similarity search. The query will help retrieve additional information from a memory bank that consists of detailed descriptions and high-level abstractions of the video, considering both the question and the provided knowledge.

Specifically, your response should contain the following two parts:
	1.	Reasoning: First, consider the question and existing knowledge. Think about whether the current information can answer the question. If not, do some reasoning about what is the exact information that is still missing and the reason why it is important to answer the question.
	2.	Answer or Search:
	•	Answer: If you can answer the question based on the provided knowledge, output [ANSWER] and provide the answer.
	•	Search: If you cannot answer the question based on the provided knowledge, output [SEARCH] and generate a query. For the query:
		•	Identify broad topics or themes that may help answer the question. These themes should cover aspects that provide useful context or background to the question, such as character names, behaviors, relationships, personality traits, actions, and key events.
		•	Make the query concise and focused on a specific piece of information that could help answer the question. 
		•	The query should target information outside of the existing knowledge that might help answer the question.
		•	For time-sensitive or chronological information (e.g., events occurring in sequence, changes over time, or specific moments in a timeline), you can generate clip-based queries that reference specific clips or moments in time. These queries should include a reference to the clip number, indicating the index of the clip in the video (a number from 1 to N, where a smaller number indicates an earlier clip). Format these queries as "CLIP_x", where x should be an integer that indicated the clip index. Note only generate clip-based queries if the question is about a specific moment in time or a sequence of events.
		•	You can also generate queries that focus on specific characters or characters' attributes using the id shown in the knowledge.
		•	Make sure your generated query focus on some aspects that are not retrieved or asked yet. Do not repeatedly generate queries that have high semantic similarity with those generated before.

Example 1:

Input:

Question: How did the argument between Alice and Bob influence their relationship in the story?
Knowledge:
[
	{{
		"query": "What happened during the argument between Alice and Bob?",
		"related memories": {{
			"CLIP_2": [
				"<face_1> and <face_2> are seen arguing in the living room."
				"<face_1> raises her voice, and <face_2> looks upset."
				"<face_1> accuses <face_2> of not listening to her."
			],
		}}
	}}
]

Output:

It seems that <face_1> and <face_2> are arguing about their relationship. I need to figure out the names of <face_1> and <face_2>.
[SEARCH] What are the names of <face_1> and <face_2>?

Example 2:

Input:

Question: How did the argument between Alice and Bob influence their relationship in the story?
Knowledge:
[
	{{
		"query": "What happened during the argument between Alice and Bob?",
		"related memories": {{
			"CLIP_2": [
				"<face_1> and <face_2> are seen arguing in the living room."
				"<face_1> raises her voice, and <face_2> looks upset."
				"<face_1> accuses <face_2> of not listening to her."
			],
		}}
	}},
	{{
		"query": "What are the names of <face_1> and <face_2>?",
		"related memories": {{
			"CLIP_1": [
				"<face_1> says to <face_2>: 'I am done with you Bob!'",
				"<face_2> says to <face_1>: 'What about now, Alice?'"
			],
		}}
	}}	
]

Output:

It seems that content in CLIP_2 shows exactly the argument between Alice and Bob. To figure out how did the argument between Alice and Bob influence their relationship, I need to see what happened next in CLIP_3.
[SEARCH] CLIP_3

Now, generate your response for the following input:

Question: {question}

Knowledge: {knowledge}

Output:"""

prompt_generate_plan = """You are given a clip from a specific video and a question about the video. There exists a memory bank that contains information about this video, but you will not be shown its contents.

The memory bank is structured as a temporally ordered sequence of entries. Each entry contains either:
	•	a fine-grained description of a specific moment in the video, or
	•	a high-level summary or abstraction of events.

Your task is to create a detailed and robust retrieval plan: a step-by-step outline describing what kinds of information should be retrieved from the memory bank to answer the question effectively.

Requirements:
	•	Do not answer the question.
	•	Instead, output a string list, where each item describes one retrieval step.
	•	Each step should specify a type of content, topic, or temporal segment to retrieve (e.g., "find entries describing character motivations" or "look for summaries of the climax").

Your plan must:
	1.	Ensure completeness:
		The plan must guide the retrieval process in such a way that all essential pieces of information required to answer the question will be retrieved — including context, reasoning chains, motivations, consequences, and temporal links, as relevant.
		Do not stop at partial evidence. Design the plan so that it systematically explores and gathers all necessary supporting elements.
	2.	Include contingency strategies:
		Anticipa what might go wrong or be missing during retrieval. For example:
		•	What if direct mentions of an event are not available?
		•	What if the memory bank contains conflicting interpretations?
		•	What if characters' intentions or relationships are implied but not explicitly stated?
		Your plan should include fallback options and indirect paths to cover these cases (e.g., using emotion cues, related scenes, earlier summaries, or surrounding context).
	3.	Follow a logical order:
		The steps should be ordered in a way that reflects effective reasoning — e.g., from specific to general, or from earlier scenes to later consequences.

Output format:
A list of strings. Example:

[
	"Step 1: Retrieve entries describing the initial context and setting of the video.",
	"Step 2: Look for interactions between the main characters relevant to the question.",
	"Step 3: Find summaries that explain the consequences of the key events."
]

Please response with only the string list of the plan (wrapped by "[]"), without any additional explanation or formatting.

Now start generating the plan.

Questions: {question}"""

prompt_generate_action_with_plan = """You are given a question and some relevant knowledge about a specific video. You are also provided with a retrieval plan, which outlines the types of information that should be retrieved from a memory bank in order to answer the question. Your task is to reason about whether the provided knowledge is sufficient to answer the question. If it is sufficient, output [ANSWER] followed by the answer. If it is not sufficient, output [SEARCH] and generate a query that will be encoded into embeddings for a vector similarity search. The query will help retrieve additional information from a memory bank that contains detailed descriptions and high-level abstractions of the video, considering the question, the provided knowledge, and the retrieval plan.

Your response should contain two parts:
1.	Reasoning
	•	Analyze the question, the knowledge, and the retrieval plan.
	•	If the current information is sufficient, explain why and what conclusions you can draw.
	•	If not, clearly identify what is missing and why it is important.
2.	Answer or Search
	•	[ANSWER]: If the answer can be derived from the provided knowledge, output [ANSWER] followed by a short, clear, and direct answer.
		•	When referring to a character, always use their specific name if available.
		•	Do not use ID tags like <character_1> or <face_1>.
	•	[SEARCH]: If the answer cannot be derived yet, output [SEARCH] followed by a single search query that would help retrieve the missing information.

Instructions for [SEARCH] queries:
	•	Use the retrieval plan to inform what type of content should be searched for next. These contents should cover aspects that provide useful context or background to the question, such as character names, behaviors, relationships, personality traits, actions, and key events.
	•	Use keyword-based queries, not command sentences. Queries should be written as compact keyword phrases, not as full sentences or instructions. Avoid using directive language like “Retrieve”, “Describe”, or question forms such as “What”, “When”, “How”.
	•	Keep each query short and focused on one point. Each query should target one specific type of information, without combining multiple ideas or aspects.
	•	Avoid over-complexity and unnecessary detail. Do not include too many qualifiers or conditions. Strip down to the most essential keywords needed to retrieve valuable content.
	•	The query should target information outside of the existing knowledge that might help answer the question.
	•	For time-sensitive or chronological information (e.g., events occurring in sequence, changes over time, or specific moments in a timeline), you can generate clip-based queries that reference specific clips or moments in time. These queries should include a reference to the clip number, indicating the index of the clip in the video (a number from 1 to N, where a smaller number indicates an earlier clip). Format these queries as "CLIP_x", where x should be an integer that indicates the clip index. Note only generate clip-based queries if the question is about a specific moment in time or a sequence of events.
	•	You can also generate queries that focus on specific characters or characters' attributes using the id shown in the knowledge.
	•	Make sure your generated query focus on some aspects that are not retrieved or asked yet. Do not repeatedly generate queries that have high semantic similarity with those generated before.

Example 1:

Input:

Question: How did the argument between Alice and Bob influence their relationship in the story?

Knowledge:
[
	{{
		"query": "What happened during the argument between Alice and Bob?",
		"related memories": {{
			"CLIP_2": [
				"<face_1> and <face_2> are seen arguing in the living room."
				"<face_1> raises her voice, and <face_2> looks upset."
				"<face_1> accuses <face_2> of not listening to her."
			],
		}}
	}}
]

Output:

It seems that <face_1> and <face_2> are arguing about their relationship. I need to figure out the names of <face_1> and <face_2>.
[SEARCH] What are the names of <face_1> and <face_2>?

Example 2:

Input:

Question: How did the argument between Alice and Bob influence their relationship in the story?

Knowledge:
[
	{{
		"query": "What happened during the argument between Alice and Bob?",
		"related memories": {{
			"CLIP_2": [
				"<face_1> and <face_2> are seen arguing in the living room."
				"<face_1> raises her voice, and <face_2> looks upset."
				"<face_1> accuses <face_2> of not listening to her."
			],
		}}
	}},
	{{
		"query": "What are the names of <face_1> and <face_2>?",
		"related memories": {{
			"CLIP_1": [
				"<face_1> says to <face_2>: 'I am done with you Bob!'",
				"<face_2> says to <face_1>: 'What about now, Alice?'"
			],
		}}
	}}	
]

Output:

It seems that content in CLIP_2 shows exactly the argument between Alice and Bob. To figure out how did the argument between Alice and Bob influence their relationship, I need to see what happened next in CLIP_3.
[SEARCH] What happened in CLIP_3?

Now, generate your response for the following input:

Question: {question}

Retrieval Plan: {retrieval_plan}

Knowledge: {knowledge}

Output:"""

prompt_generate_action_with_plan_new_direction = """You are given a question and some relevant knowledge about a specific video. You are also provided with a retrieval plan, which outlines the types of information that should be retrieved from a memory bank in order to answer the question. Your task is to reason about whether the provided knowledge is sufficient to answer the question.

Important Context:
The previous retrieval attempt did not return any useful new information. Therefore, you must now shift your approach and think differently. Specifically, you must identify new angles or unexplored directions based on the retrieval plan that have not yet been considered. Your goal is to create search queries that are distinct from the ones used before, aiming to retrieve different types of content that could lead to an answer.

Your response must include two parts:
1. Reasoning:
	•	Analyze the question, the provided knowledge, and the retrieval plan.
	•	Evaluate why the previous queries may have failed and what new avenues should be explored now.
	•	Identify what specific types of information are still missing and why they matter.
	•	Suggest alternative directions that have not been fully explored yet, based on the retrieval plan.
2. Answer or Search:
	•	[ANSWER]: If the answer can now be derived from the current knowledge, output [ANSWER] followed by a short, clear, and direct answer.
		•	Use specific character names if available.
		•	Do not use generic tags like <character_1> or <face_1>.
	•	[SEARCH]: If more information is needed, output [SEARCH] followed by a new search query that are different from those used in the previous retrieval attempt.
		•	The new query must reflect a change in strategy, targeting unexplored or less obvious aspects.
		•	Use the retrieval plan to guide what different types of content should be searched for (e.g., overlooked characters, background events, personality traits, contextual clues).
		•	Include CLIP-based queries only if the question relates to specific moments or sequences in time, formatted as "CLIP_x" (noting that the clip ids are ordered chronologically).
		•	Avoid repeating previous query patterns or focusing on the same semantic areas.
		•	Use keyword-based queries, not command sentences. Queries should be written as compact keyword phrases, not as full sentences or instructions. Avoid using directive language like “Retrieve”, “Describe”, or question forms such as “What”, “When”, “How”.
		•	Keep each query short and focused on one point. Each query should target one specific type of information, without combining multiple ideas or aspects.
		•	Avoid over-complexity and unnecessary detail. Do not include too many qualifiers or conditions. Strip down to the most essential keywords needed to retrieve valuable content.

Instructions for [SEARCH] queries:
	•	Reflect on what was not captured by previous queries, and pivot towards different aspects (e.g., from actions to motivations, from individuals to relationships, from events to consequences).
	•	Think about what has not yet been considered: Are there minor characters, secondary events, or hidden dynamics that might now be worth retrieving?
	•	Aim for maximum diversity and originality in your search suggestions.

Example 1:

Input:

Question: How did the argument between Alice and Bob influence their relationship in the story?

Knowledge:
[
	{{
		"query": "What happened during the argument between Alice and Bob?",
		"related memories": {{
			"CLIP_2": [
				"<face_1> and <face_2> are seen arguing in the living room."
				"<face_1> raises her voice, and <face_2> looks upset."
				"<face_1> accuses <face_2> of not listening to her."
			],
		}}
	}}
]

Output:

It seems that <face_1> and <face_2> are arguing about their relationship. I need to figure out the names of <face_1> and <face_2>.
[SEARCH] What are the names of <face_1> and <face_2>?

Example 2:

Input:

Question: How did the argument between Alice and Bob influence their relationship in the story?

Knowledge:
[
	{{
		"query": "What happened during the argument between Alice and Bob?",
		"related memories": {{
			"CLIP_2": [
				"<face_1> and <face_2> are seen arguing in the living room."
				"<face_1> raises her voice, and <face_2> looks upset."
				"<face_1> accuses <face_2> of not listening to her."
			],
		}}
	}},
	{{
		"query": "What are the names of <face_1> and <face_2>?",
		"related memories": {{
			"CLIP_1": [
				"<face_1> says to <face_2>: 'I am done with you Bob!'",
				"<face_2> says to <face_1>: 'What about now, Alice?'"
			],
		}}
	}}	
]

Output:

It seems that content in CLIP_2 shows exactly the argument between Alice and Bob. To figure out how did the argument between Alice and Bob influence their relationship, I need to see what happened next in CLIP_3.
[SEARCH] What happened in CLIP_3?

Now, generate your response for the following input:

Question: {question}

Retrieval Plan: {retrieval_plan}

Knowledge: {knowledge}

Output:"""

prompt_generate_action_with_plan_multiple_queries = """You are given a question and some relevant knowledge about a specific video. You are also provided with a retrieval plan, which outlines the types of information that should be retrieved from a memory bank in order to answer the question. Your task is to reason about whether the provided knowledge is sufficient to answer the question.

If the knowledge is sufficient, output [ANSWER] followed by the answer. If it is not sufficient, output [SEARCH] and generate five diverse queries (in the form of string list wrapped by "[]") that can be used to retrieve more information from the memory bank. The memory bank contains detailed descriptions and high-level abstractions of the video. Your queries should take into account the question, the provided knowledge, and the retrieval plan.

Your response should contain two parts:
	1.	Reasoning
	•	Analyze the question, the knowledge, and the retrieval plan.
	•	If the current information is sufficient, explain why and what conclusions you can draw.
	•	If not, clearly identify what is missing and why it is important.
	2.	Answer or Search
	•	[ANSWER]: If the answer can be derived from the provided knowledge, output [ANSWER] followed by a short, clear, and direct answer.
		•	When referring to a character, always use their specific name if available.
		•	Do not use ID tags like <character_1> or <face_1>.
	•	[SEARCH]: If the answer cannot be derived yet, output [SEARCH] followed by a list of 5 diverse search queries that would help retrieve the missing information.

Instructions for [SEARCH] queries:
	•	Use the retrieval plan to inform what type of content should be searched for next. These contents should cover aspects that provide useful context or background to the question, such as character names, behaviors, relationships, personality traits, actions, and key events.
	•	Use keyword-based queries, not command sentences. Queries should be written as compact keyword phrases, not as full sentences or instructions. Avoid using directive language like “Retrieve”, “Describe”, or question forms such as “What”, “When”, “How”.
	•	Keep each query short and focused on one point. Each query should target one specific type of information, without combining multiple ideas or aspects.
	•	Avoid over-complexity and unnecessary detail. Do not include too many qualifiers or conditions. Strip down to the most essential keywords needed to retrieve valuable content.
	•	The query should target information outside of the existing knowledge that might help answer the question.
	•	For time-sensitive or chronological information (e.g., events occurring in sequence, changes over time, or specific moments in a timeline), you can generate clip-based queries that reference specific clips or moments in time. These queries should include a reference to the clip number, indicating the index of the clip in the video (a number from 1 to N, where a smaller number indicates an earlier clip). Format these queries as "CLIP_x", where x should be an integer that indicates the clip index. Note only generate clip-based queries if the question is about a specific moment in time or a sequence of events.
	•	You can also generate queries that focus on specific characters or characters' attributes using the id shown in the knowledge.
	•	Make sure your generated query focus on some aspects that are not retrieved or asked yet. Do not repeatedly generate queries that have high semantic similarity with those generated before.
	•	Ensure diversity: the five queries must not be semantically redundant. Each query should explore a distinct direction toward answering the question.
	•	Format the queries as a **Python-style string list wrapped by "[]"**: [SEARCH] ["What does Bob do after the argument?", "How does Alice react in CLIP_3?", "What is the emotional state of Alice after CLIP_2?", "What conclusions are drawn in high-level summaries about Alice and Bob's relationship?", "Does CLIP_4 show any reconciliation or continued conflict?"]

Example 1:

Input:

Question: How did the argument between Alice and Bob influence their relationship in the story?

Knowledge:
[
	{{
		"query": "What happened during the argument between Alice and Bob?",
		"related memories": {{
			"CLIP_2": [
				"<face_1> and <face_2> are seen arguing in the living room."
				"<face_1> raises her voice, and <face_2> looks upset."
				"<face_1> accuses <face_2> of not listening to her."
			],
		}}
	}}
]

Output:

It seems that <face_1> and <face_2> are engaged in an argument, but their identities are not yet known, and there is no information about the consequences of the argument. To understand how it influenced their relationship, I need more contextual information about their identities, reactions, and what happened after.
[SEARCH] ["What are the names of <face_1> and <face_2>?", "What is the emotional state of <face_1> and <face_2> after the argument?", "What happens immediately after CLIP_2?", "Is there a summary indicating a change in the relationship between these two characters?", "Do any later clips show reconciliation or continued conflict between <face_1> and <face_2>?"]

Example 2:

Input:

Question: How did the argument between Alice and Bob influence their relationship in the story?

Knowledge:
[
	{{
		"query": "What happened during the argument between Alice and Bob?",
		"related memories": {{
			"CLIP_2": [
				"<face_1> and <face_2> are seen arguing in the living room."
				"<face_1> raises her voice, and <face_2> looks upset."
				"<face_1> accuses <face_2> of not listening to her."
			],
		}}
	}},
	{{
		"query": "What are the names of <face_1> and <face_2>?",
		"related memories": {{
			"CLIP_1": [
				"<face_1> says to <face_2>: 'I am done with you Bob!'",
				"<face_2> says to <face_1>: 'What about now, Alice?'"
			],
		}}
	}}	
]

Output:

CLIP_1 identifies <face_1> as Alice and <face_2> as Bob. CLIP_2 shows the argument between them. However, the influence of this argument on their relationship is not yet clear — we need to know what happened afterward and whether their interaction changed.
[SEARCH] ["What happens in CLIP_3 after the argument?", "How does Alice behave toward Bob after the argument?", "Are there any summaries indicating a shift in Alice and Bob's relationship?", "Do Alice and Bob interact again in later clips?", "Is there any indication that their relationship improves or deteriorates after CLIP_2?"]

Now, generate your response for the following input:

Question: {question}

Retrieval Plan: {retrieval_plan}

Knowledge: {knowledge}

Output:"""

prompt_generate_action_with_plan_multiple_queries_new_direction = """You are given a question and some relevant knowledge about a specific video. You are also provided with a retrieval plan, which outlines the types of information that should be retrieved from a memory bank in order to answer the question. Your task is to reason about whether the provided knowledge is sufficient to answer the question.

Important Note:
The previous retrieval attempt did not return any useful new information. Therefore, you must now change your approach.
You need to think differently and generate new types of queries that explore alternative directions based on the retrieval plan. Your new queries must be distinct from the ones used before, targeting different aspects or underexplored areas in order to uncover useful content.


If the knowledge is sufficient, output [ANSWER] followed by the answer. If it is not sufficient, output [SEARCH] and generate five diverse and novel queries (in the form of a string list wrapped by "[]") that can be used to retrieve more information from the memory bank. The memory bank contains detailed descriptions and high-level abstractions of the video. Your queries should take into account the question, the provided knowledge, and the retrieval plan.

Your response must contain two parts:
1. Reasoning:
	•	Analyze the question, the knowledge, and the retrieval plan.
	•	Evaluate why the previous queries might have failed, and identify what new areas or different perspectives can be explored now.
	•	Clearly explain what information is still missing, and why it matters.
	•	Suggest what alternative retrieval directions could be valuable, based on the retrieval plan but not yet fully explored.
2. Answer or Search:
	•	[ANSWER]: If the answer can now be derived from the current knowledge, output [ANSWER] followed by a short, clear, and direct answer.
		•	Always use specific character names if available.
		•	Do not use ID tags like <character_1> or <face_1>.
	•	[SEARCH]: If more information is needed, output [SEARCH] followed by a list of 5 new, diverse, and exploratory search queries that reflect a shift in strategy.
		•	These queries must be different in nature from those used in previous retrievals.
		•	Use the retrieval plan to focus on alternative types of content, such as:
			•	Less obvious character relationships or dynamics.
			•	Emotional states, motivations, background context.
			•	Events not directly related but potentially influential.
		•	Include clip-based queries (formatted as "CLIP_x") only if the question relates to specific moments or sequences in time.
		•	Use keyword-based queries, not command sentences. Queries should be written as compact keyword phrases, not as full sentences or instructions. Avoid using directive language like “Retrieve”, “Describe”, or question forms such as “What”, “When”, “How”.
		•	Keep each query short and focused on one point. Each query should target one specific type of information, without combining multiple ideas or aspects.
		•	Avoid over-complexity and unnecessary detail. Do not include too many qualifiers or conditions. Strip down to the most essential keywords needed to retrieve valuable content.
		•	Ensure the five queries are semantically diverse, each probing a unique angle.
		•	Avoid repetition or slight variations of past queries.

Formatting for Search Queries:
	•	Output in Python-style string list, e.g.
		[SEARCH] ["What are Alice's intentions during CLIP_5?", "How does the group react to the decision in CLIP_2?", "What traits define Bob's personality throughout the video?", "What tension exists between secondary characters?", "What themes are highlighted in the summary of CLIP_7?"]

Guidance for New Search Angles:
	•	Think: What have I not asked about yet?
	•	Focus on secondary factors, overlooked characters, indirect causes, or high-level themes.
	•	Consider shifts from actions to intentions, from events to emotions, or from individuals to group dynamics.

Example 1:

Input:

Question: How did the argument between Alice and Bob influence their relationship in the story?

Knowledge:
[
	{{
		"query": "What happened during the argument between Alice and Bob?",
		"related memories": {{
			"CLIP_2": [
				"<face_1> and <face_2> are seen arguing in the living room."
				"<face_1> raises her voice, and <face_2> looks upset."
				"<face_1> accuses <face_2> of not listening to her."
			],
		}}
	}}
]

Output:

It seems that <face_1> and <face_2> are engaged in an argument, but their identities are not yet known, and there is no information about the consequences of the argument. To understand how it influenced their relationship, I need more contextual information about their identities, reactions, and what happened after.
[SEARCH] ["What are the names of <face_1> and <face_2>?", "What is the emotional state of <face_1> and <face_2> after the argument?", "What happens immediately after CLIP_2?", "Is there a summary indicating a change in the relationship between these two characters?", "Do any later clips show reconciliation or continued conflict between <face_1> and <face_2>?"]

Example 2:

Input:

Question: How did the argument between Alice and Bob influence their relationship in the story?

Knowledge:
[
	{{
		"query": "What happened during the argument between Alice and Bob?",
		"related memories": {{
			"CLIP_2": [
				"<face_1> and <face_2> are seen arguing in the living room."
				"<face_1> raises her voice, and <face_2> looks upset."
				"<face_1> accuses <face_2> of not listening to her."
			],
		}}
	}},
	{{
		"query": "What are the names of <face_1> and <face_2>?",
		"related memories": {{
			"CLIP_1": [
				"<face_1> says to <face_2>: 'I am done with you Bob!'",
				"<face_2> says to <face_1>: 'What about now, Alice?'"
			],
		}}
	}}	
]

Output:

CLIP_1 identifies <face_1> as Alice and <face_2> as Bob. CLIP_2 shows the argument between them. However, the influence of this argument on their relationship is not yet clear — we need to know what happened afterward and whether their interaction changed.
[SEARCH] ["What happens in CLIP_3 after the argument?", "How does Alice behave toward Bob after the argument?", "Are there any summaries indicating a shift in Alice and Bob's relationship?", "Do Alice and Bob interact again in later clips?", "Is there any indication that their relationship improves or deteriorates after CLIP_2?"]

Now, generate your response for the following input:

Question: {question}

Retrieval Plan: {retrieval_plan}

Knowledge: {knowledge}

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

prompt_answer_with_retrieval_final = """You are given a question about a specific video and a dictionary of some related information about the video. Each key in the dictionary is a clip ID (an integer), representing the index of a video clip. The corresponding value is a list of video descriptions from that clip.

Your task is to analyze the provided information, reason over it, and produce the most reasonable and well-supported answer to the question.

Output Requirements:
	•	Your response must begin with a brief reasoning process that explains how you arrive at the answer.
	•	Then, output [ANSWER] followed by your final answer.
	•	The format must be: Here is the reasoning... [ANSWER] Your final answer here.
	•	Your final answer must be definite and specific — even if the information is partial or ambiguous, you must infer and provide the most reasonable answer based on the given evidence.
	•	Do not refuse to answer or say that the answer is unknowable. Use reasoning to reach the best possible conclusion.

Additional Guidelines:
	•	When referring to a character, always use their specific name if it appears in the video information.
	•	Do not use placeholder tags like <character_1> or <face_1>.
	•	Avoid summarizing or repeating the video information. Focus on reasoning and answering.
	•	The final answer should be short, clear, and directly address the question.

Input:
	•	Question: {question}
	•	Video Information: {information}

Output:"""

prompt_refine_qa_list = """You are given a list of question-answer (QA) pairs based on specific videos, along with corresponding reasoning processes written in Chinese. Your task is to:
	1.	Translate the reasoning processes into concise and fluent English, without changing their original meaning.
	2.	Refine the reasoning to make it clearer and more precise, especially in terms of how the answer is logically derived from the video content.
	3.	Revise the question and answer to ensure they are:
	•	Natural and grammatically correct in English
	•	Logically consistent with the reasoning
	•	Expressed in a clear, specific, and rigorous way

Additional Constraints:
	•	Question:
		•	Do not oversimplify or generalize the original question.
		•	Preserve important contextual details or conditions from the original input.
		•	The revised question should be more specific and tightly scoped, such that the video provides a unique and unambiguous answer.
		•	Avoid vagueness or ambiguity; the revised question should not allow multiple valid answers.
	•	Answer:
		•	Ensure the answer is concise, grammatically correct, and semantically precise.
		•	It should be directly supported by the reasoning and the video.
		•	The answer should be unambiguous and standardized, suitable for use as ground truth in evaluation tasks.

Output Format:

Return a valid JSON list where each item contains:
	•	"question": the revised English question
	•	"answer": the revised English answer (refined for clarity and evaluability)
	•	"reasoning": the translated and refined English reasoning explaining how the answer is derived from the video


Example Input:

[
	{{"question": "what's the man doing?", "answer": "he fixing the car.", "reasoning": "需要从视频中找到男人正在修理汽车的场景。"}},
	{{"question": "why she looks angry?", "answer": "because someone take her bag.", "reasoning": "需要从视频中找到女人生气的原因。"}}
]

Expected Output:

[
	{{"question": "What is the man doing to the vehicle in the garage?", "answer": "He is repairing the car.", "reasoning": "There is a clip showing the man repairing a car in the garage."}},
	{{"question": "Why does the woman appear upset after walking into the hallway?", "answer": "Because someone took her bag.", "reasoning": "The video shows the woman appearing upset in the hallway after someone takes her bag."}}
]

Now, apply the same logic to the following input:

Input:

{qa_list}

Output:
(Only return the translated JSON list.)"""

prompt_agent_verify_answer = """You are provided with a question, the ground truth answer, and the answer from an agent model. Your task is to assess whether the agent answer is semantically consistent with the ground truth answer, in the context of the question.

If the meaning expressed by the agent answer aligns with the meaning of the ground truth answer — even if the wording or format is different — return "Yes". If the agent answer expresses a different or incorrect meaning, return "No".

Do not require exact wording or surface form match. Semantic equivalence, given the context of the question, is sufficient.

Please only return "Yes" or "No", with no additional explanation or formatting."""

prompt_agent_verify_answer_referencing = """You are provided with a question, a ground truth answer, and an answer from an agent model. Your task is to determine whether the ground truth answer can be logically inferred from the agent's answer, in the context of the question.

Do not directly compare the surface forms of the agent answer and the ground truth answer. Instead, assess whether the meaning expressed by the agent answer supports or implies the ground truth answer. If the ground truth can be reasonably derived from the agent answer, return "Yes". If it cannot, return "No".

Important notes:
	•	Do not require exact wording or matching structure.
	•	Semantic inference is sufficient, as long as the agent answer entails or implies the meaning of the ground truth answer, given the question.
	•	Only return "Yes" or "No", with no additional explanation or formatting.

Input fields:
	•	question: the question asked
	•	ground_truth_answer: the correct answer
	•	agent_answer: the model's answer to be evaluated

Now evaluate the following input:

Input:
	•	question: {question}
	•	ground_truth_answer: {ground_truth_answer}
	•	agent_answer: {agent_answer}

Output ('Yes' or 'No'):"""

prompt_agent_verify_answer_with_reasoning = """You are provided with a question, a ground truth answer, a reasoning that supports the ground truth answer (based on video content), and an answer from an agent model.

Your task is to assess whether the agent answer is semantically valid, based on the question and the provided reasoning.

Specifically:
	•	If the agent answer expresses a meaning that is consistent with what can be reasonably inferred from the reasoning, return "Yes", even if it differs slightly from the ground truth answer in wording or phrasing.
	•	If the agent answer conveys a different or incorrect meaning that is not supported by the reasoning, return "No".

Do not require exact match between the agent answer and the ground truth answer. Use the reasoning as the primary source of truth — if the agent answer can be plausibly derived from the reasoning in the context of the question, it should be accepted as correct.

Only return "Yes" or "No", with no additional explanation or formatting.

Input fields:
	•	question: the question asked about the video
	•	ground_truth_answer: the correct answer
	•	reasoning: explanation of how the answer is derived from the video
	•	agent_answer: the answer generated by the model"""

prompt_refine_answer = """You are given a question and its corresponding answer. Your task is to rewrite the answer to make it more concise, direct, and focused, so that it can be more easily and reliably evaluated for correctness.

The revised answer should:
	•	Clearly and explicitly address the question
	•	Remove vague, redundant, or indirect phrasing
	•	Preserve the original meaning, but express it in a shorter and more precise way
	•	Be grammatically correct and unambiguous
	•	Be suitable for use as a ground truth reference when evaluating other model outputs

Input Format Example:

{{
	"question": "What did the woman do after entering the room?",
	"answer": "After she entered the room, she looked around and seemed confused."
}}

Expected Output:

She looked around and seemed confused.

Now apply the same logic to the following input:

Input:

{qa_pair}

Output:
(Directly return the revised answer.)"""