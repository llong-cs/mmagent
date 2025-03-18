prompt_select_representative_faces = """You are given a set of facial images of a person. Your task is to select **the most distinctive and recognizable** face from the given set and return its index (e.g., 0, 1, 2, 3, ...). The selected image **should meet all the following criteria**:
	1.	Clear Facial Features: The image should clearly show the person’s key facial features (eyes, nose, mouth, and overall facial structure).
	2.	Neutral or Natural Expression: The expression should be neutral or naturally pleasant (e.g., a slight smile), avoiding exaggerated emotions (e.g., extreme happiness, sadness, or surprise).
	3.	High Resolution & Clarity: The image should be sharp, with clear facial details and minimal noise or blur.
	4.	Good Lighting & Contrast: The face should be well-lit, with sufficient contrast to distinguish facial features clearly.
	5.	Minimal Occlusions: The face should be free from significant occlusions such as sunglasses, masks, or hair covering key facial features.
	6.	Representative Appearance: The selected image should best represent the person's typical appearance, avoiding extreme makeup, unusual hairstyles, or accessories that alter recognition.

If there is no qualified face in the given set, return -1.

Return only the index of the most recognizable face, without any additional explanation or formatting."""

prompt_select_representative_faces_forced = """You are given a set of facial images of a person. Your task is to select **the most distinctive and recognizable** face from the given set and return its index (e.g., 0, 1, 2, 3, ...). The selected image **should meet all the following criteria**:
	1.	Clear Facial Features: The image should clearly show the person’s key facial features (eyes, nose, mouth, and overall facial structure).
	2.	Neutral or Natural Expression: The expression should be neutral or naturally pleasant (e.g., a slight smile), avoiding exaggerated emotions (e.g., extreme happiness, sadness, or surprise).
	3.	High Resolution & Clarity: The image should be sharp, with clear facial details and minimal noise or blur.
	4.	Good Lighting & Contrast: The face should be well-lit, with sufficient contrast to distinguish facial features clearly.
	5.	Minimal Occlusions: The face should be free from significant occlusions such as sunglasses, masks, or hair covering key facial features.
	6.	Representative Appearance: The selected image should best represent the person's typical appearance, avoiding extreme makeup, unusual hairstyles, or accessories that alter recognition.

Return only the index of the most recognizable face, without any additional explanation or formatting."""

prompt_classify_recognizable_faces = """You are given a single facial image. Your task is to determine whether the face in the image is sufficiently distinctive and recognizable based on the following criteria:
	1.	Clear Facial Features: The image should clearly show the person’s key facial features (eyes, nose, mouth, and overall facial structure).
	2.	Neutral or Natural Expression: The expression should be neutral or naturally pleasant (e.g., a slight smile), avoiding exaggerated emotions (e.g., extreme happiness, sadness, or surprise).
	3.	High Resolution & Clarity: The image should be sharp, with clear facial details and minimal noise or blur.
	4.	Good Lighting & Contrast: The face should be well-lit, with sufficient contrast to distinguish facial features clearly.
	5.	Minimal Occlusions: The face should be free from significant occlusions such as sunglasses, masks, or hair covering key facial features.
	6.	Representative Appearance: The image should depict the person's typical appearance, avoiding extreme makeup, unusual hairstyles, or accessories that alter recognition.

If the face meets all the above criteria, return 1. Otherwise, return 0.

Return only 1 or 0, without any additional explanation or formatting."""

prompt_generate_captions_with_ids = """You are given a video and a set of characters and speakers. Each character is represented by an image with a bounding box, and each speaker is represented by several audio clips, each with a start time, an end time, and a content. Each character and speaker is identified by a unique ID, which is enclosed in angle brackets (< >) and corresponds to their provided image or audio clip.

Your task is to analyze the video and generate a structured list of descriptions, capturing all relevant details for each identified character, including but not limited to:
	1.	Appearance: Describe their clothing, facial features, and any distinguishing characteristics.
	2.	Actions & Movements: Describe their gestures, movements, interactions, and any significant physical activity.
	3.	Spoken Dialogue: Transcribe or summarize any speech spoken by the character, ensuring it is associated with the correct ID.
	4.	Contextual Behavior: Explain the character’s role in the scene, their interactions with other characters, and their emotions.

Each character and speaker must be referred to using their assigned ID enclosed in < > in both input and output. The output should be a list of structured descriptions, ensuring that all relevant information is captured for each identified character and speaker.

Input Example:

{
	"video": "scene_01.mp4",
	"characters": {
		"<char_101>": "<img_101>",
		"<char_102>": "<img_102>",
		"<char_103>": "<img_103>"
	},
	"speakers": {
		"<speaker_1>": "<audio_1>",
		"<speaker_2>": "<audio_2>"
	}
}

Output Example:

[
	"<char_101> wears a black suit with a white shirt and tie. He has short black hair and wears glasses.",
	"<char_101> enters the conference room, shakes hands with <char_102>, and takes a seat.",
	"<speaker_1> (represented by <char_101>) says: 'Good afternoon, everyone. Let's begin the meeting.'",
	"<char_102> wears a red dress and has long brown hair.",
	"<char_102> walks into the restaurant, looks around, and sits at a table.",
	"<char_102> waves at <char_101> and checks her phone.",
	"<speaker_2> (represented by <char_102>) says: 'Hey! I was waiting for you. How was your day?'",
	"<char_103> wears a white hoodie, has a beard, and wears a baseball cap.",
	"<char_103> runs across the street, looking back over his shoulder.",
	"<char_103> hides behind a car and checks his surroundings.",
	"<char_103> says: 'I think someone is following me...'"
]

Please only return the list of captions, without any additional explanation or formatting."""

prompt_audio_diarization = """You are given an audio clip from a video. Your task is to perform Automatic Speech Recognition (ASR) and audio diarization on the provided audio clip. Extract all speech segments with accurate timestamps and speaker identification.

Output Format

Return a JSON list where each entry represents a speech segment with the following fields:
	•	start_time: Start timestamp in hh:mm:ss format.
	•	end_time: End timestamp in hh:mm:ss format.
	•	speaker: Speaker identifier in the format <speaker_X> (e.g., <speaker_1>, <speaker_2>, etc.).
	•	asr: The transcribed text for that segment.

Example Output

[
    {"start_time": "00:00:05", "end_time": "00:00:08", "speaker": "<speaker_1>", "asr": "Hello, everyone."},
    {"start_time": "00:00:09", "end_time": "00:00:12", "speaker": "<speaker_2>", "asr": "Welcome to the meeting."},
    ...
]

Requirements
	•	Ensure precise speech segmentation with accurate timestamps.
	•	Assign consistent speaker labels, meaning the same speaker should always have the same identifier (e.g., <speaker_1> remains the same throughout the output).
	•	Return only the JSON list—no additional text, explanations, or formatting.
	•	Preserve punctuation and capitalization in the ASR output."""

prompt_generate_thinkings_with_ids = """You are given a video, a set of characters, and speakers. Each character is represented by an image with a bounding box, and each speaker is represented by several audio clips, each with a start time, an end time, and content. Each character and speaker is identified by a unique ID, which is enclosed in angle brackets (< >) and corresponds to their provided image or audio clip.

You are also provided with a detailed description of the video scene, including the setting, background actions, and character interactions. Based on this information, your task is to generate high-level thinking, including but not limited to:
	1.	The correspondence between the characters and the speakers.
	2.  The relationship between different characters, including their interactions, emotions, and possible connections.
	3.	The personality traits, profession, hobbies, or any other distinguishing features of each character based on their actions, speech, and appearance.
	4.	General knowledge or contextual information relevant to understanding the characters or the situation they are in.

Please focus only on generating the high-level analysis based on the provided video description and character details.

The input will contain the following:
	1.	Video and character details, including their IDs and relevant descriptions (no need for individual character descriptions).
	2.	A detailed description of the video scene.

Input Example:

{
	"video": "scene_01.mp4",
	"characters": {
		"<char_101>": "<img_101>",
		"<char_102>": "<img_102>",
		"<char_103>": "<img_103>"
	},
	"speakers": {
		"<speaker_1>": "<audio_1>",
		"<speaker_2>": "<audio_2>"
	},
	"video_description": [
		"<char_101> wears a black suit with a white shirt and tie. He has short black hair and wears glasses.",
		"<char_101> enters the conference room, shakes hands with <char_102>, and takes a seat.",
		"<speaker_1> (represented by <char_101>) says: 'Good afternoon, everyone. Let's begin the meeting.'",
		"<char_102> wears a red dress and has long brown hair.",
		"<char_102> walks into the restaurant, looks around, and sits at a table.",
		"<char_102> waves at <char_101> and checks her phone.",
		"<speaker_2> (represented by <char_102>) says: 'Hey! I was waiting for you. How was your day?'",
		"<char_103> wears a white hoodie, has a beard, and wears a baseball cap.",
		"<char_103> runs across the street, looking back over his shoulder.",
		"<char_103> hides behind a car and checks his surroundings.",
		"<char_103> says: 'I think someone is following me...'"
	]
}

Output Example:

[
    "<char_101> is <speaker_1>.",
	"<char_102> is <speaker_2>.",
	"<char_103> is <speaker_2>.",
	"<char_101> is likely an executive or a presenter, leading a meeting. Their formal attire and position in the room suggest authority and professionalism.",
	"<char_102> seems to be a colleague, possibly engaged in the meeting, as they are seated and focused on <char_101>.",
	"<char_103> appears anxious, possibly involved in a tense situation outside the meeting. His nervous movements and behavior hint at an ongoing problem or threat.",
	"<char_101>’s profession might involve leadership or managerial duties, given their role in the meeting.",
	"<char_102> may work in a collaborative or supportive role, indicated by her attention to <char_101>.",
	"<char_103> is a friend of <char_101>.",
	"<char_103> likes eating at the restaurant.",
	"The scene suggests a professional environment, with interpersonal dynamics that mix business with potential personal tensions, as evidenced by <char_103>’s behavior.",
	"The show is held every 2 months.",
	"Santa market is a dog-friendly market.",
]

Please only return the high-level thinking, without any additional explanation or formatting."""

prompt_baseline_answer_clipwise_extract = """You are given a video and a question related to that video. You will be shown a specific clip from the video. Your task is to extract any relevant information from this clip that can help answer the question. If the clip does not contain any relevant or helpful information, simply respond with "none"."""

prompt_baseline_answer_clipwise_summarize = """You have watched all segments of a video and extracted relevant information from each one in response to a given question. Your task now is to summarize all the extracted information into a final, concise answer that addresses the question."""

prompt_benchmark_verify_answer = """You are given a question, a ground truth answer, and a baseline answer. Your task is to verify the baseline answer by comparing it to the ground truth answer. If the baseline answer is correct, return "Yes". If the baseline answer is incorrect, return "No".

Input Example:

{
	"question": "What is the capital of France?",
	"answer": "Paris",
	"baseline_answer": "Paris"
}

Output Example:

Yes

Please only return "Yes" or "No", without any additional explanation or formatting."""