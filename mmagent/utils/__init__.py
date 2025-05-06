import json
processing_config = json.load(open("configs/processing_config.json"))
model = processing_config["model"]

from . import chat_api
if model == "qwen2.5-omni":
    from . import chat_qwen
from . import general
from . import tos
from . import video_processing
from . import video_verification

__all__ = [
    "chat_api",
    "chat_qwen" if model == "qwen2.5-omni" else None,
    "general",
    "tos",
    "video_processing",
    "video_verification",
]