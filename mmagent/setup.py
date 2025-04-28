from setuptools import setup, find_packages

setup(
    name="mmagent",
    version="0.1.0",
    author="KLong",
    author_email="kylin0long@gmail.com",
    description="A multimodal processing package with face, voice, memory and QA modules.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://code.byted.org/longlin.kylin/mmagent",
    packages=find_packages(include=["mmagent", "mmagent.*"]),
    include_package_data=True,
    # install_requires=open("requirements.txt", encoding="utf-8").read().splitlines(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">3.6",
)

from mmagent.utils.general import load_video_graph
from mmagent.retrieve import answer_with_retrieval
import sys
import mmagent.videograph

sys.modules["videograph"] = mmagent.videograph

video_graph_path = "/mnt/hdfs/foundation/longlin.kylin/mmagent/data/mems/CZ_1/-UhacNNM_HU_30_5_-1_10_20_0.3_0.6.pkl"
video_graph = load_video_graph(video_graph_path)

question = "Do Hannahlei's parents support her dancing?"
answer = answer_with_retrieval(video_graph, question)