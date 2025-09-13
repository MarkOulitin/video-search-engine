# Video search engine
# Design description
I created a ChromaDB vector database of indexed video chunks. Each video is processed by 3 different models:
- Video captioning model: [VideoLLaMA3-2B](https://huggingface.co/DAMO-NLP-SG/VideoLLaMA3-2B) to get a high level textual representation of the video
- Trascription of the audio trach in the video: [canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2) to get transcription segments with timestamps
- Applying Frame captioning every 5 seconds using [FastVLM-0.5B](https://huggingface.co/apple/FastVLM-0.5B).

Each video is devided to chunks of 30 seconds, each chunk is represented by:
- Video caption
- Transcription segments that overlap with time interval of the chunk
- Frame captions whose frames are in the chunk

For each chunk a prompt is created and then embedded using [Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B).

Supported features:
- Search per user query
- Gradio UI that shows the video chunk
- Video file upload and indexing

For each step there is a separate component:
- Video captioning: `./server/scripts/process_video_captions.py`
- Transcription generation: `./server/scripts/process_transcriptions.py`
- Keyframe captioning (every 5 seconds): `./server/scripts/process_keyframe_captioning.py`
- Chunking videos: `./server/scripts/chunk_videos.py`
- Ingest to ChromaDB: `./server/scripts/ingest_video_embeddings.py`
- Server side control flow: `./server/video_pipeline.py`
- Video trimming: `./server/video_processor.py`
# Hardware requirements
- Minimal VRAM needed (in idle state) 26 GB. It was tested on RTX A6000 with 48 GB VRAM.
- Minimal storage FREE 60 GB for docker image that contains model weights.
- All the components are self hosted in one docker image so there is need for only one node
# Things that can be imporved
- In current approach different chunks of same video can be returned in same result set if set top K to more than 1. Can be improved by deciding the relevant video and then extracting the relevant chunks from the video.
- The prompt representation can be noisy due to captioning each frame, potentially leading to irrelevant or misleading information by FastVLM. This can be fixed by iniitally chunking the video to short videos of 30 seconds and then apply video captioning per chunk. This will reduce the prompt representation and also the cost and time of indexing the video.
- Indexing the video during server runtime (user upload), currently done sequencially. Can be done in parrallel.
- Working in multi-part upload and download in server-client interaction.

# Indexing exisitng directory of videos
## Setup
```sh
conda create --name server python=3.12 -y
conda activate server
pip install -r ./server/requirements.txt
pip install flash-attn --no-build-isolation
pip install transformers==4.53.0

HF_HOME=./models/huggingface

huggingface-cli download Qwen/Qwen3-Embedding-8B
huggingface-cli download apple/FastVLM-0.5B
huggingface-cli download DAMO-NLP-SG/VideoLLaMA3-2B

# Fix VideoLLaMA3 import issues
cp ./server/fix/image_processing_videollama3.py $(find ./models/huggingface/hub/models--DAMO-NLP-SG--VideoLLaMA3-2B -name "image_processing_videollama3.py" | head -1)
```
## Indexing
**Assume that you have `./data/videos` directory to index**
```sh
python ./server/scripts/process_video_captions.py
python ./server/scripts/process_transcriptions.py
python ./server/scripts/process_keyframe_captioning.py
python ./server/scripts/chunk_videos.py
python ./server/scripts/generate_embeddings.py
```

# Lauch server
## Setup
```sh
# build server
docker build -t rag-server:latest ./server

# launch server with chromadb
docker compose up -d

# ingest embeddings from offline indexing
docker exec -it server python ./scripts/ingest_video_embeddings.py
```

# Launch client
Setup in `.env`:
```sh
SERVER_HOST="server host"
```

```sh
conda create --name client python=3.12 -y
conda activate client
pip install -r ./client/requirements.txt

python client/client.py
```
Follow the instructions of the Gradio terminal prompts