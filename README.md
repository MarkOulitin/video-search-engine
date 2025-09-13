# Video search engine
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
Assume that you have `./data/videos` directory to index
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