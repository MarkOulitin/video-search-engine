FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg git

COPY ./requirements.txt .
RUN pip install -r requirements.txt
RUN pip install flash-attn --no-build-isolation
RUN pip install transformers==4.53.0

ENV HF_HOME=./models/huggingface

RUN huggingface-cli download Qwen/Qwen3-Embedding-8B
RUN huggingface-cli download apple/FastVLM-0.5B
RUN huggingface-cli download DAMO-NLP-SG/VideoLLaMA3-2B

COPY ./ .

# Fix VideoLLaMA3 import issues
RUN cp ./fix/image_processing_videollama3.py $(find ./models/huggingface/hub/models--DAMO-NLP-SG--VideoLLaMA3-2B -name "image_processing_videollama3.py" | head -1)

CMD ["python", "main.py"]