import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from datetime import datetime
import torch
from logger import logger

class VideoEmbeddingGenerator:
    def __init__(self):
        self.model = SentenceTransformer(
            "Qwen/Qwen3-Embedding-8B",
            model_kwargs={
                "attn_implementation": "flash_attention_2",
                "device_map": "auto",
                "torch_dtype": torch.bfloat16
            },
            tokenizer_kwargs={"padding_side": "left"},
        )
        logger.info("Model loaded successfully")        
        self.chunked_videos_dir = Path('./data/chunked_videos')
        self.embeddings_dir = Path('./data/embeddings')        
        self.embeddings_dir.mkdir(exist_ok=True)
        
    def load_chunk_data(self, video_id):
        chunk_file = self.chunked_videos_dir / f"{video_id}.json"
        chunk_data = None
        
        if chunk_file.exists():
            with open(chunk_file, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
        else:
            logger.warning(f"Chunk file not found: {chunk_file}")
            
        return chunk_data
    
    def create_chunk_prompt(self, chunk_data, chunk):
        prompt_parts = []
        
        video_caption = chunk_data.get('video_caption', '')
        if video_caption:
            prompt_parts.append(f"Video Description: {video_caption}")
        
        # Add transcription segments (already filtered for this chunk)
        segments = chunk.get('segments', [])
        if segments:
            transcription_text = "Transcription: "
            for segment in segments:
                segment_text = segment.get('segment', '').strip()
                if segment_text:
                    start_time = segment.get('start', 0)
                    end_time = segment.get('end', 0)
                    transcription_text += f"[{start_time:.2f}s-{end_time:.2f}s] {segment_text} "
            prompt_parts.append(transcription_text.strip())
        
        # Add keyframe captions
        keyframes = chunk.get('key_frames', [])
        if keyframes:
            keyframe_text = "Visual Content: "
            for i, keyframe in enumerate(keyframes):
                timestamp = keyframe.get('timestamp', 0)
                caption = keyframe.get('caption', '').strip()
                if caption:
                    # Truncate very long captions
                    if len(caption) > 300:
                        caption = caption[:300] + "..."
                    keyframe_text += f"[{timestamp:.2f}s] {caption} "
            prompt_parts.append(keyframe_text.strip())
        
        full_prompt = " ".join(prompt_parts)
        return full_prompt
    
    def process_video_chunks(self, video_id):
        chunk_data = self.load_chunk_data(video_id)
        
        chunks_with_prompts = []
        
        for chunk in chunk_data.get('chunks', []):
            start_time = chunk.get('start_time', 0)
            end_time = chunk.get('end_time', 0)
            chunk_id = chunk.get('chunk_id', 0)
            
            prompt = self.create_chunk_prompt(chunk_data, chunk)
            
            metadata = {
                'video_id': video_id,
                'chunk_id': chunk_id,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
            }
            
            chunks_with_prompts.append((prompt, metadata))
            
        return chunks_with_prompts
    
    def generate_embeddings_for_chunks(self, chunks_with_prompts, batch_size=64):
        if not chunks_with_prompts:
            return np.array([]), []
        
        prompts = [chunk[0] for chunk in chunks_with_prompts]
        metadata_list = [chunk[1] for chunk in chunks_with_prompts]
        
        total_prompts = len(prompts)
        total_batches = (total_prompts + batch_size - 1) // batch_size
        
        logger.info(f"Generating embeddings for {total_prompts} chunks in {total_batches} batches (batch_size={batch_size})")
        
        all_embeddings = []
        
        for i in range(0, total_prompts, batch_size):
            batch_end = min(i + batch_size, total_prompts)
            batch_prompts = prompts[i:batch_end]
            batch_num = i // batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_prompts)} prompts)")
            
            # Generate embeddings for this batch
            batch_embeddings = self.model.encode(
                batch_prompts,
                batch_size=min(32, len(batch_prompts))  # Internal batch size for the model
            )
            
            all_embeddings.append(batch_embeddings)
            
            torch.cuda.empty_cache()
            logger.debug(f"Cleared GPU cache after batch {batch_num}")
        
        # Combine all batch embeddings
        combined_embeddings = np.vstack(all_embeddings)
        
        logger.info(f"Generated embeddings with shape: {combined_embeddings.shape}")
        return combined_embeddings, metadata_list
    
    def process_all_videos(self):
        chunk_files = list(self.chunked_videos_dir.glob("*.json"))
        
        logger.info(f"Processing {len(chunk_files)} videos")
        
        all_embeddings = []
        all_metadata = []
        all_prompts = []
        
        processed_count = 0
        failed_count = 0
        
        for chunk_file in chunk_files:
            video_id = chunk_file.stem
            
            try:
                logger.info(f"Processing video {processed_count + 1}/{len(chunk_files)}: {video_id}")
                
                chunks_with_prompts = self.process_video_chunks(video_id)
                
                if chunks_with_prompts:
                    embeddings, metadata_list = self.generate_embeddings_for_chunks(chunks_with_prompts)
                    
                    if len(embeddings) > 0:
                        all_embeddings.append(embeddings)
                        all_metadata.extend(metadata_list)
                        all_prompts.extend([chunk[0] for chunk in chunks_with_prompts])
                        
                        logger.info(f"Successfully processed {len(chunks_with_prompts)} chunks for video {video_id}")
                    else:
                        logger.warning(f"No embeddings generated for video {video_id}")
                        failed_count += 1
                else:
                    logger.warning(f"No chunks found for video {video_id}")
                    failed_count += 1
                    
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing video {video_id}: {e}")
                failed_count += 1
                continue
        
        # Combine all embeddings
        if all_embeddings:
            combined_embeddings = np.vstack(all_embeddings)
            
            logger.info(f"Combined embeddings shape: {combined_embeddings.shape}")
            logger.info(f"Total metadata entries: {len(all_metadata)}")
            logger.info(f"Total prompts: {len(all_prompts)}")
            
            # Save embeddings and metadata
            success = self.save_embeddings_and_metadata(combined_embeddings, all_metadata, all_prompts)
            
            if success:
                logger.info(f"Successfully processed {processed_count} videos, failed: {failed_count}")
                logger.info(f"Total chunks processed: {len(all_metadata)}")
                return True
            else:
                logger.error("Failed to save embeddings and metadata")
                return False
        else:
            logger.error("No embeddings were generated")
            return False
    
    def save_embeddings_and_metadata(self, embeddings: np.ndarray, metadata_list: List[Dict], prompts: List[str]) -> bool:
        # Save embeddings as .npy file
        embeddings_file = self.embeddings_dir / 'video_embeddings.npy'
        np.save(embeddings_file, embeddings)
        logger.info(f"Saved embeddings to: {embeddings_file}")
        
        # Save metadata as JSON
        metadata_file = self.embeddings_dir / 'video_metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved metadata to: {metadata_file}")
        
        # Save prompts for reference
        prompts_file = self.embeddings_dir / 'video_prompts.json'
        with open(prompts_file, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved prompts to: {prompts_file}")

if __name__ == "__main__":
    generator = VideoEmbeddingGenerator()
    generator.process_all_videos()