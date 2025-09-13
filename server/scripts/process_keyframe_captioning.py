import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import cv2
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from logger import logger

MID = "apple/FastVLM-0.5B"
IMAGE_TOKEN_INDEX = -200  # what the model code looks for


class FrameCaptioningProcessor:
    def __init__(self, batch_size: int = 4):
        self.batch_size = batch_size
        self.model = AutoModelForCausalLM.from_pretrained(
            MID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)    
        
    def extract_key_frames_from_video(self, video_path, frame_interval=2.0):
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            logger.info(f"Video: {os.path.basename(video_path)} - FPS: {fps:.2f}, Duration: {duration:.2f}s")
            
            frames_data = []
            frame_interval_frames = int(fps * frame_interval)
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract frame at specified intervals (key frames only)
                if frame_count % frame_interval_frames == 0:
                    timestamp = frame_count / fps
                    
                    # Convert BGR to RGB for PIL
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    frames_data.append({
                        'timestamp': timestamp,
                        'frame_number': frame_count,
                        'image': pil_image
                    })
                
                frame_count += 1
            
            cap.release()
            logger.info(f"Extracted {len(frames_data)} key frames from {os.path.basename(video_path)} (every {frame_interval}s)")
            return frames_data
            
        except Exception as e:
            logger.error(f"Error extracting key frames from {video_path}: {e}")
            raise
    
    def caption_images_batch(self, images):
        if not images:
            return []
        
        try:
            batch_size = len(images)
            logger.debug(f"Processing batch of {batch_size} images")

            # Prepare batch of prompts
            messages = [
                {"role": "user", "content": "<image>\nDescribe this scene in detail."}
            ]
            rendered = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            
            pre, post = rendered.split("<image>", 1)
            
            # Tokenize the text *around* the image token (no extra specials!)
            pre_ids = self.tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids
            post_ids = self.tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids
            
            # Create batch input_ids for all images
            img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
            
            # Repeat for batch
            batch_pre_ids = pre_ids.repeat(batch_size, 1)
            batch_post_ids = post_ids.repeat(batch_size, 1)
            batch_img_tok = img_tok.repeat(batch_size, 1)
            
            # Concatenate for batch
            batch_input_ids = torch.cat([batch_pre_ids, batch_img_tok, batch_post_ids], dim=1).to(self.model.device)
            batch_attention_mask = torch.ones_like(batch_input_ids, device=self.model.device)
            
            # Preprocess images via the model's own processor
            batch_px = self.model.get_vision_tower().image_processor(images=images, return_tensors="pt")["pixel_values"]
            batch_px = batch_px.to(self.model.device, dtype=self.model.dtype)
            
            # Generate captions for batch
            with torch.no_grad():
                start_time = time.time()
                batch_out = self.model.generate(
                    inputs=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    images=batch_px,
                    max_new_tokens=128,
                    do_sample=False,  # For consistent results
                )
                inference_time = time.time() - start_time
            
            # Decode the generated texts
            captions = []
            for i in range(batch_size):
                full_response = self.tokenizer.decode(batch_out[i], skip_special_tokens=True)
                
                # Extract only the generated part (after the prompt)
                if "assistant" in full_response.lower():
                    parts = full_response.split("assistant")
                    if len(parts) > 1:
                        caption = parts[-1].strip()
                    else:
                        caption = full_response.strip()
                else:
                    caption = full_response.strip()
                
                captions.append(caption)
            
            avg_time_per_image = inference_time / batch_size
            logger.debug(f"Batch of {batch_size} images processed in {inference_time:.3f}s ({avg_time_per_image:.3f}s per image)")
            return captions
            
        except Exception as e:
            logger.error(f"Error generating batch captions: {e}")
            logger.exception(e)
            raise e

    def caption_image(self, image):
        try:
            messages = [
                {"role": "user", "content": "<image>\nDescribe this scene in detail."}
            ]
            rendered = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            
            pre, post = rendered.split("<image>", 1)
            
            # Tokenize the text *around* the image token (no extra specials!)
            pre_ids = self.tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids
            post_ids = self.tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids
            
            # Splice in the IMAGE token id (-200) at the placeholder position
            img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
            input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(self.model.device)
            attention_mask = torch.ones_like(input_ids, device=self.model.device)
            
            # Preprocess image via the model's own processor
            px = self.model.get_vision_tower().image_processor(images=image, return_tensors="pt")["pixel_values"]
            px = px.to(self.model.device, dtype=self.model.dtype)
            
            # Generate caption
            with torch.no_grad():
                start_time = time.time()
                out = self.model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    images=px,
                    max_new_tokens=128,
                )
                inference_time = time.time() - start_time
            
            # Decode the generated text
            full_response = self.tokenizer.decode(out[0], skip_special_tokens=True)
            
            # Extract only the generated part (after the prompt)
            # Find the assistant's response part
            if "assistant" in full_response.lower():
                parts = full_response.split("assistant")
                if len(parts) > 1:
                    caption = parts[-1].strip()
                else:
                    caption = full_response.strip()
            else:
                caption = full_response.strip()
            
            logger.debug(f"Caption generated in {inference_time:.3f}s: {caption[:50]}...")
            return caption
            
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return "Error generating caption"
    
    def process_video_key_frames(self, video_path, frame_interval=2.0):
        try:
            frames_data = self.extract_key_frames_from_video(video_path, frame_interval)
            total_frames = len(frames_data)
            
            logger.info(f"Processing {total_frames} key frames in batches of {self.batch_size}")
            
            captioned_frames = []
            
            for batch_start in tqdm(list(range(0, total_frames, self.batch_size))):
                batch_end = min(batch_start + self.batch_size, total_frames)
                batch_frames = frames_data[batch_start:batch_end]
                
                logger.info(f"Processing batch {batch_start//self.batch_size + 1}/{(total_frames + self.batch_size - 1)//self.batch_size}: key frames {batch_start+1}-{batch_end}")
                batch_images = [frame_data['image'] for frame_data in batch_frames]
                batch_captions = self.caption_images_batch(batch_images)
                
                # Combine results
                for frame_data, caption in zip(batch_frames, batch_captions):
                    captioned_frames.append({
                        'timestamp': frame_data['timestamp'],
                        'frame_number': frame_data['frame_number'],
                        'caption': caption
                    })
                
                del batch_images
                torch.cuda.empty_cache()
            
            return captioned_frames
            
        except Exception as e:
            logger.error(f"Error processing video key frames: {e}")
            raise


def save_captions_json(captions: List[Dict[str, Any]], output_path: str, video_name: str) -> None:
    """Save key frame captions to JSON file
    
    Args:
        captions: List of key frame caption dictionaries
        output_path: Directory to save JSON file
        video_name: Original video filename (without extension)
    """
    try:
        json_data = {
            'video_file': video_name,
            'total_key_frames': len(captions),
            'key_frame_captions': captions
        }
        
        json_file = os.path.join(output_path, f"{video_name}_key_frame_captions.json")
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved key frame captions: {json_file}")
        
    except Exception as e:
        logger.error(f"Error saving JSON for {video_name}: {e}")
        raise

def process_videos(video_dir, output_dir, frame_interval=2.0, batch_size=4):
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Get all video files
    video_files = list(video_dir.glob("*.mp4"))
    total_videos = len(video_files)
    
    logger.info(f"Found {total_videos} video files to process")
    logger.info(f"Key frame extraction interval: {frame_interval}s")
    logger.info(f"Batch size: {batch_size}")
    
    start_total_time = time.time()
    processor = FrameCaptioningProcessor(batch_size=batch_size)
    
    successful = 0
    for i, video_path in enumerate(video_files, 1):
        video_name = video_path.stem
        logger.info(f"Processing {i}/{total_videos}: {video_name}")
        
        try:
            start_time = time.time()
            captions = processor.process_video_key_frames(str(video_path), frame_interval)
            save_captions_json(captions, str(output_dir), video_name)
            processing_time = time.time() - start_time
            logger.info(f"Completed {video_name} in {processing_time:.2f}s with {len(captions)} key frames")
            
            successful += 1
            
        except Exception as e:
            logger.error(f"Failed to process {video_name}: {e}")
            continue
    
    total_time = time.time() - start_total_time
    logger.info(f"Processing completed! {successful}/{total_videos} videos processed successfully in {total_time:.2f}s")

if __name__ == "__main__":
    process_videos(
        video_dir=Path("data/videos"),
        output_dir=Path("data/video_key_frame_captions"),
        frame_interval=5,
        batch_size=64
    )
    logger.info("Video VLM key frame captioning processing finished!")
