import os
import json
import time
from pathlib import Path
from datetime import datetime
import traceback
from logger import logger
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

class VideoCaptioningProcessor:    
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading video caption model and processor...")
        start_time = time.time()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "DAMO-NLP-SG/VideoLLaMA3-2B",
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        
        self.processor = AutoProcessor.from_pretrained(
            "DAMO-NLP-SG/VideoLLaMA3-2B", 
            trust_remote_code=True
        )
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f}s")
        logger.info(f"Model device: {next(self.model.parameters()).device}")
    
    def caption_video(self, video_path, question) -> str:
        try:
            logger.debug(f"Processing video: {os.path.basename(video_path)}")
            start_time = time.time()
            
            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": 128}},
                        {"type": "text", "text": question},
                    ]
                },
            ]
            
            inputs = self.processor(conversation=conversation, return_tensors="pt")
            
            inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=256)
            
            response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
            processing_time = time.time() - start_time
            logger.debug(f"Video processed in {processing_time:.2f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Error: Failed to process video - {str(e)}"
    
class VideoProcessor:    
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.processor = None
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate input directory
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def get_video_files(self):
        video_files = []
        
        for ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']:
            video_files.extend(self.input_dir.glob(f"*{ext}"))
            video_files.extend(self.input_dir.glob(f"*{ext.upper()}"))
        
        video_files.sort()
        logger.info(f"Found {len(video_files)} video files")
        
        return video_files
    
    def process_videos(self, question, max_videos=None):
        """Process all videos in the input directory"""
        try:
            logger.info("Initializing Vido Captioning processor...")
            self.processor = VideoCaptioningProcessor()
            
            video_files = self.get_video_files()
            
            if max_videos:
                video_files = video_files[:max_videos]
                logger.info(f"Processing limited to {max_videos} videos")
            
            if not video_files:
                logger.warning("No video files found to process")
                return []
            
            results = []
            total_videos = len(video_files)
            start_total_time = time.time()
            
            logger.info(f"Starting to process {total_videos} videos...")
            logger.info(f"Question: '{question}'")
            
            for i, video_path in enumerate(video_files, 1):
                video_name = video_path.name
                logger.info(f"Processing {i}/{total_videos}: {video_name}")
                
                start_time = time.time()
                
                try:
                    caption = self.processor.caption_video(str(video_path), question)
                    processing_time = time.time() - start_time
                    
                    result = {
                        'video_file': video_name,
                        'video_path': str(video_path),
                        'status': 'success',
                        'caption': caption,
                        'processing_time': round(processing_time, 2),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    caption_preview = caption[:100] + "..." if len(caption) > 100 else caption
                    logger.info(f"Caption preview: {caption_preview}")
                    logger.info(f"Processed in {processing_time:.2f}s")
                    
                except Exception as e:
                    processing_time = time.time() - start_time
                    error_msg = str(e)
                    
                    result = {
                        'video_file': video_name,
                        'video_path': str(video_path),
                        'status': 'error',
                        'caption': '',
                        'processing_time': round(processing_time, 2),
                        'error': error_msg,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    logger.error(f"Failed to process {video_name}: {error_msg}")
                
                results.append(result)
                
                # Print progress summary every 10 videos
                if i % 10 == 0 or i == total_videos:
                    successful = sum(1 for r in results if r['status'] == 'success')
                    avg_time = sum(r['processing_time'] for r in results) / len(results)
                    logger.info(f"Progress: {i}/{total_videos} | Success: {successful} | Avg time: {avg_time:.2f}s")
            
            # Final summary
            total_time = time.time() - start_total_time
            successful = sum(1 for r in results if r['status'] == 'success')
            failed = total_videos - successful
            avg_time = sum(r['processing_time'] for r in results) / len(results) if results else 0
            
            logger.info("=" * 50)
            logger.info("PROCESSING COMPLETE")
            logger.info(f"Total videos: {total_videos}")
            logger.info(f"Successful: {successful}")
            logger.info(f"Failed: {failed}")
            logger.info(f"Total time: {total_time:.2f}s")
            logger.info(f"Average time per video: {avg_time:.2f}s")
            logger.info("=" * 50)
            
            return results
            
        except Exception as e:
            logger.error(f"Critical error during video processing: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

def main():
    try:
        processor = VideoProcessor(input_dir="data/videos", output_dir="data/video_captions")
        results = processor.process_videos(question="Describe this video in detail.")

        json_file = "data/video_captions.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_videos': len(results),
                'results': results
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {json_file}")
        
        logger.info("Script completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Script failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
