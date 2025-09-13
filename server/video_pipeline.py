from pathlib import Path
from typing import Dict, Optional, Callable
from logger import logger
from nemo.collections.asr.models import ASRModel
from scripts.process_transcriptions import extract_audio_from_video, transcribe_audio, save_timestamps_json
from scripts.process_video_captions import VideoCaptioningProcessor
from scripts.process_keyframe_captioning import FrameCaptioningProcessor, save_captions_json
from scripts.chunk_videos import VideoChunker
from scripts.generate_embeddings import VideoEmbeddingGenerator
from scripts.ingest_video_embeddings import VideoEmbeddingIngest
from moviepy import VideoFileClip

class VideoPipelineProcessor:
    
    def __init__(self):
        self.videos_dir = Path('data/videos')
        self.asr_dir = Path("data/asr_timestamps")
        self.video_captions_dir = Path("data/video_captions")
        self.keyframe_captions_dir = Path("data/key_frame_captions")
        self.temp_audio_dir = Path("temp_audio")
        # Ensure directories exist
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.asr_dir.mkdir(exist_ok=True)
        self.video_captions_dir.mkdir(exist_ok=True)
        self.keyframe_captions_dir.mkdir(exist_ok=True)
        self.temp_audio_dir.mkdir(exist_ok=True)
        

        self.asr_model = ASRModel.from_pretrained(model_name="nvidia/canary-1b-v2")
        self.video_captioning_processor = VideoCaptioningProcessor()
        self.key_frame_captioning_processor = FrameCaptioningProcessor(batch_size=16)
        self.embedding_generator = VideoEmbeddingGenerator()
        self.embedding_ingester = VideoEmbeddingIngest()
    
    def save_uploaded_video(self, video_data, video_id):
        video_path = str(self.videos_dir / f"{video_id}.mp4")
        with open(video_path, 'wb') as f:
            f.write(video_data)
        logger.info(f"Saved uploaded video: {video_path}")
        return video_path
    
    def transcribe(self, video_id, video_path):
        temp_audio_path = self.temp_audio_dir / f"{video_id}.wav"
        try:
            extract_audio_from_video(str(video_path), str(temp_audio_path))
            segments = transcribe_audio(str(temp_audio_path), self.asr_model)
            transcribed_file = save_timestamps_json(segments, str(self.asr_dir), video_id)
            return transcribed_file
        finally:
            # Clean up temporary audio file
            if temp_audio_path.exists():
                temp_audio_path.unlink()

    def video_captioning(self, video_id, video_path):
        video_caption = self.video_captioning_processor.caption_video(video_path, "Describe this video in detail.")
        return video_caption

    async def process_video(self, video_id: str, video_path: str, progress_callback: Optional[Callable] = None) -> bool:
        try:
            logger.info(f"Starting video processing pipeline for {video_id}")
            
            if progress_callback:
                await progress_callback(f"Starting processing pipeline for video {video_id}")
            
            transcribed_file = self.transcribe(video_id, video_path)
            if progress_callback:
                await progress_callback(f"Step 1/6: ASR Transcription completed")
            video_caption = self.video_captioning(video_id, video_path)
            if progress_callback:
                await progress_callback(f"Step 2/6: Video Captioning completed")
            with VideoFileClip(str(video_path)) as clip:
                duration = clip.duration
            captions = self.key_frame_captioning_processor.process_video_key_frames(str(video_path), frame_interval=5.0)
            save_captions_json(captions, str(self.keyframe_captions_dir), video_id)
            if progress_callback:
                await progress_callback(f"Step 3/6: Key Frame Captioning completed")
            chunking_processor = VideoChunker(video_id=video_id, video_path=video_path, video_caption=video_caption)
            chunking_processor.process_video(video_id, duration)
            if progress_callback:
                await progress_callback(f"Step 4/6: Video Chunking completed")
            self.embedding_generator.process_all_videos()
            chunks_with_prompts = self.embedding_generator.process_video_chunks(video_id)
            embeddings, metadata_list = self.embedding_generator.generate_embeddings_for_chunks(chunks_with_prompts)
            if progress_callback:
                await progress_callback(f"Step 5/6: Video Embedding Generation completed")
            prompts = [chunk[0] for chunk in chunks_with_prompts]
            self.embedding_ingester.save_embeddings_to_chromadb(embeddings, metadata_list, prompts)
            # Pipeline completed successfully
            success_msg = f"✅ Video processing pipeline completed successfully for {video_id}"
            logger.info(success_msg)
            if progress_callback:
                await progress_callback(success_msg)
            
            return True
            
        except Exception as e:
            error_msg = f"Error in video processing pipeline: {e}"
            logger.error(error_msg)
            if progress_callback:
                await progress_callback(f"❌ {error_msg}")
            return False
    
    async def process_uploaded_video(self, video_data: bytes, video_id: str, progress_callback: Optional[Callable] = None) -> Dict:
        try:
            if progress_callback:
                await progress_callback("Saving uploaded video...")
            
            video_path = self.save_uploaded_video(video_data, video_id)
            
            if progress_callback:
                await progress_callback(f"Video saved")
            
            success = await self.process_video(video_id, video_path, progress_callback)
            
            result = {
                'video_id': video_id,
                'filename': f'{video_id}.mp4',
                'success': success,
                'message': 'Video processing completed successfully' if success else 'Video processing failed'
            }
            
            return result
            
        except Exception as e:
            error_msg = f"Error processing uploaded video: {e}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'filename': f'{video_id}.mp4',
            }
