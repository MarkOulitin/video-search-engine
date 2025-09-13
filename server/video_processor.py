import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
import tempfile
import base64
from logger import logger

class VideoProcessor:
    
    def __init__(self):
        self.source_videos_dir = Path('./data/videos')        
        
    def find_source_video_file(self, video_id: str) -> Optional[Path]:
        for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']:
            video_file = self.source_videos_dir / f"{video_id}{ext}"
            if video_file.exists():
                return video_file
        logger.error(f"Source video file not found for video_id: {video_id}")
        return None
    
    def extract_video_chunk(self, video_id: str, chunk_id: int, start_time: float, end_time: float) -> Optional[str]:
        """Extract a video chunk and return it as base64 encoded string"""
        try:
            # Find source video file
            source_video = self.find_source_video_file(video_id)
            if not source_video:
                return None
            
            # Create temporary file for the chunk
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Use ffmpeg to extract the chunk
                duration = end_time - start_time
                cmd = [
                    'ffmpeg',
                    '-i', str(source_video),
                    '-ss', str(start_time),
                    '-t', str(duration),
                    '-c:v', 'libx264',
                    '-c:a', 'copy',
                    '-avoid_negative_ts', 'make_zero',
                    '-y',  # Overwrite output file
                    temp_path
                ]
                
                logger.info(f"Extracting chunk: {video_id} [{start_time:.2f}s - {end_time:.2f}s]")
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode != 0:
                    logger.error(f"FFmpeg error: {result.stderr}")
                    return None
                
                # Read the extracted chunk and encode as base64
                with open(temp_path, 'rb') as f:
                    video_data = f.read()
                
                base64_data = base64.b64encode(video_data).decode('utf-8')
                logger.info(f"Successfully extracted chunk {chunk_id} from video {video_id}")
                
                return base64_data
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout extracting chunk from {video_id}")
            return None
        except Exception as e:
            logger.error(f"Error extracting chunk from {video_id}: {e}")
            return None
    
    def extract_multiple_chunks(self, chunk_requests: List[Tuple[str, int, float, float]]) -> List[Dict]:
        results = []
        
        for video_id, chunk_id, start_time, end_time in chunk_requests:
            try:
                video_data = self.extract_video_chunk(video_id, chunk_id, start_time, end_time)
                
                result = {
                    'video_id': video_id,
                    'chunk_id': chunk_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'video_data': video_data,
                    'success': video_data is not None
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_id} from video {video_id}: {e}")
                results.append({
                    'video_id': video_id,
                    'chunk_id': chunk_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'error': str(e),
                    'success': False
                })
        
        return results
