import json
import pandas as pd
from pathlib import Path
from logger import logger

class VideoChunker:
    def __init__(self):
        self.chunk_duration = 30.0  # 30 seconds per chunk
        self.min_last_chunk = 15.0  # Minimum duration for last chunk (merge if shorter)
        
        self.video_ids = []
        self.video_durations = {}
        self.video_captions = {}
        
        self.base_path = Path(".")
        self.asr_path = self.base_path / "data" / "asr_timestamps"
        self.video_captions_path = self.base_path / "data" / "video_captions"
        self.key_frames_path = self.base_path / "data" / "key_frame_captions"
        self.output_path = self.base_path / "data" / "chunked_videos"
        
        self.output_path.mkdir(exist_ok=True)
        
    def load_video_metadata(self):
        df = pd.read_csv('data/video_ids.csv')
        self.video_ids = df['video_id']
        self.video_durations = df['duration_seconds']
        with open('data/video_captions/video_captions.json', 'r', encoding='utf-8') as f:
            self.video_captions = json.load(f)
        logger.info(f"Loaded {len(self.video_ids)} video IDs")
    
    def load_asr_data(self, video_id):
        asr_file = self.asr_path / f"{video_id}_timestamps.json"
        if asr_file.exists():
            with open(asr_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.warning(f"ASR file not found for video {video_id}")
            return None
    
    def load_key_frame_data(self, video_id):
        key_frame_file = self.key_frames_path / f"{video_id}_key_frame_captions.json"
        if key_frame_file.exists():
            with open(key_frame_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.warning(f"Key frame file not found for video {video_id}")
            return None
    
    def get_overlapping_segments(self, segments, chunk_start, chunk_end):
        overlapping = []
        
        for segment in segments:
            seg_start = segment.get('start', 0)
            seg_end = segment.get('end', 0)
            
            # Check if segment overlaps with chunk
            if (seg_start < chunk_end) and (seg_end > chunk_start):
                overlapping.append(segment)
        
        return overlapping
    
    def get_chunk_key_frames(self, key_frames, chunk_start, chunk_end):
        """Get key frames that fall within the chunk time interval."""
        chunk_frames = []
        
        for frame in key_frames:
            timestamp = frame.get('timestamp', 0)
            
            # Include frame if timestamp is within chunk bounds
            if (chunk_start <= timestamp) and (timestamp <= chunk_end):
                chunk_frames.append(frame)
        
        return chunk_frames
    
    def create_chunks(self, duration, asr_data, key_frame_data):
        chunks = []
        
        # Get segments and key frames
        segments = asr_data.get('segments', []) if asr_data else []
        key_frames = key_frame_data.get('key_frame_captions', []) if key_frame_data else []
        
        # Calculate number of chunks needed
        num_chunks = int(duration / self.chunk_duration)
        if duration % self.chunk_duration > 0:
            num_chunks += 1
        
        for i in range(num_chunks):
            chunk_start = i * self.chunk_duration
            chunk_end = min((i + 1) * self.chunk_duration, duration)
            
            # Check if this is the last chunk and if it's too short
            is_last_chunk = (i == num_chunks - 1)
            chunk_duration_actual = chunk_end - chunk_start
            
            # If last chunk is too short, merge with previous chunk
            if is_last_chunk and chunk_duration_actual < self.min_last_chunk and len(chunks) > 0:
                # Extend the previous chunk to include this one
                prev_chunk = chunks[-1]
                prev_chunk['end_time'] = chunk_end
                prev_chunk['duration'] = chunk_end - prev_chunk['start_time']
                
                # Add segments and key frames from this chunk to previous chunk
                overlapping_segments = self.get_overlapping_segments(segments, chunk_start, chunk_end)
                chunk_key_frames = self.get_chunk_key_frames(key_frames, chunk_start, chunk_end)
                
                # Merge segments (avoid duplicates)
                existing_segments = {(s['start'], s['end']) for s in prev_chunk['segments']}
                for seg in overlapping_segments:
                    if (seg['start'], seg['end']) not in existing_segments:
                        prev_chunk['segments'].append(seg)
                
                # Merge key frames (avoid duplicates)
                existing_frames = {f['timestamp'] for f in prev_chunk['key_frames']}
                for frame in chunk_key_frames:
                    if frame['timestamp'] not in existing_frames:
                        prev_chunk['key_frames'].append(frame)
                
                # Update segment and frame counts
                prev_chunk['total_segments'] = len(prev_chunk['segments'])
                prev_chunk['total_key_frames'] = len(prev_chunk['key_frames'])
                
                continue
            
            # Get overlapping segments and key frames for this chunk
            overlapping_segments = self.get_overlapping_segments(segments, chunk_start, chunk_end)
            chunk_key_frames = self.get_chunk_key_frames(key_frames, chunk_start, chunk_end)
            
            chunk = {
                'chunk_id': i,
                'start_time': chunk_start,
                'end_time': chunk_end,
                'duration': chunk_duration_actual,
                'total_segments': len(overlapping_segments),
                'segments': overlapping_segments,
                'total_key_frames': len(chunk_key_frames),
                'key_frames': chunk_key_frames
            }
            
            chunks.append(chunk)
        
        return chunks
    
    def process_video(self, video_id, duration):
        try:
            asr_data = self.load_asr_data(video_id)
            key_frame_data = self.load_key_frame_data(video_id)
            video_caption_data = self.video_captions.get(video_id, {})

            chunks = self.create_chunks(duration, asr_data, key_frame_data)
            
            output_data = {
                'video_id': video_id,
                'duration': duration,
                'video_caption': video_caption_data.get('caption', ''),
                'caption_status': video_caption_data.get('status', 'unknown'),
                'total_chunks': len(chunks),
                'chunk_duration': self.chunk_duration,
                'chunks': chunks,
                'metadata': {
                    'original_asr_segments': len(asr_data.get('segments', [])) if asr_data else 0,
                    'original_key_frames': len(key_frame_data.get('key_frame_captions', [])) if key_frame_data else 0,
                    'has_asr_data': asr_data is not None,
                    'has_key_frame_data': key_frame_data is not None,
                    'has_video_caption': bool(video_caption_data.get('caption', '').strip())
                }
            }
            
            # Save to file
            output_file = self.output_path / f"{video_id}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Successfully processed video {video_id} -> {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error processing video {video_id}: {e}")
            return False
    
    def process_all_videos(self):
        successful = 0
        failed = 0
        
        logger.info(f"Starting to process {len(self.video_ids)} videos...")
        
        for i, (video_id, duration) in enumerate(zip(self.video_ids, self.video_durations), 1):
            logger.info(f"Processing video {i}/{len(self.video_ids)}: {video_id}")
            
            if self.process_video(video_id, duration):
                successful += 1
            else:
                failed += 1
        
        return successful, failed
    
    def run(self):
        logger.info("Starting video chunking process...")
        
        self.load_video_metadata()
        
        successful, failed = self.process_all_videos()
        
        logger.info(f"Processing complete!")
        logger.info(f"Successfully processed: {successful} videos")
        logger.info(f"Failed to process: {failed} videos")
        logger.info(f"Output directory: {self.output_path}")
        
        return failed == 0
    
if __name__ == "__main__":
    chunker = VideoChunker()
    chunker.run()
