import os
import json
from pathlib import Path
import ffmpeg
from nemo.collections.asr.models import ASRModel
from logger import logger

def extract_audio_from_video(video_path, audio_path):
    try:
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, acodec='pcm_s16le', ac=1, ar='16000')
            .overwrite_output()
            .run(quiet=True)
        )
        logger.info(f"Audio extracted: {video_path} -> {audio_path}")
        return audio_path
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg error processing {video_path}: {e}")
        raise

def transcribe_audio(audio_path, asr_model):
    output = asr_model.transcribe(
        [audio_path], 
        source_lang='en', 
        target_lang='en', 
        timestamps=True
    )        
    segment_timestamps = output[0].timestamp['segment']
    segments = []
    for stamp in segment_timestamps:
        segments.append({
            'start': float(stamp['start']),
            'end': float(stamp['end']),
            'segment': str(stamp['segment'])
        })
    
    logger.info(f"Transcribed {len(segments)} segments from {audio_path}")
    return segments

def save_timestamps_json(segments, output_path, video_name):
    json_data = {
        'video_file': video_name,
        'total_segments': len(segments),
        'segments': segments
    }
    
    json_file = os.path.join(output_path, f"{video_name}_timestamps.json")
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved timestamps: {json_file}")
    return json_file

def process_videos():
    
    video_dir = Path("data/videos")
    output_dir = Path("data/asr_timestamps")
    temp_audio_dir = Path("temp_audio")
    
    output_dir.mkdir(exist_ok=True)
    temp_audio_dir.mkdir(exist_ok=True)
    
    try:
        logger.info("Loading NeMo Canary ASR model...")
        asr_model = ASRModel.from_pretrained(model_name="nvidia/canary-1b-v2")
        logger.info("ASR model loaded successfully")
        
        video_files = list(video_dir.glob("*.mp4"))
        total_videos = len(video_files)
        
        logger.info(f"Found {total_videos} video files to process")
        
        for i, video_path in enumerate(video_files, 1):
            video_name = video_path.stem
            logger.info(f"Processing {i}/{total_videos}: {video_name}")
            
            # Create temporary audio file
            temp_audio_path = temp_audio_dir / f"{video_name}.wav"
            
            try:
                extract_audio_from_video(str(video_path), str(temp_audio_path))
                segments = transcribe_audio(str(temp_audio_path), asr_model)
                save_timestamps_json(segments, str(output_dir), video_name)
                if segments:
                    logger.info(f"Sample segments for {video_name}:")
                    for segment in segments[:3]:  # Show first 3 segments
                        logger.info(f"  {segment['start']:.2f}s - {segment['end']:.2f}s: {segment['segment']}")
                
            except Exception as e:
                logger.error(f"Failed to process {video_name}: {e}")
                continue
                
            finally:
                # Clean up temporary audio file
                if temp_audio_path.exists():
                    temp_audio_path.unlink()
        
        logger.info("Processing completed!")
        
    except Exception as e:
        logger.error(f"Critical error: {e}")
        raise
    
    finally:
        # Clean up temporary directory
        if temp_audio_dir.exists():
            try:
                temp_audio_dir.rmdir()
            except OSError:
                logger.warning(f"Could not remove temporary directory: {temp_audio_dir}")


if __name__ == "__main__":
    logger.info("Starting video ASR processing...")
    process_videos()
    logger.info("Video ASR processing finished!")
