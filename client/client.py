import gradio as gr
import json
import websockets
import websockets.exceptions
import os
import base64
import tempfile
import asyncio
import time
from logger import logger
from dotenv import load_dotenv

load_dotenv()

class Client:
    def __init__(self):
        self.server_host = os.getenv('SERVER_HOST', 'localhost')
        self.server_port = int(os.getenv('SERVER_PORT', 7766))
        self.current_results = []
        self.current_query = ""
        
    async def upload_video(self, video_file_path: str, status_callback=None, completion_callback=None):
        """Upload a video file for processing"""
        try:
            if not video_file_path or not os.path.exists(video_file_path):
                error_msg = "Invalid video file path"
                logger.error(error_msg)
                if status_callback:
                    status_callback(f"❌ {error_msg}")
                return
            
            # Read video file
            filename = os.path.basename(video_file_path)
            logger.info(f"Reading video file: {filename}")
            
            if status_callback:
                status_callback(f"📖 Reading video file: {filename}")
            
            with open(video_file_path, 'rb') as f:
                video_data = f.read()
            
            # Encode to base64
            if status_callback:
                status_callback("🔐 Encoding video data...")
            
            video_b64 = base64.b64encode(video_data).decode('utf-8')
            
            # Connect to server
            uri = f"ws://{self.server_host}:{self.server_port}/"
            logger.info(f"Connecting to {uri}")
            
            if status_callback:
                status_callback("🔗 Connecting to server...")
            
            async with websockets.connect(uri, ping_timeout=300, ping_interval=60) as websocket:
                logger.info(f"Connected to server")
                
                # Send upload request
                message = {
                    "action": "upload",
                    "video_data": video_b64,
                    "filename": filename
                }
                
                if status_callback:
                    status_callback("📤 Uploading video...")
                
                await websocket.send(json.dumps(message))
                
                # Listen for responses
                while True:
                    response_json = await websocket.recv()
                    response_data = json.loads(response_json)
                    event_type = response_data.get('event')
                    data = response_data.get('data')
                    
                    logger.info(f"Received event: {event_type}")
                    
                    if event_type == 'upload_progress':
                        if status_callback:
                            status_callback(data)
                            
                    elif event_type == 'upload_completed':
                        success_msg = f"✅ Upload completed! Video ID: {data.get('video_id')}"
                        logger.info(success_msg)
                        if status_callback:
                            status_callback(success_msg)
                        if completion_callback:
                            completion_callback(True, data)
                        break
                        
                    elif event_type == 'upload_failed':
                        error_msg = f"❌ Upload failed: {data.get('error')}"
                        logger.error(error_msg)
                        if status_callback:
                            status_callback(error_msg)
                        if completion_callback:
                            completion_callback(False, data)
                        break
                        
                    elif event_type == 'error':
                        error_msg = f"❌ Server error: {data}"
                        logger.error(error_msg)
                        if status_callback:
                            status_callback(error_msg)
                        if completion_callback:
                            completion_callback(False, {"error": data})
                        break
                        
        except Exception as e:
            error_msg = f"❌ Upload error: {e}"
            logger.error(error_msg)
            if status_callback:
                status_callback(error_msg)
            if completion_callback:
                completion_callback(False, {"error": str(e)})

    async def search_videos(self, query: str, status_callback=None, results_callback=None):
        """Search for videos using the query"""
        self.current_query = query
        self.current_results = []
        
        try:
            uri = f"ws://{self.server_host}:{self.server_port}/"
            logger.info(f"Connecting to {uri}")
            
            async with websockets.connect(
                uri, 
                max_size=50 * 1024 * 1024,
                ping_timeout=300, 
                ping_interval=60,
                close_timeout=10
            ) as websocket:
                logger.info(f"Connected to server")
                
                # Send search request
                message = {
                    "action": "search",
                    "query": query
                }
                await websocket.send(json.dumps(message))
                
                # Listen for responses
                while True:
                    response_json = await websocket.recv()
                    response_data = json.loads(response_json)
                    event_type = response_data.get('event')
                    data = response_data.get('data')
                    
                    logger.info(f"Received event: {event_type}")
                    
                    if event_type == 'encoding_query':
                        if status_callback:
                            status_callback("🔍 Encoding query...")
                            
                    elif event_type == 'searching_chunks':
                        if status_callback:
                            status_callback("📹 Searching for similar video chunks...")
                            
                    elif event_type == 'extracting_videos':
                        if status_callback:
                            status_callback("✂️ Extracting video chunks...")
                            
                    elif event_type == 'search_completed':
                        self.current_results = data.get('chunks', [])
                        if status_callback:
                            status_callback(f"✅ Search completed! Found {len(self.current_results)} video chunks.")
                        if results_callback:
                            results_callback(self.current_results)
                        break
                        
                    elif event_type == 'error':
                        error_msg = f"❌ Error: {data}"
                        logger.error(error_msg)
                        if status_callback:
                            status_callback(error_msg)
                        break
                        
        except websockets.exceptions.ConnectionClosed as e:
            error_msg = f"❌ Connection closed unexpectedly: {e}"
            logger.error(error_msg)
            if status_callback:
                status_callback(error_msg)
        except ConnectionResetError as e:
            error_msg = f"❌ Connection was reset by server - this is normal after search completion"
            logger.info(error_msg)  # Log as info since this is often normal
            # Don't call status_callback for connection reset as it's often expected
        except Exception as e:
            error_msg = f"❌ Connection error: {e}"
            logger.error(error_msg)
            if status_callback:
                status_callback(error_msg)
    
    def create_video_from_base64(self, video_data: str, video_id: str, chunk_id: int) -> str:
        """Create a temporary video file from base64 data"""
        try:
            if not video_data:
                return None
            
            # Decode base64 data
            video_bytes = base64.b64decode(video_data)
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(
                suffix=f'_{video_id}_{chunk_id}.mp4',
                delete=False
            )
            
            with open(temp_file.name, 'wb') as f:
                f.write(video_bytes)
            
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Error creating video file: {e}")
            return None
    
    def format_results_summary(self) -> str:
        """Format search results summary"""
        if not self.current_results:
            return "No results yet. Enter a query and click Search!"
        
        summary = f"# Search Results for: '{self.current_query}'\n\n"
        summary += f"Found **{len(self.current_results)}** video chunks:\n\n"
        
        for i, result in enumerate(self.current_results, 1):
            video_id = result.get('video_id', 'Unknown')
            chunk_id = result.get('chunk_id', 'Unknown')
            start_time = result.get('start_time', 0)
            end_time = result.get('end_time', 0)
            similarity = result.get('similarity_score', 0)
            
            summary += f"**{i}. Video {video_id} - Chunk {chunk_id}**\n"
            summary += f"   - Time: {start_time:.1f}s - {end_time:.1f}s\n"
            summary += f"   - Similarity: {similarity:.3f}\n"
            summary += f"   - Extracted: {'✅' if result.get('extraction_success') else '❌'}\n\n"
        
        return summary

def create_gradio_app():
    """Create the Gradio video search interface"""
    client = Client()
    
    async def upload_video(video_file, progress=gr.Progress()):
        """Handle video upload with real-time progress updates"""
        if not video_file:
            return (
                "Please select a video file to upload!",
                0,  # progress
                gr.update(visible=True),   # upload button
                gr.update(visible=False),  # search button
                gr.update(visible=False)   # retry button
            )
        
        # Initialize progress
        progress(0, desc="Starting upload...")
        
        # Status updates with progress tracking
        current_progress = 0
        status_updates = []
        
        def update_status_with_progress(status):
            nonlocal current_progress, status_updates
            status_updates.append(f"⏰ {time.strftime('%H:%M:%S')} - {status}")
            
            # Update progress based on status keywords with detailed server-side tracking
            if "Reading video" in status:
                current_progress = 5
                progress(0.05, desc="Reading video file...")
            elif "Encoding video" in status:
                current_progress = 10
                progress(0.10, desc="Encoding video data...")
            elif "Connecting" in status:
                current_progress = 15
                progress(0.15, desc="Connecting to server...")
            elif "Uploading" in status:
                current_progress = 20
                progress(0.20, desc="Uploading video...")
            elif "Starting processing pipeline" in status:
                current_progress = 25
                progress(0.25, desc="Starting video processing pipeline...")
            elif "Step 1/6: ASR Transcription completed" in status:
                current_progress = 35
                progress(0.35, desc="Audio transcription completed...")
            elif "Step 2/6: Video Captioning completed" in status:
                current_progress = 50
                progress(0.50, desc="Video captioning completed...")
            elif "Step 3/6: Key Frame Captioning completed" in status:
                current_progress = 65
                progress(0.65, desc="Key frame analysis completed...")
            elif "Step 4/6: Video Chunking completed" in status:
                current_progress = 75
                progress(0.75, desc="Video chunking completed...")
            elif "Step 5/6: Video Embedding Generation completed" in status:
                current_progress = 85
                progress(0.85, desc="Embedding generation completed...")
            elif "Step 6/6: Database Ingestion completed" in status:
                current_progress = 95
                progress(0.95, desc="Database ingestion completed...")
            elif "Video processing pipeline completed successfully" in status:
                current_progress = 100
                progress(1.0, desc="All processing completed!")
            elif "processing" in status.lower() and "step" not in status.lower():
                # Generic processing updates (fallback)
                current_progress = min(current_progress + 5, 90)
                progress(current_progress / 100, desc="Processing video...")
            elif "completed" in status.lower() and "Upload completed" in status:
                current_progress = 100
                progress(1.0, desc="Upload completed!")
            
            # Format status display with recent updates
            recent_updates = status_updates[-5:]  # Show last 5 updates
            formatted_status = "### 📤 Upload Progress\n\n"
            formatted_status += f"**Progress: {current_progress}%**\n\n"
            for update in recent_updates:
                formatted_status += f"{update}\n\n"
            
            return formatted_status
        
        upload_result = {"success": False}
        
        def upload_completion(success, data):
            nonlocal upload_result
            upload_result = {"success": success, "data": data}
        
        # Perform the upload
        final_status = ""
        try:
            await client.upload_video(
                video_file,
                status_callback=update_status_with_progress,
                completion_callback=upload_completion
            )
        except Exception as e:
            error_msg = f"❌ Upload error: {e}"
            logger.error(error_msg)
            progress(0, desc="Upload failed!")
            final_status = f"### ❌ Upload Failed\n\n{error_msg}\n\nPlease try again or check the server connection."
            return (
                final_status,
                0,  # progress
                gr.update(visible=True),   # upload button
                gr.update(visible=False),  # search button
                gr.update(visible=True)    # retry button
            )
        
        # Update UI based on results
        if upload_result["success"]:
            video_id = upload_result["data"].get("video_id", "Unknown")
            final_status = f"""### ✅ Upload & Indexing Complete!

**Video ID: {video_id}**

Your video has been successfully uploaded and fully processed:

**📤 Upload Phase:**
- ✅ File upload completed

**🔧 Processing Pipeline:**
- ✅ Audio transcription (ASR) completed
- ✅ Video captioning completed  
- ✅ Key frame analysis completed
- ✅ Video chunking (30-second segments) completed
- ✅ Embedding generation (Qwen3-Embedding-8B) completed
- ✅ Database indexing (ChromaDB) completed

**🎯 Ready for Search:**
Your video is now fully indexed and searchable! You can search for content using natural language queries in the search tab."""
            
            progress(1.0, desc="Upload completed successfully!")
            return (
                final_status,
                100,  # progress
                gr.update(visible=False),  # hide upload button
                gr.update(visible=True),   # show search button
                gr.update(visible=False)   # hide retry button
            )
        else:
            error = upload_result.get("data", {}).get("error", "Unknown error")
            final_status = f"""### ❌ Upload Failed

**Error:** {error}

The upload process failed. Please check:
- Video file format is supported (MP4, AVI, MOV, MKV, WebM)
- File size is reasonable
- Server is running and accessible
- Network connection is stable"""
            
            progress(0, desc="Upload failed!")
            return (
                final_status,
                0,  # progress
                gr.update(visible=True),   # show upload button
                gr.update(visible=False),  # hide search button
                gr.update(visible=True)    # show retry button
            )
    
    
    async def search_videos(query, progress=gr.Progress()):
        """Handle video search with real-time progress updates"""
        if not query.strip():
            return (
                "Please enter a search query!",
                client.format_results_summary(),
                gr.update(visible=False),  # search again button
                *[gr.update(visible=False, value=None)] * 5  # Hide all video players
            )
        
        # Initialize progress
        progress(0, desc="Starting search...")
        
        # Status updates with progress tracking
        current_progress = 0
        status_updates = []
        results_summary = ""
        video_outputs = [gr.update(visible=False, value=None)] * 5
        
        def update_status_with_progress(status):
            nonlocal current_progress, status_updates
            status_updates.append(f"⏰ {time.strftime('%H:%M:%S')} - {status}")
            
            # Update progress based on status keywords
            if "Encoding query" in status:
                current_progress = 25
                progress(0.25, desc="Encoding search query...")
            elif "Searching" in status:
                current_progress = 50
                progress(0.5, desc="Searching video database...")
            elif "Extracting" in status:
                current_progress = 75
                progress(0.75, desc="Extracting video chunks...")
            elif "completed" in status.lower():
                current_progress = 100
                progress(1.0, desc="Search completed!")
            
            # Format status display with recent updates
            recent_updates = status_updates[-3:]  # Show last 3 updates
            formatted_status = "### 🔍 Search Progress\n\n"
            formatted_status += f"**Query:** {query}\n\n"
            formatted_status += f"**Progress: {current_progress}%**\n\n"
            for update in recent_updates:
                formatted_status += f"{update}\n\n"
            
            return formatted_status
        
        def update_results(results):
            nonlocal results_summary, video_outputs
            results_summary = client.format_results_summary()
            
            # Process video results with better feedback
            processed_count = 0
            for i, result in enumerate(results[:5]):  # Only show top 5
                if result.get('extraction_success') and result.get('video_data'):
                    # Create temporary video file
                    video_file = client.create_video_from_base64(
                        result['video_data'],
                        result['video_id'],
                        result['chunk_id']
                    )
                    
                    if video_file:
                        processed_count += 1
                        similarity_score = result.get('similarity_score', 0)
                        video_info = f"🎬 Video {result['video_id']} - Chunk {result['chunk_id']} | ⏱️ {result['start_time']:.1f}s-{result['end_time']:.1f}s | 🎯 Similarity: {similarity_score:.3f}"
                        video_outputs[i] = gr.update(
                            visible=True,
                            value=video_file,
                            label=video_info
                        )
                        
                        # Update progress as videos are processed
                        video_progress = 85 + (processed_count * 3)  # 85-100% for video processing
                        progress(min(video_progress / 100, 1.0), desc=f"Loading video {processed_count}...")
                    else:
                        video_outputs[i] = gr.update(visible=False, value=None)
                else:
                    video_outputs[i] = gr.update(visible=False, value=None)
        
        # Perform the search
        final_status = ""
        try:
            await client.search_videos(
                query,
                status_callback=update_status_with_progress,
                results_callback=update_results
            )
            
            # Final status update
            num_results = len(client.current_results)
            if num_results > 0:
                final_status = f"""### ✅ Search Completed!

**Query:** {query}
**Results:** Found {num_results} matching video chunks

The search has completed successfully. Results are ranked by similarity score, with the most relevant videos shown first."""
                progress(1.0, desc="Search completed successfully!")
            else:
                final_status = f"""### 🔍 No Results Found

**Query:** {query}

No matching video chunks were found. Try:
- Using different keywords
- Making your query more specific
- Checking if videos have been uploaded and processed"""
                progress(1.0, desc="Search completed - no results")
                
        except ConnectionResetError as e:
            # Connection reset is often normal after search completion on Windows
            logger.info(f"Connection reset after search - this is normal on Windows: {e}")
            if not final_status:  # Only show error if we haven't already set a final status
                final_status = f"""### ✅ Search Completed

**Query:** {query}

Search completed successfully. Some connection cleanup errors on Windows are normal and can be ignored."""
                progress(1.0, desc="Search completed!")
        except Exception as e:
            error_msg = f"❌ Search error: {e}"
            logger.error(error_msg)
            progress(0, desc="Search failed!")
            final_status = f"""### ❌ Search Failed

**Query:** {query}
**Error:** {error_msg}

The search process failed. Please check:
- Server is running and accessible
- Network connection is stable
- Try a different search query"""
        
        return (
            final_status, 
            results_summary, 
            gr.update(visible=True),  # show search again button
            *video_outputs
        )
    
    
    # Create the interface
    with gr.Blocks(title="Video RAG Search", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎬 Video RAG Search")
        gr.Markdown("Upload videos for indexing and search for video chunks using natural language queries.")
        
        with gr.Tabs() as tabs:
            # Upload Tab
            with gr.TabItem("📤 Upload Video", id="upload_tab"):
                gr.Markdown("## Upload a Video for Processing")
                gr.Markdown("Upload a video file to be processed through the complete pipeline: ASR transcription, video captioning, key frame analysis, chunking, embedding generation, and database ingestion.")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        video_upload = gr.File(
                            label="Select Video File",
                            file_types=[".mp4", ".avi", ".mov", ".mkv", ".webm"],
                            file_count="single"
                        )
                    with gr.Column(scale=1):
                        upload_btn = gr.Button("📤 Upload & Process", variant="primary", size="lg")
                        reset_upload_btn = gr.Button("🔄 Reset", variant="secondary")
                        retry_upload_btn = gr.Button("🔄 Retry Upload", variant="secondary", visible=False)
                
                # Progress bar for upload
                upload_progress = gr.Number(
                    label="Upload Progress (%)", 
                    value=0,
                    minimum=0,
                    maximum=100,
                    interactive=False,
                    visible=True
                )
                
                upload_status = gr.Markdown("Ready to upload a video!", label="Upload Status")
                
                with gr.Row():
                    go_to_search_btn = gr.Button("🔍 Go to Search", variant="secondary", visible=False)
                
                gr.Markdown("""
                ### 📋 Upload Process:
                1. **Select a video file** (supported formats: MP4)
                2. **Click Upload & Process** to start the pipeline
                3. **Wait for processing** - this may take several minutes depending on video length
                4. **Processing includes:**
                   - Audio transcription with timestamps
                   - Video captioning
                   - Key frame captioning
                   - Video chunking (30-second segments)
                   - Embedding generation with Qwen3-Embedding-8B
                   - Database ingestion into ChromaDB
                """)
            
            # Search Tab
            with gr.TabItem("🔍 Search Videos", id="search_tab"):
                gr.Markdown("## Search Video Content")
                gr.Markdown("Search for video chunks using natural language queries. The system uses Qwen3-Embedding-8B for semantic search.")
                
                with gr.Row():
                    with gr.Column(scale=4):
                        query_input = gr.Textbox(
                            placeholder="Enter your search query (e.g., 'girl with purple hair', 'butterfly flying', 'clock animation')...",
                            label="Search Query",
                            lines=2
                        )
                    with gr.Column(scale=1):
                        search_btn = gr.Button("🔍 Search", variant="primary", size="lg")
                        clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                        search_again_btn = gr.Button("🔍 Search Again", variant="secondary", visible=False)
                
                # Status and results
                with gr.Row():
                    with gr.Column(scale=1):
                        status_output = gr.Markdown("Ready to search!", label="Status")
                    with gr.Column(scale=2):
                        results_summary = gr.Markdown("No results yet. Enter a query and click Search!", label="Results Summary")
                
                # Video players for results
                gr.Markdown("## 📹 Video Results")
                
                with gr.Row():
                    video1 = gr.Video(label="Result 1", visible=False)
                    video2 = gr.Video(label="Result 2", visible=False)
                
                with gr.Row():
                    video3 = gr.Video(label="Result 3", visible=False)
                    video4 = gr.Video(label="Result 4", visible=False)
                
                video5 = gr.Video(label="Result 5", visible=False)
                
                # Example queries
                gr.Examples(
                    examples=[
                        ["girl with purple hair"],
                        ["butterfly flying"],
                        ["clock animation"],
                        ["person looking in mirror"],
                        ["diamond moving across screen"]
                    ],
                    inputs=[query_input]
                )
                
                # Instructions
                gr.Markdown("""
                ### 📋 Search Instructions:
                1. **Enter a search query** describing the video content you're looking for
                2. **Click Search** or press Enter to find similar video chunks
                3. **View the results** - up to 5 video chunks will be displayed
                4. Each result shows the video segment with similarity score and time range
                """)
        
        # Upload Event handlers
        upload_btn.click(
            fn=upload_video,
            inputs=[video_upload],
            outputs=[upload_status, upload_progress, upload_btn, go_to_search_btn, retry_upload_btn]
        )
        
        retry_upload_btn.click(
            fn=upload_video,
            inputs=[video_upload],
            outputs=[upload_status, upload_progress, upload_btn, go_to_search_btn, retry_upload_btn]
        )
        
        def reset_upload_with_progress():
            """Reset upload interface with progress"""
            return (
                "Ready to upload a video!",
                0,  # progress
                gr.update(visible=True),   # upload button
                gr.update(visible=False),  # search button
                gr.update(visible=False)   # retry button
            )
        
        reset_upload_btn.click(
            fn=reset_upload_with_progress,
            outputs=[upload_status, upload_progress, upload_btn, go_to_search_btn, retry_upload_btn]
        )
        
        # Function to switch to search tab
        def switch_to_search():
            return gr.Tabs.update(selected="search_tab")
        
        go_to_search_btn.click(
            fn=switch_to_search,
            outputs=[tabs]
        )
        
        # Search Event handlers
        search_btn.click(
            fn=search_videos,
            inputs=[query_input],
            outputs=[status_output, results_summary, search_again_btn, video1, video2, video3, video4, video5]
        )
        
        search_again_btn.click(
            fn=search_videos,
            inputs=[query_input],
            outputs=[status_output, results_summary, search_again_btn, video1, video2, video3, video4, video5]
        )
        
        query_input.submit(
            fn=search_videos,
            inputs=[query_input],
            outputs=[status_output, results_summary, search_again_btn, video1, video2, video3, video4, video5]
        )
        
        def clear_results_with_button():
            """Clear all results and hide search again button"""
            client.current_results = []
            client.current_query = ""
            return (
                "Ready to search!",
                "No results yet. Enter a query and click Search!",
                gr.update(visible=False),  # hide search again button
                *[gr.update(visible=False, value=None)] * 5
            )
        
        clear_btn.click(
            fn=clear_results_with_button,
            outputs=[status_output, results_summary, search_again_btn, video1, video2, video3, video4, video5]
        )
    
    return app

if __name__ == "__main__":
    app = create_gradio_app()
    port = int(os.getenv('CLIENT_PORT', 7861))
    app.launch(server_name="localhost", server_port=port, share=False)
