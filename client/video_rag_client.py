import gradio as gr
import json
import websockets
import os
import base64
import tempfile
from logger import logger
from dotenv import load_dotenv

load_dotenv()

class Client:
    def __init__(self):
        self.server_host = os.getenv('SERVER_HOST', 'localhost')
        self.server_port = int(os.getenv('SERVER_PORT', 7766))
        self.current_results = []
        self.current_query = ""
        
    async def search_videos(self, query: str, status_callback=None, results_callback=None):
        """Search for videos using the query"""
        self.current_query = query
        self.current_results = []
        
        try:
            uri = f"ws://{self.server_host}:{self.server_port}/"
            logger.info(f"Connecting to {uri}")
            
            async with websockets.connect(uri) as websocket:
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
    
    async def search_videos(query):
        """Handle video search"""
        if not query.strip():
            return (
                "Please enter a search query!",
                client.format_results_summary(),
                *[gr.update(visible=False, value=None)] * 5  # Hide all video players
            )
        
        # Status updates
        status_text = ""
        results_summary = ""
        video_outputs = [gr.update(visible=False, value=None)] * 5
        
        def update_status(status):
            nonlocal status_text
            status_text = status
        
        def update_results(results):
            nonlocal results_summary, video_outputs
            results_summary = client.format_results_summary()
            
            # Process video results
            for i, result in enumerate(results[:5]):  # Only show top 5
                if result.get('extraction_success') and result.get('video_data'):
                    # Create temporary video file
                    video_file = client.create_video_from_base64(
                        result['video_data'],
                        result['video_id'],
                        result['chunk_id']
                    )
                    
                    if video_file:
                        video_info = f"Video {result['video_id']} - Chunk {result['chunk_id']} | {result['start_time']:.1f}s-{result['end_time']:.1f}s | Similarity: {result.get('similarity_score', 0):.3f}"
                        video_outputs[i] = gr.update(
                            visible=True,
                            value=video_file,
                            label=video_info
                        )
                    else:
                        video_outputs[i] = gr.update(visible=False, value=None)
                else:
                    video_outputs[i] = gr.update(visible=False, value=None)
        
        # Perform the search
        try:
            await client.search_videos(
                query,
                status_callback=update_status,
                results_callback=update_results
            )
        except Exception as e:
            status_text = f"❌ Error: {e}"
        
        return (status_text, results_summary, *video_outputs)
    
    def clear_results():
        """Clear all results"""
        client.current_results = []
        client.current_query = ""
        return (
            "Ready to search!",
            "No results yet. Enter a query and click Search!",
            *[gr.update(visible=False, value=None)] * 5
        )
    
    # Create the interface
    with gr.Blocks(title="Video RAG Search", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎬 Video RAG Search")
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
        
        # Event handlers
        search_btn.click(
            fn=search_videos,
            inputs=[query_input],
            outputs=[status_output, results_summary, video1, video2, video3, video4, video5]
        )
        
        query_input.submit(
            fn=search_videos,
            inputs=[query_input],
            outputs=[status_output, results_summary, video1, video2, video3, video4, video5]
        )
        
        clear_btn.click(
            fn=clear_results,
            outputs=[status_output, results_summary, video1, video2, video3, video4, video5]
        )
        
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
        ### 📋 Instructions:
        1. **Enter a search query** describing the video content you're looking for
        2. **Click Search** or press Enter to find similar video chunks
        3. **View the results** - up to 5 video chunks will be displayed
        4. Each result shows the video segment with similarity score and time range
        
        ### 🔧 Features:
        - **Semantic Search**: Uses Qwen3-Embedding-8B for understanding natural language queries
        - **Video Extraction**: Automatically extracts relevant video segments
        - **Similarity Scoring**: Shows how well each result matches your query
        - **Real-time Status**: Live updates during the search process
        """)
    
    return app

if __name__ == "__main__":
    app = create_gradio_app()
    port = int(os.getenv('CLIENT_PORT', 7861))
    app.launch(server_name="localhost", server_port=port, share=False)
