import os
import json
import time
import chromadb
import torch
from typing import Dict, List
from sentence_transformers import SentenceTransformer
from logger import logger
from dotenv import load_dotenv
load_dotenv()

from video_processor import VideoProcessor

class Server:
    def __init__(self):
        self.chromadb_host = os.getenv('CHROMADB_HOST', 'localhost')
        self.chromadb_port = int(os.getenv('CHROMADB_PORT', 8000))
        self.top_k = int(os.getenv('RETRIEVAL_TOP_K', 5))
        self.video_processor = VideoProcessor()

        logger.info(f"Connecting to ChromaDB at {self.chromadb_host}:{self.chromadb_port}")
        self.client = chromadb.HttpClient(
            host=self.chromadb_host,
            port=self.chromadb_port
        )
        
        collection_name = "video_content"
        self.collection = self.client.get_collection(name=collection_name)
        logger.info(f"Connected to collection: {collection_name}")

        self.model = SentenceTransformer(
            "Qwen/Qwen3-Embedding-8B",
            model_kwargs={
                "attn_implementation": "flash_attention_2",
                "device_map": "auto",
                "torch_dtype": torch.bfloat16
            },
            tokenizer_kwargs={"padding_side": "left"},
        )
    
    async def search_video_chunks(self, query: str, websocket=None) -> List[Dict]:
        try:
            # Send status update
            if websocket:
                await websocket.send(json.dumps({
                    "event": "encoding_query",
                    "data": "Encoding query..."
                }))
            
            # Encode the query
            start_time = time.time()
            query_embedding = self.model.encode([query], prompt_name="query")
            query_embedding = query_embedding[0].tolist()
            encoding_time = time.time() - start_time
            
            logger.info(f"Query encoded in {encoding_time:.2f}s")
            
            # Send status update
            if websocket:
                await websocket.send(json.dumps({
                    "event": "searching_chunks",
                    "data": f"Searching for top {self.top_k} similar video chunks..."
                }))
            
            # Search in ChromaDB
            start_time = time.time()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=self.top_k,
                include=['metadatas', 'distances']
            )
            search_time = time.time() - start_time
            
            logger.info(f"Vector search completed in {search_time:.2f}s")
            logger.info(f"Found {len(results['ids'][0])} results")
            
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            chunks = []
            for i, (metadata, distance) in enumerate(zip(metadatas, distances)):
                chunk_data = {
                    'rank': i + 1,
                    'video_id': metadata.get('video_id'),
                    'chunk_id': metadata.get('chunk_id'),
                    'start_time': metadata.get('start_time'),
                    'end_time': metadata.get('end_time'),
                    'duration': metadata.get('duration'),
                    'similarity_score': 1 - distance,
                }
                chunks.append(chunk_data)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error searching video chunks: {e}")
            raise
    
    async def extract_video_chunks(self, chunks: List[Dict], websocket=None) -> List[Dict]:
        """Extract video data for the retrieved chunks"""
        try:
            if websocket:
                await websocket.send(json.dumps({
                    "event": "extracting_videos",
                    "data": f"Extracting {len(chunks)} video chunks..."
                }))
            
            # Prepare chunk requests
            chunk_requests = []
            for chunk in chunks:
                chunk_requests.append((
                    chunk['video_id'],
                    chunk['chunk_id'],
                    chunk['start_time'],
                    chunk['end_time']
                ))
            
            # Extract video chunks
            start_time = time.time()
            extracted_chunks = self.video_processor.extract_multiple_chunks(chunk_requests)
            extraction_time = time.time() - start_time
            
            logger.info(f"Video extraction completed in {extraction_time:.2f}s")
            
            # Merge search results with extracted video data
            final_results = []
            for i, chunk in enumerate(chunks):
                if i < len(extracted_chunks):
                    extracted = extracted_chunks[i]
                    merged_result = {
                        **chunk,
                        'video_data': extracted.get('video_data'),
                        'extraction_success': extracted.get('success', False),
                        'extraction_error': extracted.get('error')
                    }
                else:
                    merged_result = {
                        **chunk,
                        'extraction_success': False,
                        'extraction_error': 'Not processed'
                    }
                
                final_results.append(merged_result)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error extracting video chunks: {e}")
            raise

    async def handle_video_search(self, websocket, message_data):
        query = message_data.get("query", "")
        
        logger.info(f"Video search query: {query}")
        
        try:
            chunks = await self.search_video_chunks(query, websocket)
            
            results = await self.extract_video_chunks(chunks, websocket)
            
            await websocket.send(json.dumps({
                "event": "search_completed",
                "data": {
                    "query": query,
                    "total_results": len(results),
                    "chunks": results
                }
            }))
            logger.info(f"Video search completed successfully. Found {len(results)} chunks.")
            
        except Exception as e:
            error_msg = f"Error processing video search: {e}"
            logger.error(error_msg)
            await websocket.send(json.dumps({
                "event": "error",
                "data": error_msg
            }))
