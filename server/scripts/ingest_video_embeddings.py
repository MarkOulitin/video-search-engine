import numpy as np
import chromadb
import uuid
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from logger import logger
load_dotenv()

class VideoEmbeddingIngest:
    def __init__(self):
        self.chromadb_host = os.getenv('CHROMADB_HOST', 'localhost')
        self.chromadb_port = int(os.getenv('CHROMADB_PORT', 8000))
        self.embeddings_dir = Path('./data/embeddings')
        
        logger.info(f"Connecting to ChromaDB at {self.chromadb_host}:{self.chromadb_port}")
        self.client = chromadb.HttpClient(
            host=self.chromadb_host,
            port=self.chromadb_port
        )
        logger.info(f"Connected to ChromaDB at {self.chromadb_host}:{self.chromadb_port}")
        
        collection_name = 'video_content'
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            configuration={
                "hnsw": {
                    "space": "cosine"
                }
            }
        )
        logger.info(f"Using collection: {collection_name}")
    
    def load_video_embeddings_and_metadata(self):
        embeddings_file = self.embeddings_dir / 'video_embeddings.npy'
        metadata_file = self.embeddings_dir / 'video_metadata.json'
        prompts_file = self.embeddings_dir / 'video_prompts.json'
        
        if not embeddings_file.exists():
            raise FileNotFoundError(f"Video embeddings file not found: {embeddings_file}")
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"Video metadata file not found: {metadata_file}")
            
        if not prompts_file.exists():
            raise FileNotFoundError(f"Video prompts file not found: {prompts_file}")
        
        logger.info(f"Loading embeddings from {embeddings_file}")
        embeddings = np.load(embeddings_file)
        
        logger.info(f"Loading metadata from {metadata_file}")
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        logger.info(f"Loading prompts from {prompts_file}")
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        
        logger.info(f"Loaded {len(embeddings)} embeddings, {len(metadata)} metadata entries, and {len(prompts)} prompts")
        
        if len(embeddings) != len(metadata) or len(prompts) != len(metadata):
            raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs {len(metadata)} metadata entries vs {len(prompts)} prompts")
        
        return embeddings, metadata, prompts
    
    def prepare_metadata_for_chromadb(self, metadata_list):
        prepared_metadata = []
        
        for metadata in metadata_list:
            chromadb_metadata = {
                'video_id': str(metadata.get('video_id', '')),
                'chunk_id': int(metadata.get('chunk_id', 0)),
                'start_time': float(metadata.get('start_time', 0.0)),
                'end_time': float(metadata.get('end_time', 0.0)),
                'duration': float(metadata.get('duration', 0.0)),
            }
            
            prepared_metadata.append(chromadb_metadata)
        
        return prepared_metadata
    
    def save_embeddings_to_chromadb(self, embeddings, metadata_array, prompts):
        ids = [str(uuid.uuid4()) for _ in range(len(embeddings))]
        embeddings_list = embeddings.tolist()
        
        prepared_metadata = self.prepare_metadata_for_chromadb(metadata_array)
        
        batch_size = 1000
        total_batches = (len(embeddings) + batch_size - 1) // batch_size
        
        logger.info(f"Adding {len(embeddings)} video chunk embeddings to ChromaDB in {total_batches} batches")
        
        for i in range(0, len(embeddings), batch_size):
            batch_end = min(i + batch_size, len(embeddings))
            batch_ids = ids[i:batch_end]
            batch_embeddings = embeddings_list[i:batch_end]
            batch_metadata = prepared_metadata[i:batch_end]
            batch_prompts = prompts[i:batch_end]
            
            logger.info(f"Adding batch {i//batch_size + 1}/{total_batches} ({len(batch_ids)} items)")
            
            self.collection.add(
                embeddings=batch_embeddings,
                documents=batch_prompts,
                metadatas=batch_metadata,
                ids=batch_ids
            )
            
            logger.info(f"Successfully added batch {i//batch_size + 1}")
        
        logger.info(f"Successfully added {len(embeddings)} video chunk embeddings to collection")
        
        # Test the collection with a query
        self._test_collection(embeddings_list[0])
        
    def _test_collection(self, test_embedding):
        """Test the collection with a sample query"""
        try:
            logger.info("Testing collection with sample query...")
            results = self.collection.query(
                query_embeddings=[test_embedding],
                n_results=5
            )
            
            logger.info(f"Test query returned {len(results['ids'][0])} results")
            
            # Log some sample results
            for i, (doc_id, metadata, document) in enumerate(zip(results['ids'][0], results['metadatas'][0], results['documents'][0])):
                video_id = metadata.get('video_id', 'Unknown')
                chunk_id = metadata.get('chunk_id', 'Unknown')
                start_time = metadata.get('start_time', 0)
                
                # Truncate document for logging
                doc_preview = document[:100] + "..." if len(document) > 100 else document
                
                logger.info(f"  Result {i+1}: Video {video_id}, Chunk {chunk_id}, Start: {start_time}s")
                logger.info(f"    Preview: {doc_preview}")
                
        except Exception as e:
            logger.warning(f"Collection test failed: {e}")

def main():
    ingest = VideoEmbeddingIngest()
    logger.info("Starting video embedding ingestion process")
    embeddings, metadata, prompts = ingest.load_video_embeddings_and_metadata()
    ingest.save_embeddings_to_chromadb(embeddings, metadata, prompts)
    logger.info("Video embedding ingestion completed successfully!")
    
if __name__ == "__main__":
    main()
