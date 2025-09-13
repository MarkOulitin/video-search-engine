#!/usr/bin/env python3
"""
Test script for Video RAG server
Tests connection and search functionality with a sample query
"""

import asyncio
import json
import websockets
import time
import logging
import os
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_video_search():
    """Test video search with CSGO championship query"""
    
    # Configuration
    server_host = os.getenv('VIDEO_RAG_SERVER_HOST', '65.108.32.144')
    server_port = int(os.getenv('VIDEO_RAG_SERVER_PORT', 7766))
    test_query = "CSGO championship"
    
    uri = f"ws://{server_host}:{server_port}/"
    
    print("=" * 60)
    print("🧪 Video RAG Server Test Script")
    print("=" * 60)
    print(f"Server: {server_host}:{server_port}")
    print(f"Query: '{test_query}'")
    print("=" * 60)
    
    try:
        print(f"🔗 Connecting to {uri}...")
        
        async with websockets.connect(
            uri, 
            ping_timeout=30, 
            ping_interval=30,
            max_size=50 * 1024 * 1024  # 50MB message size limit
        ) as websocket:
            print("✅ Connected successfully!")
            
            # Send search request
            message = {
                "action": "search",
                "query": test_query
            }
            
            print(f"📤 Sending search request...")
            await websocket.send(json.dumps(message))
            print(f"   Query: '{test_query}'")
            
            # Track timing
            start_time = time.time()
            results_received = False
            
            # Listen for responses
            print("\n📨 Listening for responses...")
            while True:
                try:
                    response_json = await asyncio.wait_for(websocket.recv(), timeout=60.0)
                    response_data = json.loads(response_json)
                    event_type = response_data.get('event')
                    data = response_data.get('data')
                    
                    elapsed = time.time() - start_time
                    
                    if event_type == 'encoding_query':
                        print(f"   [{elapsed:.1f}s] 🔍 {data}")
                        
                    elif event_type == 'searching_chunks':
                        print(f"   [{elapsed:.1f}s] 📹 {data}")
                        
                    elif event_type == 'extracting_videos':
                        print(f"   [{elapsed:.1f}s] ✂️ {data}")
                        
                    elif event_type == 'search_completed':
                        results_received = True
                        total_time = time.time() - start_time
                        chunks = data.get('chunks', [])
                        
                        print(f"\n✅ Search completed in {total_time:.2f}s")
                        print(f"📊 Results: {len(chunks)} video chunks found")
                        
                        if chunks:
                            print("\n🎬 Top Results:")
                            for i, chunk in enumerate(chunks[:3], 1):  # Show top 3
                                video_id = chunk.get('video_id', 'Unknown')
                                chunk_id = chunk.get('chunk_id', 'Unknown')
                                start_time = chunk.get('start_time', 0)
                                end_time = chunk.get('end_time', 0)
                                similarity = chunk.get('similarity_score', 0)
                                extraction_success = chunk.get('extraction_success', False)
                                
                                print(f"   {i}. Video {video_id} - Chunk {chunk_id}")
                                print(f"      Time: {start_time:.1f}s - {end_time:.1f}s")
                                print(f"      Similarity: {similarity:.3f}")
                                print(f"      Extracted: {'✅' if extraction_success else '❌'}")
                                
                                if chunk.get('document'):
                                    doc_preview = chunk['document'][:100] + "..." if len(chunk['document']) > 100 else chunk['document']
                                    print(f"      Content: {doc_preview}")
                                print()
                        else:
                            print("   No results found for this query.")
                        break
                        
                    elif event_type == 'error':
                        print(f"❌ Server Error: {data}")
                        break
                        
                    else:
                        print(f"   [{elapsed:.1f}s] Unknown event: {event_type}")
                        
                except asyncio.TimeoutError:
                    print("⏰ Timeout waiting for response")
                    break
                except Exception as e:
                    print(f"❌ Error receiving response: {e}")
                    break
            
            if not results_received:
                print("⚠️ No search results received")
                
    except websockets.exceptions.ConnectionClosed as e:
        print(f"❌ Connection closed: {e}")
    except websockets.exceptions.InvalidURI as e:
        print(f"❌ Invalid URI: {e}")
    except ConnectionRefusedError as e:
        print(f"❌ Connection refused: {e}")
        print("   Make sure the server is running and port 7766 is exposed")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

async def test_health_check():
    """Test server health check"""
    server_host = os.getenv('VIDEO_RAG_SERVER_HOST', '65.108.32.144')
    server_port = int(os.getenv('VIDEO_RAG_SERVER_PORT', 7766))
    uri = f"ws://{server_host}:{server_port}/"
    
    try:
        print("\n🏥 Testing health check...")
        async with websockets.connect(
            uri, 
            ping_timeout=10, 
            ping_interval=10,
            max_size=50 * 1024 * 1024  # 50MB message size limit
        ) as websocket:
            
            # Send health check
            health_message = {"action": "health"}
            await websocket.send(json.dumps(health_message))
            
            # Wait for response
            response_json = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            response_data = json.loads(response_json)
            
            if response_data.get('event') == 'health_check':
                print("✅ Health check passed!")
                print(f"   Status: {response_data['data']['status']}")
            else:
                print(f"⚠️ Unexpected health check response: {response_data}")
                
    except Exception as e:
        print(f"❌ Health check failed: {e}")

async def main():
    """Main test function"""
    print("Starting Video RAG Server Tests...\n")
    
    # Test 1: Health check
    # await test_health_check()
    
    # Test 2: Video search
    await test_video_search()
    
    print("\n" + "=" * 60)
    print("🏁 Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()