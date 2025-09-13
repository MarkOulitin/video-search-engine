import asyncio
import json
import websockets
import traceback
import os
import time
from logger import logger
from server import Server
from dotenv import load_dotenv
load_dotenv()

server = Server()

async def websocket_handler(websocket):
    try:
        async for message in websocket:
            
            try:
                data = json.loads(message)
                action = data.get("action", "search")
                logger.info(f"Received message with action: {action}")
                
                if action == "search":
                    if "query" in data:
                        await server.handle_video_search(websocket, data)
                    else:
                        error_response = {
                            "event": "error",
                            "data": "Missing 'query' in request"
                        }
                        await websocket.send(json.dumps(error_response))
                        
                elif action == "upload":
                    await server.handle_video_upload(websocket, data)
                        
                elif action == "health":
                    # Health check
                    health_response = {
                        "event": "health_check",
                        "data": {
                            "status": "healthy",
                            "timestamp": time.time()
                        }
                    }
                    await websocket.send(json.dumps(health_response))
                    
                else:
                    error_response = {
                        "event": "error",
                        "data": f"Unknown action: {action}"
                    }
                    await websocket.send(json.dumps(error_response))
                    
            except json.JSONDecodeError:
                error_response = {
                    "event": "error",
                    "data": "Invalid JSON format"
                }
                await websocket.send(json.dumps(error_response))
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                error_response = {
                    "event": "error",
                    "data": f"Server error: {str(e)}"
                }
                await websocket.send(json.dumps(error_response))
                
    except websockets.exceptions.ConnectionClosedOK:
        logger.info("Client disconnected gracefully")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.error(f"Client disconnected with error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket handler: {e}")
    finally:
        logger.info("Client connection closed")

async def main():
    host = "0.0.0.0"
    port = int(os.getenv('SERVER_PORT', 7766))
    max_size = 50 * 1024 * 1024 # 50MB message size limit
    try:
        async with websockets.serve(websocket_handler, host, port, max_size=max_size, ping_timeout=60, ping_interval=30):
            logger.info(f"Server started on ws://{host}:{port}")
            await asyncio.Future()  # Run forever
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nServer stopped by user.")
    except Exception as e:
        logger.error(f"Server error: {e}")
        traceback.print_exc()
