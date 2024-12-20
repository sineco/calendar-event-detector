import argparse
import asyncio
import websockets
import json
from event_detector import EventDetector
#from fastapi import FastAPI

async def listen(url, detector):
    async with websockets.connect(url) as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            print(data) # remove
            detector.process_message(data) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A WebSocket client to process data stream for calendar events.")
    parser.add_argument("url", type=str, help="A URL corresponding to the WebSocket server (e.g., ws://143.110.238.245:8000/stream)")
    args = parser.parse_args()

    detector = EventDetector(results_dir="results")
    asyncio.run(listen(args.url, detector)) 

    

