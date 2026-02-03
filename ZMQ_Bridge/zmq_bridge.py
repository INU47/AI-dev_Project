import zmq
import zmq.asyncio
import asyncio
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ZMQ_Bridge")

class ZMQBridge:
    def __init__(self, pub_port=5556, pull_port=5557, push_port=5558):
        """
        Initialize ZeroMQ Context and Sockets.
        
        Architecture:
        - SUB_SOCKET (Connect to MT5 PUB): Receive real-time Tick Data.
        - PUSH_SOCKET (Connect to MT5 PULL): Send Trade Commands/Orders.
        - PULL_SOCKET (Connect to MT5 PUSH): Receive Trade Responses/Account Info.
        
        Note: The ports must match the MQL5 EA settings.
        """
        self.context = zmq.asyncio.Context()
        
        # 1. DATA SOCKET (SUB) - Receives Market Data from MT5
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://127.0.0.1:{pub_port}")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all topics
        
        # 2. COMMAND SOCKET (PUSH) - Sends Orders to MT5
        self.push_socket = self.context.socket(zmq.PUSH)
        self.push_socket.bind(f"tcp://127.0.0.1:{push_port}")
        
        # 3. RESPONSE SOCKET (PULL) - Receives execution results from MT5
        self.pull_socket = self.context.socket(zmq.PULL)
        self.pull_socket.bind(f"tcp://127.0.0.1:{pull_port}")
        
        self.running = False
        logger.info(f"ZMQ Bridge initialized. Listening on {pub_port} (SUB), {pull_port} (PULL). Sending on {push_port} (PUSH).")

    async def start(self):
        self.running = True
        logger.info("ZMQ Bridge Started.")

    async def stop(self):
        self.running = False
        self.sub_socket.close()
        self.push_socket.close()
        self.pull_socket.close()
        self.context.term()
        logger.info("ZMQ Bridge Stopped.")

    async def receive_data(self):
        """
        generator to yield market data asynchronously.
        """
        while self.running:
            try:
                # Expecting multipart: [topic, message]
                # But for simplicity, we might just get a json string if MT5 sends just string
                # Let's assume MT5 sends: "SYMBOL | BID | ASK" or JSON
                msg = await self.sub_socket.recv_string()
                # logger.debug(f"Received Data: {msg}")
                yield msg
            except zmq.ZMQError as e:
                logger.error(f"ZMQ Receive Error: {e}")
                await asyncio.sleep(1)

    async def send_order(self, order_dict):
        """
        Send a validated order to MT5.
        """
        try:
            msg = json.dumps(order_dict)
            await self.push_socket.send_string(msg)
            logger.info(f"Sent Order: {msg}")
        except Exception as e:
            logger.error(f"Failed to send order: {e}")

    async def check_command_response(self):
        """
        Listen for responses from MT5 (e.g., Order Filled, Error).
        """
        while self.running:
            try:
                msg = await self.pull_socket.recv_string()
                logger.info(f"MT5 Response: {msg}")
                # Process response (update DB, etc.) - To be implemented
            except zmq.ZMQError:
                break

if __name__ == "__main__":
    # Test Run
    async def main():
        bridge = ZMQBridge()
        await bridge.start()
        
        # Simulate listening
        async for data in bridge.receive_data():
            print(f"Main Loop Recv: {data}")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
