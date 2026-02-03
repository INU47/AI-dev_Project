import time
import asyncio
import logging

logger = logging.getLogger("Watchdog")

class Watchdog:
    def __init__(self, bridge, threshold_seconds=10):
        self.bridge = bridge
        self.last_heartbeat = time.time()
        self.threshold = threshold_seconds
        self.healthy = True

    def update_heartbeat(self):
        self.last_heartbeat = time.time()
        if not self.healthy:
            logger.info("System recovered! Heartbeat received.")
            self.healthy = True

    async def monitor(self):
        """
        Runs in background to check if data stream is alive.
        """
        logger.info("Watchdog started monitoring...")
        while True:
            elapsed = time.time() - self.last_heartbeat
            if elapsed > self.threshold:
                if self.healthy:
                    logger.warning(f"CRITICAL: No data received for {elapsed:.1f} seconds! functionality degraded.")
                    self.healthy = False
                # Trigger alerts or reconnection attempts here
            
            await asyncio.sleep(1)

