import asyncpg
import logging
import os
import json
from datetime import datetime

logger = logging.getLogger("DBHandler")

class DBHandler:
    def __init__(self, config_path="Config/server_config.json"):
        self.config_path = config_path
        self.pool = None

    async def connect(self):
        try:
            # Load from server_config
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                db_cfg = config.get("db_config", {})
                host = db_cfg.get("host", "127.0.0.1")
                user = db_cfg.get("user", "postgres")
                db_name = db_cfg.get("database", "quant_db")
                port = db_cfg.get("port", 5432)
                
                logger.info(f"Connecting to DB: {host}:{port} | User: {user} | DB: {db_name}")

                self.pool = await asyncpg.create_pool(
                    user=user,
                    password=db_cfg.get("password", "password"),
                    database=db_name,
                    host=host,
                    port=port
                )
                logger.info("Connected to PostgreSQL successfully.")
                await self.initialize_schema()
            else:
                logger.error(f"Config file not found: {self.config_path}")
        except Exception as e:
            logger.error(f"DB Connection Failed: {e}")
            self.pool = None

    async def is_healthy(self):
        """Checks if DB connection is alive and reachable"""
        if not self.pool: return False
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False

    async def initialize_schema(self):
        """Automatically creates tables if they don't exist"""
        if not self.pool: return
        try:
            schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
            if os.path.exists(schema_path):
                with open(schema_path, 'r') as f:
                    schema_sql = f.read()
                
                async with self.pool.acquire() as conn:
                    await conn.execute(schema_sql)
                logger.info("Database schema verified/initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")

    async def log_market_data(self, symbol, bid, ask, volume):
        if not self.pool: return
        try:
            query = """
                INSERT INTO market_data (time, symbol, bid, ask, volume)
                VALUES (NOW(), $1, $2, $3, $4)
            """
            async with self.pool.acquire() as conn:
                await conn.execute(query, symbol, bid, ask, volume)
        except Exception as e:
            logger.error(f"Failed to log market data: {e}")

    async def log_signal(self, symbol, signal_data):
        if not self.pool: return
        try:
            query = """
                INSERT INTO ai_signals (time, symbol, pattern_type, cnn_confidence, lstm_trend_pred, lstm_confidence, final_signal)
                VALUES (NOW(), $1, $2, $3, $4, $5, $6)
            """
            
            # Extract raw model data (available even in Exploration Mode)
            # Priorities: raw_cnn_class > pattern (text)
            # Priorities: raw_lstm_trend > trend
            
            raw_pattern = signal_data.get('raw_cnn_class')
            if raw_pattern is None: 
                # Fallback mapping if only text is available (less accurate)
                raw_pattern = 0 # Default/Unknown
            
            raw_trend = signal_data.get('raw_lstm_trend')
            if raw_trend is None:
                raw_trend = signal_data.get('trend', 0.0) # Fallback
                
            raw_conf = signal_data.get('raw_lstm_conf')
            if raw_conf is None:
                raw_conf = 0.0

            async with self.pool.acquire() as conn:
                await conn.execute(query, 
                                   symbol, 
                                   int(raw_pattern) if raw_pattern is not None else 0,
                                   float(signal_data.get('confidence', 0.0)),
                                   float(raw_trend),
                                   float(raw_conf),
                                   signal_data.get('action'))
        except Exception as e:
            logger.error(f"Failed to log signal: {e}")

    async def get_recent_signals(self, limit=5):
        if not self.pool: return []
        try:
            query = "SELECT time, symbol, final_signal, cnn_confidence FROM ai_signals ORDER BY time DESC LIMIT $1"
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, limit)
                return rows
        except Exception as e:
            logger.error(f"Failed to fetch recent signals: {e}")
            return []

    async def close(self):
        if self.pool:
            await self.pool.close()
            logger.info("DB Connection closed.")
