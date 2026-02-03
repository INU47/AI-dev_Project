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

    async def log_candle(self, symbol, timeframe, candle_data):
        if not self.pool: return
        try:
            query = """
                INSERT INTO market_candles (time, symbol, timeframe, open, high, low, close, volume)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (time, symbol, timeframe) DO NOTHING
            """
            # Convert time from ms integer if needed, or assume datetime object if pre-processed
            # In native polling, candle['time'] is int (msc). Convert to datetime.
            if isinstance(candle_data['time'], (int, float)):
                ts = datetime.fromtimestamp(candle_data['time']/1000.0)
            else:
                ts = candle_data['time']

            async with self.pool.acquire() as conn:
                await conn.execute(query, 
                                   ts, 
                                   symbol, 
                                   timeframe,
                                   float(candle_data['open']), 
                                   float(candle_data['high']),
                                   float(candle_data['low']), 
                                   float(candle_data['close']),
                                   float(candle_data.get('volume', candle_data.get('tick_volume', 0)) if isinstance(candle_data, dict) else 0))
        except Exception as e:
            logger.error(f"Failed to log candle: {e}")

    async def log_candles_batch(self, candles_list):
        """
        Efficiently logs a batch of candles.
        candles_list: List of dicts with keys: symbol, timeframe, time, open, high, low, close, tick_volume
        """
        if not self.pool or not candles_list: return
        try:
            query = """
                INSERT INTO market_candles (time, symbol, timeframe, open, high, low, close, volume)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (time, symbol, timeframe) DO NOTHING
            """
            
            data_tuples = []
            for c in candles_list:
                # Convert time
                if isinstance(c['time'], (int, float)):
                    ts = datetime.fromtimestamp(c['time']/1000.0)
                else:
                    ts = c['time']
                
                data_tuples.append((
                    ts,
                    c['symbol'],
                    c['timeframe'],
                    float(c['open']),
                    float(c['high']),
                    float(c['low']),
                    float(c['close']),
                    float(c.get('tick_volume', c.get('volume', 0)))
                ))

            async with self.pool.acquire() as conn:
                await conn.executemany(query, data_tuples)
            logger.info(f"Batch logged {len(data_tuples)} candles.")
        except Exception as e:
            logger.error(f"Failed to log candle batch: {e}")

    async def log_trade_entry(self, symbol, action, lot_size, price, signal_data):
        """
        Logs the OPENING of a trade. 
        Returns the database ID of the log entry to be used for closing later.
        """
        if not self.pool: return None
        try:
            query = """
                INSERT INTO trade_logs (
                    open_time, symbol, action, open_price, lot_size,
                    ai_mode, pattern_type, cnn_confidence, lstm_trend_pred, lstm_confidence, status
                ) VALUES (NOW(), $1, $2, $3, $4, $5, $6, $7, $8, $9, 'OPEN')
                RETURNING id
            """
            
            # Extract AI State
            raw_pattern = signal_data.get('raw_cnn_class', 0)
            raw_trend = signal_data.get('raw_lstm_trend', 0.0)
            raw_conf = signal_data.get('raw_lstm_conf', 0.0)
            
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(query, 
                                          symbol, 
                                          action, 
                                          float(price), 
                                          float(lot_size),
                                          "CONSERVATIVE", # Or pass from signal_data
                                          str(raw_pattern),
                                          float(signal_data.get('confidence', 0.0)),
                                          float(raw_trend),
                                          float(raw_conf))
                return row['id'] if row else None
        except Exception as e:
            logger.error(f"Failed to log trade entry: {e}")
            return None

    async def log_trade_exit(self, db_id, close_price, profit, net_profit):
        """Updates the trade log with CLOSE details"""
        if not self.pool or not db_id: return
        try:
            query = """
                UPDATE trade_logs 
                SET close_time = NOW(), 
                    close_price = $1, 
                    gross_profit = $2, 
                    net_profit = $3, 
                    status = 'CLOSED'
                WHERE id = $4
            """
            async with self.pool.acquire() as conn:
                await conn.execute(query, float(close_price), float(profit), float(net_profit), db_id)
        except Exception as e:
            logger.error(f"Failed to log trade exit: {e}")

    async def close_latest_trade(self, symbol, close_price, profit, net_profit):
        """Closes the latest OPEN trade for a symbol (FIFO logic for RL logging)"""
        if not self.pool: return
        try:
            # Find the latest OPEN trade for this symbol
            find_query = """
                SELECT id FROM trade_logs 
                WHERE symbol = $1 AND status = 'OPEN' 
                ORDER BY open_time DESC LIMIT 1
            """
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(find_query, symbol)
                if row:
                    await self.log_trade_exit(row['id'], close_price, profit, net_profit)
                else:
                    logger.warning(f"No OPEN trade found to close for {symbol}")
        except Exception as e:
            logger.error(f"Failed to close latest trade: {e}")

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
