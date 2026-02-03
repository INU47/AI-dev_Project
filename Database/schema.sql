-- Database Schema for Quant Trading System

-- 1. Create Tables
CREATE TABLE IF NOT EXISTS market_data (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    bid DOUBLE PRECISION NOT NULL,
    ask DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS ai_signals (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    pattern_type TEXT,
    cnn_confidence DOUBLE PRECISION,
    lstm_trend_pred DOUBLE PRECISION,
    lstm_confidence DOUBLE PRECISION,
    final_signal TEXT
);

-- 2. Optional: Create Indexes for performance
CREATE INDEX IF NOT EXISTS idx_market_data_time ON market_data (time DESC);
CREATE INDEX IF NOT EXISTS idx_ai_signals_time ON ai_signals (time DESC);

-- 3. Note for TimescaleDB users:
-- If you have TimescaleDB extension installed, you can turn these into hypertables:
-- SELECT create_hypertable('market_data', 'time');
-- SELECT create_hypertable('ai_signals', 'time');
