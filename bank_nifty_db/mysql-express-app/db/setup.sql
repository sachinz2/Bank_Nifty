CREATE DATABASE IF NOT EXISTS bank_nifty;

USE bank_nifty;

-- Table for storing OHLC (Open, High, Low, Close) data for instruments
CREATE TABLE IF NOT EXISTS ohlc_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    instrument VARCHAR(50) NOT NULL,
    timestamp DATETIME NOT NULL,
    open DECIMAL(10, 2) NOT NULL,
    high DECIMAL(10, 2) NOT NULL,
    low DECIMAL(10, 2) NOT NULL,
    close DECIMAL(10, 2) NOT NULL,
    volume BIGINT UNSIGNED,
    -- Ensure that we don't store duplicate candles for the same instrument and time
    UNIQUE KEY idx_instrument_timestamp (instrument, timestamp)
);