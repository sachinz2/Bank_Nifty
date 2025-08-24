const pool = require('../config/db');

class DataModel {
    /**
     * Stores new OHLC data in the database.
     * @param {Array<object>} ohlcDataArray - An array of OHLC objects to store.
     * @returns {Promise<object>} The result from the database query.
     */
    async storeOhlcData(ohlcDataArray) {
        // Map the array of objects to an array of arrays for bulk insertion
        const values = ohlcDataArray.map(d => [
            d.instrument,
            d.timestamp,
            d.open,
            d.high,
            d.low,
            d.close,
            d.volume
        ]);

        const sql = `
            INSERT INTO ohlc_data (instrument, timestamp, open, high, low, close, volume) 
            VALUES ?
        `;

        const [result] = await pool.query(sql, [values]);
        return result;
    }

    /**
     * Fetches the latest 100 OHLC records from the database.
     * @returns {Promise<Array<object>>} An array of OHLC records.
     */
    async fetchOhlcData() {
        const [rows] = await pool.query('SELECT * FROM ohlc_data ORDER BY timestamp DESC LIMIT 100');
        return rows;
    }
}

module.exports = new DataModel();