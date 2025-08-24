require('dotenv').config({ path: '../../.env' }); // Make sure env variables are loaded

const dataModel = require('../models/dataModel');
const pool = require('../config/db');

/**
 * Simulates fetching the latest OHLC data from an external API like Kite Connect.
 * In a real application, this would involve making an HTTP request.
 * @returns {Promise<Array<object>>} A promise that resolves to an array of OHLC data.
 */
const fetchBankNiftyDataFromKite = async () => {
    console.log('Simulating fetch from Kite API...');
    // This is mock data. Replace with actual API call.
    // Generating 5 minutes of data for the last 25 minutes.
    const mockData = [];
    const now = new Date();

    for (let i = 5; i > 0; i--) {
        const timestamp = new Date(now.getTime() - i * 5 * 60 * 1000);
        const open = 45000 + (Math.random() * 100);
        const close = open + (Math.random() * 50) - 25;
        const high = Math.max(open, close) + (Math.random() * 20);
        const low = Math.min(open, close) - (Math.random() * 20);

        mockData.push({
            instrument: 'NIFTY BANK',
            timestamp: timestamp.toISOString().slice(0, 19).replace('T', ' '), // Format for MySQL DATETIME
            open: open.toFixed(2),
            high: high.toFixed(2),
            low: low.toFixed(2),
            close: close.toFixed(2),
            volume: Math.floor(100000 + Math.random() * 50000)
        });
    }
    return mockData;
};

const collectData = async () => {
    try {
        const ohlcData = await fetchBankNiftyDataFromKite();
        console.log(`Fetched ${ohlcData.length} data points. Storing in database...`);
        const result = await dataModel.storeOhlcData(ohlcData);
        console.log('Data stored successfully.', { affectedRows: result.affectedRows });
    } catch (error) {
        console.error('Error storing data:', error);
    } finally {
        await pool.end(); // Close the pool as this is a standalone script
    }
};

collectData();