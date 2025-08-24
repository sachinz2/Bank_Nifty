class DataController {
    constructor(dataModel) {
        if (!dataModel) {
            throw new Error('DataModel is required for DataController.');
        }
        this.dataModel = dataModel;
    }

    async collectData(req, res) {
        try {
            // Expecting an array of OHLC data objects in the body
            const data = req.body;
            if (!Array.isArray(data) || data.length === 0) {
                return res.status(400).json({ message: 'Invalid data format. Expected an array of OHLC records.' });
            }
            const result = await this.dataModel.storeOhlcData(data);
            res.status(201).json({ message: 'Data stored successfully', data: result });
        } catch (error) {
            console.error('Error in collectData controller:', error);
            // Check for duplicate entry error (code from mysql2)
            if (error.code === 'ER_DUP_ENTRY') {
                return res.status(409).json({ message: 'Duplicate data entry detected.', error: error.message });
            }
            res.status(500).json({ message: 'Error storing data', error: error.message });
        }
    }

    async getData(req, res) {
        try {
            const data = await this.dataModel.fetchOhlcData();
            res.status(200).json(data);
        } catch (error) {
            console.error('Error in getData controller:', error);
            res.status(500).json({ message: 'Error fetching data', error: error.message });
        }
    }
}

module.exports = DataController;