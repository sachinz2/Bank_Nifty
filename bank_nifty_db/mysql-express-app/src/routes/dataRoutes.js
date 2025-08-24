const express = require('express');
const DataController = require('../controllers/dataController');
const dataModel = require('../models/dataModel');

const router = express.Router();
// Instantiate controller with the model so it can access the database.
const dataController = new DataController(dataModel);

// Bind the controller methods to ensure `this` context is correct when they are called by Express.
router.post('/data', dataController.collectData.bind(dataController));
router.get('/data', dataController.getData.bind(dataController));

module.exports = router;