const express = require('express');
const dataRoutes = require('./routes/dataRoutes');
const pool = require('./config/db');

const app = express();
const PORT = process.env.PORT || 3000;

// Middlewares
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Test the database connection
pool.getConnection()
    .then(connection => {
        console.log('Connected to the MySQL database.');
        connection.release();
    })
    .catch(err => console.error('Database connection failed:', err));

// Routes
app.use('/api', dataRoutes);

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});