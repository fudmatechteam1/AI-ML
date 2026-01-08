const express = require("express")
const mongoose = require("mongoose")
const cors = require("cors")
const cookieParser = require("cookie-parser")
const dbConnection = require("./Dbconnection/dbConfig.js")
const router = require("./Routes/authRoute.js")
require("dotenv").config()
const app = express()

dbConnection()

// FIXED: Removed the leading space in the URL string
const URLs = [
    'http://localhost:5173',
    'https://fudmatechteam1.github.io' // Removed subpath for broader matching
]

app.use(cookieParser()) 
app.use(express.json())

// FIXED: Standardized CORS configuration
app.use(cors({
    origin: function (origin, callback) {
        // Allow requests with no origin (like mobile apps or curl) 
        // or check if origin is in our whitelist
        if (!origin || URLs.some(url => origin.startsWith(url))) {
            callback(null, true);
        } else {
            callback(new Error('Not allowed by CORS'));
        }
    },
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization']
}))

const port = process.env.PORT || 4000

app.use("/api/auth", router)

app.listen(port, () => {
    console.log(`server is running on http://localhost:${port}`)
})