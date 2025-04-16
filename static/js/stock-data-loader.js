/**
 * stock-data-loader.js
 *
 * Provides simple functions to initialize and manage the global
 * window.stockData object, intended to hold data passed from the Flask backend.
 * This script should be loaded *before* any inline script in index.html
 * that populates the data and *before* script.js which consumes the data.
 */

console.log("stock-data-loader.js loaded.");

/**
 * Initializes the global window.stockData object.
 * Performs basic validation on the input data.
 *
 * @param {object | null} rawData The raw data object passed from Flask (or null).
 * @returns {boolean} True if data was considered valid and assigned, false otherwise.
 */
function initializeStockData(rawData) {
    // Check if the provided data is a non-null object
    if (rawData && typeof rawData === 'object' && !Array.isArray(rawData)) {
        // Basic check for essential properties expected by charting scripts
        if (rawData.labels && Array.isArray(rawData.labels) &&
            rawData.values && Array.isArray(rawData.values)) {

            window.stockData = rawData; // Assign to global variable
            console.log("stock-data-loader: window.stockData initialized successfully.", window.stockData);
            return true; // Indicate success
        } else {
            console.warn("stock-data-loader: Input data object is missing essential properties ('labels' or 'values' arrays). Setting window.stockData to null.");
            setEmptyStockData(); // Set to null if essential parts are missing
            return false; // Indicate failure
        }
    } else {
        // Handle cases where rawData is null, not an object, or an array
        console.warn(`stock-data-loader: Invalid input data provided (type: ${typeof rawData}). Setting window.stockData to null.`);
        setEmptyStockData(); // Set to null for invalid input
        return false; // Indicate failure
    }
}

/**
 * Sets the global window.stockData object to null.
 * Used when no data is available or an error occurs during initialization.
 */
function setEmptyStockData() {
    window.stockData = null;
    console.log("stock-data-loader: window.stockData set to null.");
}

// Optionally, initialize window.stockData to null immediately when the script loads,
// just to ensure the variable exists globally even before DOMContentLoaded runs in index.html.
// This can prevent "variable not defined" errors in rare edge cases.
if (typeof window.stockData === 'undefined') {
    window.stockData = null;
    console.log("stock-data-loader: Initialized window.stockData to null globally.");
}