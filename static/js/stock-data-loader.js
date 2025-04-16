/**
 * Stock data loader
 * This file handles loading the stock data from the server
 */

function initializeStockData(rawData) {
    if (!rawData || typeof rawData !== 'object' || !rawData.labels) {
        console.error("Stock data invalid or missing required properties");
        window.stockData = null;
        return false;
    }
    
    window.stockData = rawData;
    console.log("Stock data loaded successfully");
    return true;
}

// For cases when no data is available
function setEmptyStockData() {
    window.stockData = null;
    console.log("No stock data available");
}
