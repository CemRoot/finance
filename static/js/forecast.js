// static/js/forecast.js
console.log("forecast.js loaded.");

// --- Global Variables ---
let forecastChartInstance = null; // Reference to the Plotly chart object
const FORECAST_CHART_DIV_ID = 'forecastChart'; // Target div for Plotly
const FORECAST_STATUS_DIV_ID = 'forecastStatus';
const FORECAST_META_DIV_ID = 'forecast-meta';
const MAX_RETRIES = 2;
const RETRY_DELAY = 2500; // ms
let currentStockSymbol = null; // Store the currently loaded symbol
let isLoadingForecast = false; // Prevent multiple simultaneous loads

// --- Plotly Chart Functions ---

function showForecastLoading(message = "Loading forecast data...") {
    const chartDiv = document.getElementById(FORECAST_CHART_DIV_ID);
    const statusDiv = document.getElementById(FORECAST_STATUS_DIV_ID);
    if (!chartDiv) {
        console.error("showForecastLoading: Chart div not found:", FORECAST_CHART_DIV_ID);
        return;
    }

    // Clear previous chart instance if it exists
    if (forecastChartInstance) {
        try {
            Plotly.purge(chartDiv);
            console.log("Previous forecast chart purged.");
        } catch (e) {
            console.error("Error purging forecast chart:", e);
        }
        forecastChartInstance = null;
    }

    // Display loading message using Plotly layout (or fallback)
     try {
        Plotly.newPlot(chartDiv, [], {
             title: 'Loading...', // Simple title
             xaxis: { visible: false },
             yaxis: { visible: false },
             plot_bgcolor: '#FFFFFF', paper_bgcolor: '#FFFFFF',
             margin: { l: 40, r: 20, t: 60, b: 40 },
             annotations: [{
                 text: message, xref: "paper", yref: "paper",
                 x: 0.5, y: 0.5, showarrow: false,
                 font: { size: 14, color: '#6c757d' }
             }]
        }, { responsive: true });
        forecastChartInstance = chartDiv; // Assign the div element as the instance reference
        console.log("Forecast chart placeholder shown.");
     } catch (e) {
        console.error("Error creating Plotly loading placeholder:", e);
         // Fallback to simple text if Plotly fails
         chartDiv.innerHTML = `<div class="alert alert-light text-center p-5">${message}</div>`;
     }


    // Update status div as well
    if (statusDiv) {
        statusDiv.textContent = message;
        statusDiv.className = 'text-center text-muted small mt-2'; // Reset class
    }
    // Reset metrics/summary placeholders
    updateForecastMetrics(null); // Clears metrics
    updateForecastSummary(null); // Clears summary
}

function showForecastError(errorMessage) {
    const chartDiv = document.getElementById(FORECAST_CHART_DIV_ID);
    const statusDiv = document.getElementById(FORECAST_STATUS_DIV_ID);
    console.error("Forecast Error:", errorMessage);

    if (chartDiv) {
         // Clear previous chart instance
        if (forecastChartInstance) {
            try { Plotly.purge(chartDiv); } catch (e) { console.error("Error purging chart on error:", e); }
            forecastChartInstance = null;
        }
         // Display error message using Plotly layout
         try {
             Plotly.newPlot(chartDiv, [], {
                 title: 'Forecast Error',
                 xaxis: { visible: false }, yaxis: { visible: false },
                 plot_bgcolor: '#FFFFFF', paper_bgcolor: '#FFFFFF',
                 margin: { l: 40, r: 20, t: 60, b: 40 },
                 annotations: [{
                     text: `Error: ${errorMessage}`, xref: "paper", yref: "paper",
                     x: 0.5, y: 0.5, showarrow: false,
                     font: { size: 12, color: '#dc3545' } // Red error color
                 }]
             }, { responsive: true });
             forecastChartInstance = chartDiv;
             console.log("Forecast chart error state shown.");
         } catch(e) {
            console.error("Error displaying Plotly error state:", e);
             chartDiv.innerHTML = `<div class="alert alert-danger text-center p-3">Error loading forecast: ${errorMessage}</div>`;
         }
    }

    if (statusDiv) {
        statusDiv.textContent = `Error: ${errorMessage}`;
        statusDiv.className = 'text-center text-danger small mt-2'; // Error class
    }
    // Clear metrics/summary on error
    updateForecastMetrics(null);
    updateForecastSummary(null);
}


function updateForecastChart(data, stockSymbol) {
    const chartDiv = document.getElementById(FORECAST_CHART_DIV_ID);
    const statusDiv = document.getElementById(FORECAST_STATUS_DIV_ID);

    if (!chartDiv) { console.error('updateForecastChart: Chart div not found.'); return; }

    // Validate data structure
    if (!data || !data.forecast_values || !data.forecast_values.dates || !data.forecast_values.values || !data.forecast_values.lower_bound || !data.forecast_values.upper_bound) {
        showForecastError("Invalid or incomplete forecast data received.");
        return;
    }

    const forecastData = data.forecast_values;
    // Filter out any points where essential values are null
    const validIndices = forecastData.values.reduce((acc, v, i) => {
        if (v !== null && forecastData.dates[i] !== null && forecastData.lower_bound[i] !== null && forecastData.upper_bound[i] !== null) {
            acc.push(i);
        }
        return acc;
    }, []);

    if (validIndices.length === 0) {
        showForecastError("No valid data points found in the forecast.");
        return;
    }

    // Create arrays with only valid data
    const validDates = validIndices.map(i => forecastData.dates[i]);
    const validValues = validIndices.map(i => forecastData.values[i]);
    const validLower = validIndices.map(i => forecastData.lower_bound[i]);
    const validUpper = validIndices.map(i => forecastData.upper_bound[i]);

    // --- Plotly Traces ---
    const traceForecast = {
        x: validDates, y: validValues, name: 'Forecast',
        mode: 'lines', type: 'scatter',
        line: { color: '#3498db', width: 2.5 } // Blue forecast line
    };
    const traceUpperBound = {
        x: validDates, y: validUpper, name: 'Upper Bound',
        mode: 'lines', type: 'scatter',
        line: { color: 'rgba(52, 152, 219, 0.3)', width: 1, dash: 'dot' }, // Light blue dotted line
        fill: 'none', // No fill for upper bound itself
        showlegend: false // Don't show separate legend item for upper bound
    };
    const traceLowerBound = {
        x: validDates, y: validLower, name: 'Confidence Interval', // Legend entry for the shaded area
        mode: 'lines', type: 'scatter',
        line: { color: 'rgba(52, 152, 219, 0.3)', width: 1, dash: 'dot' }, // Light blue dotted line
        fill: 'tonexty', // Fill the area to the *next* trace (traceUpperBound)
        fillcolor: 'rgba(52, 152, 219, 0.1)', // Light blue shaded area
        showlegend: true, // Show legend for the interval
        legendgroup: 'confidence' // Group bounds in legend if needed
    };

    // --- Plotly Layout ---
    const layout = {
        title: {
            text: `${stockSymbol} Price Forecast (${validDates.length} Days)`,
            font: { size: 16 }
        },
        xaxis: {
            title: 'Date', showgrid: true, gridcolor: '#ecf0f1',
            type: 'date', tickformat: '%d %b %Y', // Format dates on axis
             // Set range slightly padded? Or let Plotly auto-range.
             // range: [validDates[0], validDates[validDates.length - 1]]
        },
        yaxis: {
            title: 'Predicted Price', showgrid: true, gridcolor: '#ecf0f1',
            // Add currency formatting if available
            tickformat: (window.stockData?.currency === 'USD' ? '$,.2f' : ',.2f'), // Basic currency format
             autorange: true // Ensure axis adjusts to data range
        },
        plot_bgcolor: '#FFFFFF', paper_bgcolor: '#FFFFFF',
        showlegend: true,
        legend: {
            x: 0.01, y: 0.99, // Position legend top-left
            bgcolor: 'rgba(255,255,255,0.7)', bordercolor: '#CCCCCC', borderwidth: 1,
            orientation: 'h' // Horizontal legend
        },
        margin: { l: 60, r: 30, t: 50, b: 50 } // Adjust margins
    };

    // --- Render Chart ---
    try {
        // Use Plotly.react for efficient updates if chart exists, or Plotly.newPlot if not
        Plotly.react(chartDiv, [traceLowerBound, traceUpperBound, traceForecast], layout, { responsive: true });
        forecastChartInstance = chartDiv; // Update instance reference
        console.log("Forecast chart updated with new data.");
        if (statusDiv) {
            // Show last updated time if available
            const timestamp = data.timestamp ? luxon.DateTime.fromISO(data.timestamp).toFormat('dd MMM yyyy HH:mm:ss ZZZZ') : 'N/A';
            statusDiv.textContent = `Forecast generated. Last updated: ${timestamp}`;
             statusDiv.className = 'text-center text-muted small mt-2'; // Reset class
        }
    } catch (e) {
        console.error("Error updating forecast chart with Plotly:", e);
        showForecastError("Failed to render the forecast chart.");
    }
}

// --- Metrics & Summary Update Functions ---

function updateForecastMetrics(data) {
    const metricsContainer = document.querySelector('.metrics-container'); // Target added in HTML
    if (!metricsContainer) {
        console.warn("updateForecastMetrics: Metrics container not found.");
        return; // Exit if container doesn't exist
    }

    // Find metric elements
    const trendDirEl = metricsContainer.querySelector('#trendDirection');
    const trendStrEl = metricsContainer.querySelector('#trendStrength');
    const volEl = metricsContainer.querySelector('#historicalVolatility');
    const seasonEl = metricsContainer.querySelector('#seasonalityStrength');
    const confIntEl = metricsContainer.querySelector('#confidenceInterval');

    // Helper to update element text or show 'N/A'
    const updateMetric = (el, value, formatFn = null) => {
        if (!el) return; // Skip if element not found
        if (value !== null && value !== undefined && !isNaN(value)) {
            el.textContent = formatFn ? formatFn(value) : value;
            // Optional: Add dynamic styling based on value
             if (el.id === 'historicalVolatility') { // Example for Volatility
                el.className = 'metric-value fw-medium'; // Reset class
                 if (value * 100 < 25) el.classList.add('text-success'); // Low vol = green
                 else if (value * 100 < 50) el.classList.add('text-warning'); // Medium vol = orange
                 else el.classList.add('text-danger'); // High vol = red
             }
        } else {
            el.textContent = 'N/A';
             el.className = 'metric-value fw-medium text-muted'; // Style N/A differently
        }
    };

    // If data or metrics are missing, clear all fields
    if (!data || !data.metrics) {
        console.warn("updateForecastMetrics: Missing data or metrics object.");
        updateMetric(trendDirEl, 'N/A'); // Explicitly set N/A
        updateMetric(trendStrEl, null);
        updateMetric(volEl, null);
        updateMetric(seasonEl, null);
        updateMetric(confIntEl, null);
        return;
    }

    const metrics = data.metrics;
    try {
        updateMetric(trendDirEl, metrics.trend_direction || 'Unknown'); // Handle empty string
        updateMetric(trendStrEl, metrics.trend_strength, v => `${(v * 100).toFixed(1)}%`);
        updateMetric(volEl, metrics.historical_volatility, v => `${(v * 100).toFixed(1)}%`);
        updateMetric(seasonEl, metrics.seasonality_strength, v => `${(v * 100).toFixed(1)}%`);
        // Confidence interval: display relative width (e.g., +/- 5%)
        updateMetric(confIntEl, metrics.confidence_interval, v => `±${(v / 2 * 100).toFixed(1)}%`);

        console.log("Forecast metrics updated.");
    } catch (e) {
        console.error("Error updating forecast metrics:", e);
        // Clear fields on error during update
        updateMetric(trendDirEl, 'Error'); updateMetric(trendStrEl, null); updateMetric(volEl, null); updateMetric(seasonEl, null); updateMetric(confIntEl, null);
    }
}

function updateForecastSummary(data) {
    const summaryElement = document.getElementById('forecastSummary');
    if (!summaryElement) return;

    // Clear summary if data is invalid
    if (!data || !data.forecast_values || !data.forecast_values.values || !data.metrics) {
        summaryElement.textContent = "Summary cannot be generated due to missing forecast data.";
        summaryElement.className = 'text-muted small'; // Reset class
        return;
    }

    try {
        const values = data.forecast_values.values.filter(v => v !== null && !isNaN(v));
        const metrics = data.metrics;
        const lastActualPrice = data.last_actual_price;

        if (values.length === 0) {
            summaryElement.textContent = "No valid forecast values available to generate summary.";
            return;
        }

        const firstForecastValue = values[0];
        const lastForecastValue = values[values.length - 1];

        // Determine base for percentage change calculation
        let changeBase = (lastActualPrice !== null && !isNaN(lastActualPrice)) ? lastActualPrice : firstForecastValue;
        let percentChangeText = "an undetermined change";
        let overallDirection = metrics.trend_direction ? metrics.trend_direction.toLowerCase() : 'uncertain';

        if (changeBase !== null && changeBase !== 0 && !isNaN(changeBase) && lastForecastValue !== null && !isNaN(lastForecastValue)) {
            const pC = ((lastForecastValue - changeBase) / Math.abs(changeBase) * 100);
            const changeDirection = pC >= 0 ? 'increase' : 'decrease';
            percentChangeText = `a ${Math.abs(pC).toFixed(1)}% ${changeDirection}`;
            if (overallDirection === 'uncertain') { // Refine direction based on calculation if trend is weak/uncertain
                 overallDirection = pC > 1 ? 'upward' : (pC < -1 ? 'downward' : 'sideways');
            }
        }

        // Assess confidence based on volatility and interval width
        let confidenceLevel = 'medium';
        const vol = metrics.historical_volatility;
        const interval = metrics.confidence_interval;
        if (vol !== null && interval !== null) {
            if (vol < 0.20 && interval < 0.15) confidenceLevel = 'high'; // Low vol, narrow interval
            else if (vol > 0.50 || interval > 0.30) confidenceLevel = 'low'; // High vol or wide interval
        } else if (vol === null || interval === null) {
             confidenceLevel = 'uncertain'; // Cannot assess if metrics missing
        }

        let summary = `The model predicts a general <strong>${overallDirection}</strong> trend over the forecast period. `;
        summary += `The price is expected to show approximately <strong>${percentChangeText}</strong> compared to the last known price. `;
        summary += `Based on historical volatility and forecast uncertainty, the confidence level for this prediction is assessed as <strong>${confidenceLevel}</strong>.`;
        if (metrics.seasonality_strength !== null && metrics.seasonality_strength > 0.1) {
            summary += ` Note: Seasonality appears to have a noticeable impact on price movements.`;
        }

        summaryElement.innerHTML = summary; // Use innerHTML to render strong tags
        summaryElement.className = 'text-dark small'; // Make text darker

    } catch (e) {
        console.error("Error generating forecast summary:", e);
        summaryElement.textContent = "An error occurred while generating the forecast summary.";
        summaryElement.className = 'text-danger small'; // Show error state
    }
}


// --- Data Loading & Fetching ---

async function loadForecastData(stockSymbol) {
    if (isLoadingForecast) {
        console.warn("Forecast loading already in progress. Skipping.");
        return;
    }
    if (!stockSymbol) {
        console.error('loadForecastData: Stock symbol is required.');
        showForecastError("No stock symbol provided.");
        return;
    }

    isLoadingForecast = true;
    currentStockSymbol = stockSymbol; // Store the symbol being loaded
    console.log(`Requesting forecast data for: ${stockSymbol}`);
    showForecastLoading(`Loading forecast for ${stockSymbol}...`); // Show loading state

    try {
        // Use fetchWithRetry for resilience
        const response = await fetchWithRetry(`/get_forecast_data?symbol=${stockSymbol}`, {}, MAX_RETRIES);

        if (!response.ok) {
            // Try to get error message from response body
            const errorData = await response.json().catch(() => ({ error: `Server error: ${response.status}` }));
            throw new Error(errorData.error || `HTTP error ${response.status}`);
        }

        const data = await response.json();

        // Check for application-level errors in the response
        if (data.error) {
            throw new Error(data.error);
        }

        console.log("Forecast data received:", data);
        // Only update if the received data is still for the current symbol
        if (stockSymbol === currentStockSymbol) {
             updateForecastChart(data, stockSymbol);
             updateForecastMetrics(data);
             updateForecastSummary(data);
        } else {
             console.log(`Forecast data received for ${stockSymbol}, but user navigated to ${currentStockSymbol}. Discarding.`);
        }

    } catch (error) {
        console.error(`Failed to load/process forecast data for ${stockSymbol}:`, error);
         // Only show error if it's for the currently selected symbol
         if (stockSymbol === currentStockSymbol) {
             showForecastError(error.message || "An unknown error occurred.");
         }
    } finally {
        isLoadingForecast = false; // Release lock
        // Ensure refresh button is enabled
        const refreshBtn = document.getElementById('refreshForecastBtn');
        if (refreshBtn) refreshBtn.disabled = false;
    }
}

// Fetch with retry logic (modified slightly)
async function fetchWithRetry(url, options = {}, retries = 0) { // Start retries from 0
     console.log(`Fetching (Attempt ${retries + 1}/${MAX_RETRIES + 1}): ${url}`);
     try {
         const response = await fetch(url, options);
         // Retry on 429 (Rate Limit) or 503/504 (Server Overload/Gateway Timeout)
         if ((response.status === 429 || response.status === 503 || response.status === 504) && retries < MAX_RETRIES) {
             const delay = RETRY_DELAY * (retries + 1); // Incremental backoff
             console.warn(`Received status ${response.status}. Retrying after ${delay}ms...`);
             await new Promise(resolve => setTimeout(resolve, delay));
             return fetchWithRetry(url, options, retries + 1); // Increment retry count
         }
         return response; // Return response if OK or non-retriable error
     } catch (error) {
         // Retry on network errors
         if (retries < MAX_RETRIES) {
              const delay = RETRY_DELAY * (retries + 1);
              console.warn(`Fetch failed (${error.message}). Retrying after ${delay}ms...`);
              await new Promise(resolve => setTimeout(resolve, delay));
              return fetchWithRetry(url, options, retries + 1);
         }
         console.error(`Fetch failed after ${MAX_RETRIES + 1} attempts: ${error.message}`);
         throw error; // Throw error after max retries
     }
 }


// --- Event Listeners ---
document.addEventListener('DOMContentLoaded', function() {
    console.log("forecast.js: DOMContentLoaded event fired.");

    const forecastMetaDiv = document.getElementById(FORECAST_META_DIV_ID);
    const initialStockSymbol = forecastMetaDiv ? forecastMetaDiv.dataset.stockSymbol : null;
    const forecastChartDiv = document.getElementById(FORECAST_CHART_DIV_ID);
    const refreshBtn = document.getElementById('refreshForecastBtn');

    if (!forecastChartDiv) {
        console.warn("forecast.js: Forecast chart div element not found. Cannot initialize.");
        return; // Stop if essential elements are missing
    }

    // Initial load logic
    if (initialStockSymbol) {
        console.log(`forecast.js: Initial stock symbol found ('${initialStockSymbol}'), triggering forecast load.`);
        loadForecastData(initialStockSymbol); // Load data for the initial symbol
    } else {
        console.log("forecast.js: No initial stock symbol found. Showing 'select stock' message.");
        showForecastLoading("Select a stock to view forecast."); // Show initial placeholder
    }

    // Refresh button listener
    if (refreshBtn) {
        refreshBtn.addEventListener('click', function() {
             const currentSymbol = document.getElementById(FORECAST_META_DIV_ID)?.dataset.stockSymbol;
            if (currentSymbol) {
                 console.log(`Forecast refresh triggered for ${currentSymbol}`);
                 this.disabled = true; // Disable button immediately
                 // Clear forecast cache before fetching new data
                 fetch(`/clear_cache?key=forecast_${currentSymbol}`)
                    .then(res => {
                         if(!res.ok) console.warn("Could not clear forecast cache before refresh.");
                         return loadForecastData(currentSymbol); // Load after attempting cache clear
                    })
                    .catch(err => {
                         console.error("Error clearing cache:", err);
                         loadForecastData(currentSymbol); // Still try to load data
                    });

            } else {
                 console.warn("Refresh forecast clicked but no symbol found.");
            }
        });
         console.log("Refresh forecast button listener attached.");
    } else {
         console.warn("Refresh forecast button not found.");
    }


    // Listener for when the forecast tab becomes visible (e.g., using Bootstrap tabs)
    // This ensures Plotly chart resizes correctly if the tab was hidden initially.
    const forecastTabLink = document.getElementById('forecasting-tab'); // ID of the tab *link*
    if (forecastTabLink) {
        forecastTabLink.addEventListener('shown.bs.tab', function () {
            const chartDiv = document.getElementById(FORECAST_CHART_DIV_ID);
            // Check if Plotly is loaded and a chart exists in the div
            if (chartDiv && typeof Plotly !== 'undefined' && chartDiv.data) {
                try {
                    console.log("Forecast tab shown, resizing Plotly chart...");
                    // Delay resize slightly to ensure container dimensions are stable
                    setTimeout(() => Plotly.Plots.resize(chartDiv), 100);
                } catch (e) {
                    console.error("Plotly resize error on tab show:", e);
                }
            } else {
                 console.warn("Forecast tab shown, but Plotly chart or library not ready for resize.");
            }
        });
         console.log("Listener attached for 'shown.bs.tab' on forecast tab.");
    } else {
        console.warn("Forecast tab link ('#forecasting-tab') not found.");
    }

}); // End DOMContentLoaded

// --- Add route to clear cache (needed for refresh button) ---
// Note: This should ideally be in your Flask app (app.py)
// Example function to add to app.py:
/*
@app.route('/clear_cache')
def clear_cache_key():
    key = request.args.get('key')
    if key:
        try:
            # Make cache key prefix aware if needed (depends on Flask-Caching setup)
            # full_key = cache.config.get('CACHE_KEY_PREFIX', '') + key
            # cache.delete(full_key)
            cache.delete(key) # Try deleting directly first
            app.logger.info(f"Cache key '{key}' deleted via API request.")
            return jsonify({"status": "success", "message": f"Cache key '{key}' cleared."}), 200
        except Exception as e:
            app.logger.error(f"Error deleting cache key '{key}': {e}", exc_info=True)
            return jsonify({"status": "error", "message": "Error clearing cache key."}), 500
    else:
        return jsonify({"status": "error", "message": "No cache key provided."}), 400
*/