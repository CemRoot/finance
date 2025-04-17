// static/js/forecast.js
console.log("forecast.js loaded.");

// --- Global Variables ---
let forecastChartInstance = null;
const FORECAST_CHART_DIV_ID = 'forecastChart';
const FORECAST_STATUS_DIV_ID = 'forecastStatus';
const FORECAST_META_DIV_ID = 'forecast-meta';
const MAX_RETRIES = 2;
const RETRY_DELAY = 2500;
let currentStockSymbol = null;
let isLoadingForecast = false;

// --- Plotly Chart Functions ---
function showForecastLoading(message = "Loading forecast data...") {
    const chartDiv = document.getElementById(FORECAST_CHART_DIV_ID);
    const statusDiv = document.getElementById(FORECAST_STATUS_DIV_ID);
    if (!chartDiv) { console.error("showForecastLoading: Chart div not found:", FORECAST_CHART_DIV_ID); return; }
    if (forecastChartInstance) { try { Plotly.purge(chartDiv); } catch (e) { console.error("Error purging forecast chart:", e); } forecastChartInstance = null; }
    try {
        Plotly.newPlot(chartDiv, [], { title: 'Loading...', xaxis: { visible: false }, yaxis: { visible: false }, plot_bgcolor: '#FFFFFF', paper_bgcolor: '#FFFFFF', margin: { l: 40, r: 20, t: 60, b: 40 }, annotations: [{ text: message, xref: "paper", yref: "paper", x: 0.5, y: 0.5, showarrow: false, font: { size: 14, color: '#6c757d' } }] }, { responsive: true });
        forecastChartInstance = chartDiv; console.log("Forecast chart placeholder shown.");
    } catch (e) { console.error("Error creating Plotly loading placeholder:", e); chartDiv.innerHTML = `<div class="alert alert-light text-center p-5">${message}</div>`; }
    if (statusDiv) { statusDiv.textContent = message; statusDiv.className = 'text-center text-muted small mt-2'; }
    updateForecastMetrics(null); updateForecastSummary(null);
}

function showForecastError(errorMessage) {
    const chartDiv = document.getElementById(FORECAST_CHART_DIV_ID);
    const statusDiv = document.getElementById(FORECAST_STATUS_DIV_ID);
    console.error("Forecast Error:", errorMessage);
    if (chartDiv) {
        if (forecastChartInstance) { try { Plotly.purge(chartDiv); } catch (e) { console.error("Error purging chart on error:", e); } forecastChartInstance = null; }
        try {
            Plotly.newPlot(chartDiv, [], { title: 'Forecast Error', xaxis: { visible: false }, yaxis: { visible: false }, plot_bgcolor: '#FFFFFF', paper_bgcolor: '#FFFFFF', margin: { l: 40, r: 20, t: 60, b: 40 }, annotations: [{ text: `Error: ${errorMessage}`, xref: "paper", yref: "paper", x: 0.5, y: 0.5, showarrow: false, font: { size: 12, color: '#dc3545' } }] }, { responsive: true });
            forecastChartInstance = chartDiv; console.log("Forecast chart error state shown.");
        } catch (e) { console.error("Error displaying Plotly error state:", e); chartDiv.innerHTML = `<div class="alert alert-danger text-center p-3">Error loading forecast: ${errorMessage}</div>`; }
    }
    if (statusDiv) { statusDiv.textContent = `Error: ${errorMessage}`; statusDiv.className = 'text-center text-danger small mt-2'; }
    updateForecastMetrics(null); updateForecastSummary(null);
}

function updateForecastChart(data, stockSymbol) {
    const chartDiv = document.getElementById(FORECAST_CHART_DIV_ID);
    const statusDiv = document.getElementById(FORECAST_STATUS_DIV_ID);
    if (!chartDiv) { console.error('updateForecastChart: Chart div not found.'); return; }
    if (!data || !data.forecast_values || !data.forecast_values.dates || !data.forecast_values.values || !data.forecast_values.lower_bound || !data.forecast_values.upper_bound) { showForecastError("Invalid or incomplete forecast data received."); return; }
    const forecastData = data.forecast_values;
    const validIndices = forecastData.values.reduce((acc, v, i) => { if (v !== null && forecastData.dates[i] !== null && forecastData.lower_bound[i] !== null && forecastData.upper_bound[i] !== null) { acc.push(i); } return acc; }, []);
    if (validIndices.length === 0) { showForecastError("No valid data points found in the forecast."); return; }
    const validDates = validIndices.map(i => forecastData.dates[i]); const validValues = validIndices.map(i => forecastData.values[i]); const validLower = validIndices.map(i => forecastData.lower_bound[i]); const validUpper = validIndices.map(i => forecastData.upper_bound[i]);
    const traceForecast = { x: validDates, y: validValues, name: 'Forecast', mode: 'lines', type: 'scatter', line: { color: '#3498db', width: 2.5 } };
    const traceUpperBound = { x: validDates, y: validUpper, name: 'Upper Bound', mode: 'lines', type: 'scatter', line: { color: 'rgba(52, 152, 219, 0.3)', width: 1, dash: 'dot' }, fill: 'none', showlegend: false };
    const traceLowerBound = { x: validDates, y: validLower, name: 'Confidence Interval', mode: 'lines', type: 'scatter', line: { color: 'rgba(52, 152, 219, 0.3)', width: 1, dash: 'dot' }, fill: 'tonexty', fillcolor: 'rgba(52, 152, 219, 0.1)', showlegend: true, legendgroup: 'confidence' };
    const layout = { title: { text: `${stockSymbol} Price Forecast (${validDates.length} Days)`, font: { size: 16 } }, xaxis: { title: 'Date', showgrid: true, gridcolor: '#ecf0f1', type: 'date', tickformat: '%d %b %Y' }, yaxis: { title: 'Predicted Price', showgrid: true, gridcolor: '#ecf0f1', tickformat: (window.stockData?.currency === 'USD' ? '$,.2f' : ',.2f'), autorange: true }, plot_bgcolor: '#FFFFFF', paper_bgcolor: '#FFFFFF', showlegend: true, legend: { x: 0.01, y: 0.99, bgcolor: 'rgba(255,255,255,0.7)', bordercolor: '#CCCCCC', borderwidth: 1, orientation: 'h' }, margin: { l: 60, r: 30, t: 50, b: 50 } };
    try {
        Plotly.react(chartDiv, [traceLowerBound, traceUpperBound, traceForecast], layout, { responsive: true });
        forecastChartInstance = chartDiv; console.log("Forecast chart updated with new data.");
        if (statusDiv) {
            // Luxon is loaded in the head, should be available here unless blocked
            const timestamp = (data.timestamp && typeof luxon !== 'undefined')
                ? luxon.DateTime.fromISO(data.timestamp).toFormat('dd MMM yyyy HH:mm:ss ZZZZ')
                : (data.timestamp || 'N/A'); // Fallback if Luxon is missing or timestamp missing
            statusDiv.textContent = `Forecast generated. Last updated: ${timestamp}`;
            statusDiv.className = 'text-center text-muted small mt-2';
        }
    } catch (e) { console.error("Error updating forecast chart with Plotly:", e); showForecastError("Failed to render the forecast chart."); }
}

// --- Metrics & Summary Update Functions ---
function updateForecastMetrics(data) {
    const metricsContainer = document.querySelector('.metrics-container');
    if (!metricsContainer) { console.warn("updateForecastMetrics: Metrics container not found."); return; }
    const trendDirEl = metricsContainer.querySelector('#trendDirection'); const trendStrEl = metricsContainer.querySelector('#trendStrength'); const volEl = metricsContainer.querySelector('#historicalVolatility'); const seasonEl = metricsContainer.querySelector('#seasonalityStrength'); const confIntEl = metricsContainer.querySelector('#confidenceInterval');
    const updateMetric = (el, value, formatFn = null) => {
        if (!el) return;
        if (value !== null && value !== undefined && value !== '' && !isNaN(value)) { // Added isNaN check and empty string check
            el.textContent = formatFn ? formatFn(value) : value;
            el.className = 'metric-value fw-medium'; // Reset class
            if (el.id === 'historicalVolatility') { if (value * 100 < 25) el.classList.add('text-success'); else if (value * 100 < 50) el.classList.add('text-warning'); else el.classList.add('text-danger'); }
        } else if (value === 'Upward' || value === 'Downward' || value === 'Sideways' || value === 'Uncertain' || value === 'Unknown') { // Handle direction strings explicitly
            el.textContent = value;
            el.className = 'metric-value fw-medium';
            if (value === 'Upward') el.classList.add('text-success');
            else if (value === 'Downward') el.classList.add('text-danger');
            else el.classList.add('text-secondary');
        } else {
            el.textContent = 'N/A';
            el.className = 'metric-value fw-medium text-muted';
        }
    };
    // *** DEBUG: Log the data object passed to metrics ***
    // console.log("Data passed to updateForecastMetrics:", data);

    if (!data || !data.metrics || data.metrics.error) { // Check for error key in metrics too
        console.warn("updateForecastMetrics: Missing data, metrics object, or metrics error.");
        updateMetric(trendDirEl, null); updateMetric(trendStrEl, null); updateMetric(volEl, null); updateMetric(seasonEl, null); updateMetric(confIntEl, null);
        return;
    }
    const metrics = data.metrics;
    try {
        // *** DEBUG: Log individual metric values ***
        // console.log("Updating metrics with:", metrics);
        updateMetric(trendDirEl, metrics.trend_direction || 'Unknown');
        updateMetric(trendStrEl, metrics.trend_strength, v => `${(v * 100).toFixed(1)}%`);
        updateMetric(volEl, metrics.historical_volatility, v => `${(v * 100).toFixed(1)}%`);
        updateMetric(seasonEl, metrics.seasonality_strength, v => `${(v * 100).toFixed(1)}%`);
        updateMetric(confIntEl, metrics.confidence_interval, v => `±${(v / 2 * 100).toFixed(1)}%`);
        console.log("Forecast metrics updated.");
    } catch (e) { console.error("Error updating forecast metrics:", e); updateMetric(trendDirEl, 'Error'); updateMetric(trendStrEl, null); updateMetric(volEl, null); updateMetric(seasonEl, null); updateMetric(confIntEl, null); }
}

function updateForecastSummary(data) {
    const summaryElement = document.getElementById('forecastSummary');
    if (!summaryElement) return;
    if (!data || !data.forecast_values || !data.forecast_values.values || !data.metrics || data.metrics.error) { summaryElement.textContent = "Summary cannot be generated due to missing forecast data or metric errors."; summaryElement.className = 'text-muted small'; return; }
    try {
        const values = data.forecast_values.values.filter(v => v !== null && !isNaN(v)); const metrics = data.metrics; const lastActualPrice = data.last_actual_price;
        if (values.length === 0) { summaryElement.textContent = "No valid forecast values available to generate summary."; return; }
        const firstForecastValue = values[0]; const lastForecastValue = values[values.length - 1];
        let changeBase = (lastActualPrice !== null && !isNaN(lastActualPrice)) ? lastActualPrice : firstForecastValue; let percentChangeText = "an undetermined change"; let overallDirection = metrics.trend_direction ? metrics.trend_direction.toLowerCase() : 'uncertain';
        if (changeBase !== null && changeBase !== 0 && !isNaN(changeBase) && lastForecastValue !== null && !isNaN(lastForecastValue)) { const pC = ((lastForecastValue - changeBase) / Math.abs(changeBase) * 100); const changeDirection = pC >= 0 ? 'increase' : 'decrease'; percentChangeText = `a ${Math.abs(pC).toFixed(1)}% ${changeDirection}`; if (overallDirection === 'uncertain') { overallDirection = pC > 1 ? 'upward' : (pC < -1 ? 'downward' : 'sideways'); } }
        let confidenceLevel = 'medium'; const vol = metrics.historical_volatility; const interval = metrics.confidence_interval;
        if (vol !== null && interval !== null) { if (vol < 0.20 && interval < 0.15) confidenceLevel = 'high'; else if (vol > 0.50 || interval > 0.30) confidenceLevel = 'low'; }
        else if (vol === null || interval === null) { confidenceLevel = 'uncertain'; }
        let summary = `The model predicts a general <strong>${overallDirection}</strong> trend. `;
        summary += `The price is expected to show approximately <strong>${percentChangeText}</strong> compared to the last known price. `;
        summary += `Prediction confidence is assessed as <strong>${confidenceLevel}</strong>.`;
        if (metrics.seasonality_strength !== null && metrics.seasonality_strength > 0.1) { summary += ` Note: Seasonality appears to have a noticeable impact.`; }
        summaryElement.innerHTML = summary; summaryElement.className = 'text-dark small';
    } catch (e) { console.error("Error generating forecast summary:", e); summaryElement.textContent = "Error generating forecast summary."; summaryElement.className = 'text-danger small'; }
}


// --- Data Loading & Fetching ---
async function loadForecastData(stockSymbol) {
    if (isLoadingForecast) { console.warn("Forecast loading already in progress."); return; }
    if (!stockSymbol) { console.error('loadForecastData: Stock symbol required.'); showForecastError("No stock symbol provided."); return; }
    isLoadingForecast = true; currentStockSymbol = stockSymbol;
    console.log(`Requesting forecast data for: ${stockSymbol}`);
    showForecastLoading(`Loading forecast for ${stockSymbol}...`);
    const refreshBtn = document.getElementById('refreshForecastBtn'); if (refreshBtn) refreshBtn.disabled = true; // Disable refresh button

    try {
        const response = await fetchWithRetry(`/get_forecast_data?symbol=${stockSymbol}`, {}, MAX_RETRIES);
        if (!response.ok) { const errorData = await response.json().catch(() => ({ error: `Server error: ${response.status}` })); throw new Error(errorData.error || `HTTP error ${response.status}`); }
        const data = await response.json();
        if (data.error) { throw new Error(data.error); }

        // *** DEBUG LOG ADDED HERE ***
        console.log("Received forecast data structure:", JSON.stringify(data, null, 2));
        console.log("Forecast data received (object):", data);
        // *** DEBUG LOG ADDED HERE ***

        if (stockSymbol === currentStockSymbol) {
            // *** DEBUG LOG ADDED HERE ***
            console.log("Metrics object before update:", data.metrics);
            updateForecastChart(data, stockSymbol);
            updateForecastMetrics(data);
            updateForecastSummary(data);
        } else { console.log(`Forecast data for ${stockSymbol} received, but user now viewing ${currentStockSymbol}. Discarding.`); }
    } catch (error) {
        console.error(`Failed to load/process forecast data for ${stockSymbol}:`, error);
        if (stockSymbol === currentStockSymbol) { showForecastError(error.message || "Unknown error occurred."); }
    } finally {
        isLoadingForecast = false;
        if (refreshBtn) refreshBtn.disabled = false; // Re-enable refresh button
    }
}

async function fetchWithRetry(url, options = {}, retries = 0) {
    console.log(`Fetching (Attempt ${retries + 1}/${MAX_RETRIES + 1}): ${url}`);
    try {
        const response = await fetch(url, options);
        if ((response.status === 429 || response.status === 503 || response.status === 504) && retries < MAX_RETRIES) { const delay = RETRY_DELAY * (retries + 1); console.warn(`Received status ${response.status}. Retrying after ${delay}ms...`); await new Promise(resolve => setTimeout(resolve, delay)); return fetchWithRetry(url, options, retries + 1); }
        return response;
    } catch (error) {
        if (retries < MAX_RETRIES) { const delay = RETRY_DELAY * (retries + 1); console.warn(`Fetch failed (${error.message}). Retrying after ${delay}ms...`); await new Promise(resolve => setTimeout(resolve, delay)); return fetchWithRetry(url, options, retries + 1); }
        console.error(`Fetch failed after ${MAX_RETRIES + 1} attempts: ${error.message}`); throw error;
    }
}


// --- Event Listeners ---
document.addEventListener('DOMContentLoaded', function () {
    console.log("forecast.js: DOMContentLoaded event fired.");
    const forecastMetaDiv = document.getElementById(FORECAST_META_DIV_ID);
    const initialStockSymbol = forecastMetaDiv ? forecastMetaDiv.dataset.stockSymbol : null;
    const forecastChartDiv = document.getElementById(FORECAST_CHART_DIV_ID);
    const refreshBtn = document.getElementById('refreshForecastBtn');
    if (!forecastChartDiv) { console.warn("Forecast chart div not found."); return; }

    if (initialStockSymbol) { console.log(`forecast.js: Initial stock symbol ('${initialStockSymbol}'), triggering forecast load.`); loadForecastData(initialStockSymbol); }
    else { console.log("forecast.js: No initial stock symbol found."); showForecastLoading("Select a stock to view forecast."); }

    if (refreshBtn) {
        refreshBtn.addEventListener('click', function () {
            const currentSymbol = document.getElementById(FORECAST_META_DIV_ID)?.dataset.stockSymbol;
            if (currentSymbol && !isLoadingForecast) { // Check isLoadingForecast flag
                console.log(`Forecast refresh triggered for ${currentSymbol}`); this.disabled = true;
                fetch(`/clear_cache?key=forecast_${currentSymbol}`)
                    .then(res => { if (!res.ok) console.warn("Could not clear forecast cache."); return loadForecastData(currentSymbol); })
                    .catch(err => { console.error("Error clearing cache:", err); loadForecastData(currentSymbol); });
            } else if (isLoadingForecast) { console.log("Refresh button clicked, but forecast is already loading."); }
            else { console.warn("Refresh forecast clicked but no symbol found."); }
        });
        console.log("Refresh forecast button listener attached.");
    } else { console.warn("Refresh forecast button not found."); }

    const forecastTabLink = document.getElementById('forecasting-tab');
    if (forecastTabLink) {
        forecastTabLink.addEventListener('shown.bs.tab', function () {
            const chartDiv = document.getElementById(FORECAST_CHART_DIV_ID);
            if (chartDiv && typeof Plotly !== 'undefined' && chartDiv.data) {
                try { console.log("Forecast tab shown, resizing Plotly chart..."); setTimeout(() => Plotly.Plots.resize(chartDiv), 100); }
                catch (e) { console.error("Plotly resize error on tab show:", e); }
            } else { console.warn("Forecast tab shown, but Plotly chart not ready for resize."); }
        });
        console.log("Listener attached for 'shown.bs.tab' on forecast tab.");
    } else { console.warn("Forecast tab link ('#forecasting-tab') not found."); }
});