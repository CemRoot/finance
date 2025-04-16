// static/js/script.js
console.log("Script.js yüklendi.");

// --- Global Chart Instances ---
window.priceChartInstance = null;
window.volumeChartInstance = null;
window.openCloseChartInstance = null;
window.detailVolumeChartInstance = null;
window.highLowChartInstance = null;

// --- Helper Functions ---
// (formatDate, parseChartData, parseCandlestickData, commonChartOptions keep the same)
function formatDate(dateStr) {
    if (!dateStr) return null;
    if (typeof dateStr === 'number' && !isNaN(dateStr)) return dateStr;
    if (dateStr instanceof Date && !isNaN(dateStr.getTime())) return dateStr.getTime();
    if (typeof dateStr === 'string') {
        try {
            // Check if Luxon is available before using it
            if (typeof luxon === 'undefined' || !luxon.DateTime) {
                 console.error("formatDate: Luxon library is not available!");
                 return null; // Cannot format without Luxon
            }
            let dt = luxon.DateTime.fromISO(dateStr, { zone: 'utc' });
            if (dt.isValid) return dt.valueOf();
            dt = luxon.DateTime.fromSQL(dateStr);
            if (dt.isValid) return dt.valueOf();
            dt = luxon.DateTime.fromFormat(dateStr, 'yyyy-MM-dd HH:mm:ss');
            if (dt.isValid) return dt.valueOf();
             dt = luxon.DateTime.fromFormat(dateStr, 'yyyy-MM-dd');
             if (dt.isValid) return dt.startOf('day').valueOf();
            console.warn(`formatDate: Could not parse date string with known formats: ${dateStr}`);
            return null;
        } catch (e) {
            console.error(`formatDate: Error parsing date string '${dateStr}':`, e);
            return null;
        }
    }
    console.warn(`formatDate: Unhandled date type: ${typeof dateStr}`);
    return null;
}

function parseChartData(labels, values, labelName = 'Value') {
    if (!labels || !values || labels.length !== values.length) {
        console.warn(`parseChartData (${labelName}): Mismatch or missing labels/values.`);
        return [];
    }
    const chartData = [];
    let invalidCount = 0;
    for (let i = 0; i < labels.length; i++) {
        const timestamp = formatDate(labels[i]);
        let value = (values[i] === null || values[i] === undefined) ? null : parseFloat(values[i]);
        if (isNaN(value)) { value = null; }
        if (timestamp !== null) {
            chartData.push({ x: timestamp, y: value });
        } else {
            invalidCount++;
        }
    }
    if (invalidCount > 0) {
        console.warn(`parseChartData (${labelName}): Skipped ${invalidCount} data points due to invalid timestamps.`);
    }
    return chartData;
}


function parseCandlestickData(candlestickData) {
    if (!candlestickData || !Array.isArray(candlestickData)) {
        console.warn("parseCandlestickData: Invalid or missing candlestick data array.");
        return [];
    }
    const result = [];
    let invalidCount = 0;
    for (let i = 0; i < candlestickData.length; i++) {
         const d = candlestickData[i];
         if (!d || typeof d !== 'object' || !d.t || !('o' in d) || !('h' in d) || !('l' in d) || !('c' in d)) {
             invalidCount++; continue;
         }
         const timestamp = formatDate(d.t);
         const o = Number(d.o); const h = Number(d.h); const l = Number(d.l); const c = Number(d.c);
         if (timestamp !== null && !isNaN(o) && !isNaN(h) && !isNaN(l) && !isNaN(c)) {
             result.push({ x: timestamp, o: o, h: h, l: l, c: c });
         } else {
             invalidCount++;
         }
    }
    if (invalidCount > 0) {
        console.warn(`parseCandlestickData: Skipped ${invalidCount} invalid candle data points.`);
    }
    return result;
}

const commonChartOptions = {
    responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false },
    scales: {
        x: { type: 'time', time: { unit: 'day', tooltipFormat: 'DD MMM YYYY HH:mm', displayFormats: { day: 'DD MMM', month: 'MMM YYYY', year: 'YYYY' } }, grid: { display: false }, ticks: { autoSkip: true, maxTicksLimit: 10, source: 'auto', color: '#6c757d' } },
        y: { beginAtZero: false, grid: { color: 'rgba(200, 200, 200, 0.1)' }, ticks: { color: '#6c757d', callback: function(value) { if (typeof value === 'number') { return value.toLocaleString(undefined, { style: 'currency', currency: window.stockData?.currency || 'USD', minimumFractionDigits: 2 }); } return value; } } }
    },
    plugins: { legend: { display: true, labels: { color: '#2c3e50' } }, tooltip: { enabled: true, backgroundColor: 'rgba(0, 0, 0, 0.8)', titleColor: '#ffffff', bodyColor: '#ffffff', padding: 10, callbacks: { title: function(tooltipItems) { const firstItem = tooltipItems[0]; if (firstItem && firstItem.parsed && typeof luxon !== 'undefined') { return luxon.DateTime.fromMillis(firstItem.parsed.x).toFormat('DD MMM YYYY HH:mm'); } return ''; }, label: function(context) { let label = context.dataset.label || ''; if (label) { label += ': '; } if (context.parsed.y !== null) { label += context.parsed.y.toLocaleString(undefined, { style: 'currency', currency: window.stockData?.currency || 'USD', minimumFractionDigits: 2 }); } else { label += 'N/A'; } return label; } } } }
};


// --- Chart Creation/Update Functions ---
// (createOrUpdateChart, createOrUpdatePriceChart, createOrUpdateVolumeChart, etc. keep the same
// BUT they will only be CALLED after libraries are confirmed loaded)
function createOrUpdateChart(canvasId, chartInstanceVar, config) {
    // Check *again* here just before creating, though the main check should prevent this call if libs aren't ready
    if (typeof Chart === 'undefined' || typeof luxon === 'undefined' || typeof Chart._adapters?._date === 'undefined') {
         console.error(`createOrUpdateChart: Attempted to create chart '#${canvasId}' but libraries are still not ready.`);
         const container = document.getElementById(canvasId)?.closest('.chart-container');
         if (container) container.innerHTML = `<div class="alert alert-warning text-center small p-2 m-0" role="alert">Chart libraries failed to load. Please refresh.</div>`;
         return null;
    }
    // ... rest of the function remains the same ...
    const ctx = document.getElementById(canvasId);
    const container = ctx ? ctx.closest('.chart-container') : null;
    if (!ctx || !container) { console.warn(`Chart element '#${canvasId}' or its container not found.`); return null; }
    console.log(`Attempting to create/update chart: #${canvasId}`);
    if (window[chartInstanceVar] && typeof window[chartInstanceVar].destroy === 'function') {
        console.log(`Destroying previous chart instance: ${chartInstanceVar}`);
        try { window[chartInstanceVar].destroy(); } catch(e){ console.warn(`Error destroying previous chart instance '${chartInstanceVar}':`, e); }
         window[chartInstanceVar] = null;
    }
    try {
        console.log(`Creating new Chart instance for #${canvasId} with type: ${config.type}`);
        window[chartInstanceVar] = new Chart(ctx.getContext('2d'), config);
        console.log(`Chart '#${canvasId}' created/updated successfully.`);
        return window[chartInstanceVar];
    } catch (e) {
        console.error(`Error creating/updating chart '#${canvasId}':`, e);
        container.innerHTML = `<div class="alert alert-danger text-center small p-2 m-0" role="alert">Could not load chart: ${e.message}</div>`;
        window[chartInstanceVar] = null;
        return null;
    }
}

// --- Other chart creation functions (createOrUpdatePriceChart, createOrUpdateVolumeChart, etc.) remain unchanged ---
// ...
function createOrUpdatePriceChart(stockData, chartType = 'candlestick') { /* ... as before ... */ }
function createOrUpdateVolumeChart(stockData) { /* ... as before ... */ }
function createOpenCloseChart(stockData) { /* ... as before ... */ }
function createDetailVolumeChart(stockData) { /* ... as before ... */ }
function createHighLowChart(stockData) { /* ... as before ... */ }
// --- AJAX Data Refresh ---
function refreshStockData() { /* ... as before ... */ }
// --- Update Header Info ---
function updateStockInfo(stockData) { /* ... as before ... */ }
// --- Flash Messages Utility ---
function showFlashMessage(message, category = 'info', duration = 5000) { /* ... as before ... */ }
function createFlashContainer() { /* ... as before ... */ }
function fadeOutAndRemove(element) { /* ... as before ... */ }
// --- Initialize Detail Charts ---
function initializeDetailCharts() { /* ... as before, but add library check inside? Or rely on main check */
    if (typeof Chart === 'undefined' || typeof luxon === 'undefined') {
        console.warn("initializeDetailCharts: Libraries not ready.");
        const detailContainers = document.querySelectorAll('.details-chart-container');
         detailContainers.forEach(container => {
             container.innerHTML = `<div class="alert alert-light text-center small p-2 m-0">Data not available for detailed charts.</div>`;
         });
        return;
    }
    if (!window.stockData || window.stockData.error) {
        console.warn("Cannot initialize detail charts: window.stockData is missing, empty, or contains an error.");
        const detailContainers = document.querySelectorAll('.details-chart-container');
         detailContainers.forEach(container => {
             container.innerHTML = `<div class="alert alert-light text-center small p-2 m-0">Data not available for detailed charts.</div>`;
         });
        return;
    }
    console.log("Initializing detail charts...");
    createOpenCloseChart(window.stockData);
    createDetailVolumeChart(window.stockData);
    createHighLowChart(window.stockData);
     console.log("Detail charts initialized.");
}

// --- DOMContentLoaded Event Listener ---
document.addEventListener('DOMContentLoaded', function() {
    console.log("script.js: DOMContentLoaded event fired.");

    // *** FIX: Add check for libraries before initializing charts ***
    const maxWaitTime = 5000; // Max wait 5 seconds for libraries
    const checkInterval = 100; // Check every 100ms
    let elapsedTime = 0;

    const libraryCheckInterval = setInterval(function() {
        elapsedTime += checkInterval;
        const chartJsReady = typeof Chart !== 'undefined';
        const luxonReady = typeof luxon !== 'undefined';
        // Also check if the adapter registered itself (might be needed)
        const adapterReady = chartJsReady && Chart._adapters && Chart._adapters._date;

        // console.log(`Checking libraries: Chart=${chartJsReady}, Luxon=${luxonReady}, Adapter=${adapterReady}`); // Debug log

        if (chartJsReady && luxonReady && adapterReady) {
            clearInterval(libraryCheckInterval); // Stop checking
            console.log("script.js: Chart.js and Luxon libraries confirmed loaded.");
            initializePageCharts(); // Proceed with chart initialization
        } else if (elapsedTime >= maxWaitTime) {
            clearInterval(libraryCheckInterval); // Stop checking after timeout
            console.error("script.js: Timeout waiting for Chart.js/Luxon libraries to load.");
            // Show error message to the user in chart areas
            displayLibraryLoadError();
        }
    }, checkInterval);

    // Function to initialize charts once libraries are ready
    function initializePageCharts() {
        if (window.stockData && !window.stockData.error) {
            console.log("script.js: Initializing overview charts with loaded data.");
            createOrUpdatePriceChart(window.stockData, 'candlestick');
            createOrUpdateVolumeChart(window.stockData);
            updateStockInfo(window.stockData);
            const overviewTabLink = document.getElementById('overview-tab');
            if (overviewTabLink && overviewTabLink.classList.contains('active')) {
                console.log("Overview tab is active on load, initializing detail charts.");
                 initializeDetailCharts();
            } else {
                 console.log("Overview tab not initially active, detail charts will load when tab is selected.");
            }
        } else {
            console.log("script.js: No initial valid stock data found. Charts not initialized.");
             updateStockInfo(window.stockData); // Update header to show N/A etc.
        }

        // Attach event listeners *after* libraries are confirmed loaded
        attachEventListeners();
    }

    // Function to display errors if libraries fail to load
    function displayLibraryLoadError() {
         showFlashMessage("Error: Charting libraries failed to load. Charts cannot be displayed.", "danger", 10000);
         const chartContainers = document.querySelectorAll('.chart-container'); // Target all chart areas
         chartContainers.forEach(container => {
             container.innerHTML = `<div class="alert alert-danger text-center small p-2 m-0" role="alert">Charting libraries failed to load. Please check your connection or refresh the page.</div>`;
         });
    }

    // Function to attach event listeners
    function attachEventListeners() {
         const chartTypeRadios = document.querySelectorAll('.chart-type-selector input[name="chartType"]');
         chartTypeRadios.forEach(radio => {
             radio.addEventListener('change', function() {
                 if (this.checked && window.stockData && !window.stockData.error) {
                      console.log(`Chart type changed to: ${this.dataset.chartType}`);
                     createOrUpdatePriceChart(window.stockData, this.dataset.chartType);
                 }
             });
         });

         const refreshButton = document.getElementById('refreshStockBtn');
         if (refreshButton) { refreshButton.addEventListener('click', refreshStockData); }

         const overviewTabLinkForListener = document.getElementById('overview-tab');
         if (overviewTabLinkForListener) {
             overviewTabLinkForListener.addEventListener('shown.bs.tab', initializeDetailCharts);
         }

         const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
         if (tooltipTriggerList.length > 0) { tooltipTriggerList.map(function (tooltipTriggerEl) { return new bootstrap.Tooltip(tooltipTriggerEl); }); }

         console.log("script.js: Event listeners attached.");
    }

}); // End DOMContentLoaded