// static/js/script.js
console.log("Script.js loaded.");

// --- Global Chart Instances ---
window.priceChartInstance = null;
window.volumeChartInstance = null;
window.openCloseChartInstance = null;
window.detailVolumeChartInstance = null;
window.highLowChartInstance = null;

// --- Helper Functions ---

/**
 * Formats a date string or object into a timestamp (milliseconds).
 * Uses Luxon if available, falls back to native Date.
 * @param {string|Date|number} dateStr - The date input.
 * @returns {number|null} Timestamp in milliseconds or null on error.
 */
function formatDate(dateStr) {
    if (!dateStr) return null;
    if (typeof dateStr === 'number' && !isNaN(dateStr)) return dateStr;
    if (dateStr instanceof Date && !isNaN(dateStr.getTime())) return dateStr.getTime();

    if (typeof dateStr === 'string') {
        try {
            // Try Luxon first if available
            if (typeof luxon !== 'undefined' && luxon.DateTime) {
                let dt = luxon.DateTime.fromISO(dateStr, { zone: 'utc' });
                if (dt.isValid) return dt.valueOf();
                // Add other Luxon formats if needed
                dt = luxon.DateTime.fromSQL(dateStr);
                if (dt.isValid) return dt.valueOf();
                dt = luxon.DateTime.fromFormat(dateStr, 'yyyy-MM-dd');
                if (dt.isValid) return dt.startOf('day').valueOf();
            }

            // Fallback to native Date parsing
            const nativeDt = new Date(dateStr);
            if (!isNaN(nativeDt.getTime())) {
                if (typeof luxon === 'undefined') console.warn("formatDate: Luxon not loaded, using native Date for:", dateStr);
                else console.warn(`formatDate: Luxon failed, using native Date fallback for: ${dateStr}`);
                return nativeDt.getTime();
            }

            console.error(`formatDate: Failed to parse date string: ${dateStr}`);
            return null;
        } catch (e) {
            console.error(`formatDate: Error parsing date string '${dateStr}':`, e);
            return null;
        }
    }

    console.warn(`formatDate: Unhandled date type: ${typeof dateStr}`);
    return null;
}

/**
 * Parses labels and values into an array of {x, y} objects suitable for Chart.js.
 * @param {Array} labels - Array of date strings/objects.
 * @param {Array} values - Array of corresponding numeric values.
 * @param {string} [labelName='Value'] - Name for logging purposes.
 * @returns {Array<object>} Array of {x: timestamp, y: value}.
 */
function parseChartData(labels, values, labelName = 'Value') {
    if (!labels || !values || labels.length !== values.length) {
        console.warn(`parseChartData (${labelName}): Mismatch/missing labels/values.`);
        return [];
    }
    const chartData = [];
    let invalidCount = 0;
    for (let i = 0; i < labels.length; i++) {
        const timestamp = formatDate(labels[i]);
        let value = (values[i] === null || values[i] === undefined) ? null : parseFloat(values[i]);
        if (isNaN(value)) { value = null; } // Ensure numeric or null

        // Add if timestamp is valid (value can be null for line chart gaps)
        if (timestamp !== null) {
            chartData.push({ x: timestamp, y: value });
        } else {
            invalidCount++;
        }
    }
    if (invalidCount > 0) {
        console.warn(`parseChartData (${labelName}): Skipped ${invalidCount} points due to invalid timestamps.`);
    }
    return chartData;
}

/**
 * Parses candlestick data array into the format required by chartjs-chart-financial.
 * @param {Array<object>} candlestickData - Array of {t, o, h, l, c} objects.
 * @returns {Array<object>} Array of {x: timestamp, o, h, l, c}.
 */
function parseCandlestickData(candlestickData) {
    if (!candlestickData || !Array.isArray(candlestickData)) {
        console.warn("parseCandlestickData: Invalid candlestick data array.");
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
        console.warn(`parseCandlestickData: Skipped ${invalidCount} invalid candles.`);
    }
    return result;
}

// --- Common Chart Options ---
const commonChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    scales: {
        x: {
            type: 'linear', // Use linear scale for timestamps
            grid: { display: false },
            ticks: {
                autoSkip: true, maxTicksLimit: 10, color: '#6c757d',
                // Callback to format timestamps on the axis
                callback: function (value, index, ticks) {
                    try {
                        const dt = typeof luxon !== 'undefined' ? luxon.DateTime.fromMillis(value) : new Date(value);
                        if (typeof luxon !== 'undefined' && dt.isValid) {
                            const format = ticks.length > 15 ? 'dd MMM' : 'dd MMM yy';
                            return dt.toFormat(format);
                        } else if (!isNaN(dt.getTime())) { // Check if native Date is valid
                            return `${dt.getDate()} ${dt.toLocaleString('default', { month: 'short' })}`;
                        } return value; // Fallback to timestamp
                    } catch (e) { console.warn("Error formatting x-axis tick:", e); return value; }
                }
            }
        },
        y: {
            beginAtZero: false, grid: { color: 'rgba(200, 200, 200, 0.1)' },
            ticks: {
                color: '#6c757d', callback: function (value) {
                    if (typeof value === 'number') {
                        return value.toLocaleString(undefined, { style: 'currency', currency: window.stockData?.currency || 'USD', minimumFractionDigits: 2 });
                    } return value;
                }
            }
        }
    },
    plugins: {
        legend: { display: true, labels: { color: '#2c3e50' } },
        tooltip: {
            enabled: true, backgroundColor: 'rgba(0, 0, 0, 0.8)', titleColor: '#ffffff', bodyColor: '#ffffff', padding: 10,
            callbacks: {
                title: function (tooltipItems) {
                    const firstItem = tooltipItems[0];
                    if (firstItem?.parsed?.x) {
                        try {
                            const dt = typeof luxon !== 'undefined' ? luxon.DateTime.fromMillis(firstItem.parsed.x) : new Date(firstItem.parsed.x);
                            if (typeof luxon !== 'undefined' && dt.isValid) { return dt.toFormat('DD MMM YYYY HH:mm'); }
                            else if (!isNaN(dt.getTime())) { return dt.toLocaleDateString() + ' ' + dt.toLocaleTimeString(); }
                        } catch (e) { console.warn("Error formatting tooltip title:", e); }
                        return firstItem.parsed.x; // Fallback
                    } return '';
                },
                label: function (context) {
                    let label = context.dataset.label || '';
                    if (label) { label += ': '; }
                    if (context.parsed?.y !== null && context.parsed?.y !== undefined) {
                        label += context.parsed.y.toLocaleString(undefined, { style: 'currency', currency: window.stockData?.currency || 'USD', minimumFractionDigits: 2 });
                    } else { label += 'N/A'; }
                    return label;
                }
            }
        }
    }
};


// --- Chart Creation/Update Functions ---

/**
 * Creates or updates a Chart.js chart instance.
 * @param {string} canvasId - The ID of the canvas element.
 * @param {string} chartInstanceVar - The name of the global variable holding the chart instance (e.g., 'priceChartInstance').
 * @param {object} config - The Chart.js configuration object (type, data, options).
 * @returns {Chart|null} The created/updated Chart instance or null on error.
 */
function createOrUpdateChart(canvasId, chartInstanceVar, config) {
    // Check if Chart.js is loaded
    if (typeof Chart === 'undefined') {
        console.error(`createOrUpdateChart: Chart.js library not loaded for '#${canvasId}'.`);
        const container = document.getElementById(canvasId)?.closest('.chart-container');
        if (container) container.innerHTML = `<div class="alert alert-warning text-center small p-2 m-0">Chart library failed. Refresh.</div>`;
        return null;
    }
    // Check if Luxon is needed and loaded (only if x-axis is not linear)
    if (config?.options?.scales?.x?.type !== 'linear' && typeof luxon === 'undefined') {
        console.error(`createOrUpdateChart: Luxon library needed but not loaded for '#${canvasId}'.`);
        const container = document.getElementById(canvasId)?.closest('.chart-container');
        if (container) container.innerHTML = `<div class="alert alert-warning text-center small p-2 m-0">Date library failed. Refresh.</div>`;
        return null;
    }

    const ctx = document.getElementById(canvasId);
    const container = ctx ? ctx.closest('.chart-container') : null;
    if (!ctx || !container) { console.warn(`Chart element '#${canvasId}' or container missing.`); return null; }

    // Destroy previous instance
    if (window[chartInstanceVar] && typeof window[chartInstanceVar].destroy === 'function') {
        try { window[chartInstanceVar].destroy(); } catch (e) { console.warn(`Error destroying '${chartInstanceVar}':`, e); }
        window[chartInstanceVar] = null;
    }

    // Create new chart
    try {
        console.log(`Creating new Chart instance for #${canvasId} with type: ${config.type}`);
        window[chartInstanceVar] = new Chart(ctx.getContext('2d'), config);
        console.log(`Chart '#${canvasId}' created/updated.`);
        return window[chartInstanceVar];
    } catch (e) {
        console.error(`Error creating chart '#${canvasId}':`, e);
        container.innerHTML = `<div class="alert alert-danger text-center small p-2 m-0">Could not load chart: ${e.message}</div>`;
        window[chartInstanceVar] = null;
        return null;
    }
}

/**
 * Creates/updates the main price chart (Candlestick or Line).
 * @param {object} stockData - The stock data object from Flask.
 * @param {string} [chartType='candlestick'] - The type of chart ('candlestick' or 'line').
 */
function createOrUpdatePriceChart(stockData, chartType = 'candlestick') {
    const canvasId = 'priceChart';
    if (!stockData || typeof stockData !== 'object') { /* ... handle error ... */ console.warn(`PriceChart: Invalid stockData.`); createOrUpdateChart(canvasId, 'priceChartInstance', { type: 'line', data: { datasets: [] }, options: { plugins: { title: { display: true, text: 'No data' } } } }); return; }
    console.log(`Updating Price Chart (#${canvasId}) with type: ${chartType}`);

    let data, options = JSON.parse(JSON.stringify(commonChartOptions)); // Deep copy options
    // *** DÜZELTME: Use the passed chartType ***
    let configType = chartType;
    // *** DÜZELTME SONU ***
    options.plugins.title = { display: false }; // No title needed for main chart
    const currencyCode = stockData.currency || 'USD';
    // Update currency formatting for this specific chart
    options.scales.y.ticks.callback = function (value) { return typeof value === 'number' ? value.toLocaleString(undefined, { style: 'currency', currency: currencyCode, minimumFractionDigits: 2 }) : value; };
    options.plugins.tooltip.callbacks.label = function (context) { let label = context.dataset.label || ''; if (label) { label += ': '; } if (context.parsed?.y !== null && context.parsed?.y !== undefined) { label += context.parsed.y.toLocaleString(undefined, { style: 'currency', currency: currencyCode, minimumFractionDigits: 2 }); } else { label += 'N/A'; } return label; };

    if (configType === 'candlestick') {
        configType = 'candlestick'; // Correct type for the plugin
        const candleData = parseCandlestickData(stockData.candlestick_data);
        console.log('Price Chart - Candlestick Data (first 5):', JSON.stringify(candleData.slice(0, 5)));
        if (candleData.length === 0) { console.warn(`PriceChart (#${canvasId}): No valid candlestick data. Falling back to line.`); createOrUpdatePriceChart(stockData, 'line'); return; }
        data = { datasets: [{ label: stockData.company_name || 'Price', data: candleData, color: { up: 'rgba(25, 135, 84, 1)', down: 'rgba(220, 53, 69, 1)', unchanged: 'rgba(108, 117, 125, 1)', }, borderColor: 'rgba(0,0,0,0.5)', borderWidth: 1 }] };
        options.plugins.legend.display = false; // Hide legend for candlestick
        // Specific tooltip for candlestick
        options.plugins.tooltip.callbacks.label = function (context) { const raw = context.raw; if (raw && typeof raw.o === 'number') { const o = raw.o.toLocaleString(undefined, { style: 'currency', currency: currencyCode, minimumFractionDigits: 2 }); const h = raw.h.toLocaleString(undefined, { style: 'currency', currency: currencyCode, minimumFractionDigits: 2 }); const l = raw.l.toLocaleString(undefined, { style: 'currency', currency: currencyCode, minimumFractionDigits: 2 }); const c = raw.c.toLocaleString(undefined, { style: 'currency', currency: currencyCode, minimumFractionDigits: 2 }); return [` Open: ${o}`, ` High: ${h}`, ` Low: ${l}`, ` Close: ${c}`]; } return ''; };
        console.log("Rendering Candlestick chart.");
    } else { // Line chart
        configType = 'line';
        const lineData = parseChartData(stockData.labels, stockData.values, 'Close Price');
        console.log('Price Chart - Line Data (first 5):', JSON.stringify(lineData.slice(0, 5)));
        if (lineData.length === 0) { console.warn(`PriceChart (#${canvasId}): No valid line data.`); createOrUpdateChart(canvasId, 'priceChartInstance', { type: 'line', data: { datasets: [] }, options: { plugins: { title: { display: true, text: 'No price data' } } } }); return; }
        data = { datasets: [{ label: 'Close Price', data: lineData, borderColor: '#0d6efd', backgroundColor: 'rgba(13, 110, 253, 0.1)', borderWidth: 1.5, tension: 0.1, pointRadius: 0, fill: true }] };
        options.plugins.legend.display = true; // Show legend for line
        console.log("Rendering Line chart.");
    }
    createOrUpdateChart(canvasId, 'priceChartInstance', { type: configType, data: data, options: options });
}

/**
 * Creates/updates the small volume bar chart in the overview.
 * @param {object} stockData - The stock data object from Flask.
 */
function createOrUpdateVolumeChart(stockData) {
    const canvasId = 'volumeChart';
    if (!stockData?.labels || !stockData?.volume_values) { /* ... handle error ... */ console.warn(`VolumeChart: Missing data.`); createOrUpdateChart(canvasId, 'volumeChartInstance', { type: 'bar', data: { datasets: [] }, options: { plugins: { title: { display: true, text: 'No volume data' } } } }); return; }
    console.log(`Updating Volume Chart (#${canvasId})`);
    const volumeData = parseChartData(stockData.labels, stockData.volume_values, 'Volume');
    console.log('Volume Chart Data (first 5):', JSON.stringify(volumeData.slice(0, 5)));
    if (volumeData.length === 0) { /* ... handle error ... */ console.warn(`VolumeChart: No valid data.`); createOrUpdateChart(canvasId, 'volumeChartInstance', { type: 'bar', data: { datasets: [] }, options: { plugins: { title: { display: true, text: 'No volume data' } } } }); return; }
    const prices = stockData.values || [];
    const barColors = volumeData.map((dp, i) => { if (i > 0 && prices[i] !== null && prices[i - 1] !== null) { return prices[i] >= prices[i - 1] ? 'rgba(25, 135, 84, 0.6)' : 'rgba(220, 53, 69, 0.6)'; } return 'rgba(108, 117, 125, 0.6)'; });
    const options = { responsive: true, maintainAspectRatio: false, scales: { x: { type: commonChartOptions.scales.x.type, display: false }, y: { beginAtZero: true, display: false } }, plugins: { legend: { display: false }, tooltip: { enabled: false } } };
    createOrUpdateChart(canvasId, 'volumeChartInstance', { type: 'bar', data: { datasets: [{ label: 'Volume', data: volumeData, backgroundColor: barColors, borderWidth: 0 }] }, options: options });
}

/** Creates/updates the Open/Close line chart. */
function createOpenCloseChart(stockData) {
    const canvasId = 'openCloseChart';
    if (!stockData?.labels || !stockData?.open_values || !stockData?.values) { /* ... handle error ... */ console.warn(`OpenCloseChart: Missing data.`); createOrUpdateChart(canvasId, 'openCloseChartInstance', { type: 'line', data: { datasets: [] }, options: { plugins: { title: { display: true, text: 'Data unavailable' } } } }); return; }
    console.log(`Creating/Updating Open/Close Chart (#${canvasId})`);
    const openData = parseChartData(stockData.labels, stockData.open_values, 'Open');
    const closeData = parseChartData(stockData.labels, stockData.values, 'Close');
    console.log('Open/Close - Open Data (first 5):', JSON.stringify(openData.slice(0, 5)));
    console.log('Open/Close - Close Data (first 5):', JSON.stringify(closeData.slice(0, 5)));
    if (openData.length === 0 && closeData.length === 0) { /* ... handle error ... */ console.warn(`OpenCloseChart: No valid data.`); createOrUpdateChart(canvasId, 'openCloseChartInstance', { type: 'line', data: { datasets: [] }, options: { plugins: { title: { display: true, text: 'Data unavailable' } } } }); return; }
    const options = JSON.parse(JSON.stringify(commonChartOptions));
    const currencyCode = stockData.currency || 'USD';
    options.scales.y.ticks.callback = function (value) { return typeof value === 'number' ? value.toLocaleString(undefined, { style: 'currency', currency: currencyCode, minimumFractionDigits: 2 }) : value; };
    options.plugins.tooltip.callbacks.label = function (context) { let label = context.dataset.label || ''; if (label) { label += ': '; } if (context.parsed?.y !== null && context.parsed?.y !== undefined) { label += context.parsed.y.toLocaleString(undefined, { style: 'currency', currency: currencyCode, minimumFractionDigits: 2 }); } else { label += 'N/A'; } return label; };
    options.plugins.legend.position = 'top'; options.plugins.title = { display: false };
    createOrUpdateChart(canvasId, 'openCloseChartInstance', { type: 'line', data: { datasets: [{ label: 'Open', data: openData, borderColor: 'rgba(255, 159, 64, 1)', borderWidth: 1.5, pointRadius: 0, tension: 0.1, fill: false }, { label: 'Close', data: closeData, borderColor: 'rgba(54, 162, 235, 1)', borderWidth: 1.5, pointRadius: 0, tension: 0.1, fill: false }] }, options: options });
}

/** Creates/updates the detailed volume bar chart. */
function createDetailVolumeChart(stockData) {
    const canvasId = 'detailVolumeChart';
    if (!stockData?.labels || !stockData?.volume_values) { /* ... handle error ... */ console.warn(`DetailVolumeChart: Missing data.`); createOrUpdateChart(canvasId, 'detailVolumeChartInstance', { type: 'bar', data: { datasets: [] }, options: { plugins: { title: { display: true, text: 'Data unavailable' } } } }); return; }
    console.log(`Creating/Updating Detail Volume Chart (#${canvasId})`);
    const volumeData = parseChartData(stockData.labels, stockData.volume_values, 'Detail Volume');
    console.log('Detail Volume Data (first 5):', JSON.stringify(volumeData.slice(0, 5)));
    if (volumeData.length === 0) { /* ... handle error ... */ console.warn(`DetailVolumeChart: No valid data.`); createOrUpdateChart(canvasId, 'detailVolumeChartInstance', { type: 'bar', data: { datasets: [] }, options: { plugins: { title: { display: true, text: 'Data unavailable' } } } }); return; }
    const prices = stockData.values || [];
    const barColors = volumeData.map((dp, i) => { if (i > 0 && prices[i] !== null && prices[i - 1] !== null) { return prices[i] >= prices[i - 1] ? 'rgba(25, 135, 84, 0.6)' : 'rgba(220, 53, 69, 0.6)'; } return 'rgba(108, 117, 125, 0.6)'; });
    const options = JSON.parse(JSON.stringify(commonChartOptions));
    options.scales.y.ticks.callback = function (value) { if (typeof value === 'number') { if (value >= 1e9) return (value / 1e9).toFixed(1) + 'B'; if (value >= 1e6) return (value / 1e6).toFixed(1) + 'M'; if (value >= 1e3) return (value / 1e3).toFixed(0) + 'K'; return value.toString(); } return value; };
    options.plugins.legend.display = false; options.plugins.title = { display: false };
    options.plugins.tooltip.callbacks.label = function (context) { let label = context.dataset.label || ''; if (label) label += ': '; if (context.parsed?.y !== null && context.parsed?.y !== undefined) { const v = context.parsed.y; if (v >= 1e9) label += (v / 1e9).toFixed(2) + 'B'; else if (v >= 1e6) label += (v / 1e6).toFixed(2) + 'M'; else if (v >= 1e3) label += Math.round(v / 1e3) + 'K'; else label += v.toLocaleString(); } return label; };
    createOrUpdateChart(canvasId, 'detailVolumeChartInstance', { type: 'bar', data: { datasets: [{ label: 'Volume', data: volumeData, backgroundColor: barColors, borderWidth: 0 }] }, options: options });
}

/** Creates/updates the High/Low range line chart. */
function createHighLowChart(stockData) {
    const canvasId = 'highLowChart';
    if (!stockData?.labels || !stockData?.high_values || !stockData?.low_values) { /* ... handle error ... */ console.warn(`HighLowChart: Missing data.`); createOrUpdateChart(canvasId, 'highLowChartInstance', { type: 'line', data: { datasets: [] }, options: { plugins: { title: { display: true, text: 'Data unavailable' } } } }); return; }
    console.log(`Creating/Updating High/Low Chart (#${canvasId})`);
    const highData = parseChartData(stockData.labels, stockData.high_values, 'High');
    const lowData = parseChartData(stockData.labels, stockData.low_values, 'Low');
    console.log('High/Low - High Data (first 5):', JSON.stringify(highData.slice(0, 5)));
    console.log('High/Low - Low Data (first 5):', JSON.stringify(lowData.slice(0, 5)));
    if (highData.length === 0 && lowData.length === 0) { /* ... handle error ... */ console.warn(`HighLowChart: No valid data.`); createOrUpdateChart(canvasId, 'highLowChartInstance', { type: 'line', data: { datasets: [] }, options: { plugins: { title: { display: true, text: 'Data unavailable' } } } }); return; }
    const options = JSON.parse(JSON.stringify(commonChartOptions));
    const currencyCode = stockData.currency || 'USD';
    options.scales.y.ticks.callback = function (value) { return typeof value === 'number' ? value.toLocaleString(undefined, { style: 'currency', currency: currencyCode, minimumFractionDigits: 2 }) : value; };
    options.plugins.tooltip.callbacks.label = function (context) { let label = context.dataset.label || ''; if (label) label += ': '; if (context.parsed?.y !== null && context.parsed?.y !== undefined) { label += context.parsed.y.toLocaleString(undefined, { style: 'currency', currency: currencyCode, minimumFractionDigits: 2 }); } else { label += 'N/A'; } return label; };
    options.plugins.legend.position = 'top'; options.plugins.title = { display: false }; options.plugins.filler = { propagate: false }; options.interaction = { mode: 'nearest', axis: 'x', intersect: false };
    createOrUpdateChart(canvasId, 'highLowChartInstance', { type: 'line', data: { datasets: [{ label: 'Low', data: lowData, borderColor: 'rgba(220, 53, 69, 0.8)', backgroundColor: 'rgba(220, 53, 69, 0.1)', borderWidth: 1.5, pointRadius: 0, tension: 0.1, fill: false }, { label: 'High', data: highData, borderColor: 'rgba(25, 135, 84, 0.8)', backgroundColor: 'rgba(35, 195, 105, 0.2)', borderWidth: 1.5, pointRadius: 0, tension: 0.1, fill: '-1' }] }, options: options });
}


// --- AJAX Data Refresh ---
/** Fetches fresh stock data via AJAX and updates charts/UI. */
function refreshStockData() {
    const stockSymbolInput = document.querySelector('#stock');
    const refreshButton = document.getElementById('refreshStockBtn');
    const refreshIcon = refreshButton ? refreshButton.querySelector('i') : null;
    if (!stockSymbolInput?.value) { console.warn("refreshStockData: Stock symbol missing."); return; }
    const stockSymbol = stockSymbolInput.value.trim().toUpperCase();

    if (refreshIcon) { refreshIcon.className = 'fas fa-spinner fa-spin'; if (refreshButton) refreshButton.disabled = true; } // Simplified class change
    console.log(`Refreshing data for ${stockSymbol}...`); showFlashMessage(`Refreshing data for ${stockSymbol}...`, "info", 2000);

    fetch(`/refresh_stock?stock=${stockSymbol}`)
        .then(response => { if (!response.ok) { return response.json().then(errData => { throw new Error(errData.message || `Network error ${response.status}`); }).catch(() => { throw new Error(`Network error ${response.status}`); }); } return response.json(); })
        .then(data => {
            if (data.status === 'success' && data.stock_data) {
                console.log("Data refreshed successfully via AJAX."); window.stockData = data.stock_data; updateStockInfo(window.stockData);
                const activeChartType = document.querySelector('.chart-type-selector input[name="chartType"]:checked')?.dataset.chartType || 'candlestick';
                // *** DÜZELTME: Use activeChartType for price chart refresh ***
                createOrUpdatePriceChart(window.stockData, activeChartType);
                createOrUpdateVolumeChart(window.stockData);
                const overviewTab = document.getElementById('overview');
                if (overviewTab?.classList.contains('active')) { console.log("Refreshing detail charts."); initializeDetailCharts(); }
                else { console.log("Overview tab not active."); }
                const timestampEl = document.querySelector('.last-updated');
                if (timestampEl && window.stockData.timestamp) {
                    try {
                        const dt = typeof luxon !== 'undefined' ? luxon.DateTime.fromISO(window.stockData.timestamp).setZone('local') : new Date(window.stockData.timestamp);
                        if (typeof luxon !== 'undefined' && dt.isValid) { timestampEl.textContent = dt.toFormat('dd MMM yyyy HH:mm:ss'); }
                        else if (!isNaN(dt.getTime())) { timestampEl.textContent = dt.toLocaleString(); }
                        else { timestampEl.textContent = window.stockData.timestamp; }
                    } catch (e) { console.error("Error formatting timestamp:", e); timestampEl.textContent = window.stockData.timestamp; }
                } showFlashMessage('Stock data updated.', 'success');
            } else { console.error('Backend Error:', data.message); showFlashMessage(data.message || 'Failed to update data.', 'danger'); }
        })
        .catch(error => { console.error('AJAX Fetch Error:', error); showFlashMessage(`Error updating data: ${error.message}`, 'danger'); })
        .finally(() => { if (refreshIcon) { refreshIcon.className = 'fas fa-sync'; if (refreshButton) refreshButton.disabled = false; } console.log(`Refresh action for ${stockSymbol} completed.`); });
}

// --- Update Header Info ---
/** Updates the header elements (price, change, market status). */
function updateStockInfo(stockData) { /* ... as before ... */ console.log("Updating stock header info..."); if (!stockData || typeof stockData !== 'object') { console.warn("updateStockInfo: Invalid stockData."); document.querySelector('.current-price').textContent = 'N/A'; document.querySelector('.change-percent').textContent = 'N/A'; document.querySelector('.change-percent').className = 'change-percent small text-muted d-block'; const marketStatusBadge = document.querySelector('.market-status-badge'); if (marketStatusBadge) { marketStatusBadge.className = 'market-status-badge badge rounded-pill px-3 py-1 bg-secondary text-white'; marketStatusBadge.querySelector('i').className = 'fas fa-question-circle me-1'; marketStatusBadge.querySelector('span').textContent = 'Market Unknown'; } return; } const priceEl = document.querySelector('.current-price'); const changeEl = document.querySelector('.change-percent'); const marketStatusBadge = document.querySelector('.market-status-badge'); const marketIcon = marketStatusBadge ? marketStatusBadge.querySelector('i') : null; const marketSpan = marketStatusBadge ? marketStatusBadge.querySelector('span') : null; const currencyCode = stockData.currency || 'USD'; if (priceEl) { priceEl.textContent = (stockData.current_price !== null && stockData.current_price !== undefined) ? stockData.current_price.toLocaleString(undefined, { style: 'currency', currency: currencyCode, minimumFractionDigits: 2 }) : 'N/A'; } if (changeEl) { const changePercent = stockData.change_percent; if (changePercent !== null && changePercent !== undefined) { const isPositive = changePercent >= 0; changeEl.classList.remove('text-success', 'text-danger', 'text-muted'); changeEl.classList.add(isPositive ? 'text-success' : 'text-danger'); changeEl.innerHTML = `<i class="fas ${isPositive ? 'fa-arrow-up' : 'fa-arrow-down'} me-1"></i>${changePercent.toFixed(2)}%`; } else { changeEl.innerHTML = 'N/A'; changeEl.className = 'change-percent small text-muted d-block'; } } if (marketStatusBadge && stockData.market_status) { const status = stockData.market_status.toUpperCase(); const status_lower = status.toLowerCase(); let badge_bg = 'bg-info'; let badge_text = 'text-white'; let badge_icon = 'fa-question-circle'; let displayStatus = status.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()); if (status_lower.includes('reg') || status_lower.includes('open')) { badge_bg = 'bg-success'; badge_text = 'text-white'; badge_icon = 'fa-check-circle'; displayStatus = 'Open'; } else if (status_lower.includes('pre')) { badge_bg = 'bg-warning'; badge_text = 'text-dark'; badge_icon = 'fa-hourglass-start'; displayStatus = 'Pre-Market'; } else if (status_lower.includes('post') || status_lower.includes('extended') || status_lower.includes('after')) { badge_bg = 'bg-warning'; badge_text = 'text-dark'; badge_icon = 'fa-hourglass-end'; displayStatus = 'After Hours'; } else if (status_lower.includes('closed')) { badge_bg = 'bg-secondary'; badge_text = 'text-white'; badge_icon = 'fa-times-circle'; displayStatus = 'Closed'; } marketStatusBadge.className = `market-status-badge badge rounded-pill px-3 py-1 ${badge_bg} ${badge_text}`; if (marketIcon) marketIcon.className = `fas ${badge_icon} me-1`; if (marketSpan) marketSpan.textContent = `Market ${displayStatus}`; } else if (marketStatusBadge) { marketStatusBadge.className = 'market-status-badge badge rounded-pill px-3 py-1 bg-secondary text-white'; if (marketIcon) marketIcon.className = 'fas fa-question-circle me-1'; if (marketSpan) marketSpan.textContent = 'Market Unknown'; } console.log("Stock header info updated."); }

// --- Flash Messages Utility ---
function showFlashMessage(message, category = 'info', duration = 5000) { /* ... as before ... */ console.log(`Flash (${category}): ${message}`); const container = document.querySelector('.flash-messages-container') || createFlashContainer(); if (!container) { console.error("Flash message container not found."); return; } const alertDiv = document.createElement('div'); const alertClass = `alert-${category === 'error' ? 'danger' : category}`; const iconClass = category === 'danger' || category === 'error' ? 'fa-exclamation-triangle' : category === 'warning' ? 'fa-exclamation-circle' : category === 'success' ? 'fa-check-circle' : 'fa-info-circle'; alertDiv.className = `alert ${alertClass} alert-dismissible fade show shadow-sm mb-2`; alertDiv.setAttribute('role', 'alert'); alertDiv.innerHTML = `<i class="fas ${iconClass} me-2"></i>${message}<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>`; container.appendChild(alertDiv); setTimeout(() => { try { const bsAlert = bootstrap.Alert.getInstance(alertDiv); if (bsAlert) bsAlert.close(); else fadeOutAndRemove(alertDiv); } catch (e) { fadeOutAndRemove(alertDiv); } }, duration); }
function createFlashContainer() { /* ... as before ... */ let c = document.querySelector('.flash-messages-container'); if (!c) { c = document.createElement('div'); c.className = 'flash-messages-container position-fixed top-0 end-0 p-3'; c.style.zIndex = '1060'; document.body.appendChild(c); } return c; }
function fadeOutAndRemove(element) { /* ... as before ... */ if (!element || !element.parentNode) return; element.style.transition = 'opacity 0.5s ease'; element.style.opacity = '0'; setTimeout(() => { if (element.parentNode) element.parentNode.removeChild(element); }, 500); }

// --- Initialize Detail Charts ---
/** Initializes charts shown in the lower part of the overview tab. */
function initializeDetailCharts() {
    // Check for libraries first
    if (typeof Chart === 'undefined') { console.warn("initializeDetailCharts: Chart.js not ready."); return; }
    // Check for data
    if (!window.stockData || window.stockData.error) { console.warn("initializeDetailCharts: Stock data unavailable."); /* Show placeholder messages */ const detailContainers = document.querySelectorAll('.details-chart-container'); detailContainers.forEach(container => { container.innerHTML = `<div class="alert alert-light text-center small p-2 m-0">Data not available.</div>`; }); return; }

    console.log("Initializing detail charts...");
    createOpenCloseChart(window.stockData);
    createDetailVolumeChart(window.stockData);
    createHighLowChart(window.stockData);
    console.log("Detail charts initialized.");
}


// --- DOMContentLoaded Event Listener ---
document.addEventListener('DOMContentLoaded', function () {
    console.log("script.js: DOMContentLoaded event fired.");

    // Wait for Chart.js (and optionally Luxon) to load
    const maxWaitTime = 5000; const checkInterval = 100; let elapsedTime = 0;
    const libraryCheckInterval = setInterval(function () {
        elapsedTime += checkInterval;
        const chartJsReady = typeof Chart !== 'undefined';
        const luxonReady = typeof luxon !== 'undefined';

        if (chartJsReady) { // Only strictly wait for Chart.js
            clearInterval(libraryCheckInterval);
            console.log("script.js: Chart.js loaded.");
            if (!luxonReady) console.warn("script.js: Luxon not loaded, using native Date for formatting.");
            initializePageCharts(); // Initialize charts and listeners
        } else if (elapsedTime >= maxWaitTime) {
            clearInterval(libraryCheckInterval);
            console.error("script.js: Timeout waiting for Chart.js.");
            displayLibraryLoadError();
        }
    }, checkInterval);

    /** Initializes charts and attaches event listeners after libraries are ready. */
    function initializePageCharts() {
        if (window.stockData && !window.stockData.error) {
            console.log("Initializing overview charts.");
            // *** DÜZELTME: Başlangıçta mum grafiği göster ***
            createOrUpdatePriceChart(window.stockData, 'candlestick');
            createOrUpdateVolumeChart(window.stockData);
            updateStockInfo(window.stockData);
            // Initialize detail charts if overview tab is active
            const overviewTabLink = document.getElementById('overview-tab');
            if (overviewTabLink?.classList.contains('active')) {
                console.log("Overview tab active on load, initializing detail charts.");
                initializeDetailCharts();
            } else {
                console.log("Overview tab not active initially.");
            }
        } else {
            console.log("No initial valid stock data for charts.");
            updateStockInfo(window.stockData); // Update header to show N/A
        }
        // Attach event listeners regardless of initial data state
        attachEventListeners();
    }

    /** Displays error messages if core libraries fail to load. */
    function displayLibraryLoadError() {
        showFlashMessage("Error: Charting libraries failed to load.", "danger", 10000);
        const chartContainers = document.querySelectorAll('.chart-container');
        chartContainers.forEach(container => {
            container.innerHTML = `<div class="alert alert-danger text-center small p-2 m-0">Charting libraries failed. Refresh.</div>`;
        });
    }

    /** Attaches main event listeners. */
    function attachEventListeners() {
        // Chart type selector
        const chartTypeRadios = document.querySelectorAll('.chart-type-selector input[name="chartType"]');
        chartTypeRadios.forEach(radio => {
            radio.addEventListener('change', function () {
                if (this.checked && window.stockData && !window.stockData.error) {
                    console.log(`Chart type changed to: ${this.dataset.chartType}`);
                    // *** DÜZELTME: Zorlamayı kaldır, seçilen tipi kullan ***
                    createOrUpdatePriceChart(window.stockData, this.dataset.chartType);
                }
            });
        });
        // Refresh button
        const refreshButton = document.getElementById('refreshStockBtn');
        if (refreshButton) { refreshButton.addEventListener('click', refreshStockData); }
        // Overview tab shown listener
        const overviewTabLinkForListener = document.getElementById('overview-tab');
        if (overviewTabLinkForListener) { overviewTabLinkForListener.addEventListener('shown.bs.tab', initializeDetailCharts); }
        // Tooltips
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        if (tooltipTriggerList.length > 0) { tooltipTriggerList.map(function (tooltipTriggerEl) { return new bootstrap.Tooltip(tooltipTriggerEl); }); }
        console.log("script.js: Event listeners attached.");
    }

}); // End DOMContentLoaded