// static/js/script.js
console.log("Script.js yüklendi.");

// --- Global Chart Instances ---
window.priceChartInstance = null;
window.volumeChartInstance = null;
window.openCloseChartInstance = null;
window.detailVolumeChartInstance = null;
window.highLowChartInstance = null;

// --- Helper Functions ---
function formatDate(dateStr) {
    if (!dateStr) return null;
    // If it's already a number (timestamp), return it
    if (typeof dateStr === 'number' && !isNaN(dateStr)) return dateStr;
    // If it's a Date object
    if (dateStr instanceof Date && !isNaN(dateStr.getTime())) return dateStr.getTime();
    // If it's a string, try parsing
    if (typeof dateStr === 'string') {
        try {
            // *** Primary check: ISO 8601 (as sent by backend) ***
            let dt = luxon.DateTime.fromISO(dateStr, { zone: 'utc' });
            if (dt.isValid) return dt.valueOf(); // Return timestamp

            // Fallback attempts (less likely needed now but safe)
            dt = luxon.DateTime.fromSQL(dateStr);
            if (dt.isValid) return dt.valueOf();
            dt = luxon.DateTime.fromFormat(dateStr, 'yyyy-MM-dd HH:mm:ss');
            if (dt.isValid) return dt.valueOf();
             dt = luxon.DateTime.fromFormat(dateStr, 'yyyy-MM-dd');
             if (dt.isValid) return dt.startOf('day').valueOf(); // Use start of day for dates without time

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
    // Add console log for debugging input
    // console.log(`Parsing Chart Data for: ${labelName}`, { labels_count: labels?.length, values_count: values?.length });

    if (!labels || !values || labels.length !== values.length) {
        console.warn(`parseChartData (${labelName}): Mismatch or missing labels/values.`);
        return []; // Return empty array if data is fundamentally flawed
    }
    const chartData = [];
    let invalidCount = 0;
    for (let i = 0; i < labels.length; i++) {
        const timestamp = formatDate(labels[i]); // Get timestamp (or null)
        // Ensure value is a number or null
        let value = (values[i] === null || values[i] === undefined) ? null : parseFloat(values[i]);
        if (isNaN(value)) { value = null; } // Treat non-numeric as null

        // Only add point if timestamp is valid (value can be null for gaps in line charts)
        if (timestamp !== null) {
            chartData.push({ x: timestamp, y: value });
        } else {
            invalidCount++;
            // console.log(`parseChartData (${labelName}): Invalid timestamp for label: ${labels[i]}`);
        }
    }
    if (invalidCount > 0) {
        console.warn(`parseChartData (${labelName}): Skipped ${invalidCount} data points due to invalid timestamps.`);
    }
    // console.log(`parseChartData (${labelName}): Parsed ${chartData.length} valid points.`);
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
         // Check structure more carefully
         if (!d || typeof d !== 'object' || !d.t || !('o' in d) || !('h' in d) || !('l' in d) || !('c' in d)) {
             invalidCount++;
             // console.log("parseCandlestickData: Skipping invalid candle structure:", d);
             continue;
         }
         const timestamp = formatDate(d.t); // Get timestamp (or null)
         // Ensure OHLC values are valid numbers
         const o = Number(d.o); const h = Number(d.h); const l = Number(d.l); const c = Number(d.c);

         if (timestamp !== null && !isNaN(o) && !isNaN(h) && !isNaN(l) && !isNaN(c)) {
             result.push({ x: timestamp, o: o, h: h, l: l, c: c });
         } else {
             invalidCount++;
             // console.log(`parseCandlestickData: Skipping candle due to invalid date or OHLC value: t=${d.t}, o=${d.o}, h=${d.h}, l=${d.l}, c=${d.c}`);
         }
    }
    if (invalidCount > 0) {
        console.warn(`parseCandlestickData: Skipped ${invalidCount} invalid candle data points.`);
    }
     // console.log(`parseCandlestickData: Parsed ${result.length} valid candles.`);
    return result;
}


// Common Chart Options (using Chart.js v3 syntax)
const commonChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
        mode: 'index', // Show tooltips for all datasets at that index
        intersect: false, // Tooltip activates even if not directly hovering over point/bar
    },
    scales: {
        x: {
            type: 'time', // Use the time scale
            time: {
                unit: 'day', // Display unit
                tooltipFormat: 'DD MMM YYYY HH:mm', // Format for tooltips (Luxon format)
                displayFormats: { // How to display ticks on the axis
                    day: 'DD MMM', // e.g., 17 Apr
                    month: 'MMM YYYY', // e.g., Apr 2025
                    year: 'YYYY'
                }
            },
            grid: {
                display: false // Hide vertical grid lines
            },
            ticks: {
                autoSkip: true, // Automatically skip labels to prevent overlap
                maxTicksLimit: 10, // Max number of ticks shown
                source: 'auto', // Use 'auto' or 'data' for tick generation
                color: '#6c757d' // Tick label color
            }
        },
        y: {
            beginAtZero: false, // Don't force axis to start at 0
            grid: {
                color: 'rgba(200, 200, 200, 0.1)' // Light horizontal grid lines
            },
            ticks: {
                color: '#6c757d', // Tick label color
                 // Format Y-axis ticks as currency (example)
                 callback: function(value) {
                     // Ensure it's a number before formatting
                     if (typeof value === 'number') {
                         // You might want to adjust formatting based on currency from stockData
                         return value.toLocaleString(undefined, { style: 'currency', currency: window.stockData?.currency || 'USD', minimumFractionDigits: 2 });
                     }
                     return value; // Return as is if not a number
                 }
            }
        }
    },
    plugins: {
        legend: {
            display: true, // Show legend by default
            labels: {
                color: '#2c3e50' // Legend text color
            }
        },
        tooltip: {
            enabled: true, // Enable tooltips
            backgroundColor: 'rgba(0, 0, 0, 0.8)', // Tooltip background
            titleColor: '#ffffff', // Tooltip title color
            bodyColor: '#ffffff', // Tooltip body color
            padding: 10, // Padding inside tooltip
            // Optional: customize tooltip title/label
             callbacks: {
                 title: function(tooltipItems) {
                     // Format the date title
                     const firstItem = tooltipItems[0];
                     if (firstItem && firstItem.parsed) {
                         return luxon.DateTime.fromMillis(firstItem.parsed.x).toFormat('DD MMM YYYY HH:mm');
                     }
                     return '';
                 },
                 label: function(context) {
                     let label = context.dataset.label || '';
                     if (label) { label += ': '; }
                     if (context.parsed.y !== null) {
                         // Format body labels as currency
                         label += context.parsed.y.toLocaleString(undefined, { style: 'currency', currency: window.stockData?.currency || 'USD', minimumFractionDigits: 2 });
                     } else {
                         label += 'N/A';
                     }
                     return label;
                 }
             }
        }
    }
};

// --- Chart Creation/Update Functions ---
function createOrUpdateChart(canvasId, chartInstanceVar, config) {
    // Enhanced check for Chart.js library and adapter
    if (typeof Chart === 'undefined' || typeof Chart.registerables === 'undefined' || typeof Chart._adapters?._date === 'undefined' || typeof luxon === 'undefined') {
         console.error(`Error creating chart '#${canvasId}': Chart.js library or Luxon date adapter not fully loaded.`);
         const container = document.getElementById(canvasId)?.closest('.chart-container');
         if (container) {
             // Display a more user-friendly message
             container.innerHTML = `<div class="alert alert-warning text-center small p-2 m-0" role="alert">Chart libraries could not be loaded. Please refresh the page.</div>`;
         }
         return null; // Stop execution for this chart
    }

    const ctx = document.getElementById(canvasId);
    const container = ctx ? ctx.closest('.chart-container') : null;
    if (!ctx || !container) {
        console.warn(`Chart element '#${canvasId}' or its container not found.`);
        return null; // Cannot proceed
    }

    console.log(`Attempting to create/update chart: #${canvasId}`);

    // Destroy previous instance if it exists AND is a Chart object
    if (window[chartInstanceVar] && typeof window[chartInstanceVar].destroy === 'function') {
        console.log(`Destroying previous chart instance: ${chartInstanceVar}`);
        try {
            window[chartInstanceVar].destroy();
        } catch(e){
            console.warn(`Error destroying previous chart instance '${chartInstanceVar}':`, e);
        }
         window[chartInstanceVar] = null; // Ensure it's cleared
    }

    // Create new chart
    try {
        // Ensure canvas is clean before creating a new chart
        // container.innerHTML = ''; // This might remove the canvas itself if not careful
        // container.appendChild(ctx); // Re-append canvas if needed, or ensure it wasn't removed

        console.log(`Creating new Chart instance for #${canvasId} with type: ${config.type}`);
        window[chartInstanceVar] = new Chart(ctx.getContext('2d'), config);
        console.log(`Chart '#${canvasId}' created/updated successfully.`);
        return window[chartInstanceVar];
    } catch (e) {
        console.error(`Error creating/updating chart '#${canvasId}':`, e);
        // Provide error feedback in the chart container
        container.innerHTML = `<div class="alert alert-danger text-center small p-2 m-0" role="alert">Could not load chart: ${e.message}</div>`;
        window[chartInstanceVar] = null; // Ensure instance is null on error
        return null;
    }
}

function createOrUpdatePriceChart(stockData, chartType = 'candlestick') {
    const canvasId = 'priceChart';
    if (!stockData || typeof stockData !== 'object') {
        console.warn(`PriceChart: Invalid or missing stockData for #${canvasId}.`);
        // Optionally clear or show an error message on the chart
        createOrUpdateChart(canvasId, 'priceChartInstance', { type: 'line', data: { datasets: [] }, options: { plugins: { title: { display: true, text: 'No data available' } } } });
        return;
    }

    console.log(`Updating Price Chart (#${canvasId}) with type: ${chartType}`);

    let data, options = JSON.parse(JSON.stringify(commonChartOptions)); // Deep copy options
    let configType = chartType;
    options.plugins.title = { display: false }; // No separate title for main chart

    // Determine currency for Y-axis formatting
    const currencyCode = stockData.currency || 'USD';
    options.scales.y.ticks.callback = function(value) {
        return typeof value === 'number' ? value.toLocaleString(undefined, { style: 'currency', currency: currencyCode, minimumFractionDigits: 2 }) : value;
    };
    options.plugins.tooltip.callbacks.label = function(context) {
        let label = context.dataset.label || '';
        if (label) { label += ': '; }
        if (context.parsed.y !== null) {
            label += context.parsed.y.toLocaleString(undefined, { style: 'currency', currency: currencyCode, minimumFractionDigits: 2 });
        } else {
            label += 'N/A';
        }
        return label;
    };


    if (chartType === 'candlestick') {
        // Requires chartjs-chart-financial plugin
        configType = 'candlestick'; // Use the specific type for this plugin
        const candleData = parseCandlestickData(stockData.candlestick_data);
        if (candleData.length === 0) {
            console.warn(`PriceChart (#${canvasId}): No valid candlestick data. Falling back to line chart.`);
            createOrUpdatePriceChart(stockData, 'line'); // Retry with line chart
            return;
        }
        data = {
            datasets: [{
                label: stockData.company_name || 'Price',
                data: candleData,
                // Colors for financial chart (may need adjustment based on plugin version)
                color: {
                    up: 'rgba(25, 135, 84, 1)', // Green for up
                    down: 'rgba(220, 53, 69, 1)', // Red for down
                    unchanged: 'rgba(108, 117, 125, 1)', // Grey for unchanged
                },
                borderColor: 'rgba(0,0,0,0.5)', // Border color of candles
                borderWidth: 1
            }]
        };
        options.plugins.legend.display = false; // Hide legend for single dataset candlestick
        // Customize tooltip for candlestick
        options.plugins.tooltip.callbacks.label = function(context) {
            const raw = context.raw;
            if (raw && typeof raw.o === 'number') {
                const o = raw.o.toLocaleString(undefined, { style: 'currency', currency: currencyCode, minimumFractionDigits: 2 });
                const h = raw.h.toLocaleString(undefined, { style: 'currency', currency: currencyCode, minimumFractionDigits: 2 });
                const l = raw.l.toLocaleString(undefined, { style: 'currency', currency: currencyCode, minimumFractionDigits: 2 });
                const c = raw.c.toLocaleString(undefined, { style: 'currency', currency: currencyCode, minimumFractionDigits: 2 });
                return [` Open: ${o}`, ` High: ${h}`, ` Low: ${l}`, ` Close: ${c}`];
            }
            return '';
        };

    } else { // Line chart
        configType = 'line';
        const lineData = parseChartData(stockData.labels, stockData.values, 'Close Price');
        if (lineData.length === 0) {
            console.warn(`PriceChart (#${canvasId}): No valid line data points.`);
            createOrUpdateChart(canvasId, 'priceChartInstance', { type: 'line', data: { datasets: [] }, options: { plugins: { title: { display: true, text: 'No price data available' } } } });
            return;
        }
        data = {
            datasets: [{
                label: 'Close Price',
                data: lineData,
                borderColor: '#0d6efd', // Blue line
                backgroundColor: 'rgba(13, 110, 253, 0.1)', // Light blue fill
                borderWidth: 1.5,
                tension: 0.1, // Slight curve
                pointRadius: 0, // No points on the line
                fill: true // Fill area under the line
            }]
        };
        options.plugins.legend.display = true; // Show legend for line chart
        // Tooltip callback is already set in common options, adjusted for currency
    }
    createOrUpdateChart(canvasId, 'priceChartInstance', { type: configType, data: data, options: options });
}


function createOrUpdateVolumeChart(stockData) { // Overview small volume chart
    const canvasId = 'volumeChart';
    if (!stockData || !stockData.labels || !stockData.volume_values) {
        console.warn(`VolumeChart (#${canvasId}): No volume data.`);
        createOrUpdateChart(canvasId, 'volumeChartInstance', { type: 'bar', data: { datasets: [] }, options: { plugins: { title: { display: true, text: 'No volume data' } } } });
        return;
    }
    console.log(`Updating Volume Chart (#${canvasId})`);

    const volumeData = parseChartData(stockData.labels, stockData.volume_values, 'Volume');
    if (volumeData.length === 0) {
        console.warn(`VolumeChart (#${canvasId}): No valid volume data points.`);
         createOrUpdateChart(canvasId, 'volumeChartInstance', { type: 'bar', data: { datasets: [] }, options: { plugins: { title: { display: true, text: 'No volume data' } } } });
        return;
    }

    const prices = stockData.values || []; // Get close prices for coloring bars
    const barColors = volumeData.map((dp, i) => {
        if (i > 0 && prices[i] !== null && prices[i-1] !== null) {
            return prices[i] >= prices[i-1] ? 'rgba(25, 135, 84, 0.6)' : 'rgba(220, 53, 69, 0.6)'; // Green if up, Red if down
        }
        return 'rgba(108, 117, 125, 0.6)'; // Grey otherwise
    });

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: { type: 'time', display: false }, // Hide X axis
            y: { beginAtZero: true, display: false } // Hide Y axis
        },
        plugins: {
            legend: { display: false }, // No legend needed
            tooltip: { enabled: false } // Disable tooltips for this simple chart
        }
    };

    createOrUpdateChart(canvasId, 'volumeChartInstance', {
        type: 'bar',
        data: {
            datasets: [{
                label: 'Volume',
                data: volumeData,
                backgroundColor: barColors,
                borderWidth: 0 // No border for bars
            }]
        },
        options: options
    });
}

// --- Detail Charts (Called when Overview Tab is shown) ---

function createOpenCloseChart(stockData) {
    const canvasId = 'openCloseChart';
     if (!stockData || !stockData.labels || !stockData.open_values || !stockData.values) {
        console.warn(`OpenCloseChart (#${canvasId}): Missing required data.`);
        createOrUpdateChart(canvasId, 'openCloseChartInstance', {type:'line', data:{datasets:[]}, options:{plugins: { title: { display: true, text: 'Open/Close data unavailable' }}}});
        return;
    }
    console.log(`Creating/Updating Open/Close Chart (#${canvasId})`);

    const openData = parseChartData(stockData.labels, stockData.open_values, 'Open');
    const closeData = parseChartData(stockData.labels, stockData.values, 'Close');
    if (openData.length === 0 && closeData.length === 0) { // Check if *both* are empty
        console.warn(`OpenCloseChart (#${canvasId}): No valid data points for Open or Close.`);
        createOrUpdateChart(canvasId, 'openCloseChartInstance', {type:'line', data:{datasets:[]}, options:{plugins: { title: { display: true, text: 'Open/Close data unavailable' }}}});
        return;
    }

    const options = JSON.parse(JSON.stringify(commonChartOptions)); // Deep copy
    const currencyCode = stockData.currency || 'USD';
    // Adjust Y-axis and tooltip for currency
    options.scales.y.ticks.callback = function(value) {
        return typeof value === 'number' ? value.toLocaleString(undefined, { style: 'currency', currency: currencyCode, minimumFractionDigits: 2 }) : value;
    };
     options.plugins.tooltip.callbacks.label = function(context) {
         let label = context.dataset.label || '';
         if (label) { label += ': '; }
         if (context.parsed.y !== null) {
             label += context.parsed.y.toLocaleString(undefined, { style: 'currency', currency: currencyCode, minimumFractionDigits: 2 });
         } else { label += 'N/A'; }
         return label;
     };
    options.plugins.legend.position = 'top';
    options.plugins.title = { display: false }; // Use card title instead

    createOrUpdateChart(canvasId, 'openCloseChartInstance', {
        type: 'line',
        data: {
            datasets: [
                { label: 'Open', data: openData, borderColor: 'rgba(255, 159, 64, 1)', borderWidth: 1.5, pointRadius: 0, tension: 0.1, fill: false },
                { label: 'Close', data: closeData, borderColor: 'rgba(54, 162, 235, 1)', borderWidth: 1.5, pointRadius: 0, tension: 0.1, fill: false }
            ]
        },
        options: options
    });
}

function createDetailVolumeChart(stockData) {
    const canvasId = 'detailVolumeChart';
    if (!stockData || !stockData.labels || !stockData.volume_values) {
        console.warn(`DetailVolumeChart (#${canvasId}): Missing required data.`);
        createOrUpdateChart(canvasId, 'detailVolumeChartInstance', {type:'bar', data:{datasets:[]}, options:{plugins: { title: { display: true, text: 'Volume data unavailable' }}}});
        return;
    }
     console.log(`Creating/Updating Detail Volume Chart (#${canvasId})`);

    const volumeData = parseChartData(stockData.labels, stockData.volume_values, 'Detail Volume');
    if (volumeData.length === 0) {
        console.warn(`DetailVolumeChart (#${canvasId}): No valid volume data points.`);
        createOrUpdateChart(canvasId, 'detailVolumeChartInstance', {type:'bar', data:{datasets:[]}, options:{plugins: { title: { display: true, text: 'Volume data unavailable' }}}});
        return;
    }

    const prices = stockData.values || []; // Get close prices for coloring bars
    const barColors = volumeData.map((dp, i) => {
        if (i > 0 && prices[i] !== null && prices[i-1] !== null) {
            return prices[i] >= prices[i-1] ? 'rgba(25, 135, 84, 0.6)' : 'rgba(220, 53, 69, 0.6)';
        }
        return 'rgba(108, 117, 125, 0.6)';
    });

    const options = JSON.parse(JSON.stringify(commonChartOptions)); // Deep copy
    // Format Y-axis ticks for large volume numbers (K, M, B)
    options.scales.y.ticks.callback = function(value) {
        if (typeof value === 'number') {
            if (value >= 1e9) return (value / 1e9).toFixed(1) + 'B';
            if (value >= 1e6) return (value / 1e6).toFixed(1) + 'M';
            if (value >= 1e3) return (value / 1e3).toFixed(0) + 'K';
            return value.toString(); // Keep small numbers as is
        }
        return value;
    };
    options.plugins.legend.display = false;
    options.plugins.title = { display: false };
     // Customize tooltip for volume
     options.plugins.tooltip.callbacks.label = function(context) {
         let label = context.dataset.label || '';
         if (label) { label += ': '; }
         if (context.parsed.y !== null) {
             const v = context.parsed.y;
             if (v >= 1e9) label += (v / 1e9).toFixed(2) + 'B';
             else if (v >= 1e6) label += (v / 1e6).toFixed(2) + 'M';
             else if (v >= 1e3) label += Math.round(v / 1e3) + 'K'; // Use Math.round for K
             else label += v.toLocaleString(); // Format smaller numbers
         } else { label += 'N/A'; }
         return label;
     };

    createOrUpdateChart(canvasId, 'detailVolumeChartInstance', {
        type: 'bar',
        data: {
            datasets: [{
                label: 'Volume',
                data: volumeData,
                backgroundColor: barColors,
                borderWidth: 0
            }]
        },
        options: options
    });
}

function createHighLowChart(stockData) {
    const canvasId = 'highLowChart';
    if (!stockData || !stockData.labels || !stockData.high_values || !stockData.low_values) {
         console.warn(`HighLowChart (#${canvasId}): Missing required data.`);
         createOrUpdateChart(canvasId, 'highLowChartInstance', {type:'line', data:{datasets:[]}, options:{plugins: { title: { display: true, text: 'High/Low data unavailable' }}}});
         return;
    }
    console.log(`Creating/Updating High/Low Chart (#${canvasId})`);

    const highData = parseChartData(stockData.labels, stockData.high_values, 'High');
    const lowData = parseChartData(stockData.labels, stockData.low_values, 'Low');
    if (highData.length === 0 && lowData.length === 0) {
        console.warn(`HighLowChart (#${canvasId}): No valid data points for High or Low.`);
        createOrUpdateChart(canvasId, 'highLowChartInstance', {type:'line', data:{datasets:[]}, options:{plugins: { title: { display: true, text: 'High/Low data unavailable' }}}});
        return;
    }

    const options = JSON.parse(JSON.stringify(commonChartOptions)); // Deep copy
    const currencyCode = stockData.currency || 'USD';
     // Adjust Y-axis and tooltip for currency
    options.scales.y.ticks.callback = function(value) {
        return typeof value === 'number' ? value.toLocaleString(undefined, { style: 'currency', currency: currencyCode, minimumFractionDigits: 2 }) : value;
    };
     options.plugins.tooltip.callbacks.label = function(context) {
         let label = context.dataset.label || '';
         if (label) { label += ': '; }
         if (context.parsed.y !== null) {
             label += context.parsed.y.toLocaleString(undefined, { style: 'currency', currency: currencyCode, minimumFractionDigits: 2 });
         } else { label += 'N/A'; }
         return label;
     };
    options.plugins.legend.position = 'top';
    options.plugins.title = { display: false };
    // Fill area between high and low lines
    options.plugins.filler = { propagate: false }; // Needed for filling between datasets
    options.interaction = { mode: 'nearest', axis: 'x', intersect: false }; // Tooltip interaction

    createOrUpdateChart(canvasId, 'highLowChartInstance', {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'Low', data: lowData,
                    borderColor: 'rgba(220, 53, 69, 0.8)', // Red line
                    backgroundColor: 'rgba(220, 53, 69, 0.1)', // Light red fill (won't show unless fill: true)
                    borderWidth: 1.5, pointRadius: 0, tension: 0.1,
                    fill: false // Don't fill this line itself
                },
                {
                    label: 'High', data: highData,
                    borderColor: 'rgba(25, 135, 84, 0.8)', // Green line
                    backgroundColor: 'rgba(35, 195, 105, 0.2)', // Lighter green fill
                    borderWidth: 1.5, pointRadius: 0, tension: 0.1,
                    fill: '-1' // Fill to the previous dataset ('Low')
                }
            ]
        },
        options: options
    });
}


// --- AJAX Data Refresh ---
function refreshStockData() {
    const stockSymbolInput = document.querySelector('#stock'); // Assuming input ID is 'stock'
    const refreshButton = document.getElementById('refreshStockBtn');
    const refreshIcon = refreshButton ? refreshButton.querySelector('i') : null;

    if (!stockSymbolInput || !stockSymbolInput.value) {
        console.warn("refreshStockData: Stock symbol input not found or empty.");
        showFlashMessage("Cannot refresh: Stock symbol not selected.", "warning");
        return;
    }
    const stockSymbol = stockSymbolInput.value.trim().toUpperCase();

    if (refreshIcon) {
        refreshIcon.classList.remove('fa-sync');
        refreshIcon.classList.add('fa-spinner', 'fa-spin');
        if(refreshButton) refreshButton.disabled = true;
    }
    console.log(`Refreshing data for ${stockSymbol}...`);
    showFlashMessage(`Refreshing data for ${stockSymbol}...`, "info", 2000); // Short info message

    fetch(`/refresh_stock?stock=${stockSymbol}`)
        .then(response => {
            if (!response.ok) {
                // Try to parse error message from JSON response
                return response.json().then(errData => {
                    throw new Error(errData.message || `Network error ${response.status}`);
                }).catch(() => {
                    // If JSON parsing fails, throw generic error
                    throw new Error(`Network error ${response.status}`);
                });
            }
            return response.json(); // Parse successful JSON response
        })
        .then(data => {
            if (data.status === 'success' && data.stock_data) {
                console.log("Data refreshed successfully via AJAX.");
                window.stockData = data.stock_data; // Update global data object

                // Update UI elements that depend on stockData
                updateStockInfo(window.stockData); // Update header badges/prices

                // Determine active chart type and update main price chart
                const activeChartType = document.querySelector('.chart-type-selector input[name="chartType"]:checked')?.dataset.chartType || 'candlestick';
                createOrUpdatePriceChart(window.stockData, activeChartType);

                // Update the small volume chart
                createOrUpdateVolumeChart(window.stockData);

                // Re-initialize detail charts (assuming they are visible or might become visible)
                // Check if the overview tab is currently active before initializing
                 const overviewTab = document.getElementById('overview');
                 if (overviewTab && overviewTab.classList.contains('active')) {
                     console.log("Refreshing detail charts as overview tab is active.");
                     initializeDetailCharts();
                 } else {
                     console.log("Overview tab not active, detail charts will initialize when tab is shown.");
                 }


                // Update the timestamp display
                const timestampEl = document.querySelector('.last-updated');
                 if (timestampEl && window.stockData.timestamp) {
                     try {
                         const dt = luxon.DateTime.fromISO(window.stockData.timestamp).setZone('local'); // Display in local time
                         timestampEl.textContent = dt.toFormat('dd MMM yyyy HH:mm:ss'); // Format timestamp
                     } catch (e) {
                         console.error("Error formatting timestamp:", e);
                         timestampEl.textContent = window.stockData.timestamp; // Fallback to raw string
                     }
                 }
                showFlashMessage('Stock data updated successfully.', 'success');
            } else {
                // Handle errors reported by the backend in the JSON response
                console.error('Backend Error:', data.message);
                showFlashMessage(data.message || 'Failed to update data. Please try again.', 'danger');
            }
        })
        .catch(error => {
            // Handle network errors or errors during JSON parsing
            console.error('AJAX Fetch Error:', error);
            showFlashMessage(`Error updating data: ${error.message}`, 'danger');
        })
        .finally(() => {
            // Always re-enable button and reset icon
            if (refreshIcon) {
                refreshIcon.classList.remove('fa-spinner', 'fa-spin');
                refreshIcon.classList.add('fa-sync');
                if(refreshButton) refreshButton.disabled = false;
            }
             console.log(`Refresh action for ${stockSymbol} completed.`);
        });
}

// --- Update Header Info ---
function updateStockInfo(stockData) {
    console.log("Updating stock header info...");
    if (!stockData || typeof stockData !== 'object') {
        console.warn("updateStockInfo: Invalid or missing stockData.");
        // Reset header elements to N/A if data is invalid
        document.querySelector('.current-price').textContent = 'N/A';
        document.querySelector('.change-percent').textContent = 'N/A';
        document.querySelector('.change-percent').className = 'change-percent small text-muted d-block'; // Reset class
         const marketStatusBadge = document.querySelector('.market-status-badge');
         if(marketStatusBadge) {
            marketStatusBadge.className = 'market-status-badge badge rounded-pill px-3 py-1 bg-secondary text-white';
            marketStatusBadge.querySelector('i').className = 'fas fa-question-circle me-1';
            marketStatusBadge.querySelector('span').textContent = 'Market Unknown';
         }
        return;
    }

    const priceEl = document.querySelector('.current-price');
    const changeEl = document.querySelector('.change-percent');
    const marketStatusBadge = document.querySelector('.market-status-badge');
    const marketIcon = marketStatusBadge ? marketStatusBadge.querySelector('i') : null;
    const marketSpan = marketStatusBadge ? marketStatusBadge.querySelector('span') : null;

    // Update Price
    const currencyCode = stockData.currency || 'USD';
    if (priceEl) {
        priceEl.textContent = (stockData.current_price !== null && stockData.current_price !== undefined)
            ? stockData.current_price.toLocaleString(undefined, { style: 'currency', currency: currencyCode, minimumFractionDigits: 2 })
            : 'N/A';
    }

    // Update Change Percentage
    if (changeEl) {
        const changePercent = stockData.change_percent;
        if (changePercent !== null && changePercent !== undefined) {
            const isPositive = changePercent >= 0;
            changeEl.classList.remove('text-success', 'text-danger', 'text-muted');
            changeEl.classList.add(isPositive ? 'text-success' : 'text-danger');
            changeEl.innerHTML = `<i class="fas ${isPositive ? 'fa-arrow-up' : 'fa-arrow-down'} me-1"></i>${changePercent.toFixed(2)}%`;
        } else {
            changeEl.innerHTML = 'N/A'; // Use innerHTML to clear potential icon
            changeEl.className = 'change-percent small text-muted d-block'; // Reset class
        }
    }

    // Update Market Status Badge
    if (marketStatusBadge && stockData.market_status) {
        const status = stockData.market_status.toUpperCase(); // Normalize to uppercase
        const status_lower = status.toLowerCase(); // For easier checking

        // Determine status category
        let badge_bg = 'bg-info'; // Default for unknown/other statuses
        let badge_text = 'text-white';
        let badge_icon = 'fa-question-circle';
        let displayStatus = status.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()); // Title case

         if (status_lower.includes('reg') || status_lower.includes('open')) { // Regular market hours
             badge_bg = 'bg-success'; badge_text = 'text-white'; badge_icon = 'fa-check-circle'; displayStatus = 'Open';
         } else if (status_lower.includes('pre')) { // Pre-market
             badge_bg = 'bg-warning'; badge_text = 'text-dark'; badge_icon = 'fa-hourglass-start'; displayStatus = 'Pre-Market';
         } else if (status_lower.includes('post') || status_lower.includes('extended') || status_lower.includes('after')) { // Post-market / After hours
             badge_bg = 'bg-warning'; badge_text = 'text-dark'; badge_icon = 'fa-hourglass-end'; displayStatus = 'After Hours';
         } else if (status_lower.includes('closed')) { // Closed
             badge_bg = 'bg-secondary'; badge_text = 'text-white'; badge_icon = 'fa-times-circle'; displayStatus = 'Closed';
         }
         // Apply classes
         marketStatusBadge.className = `market-status-badge badge rounded-pill px-3 py-1 ${badge_bg} ${badge_text}`;
         if (marketIcon) marketIcon.className = `fas ${badge_icon} me-1`;
         if (marketSpan) marketSpan.textContent = `Market ${displayStatus}`;

    } else if (marketStatusBadge) {
         // Fallback if market status is missing
         marketStatusBadge.className = 'market-status-badge badge rounded-pill px-3 py-1 bg-secondary text-white';
         if (marketIcon) marketIcon.className = 'fas fa-question-circle me-1';
         if (marketSpan) marketSpan.textContent = 'Market Unknown';
    }
     console.log("Stock header info updated.");
}

// --- Flash Messages Utility ---
function showFlashMessage(message, category = 'info', duration = 5000) {
     console.log(`Flash (${category}): ${message}`);
     const container = document.querySelector('.flash-messages-container') || createFlashContainer();
     if (!container) {
         console.error("Flash message container not found and could not be created.");
         return;
     }

     const alertDiv = document.createElement('div');
     // Map category to Bootstrap alert class
     const alertClass = `alert-${category === 'error' ? 'danger' : category}`; // Map 'error' to 'danger'
     const iconClass = category === 'danger' || category === 'error' ? 'fa-exclamation-triangle'
                      : category === 'warning' ? 'fa-exclamation-circle'
                      : category === 'success' ? 'fa-check-circle'
                      : 'fa-info-circle'; // Default to info icon

     alertDiv.className = `alert ${alertClass} alert-dismissible fade show shadow-sm`;
     alertDiv.setAttribute('role', 'alert');
     alertDiv.innerHTML = `
         <i class="fas ${iconClass} me-2"></i>
         ${message}
         <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
     `;

     container.appendChild(alertDiv);

     // Auto-dismiss after duration
     setTimeout(() => {
         // Use Bootstrap's dismiss method if available, otherwise fade out manually
         try {
             const bsAlert = bootstrap.Alert.getInstance(alertDiv);
             if (bsAlert) {
                 bsAlert.close();
             } else {
                 // Fallback fade out if Bootstrap JS isn't loaded or fails
                 fadeOutAndRemove(alertDiv);
             }
         } catch (e) {
              console.warn("Error dismissing flash message, using manual fade.", e);
              fadeOutAndRemove(alertDiv);
         }
     }, duration);
}
// Helper to create the flash container if it doesn't exist
function createFlashContainer() {
    let container = document.querySelector('.flash-messages-container');
    if (!container) {
        console.log("Creating flash message container.");
        container = document.createElement('div');
        container.className = 'flash-messages-container position-fixed top-0 end-0 p-3';
        container.style.zIndex = '1060'; // Ensure it's above most elements
        document.body.appendChild(container);
    }
    return container;
}
// Helper for manual fade out (fallback)
function fadeOutAndRemove(element) {
    if (!element || !element.parentNode) return;
    element.style.transition = 'opacity 0.5s ease';
    element.style.opacity = '0';
    setTimeout(() => {
        if (element.parentNode) {
            element.parentNode.removeChild(element);
        }
    }, 500); // Match transition duration
}

// --- Initialize Detail Charts (Function to be called when tab is shown) ---
function initializeDetailCharts() {
    // Check if data is available before trying to create charts
    if (!window.stockData || window.stockData.error) {
        console.warn("Cannot initialize detail charts: window.stockData is missing, empty, or contains an error.");
        // Optionally clear existing detail chart containers or show messages
        const detailContainers = document.querySelectorAll('.details-chart-container');
         detailContainers.forEach(container => {
             container.innerHTML = `<div class="alert alert-light text-center small p-2 m-0">Data not available for detailed charts.</div>`;
         });
        return; // Stop initialization
    }

    // Check if Chart.js library is loaded (redundant check, but safe)
    if (typeof Chart === 'undefined' || typeof Chart.registerables === 'undefined' || typeof Chart._adapters?._date === 'undefined' || typeof luxon === 'undefined') {
        console.error("Cannot initialize detail charts: Chart.js library or Luxon adapter not ready.");
        const detailContainers = document.querySelectorAll('.details-chart-container');
         detailContainers.forEach(container => {
             container.innerHTML = `<div class="alert alert-warning text-center small p-2 m-0">Chart libraries failed to load.</div>`;
         });
        return; // Stop initialization
    }

    console.log("Initializing detail charts...");
    // Create the charts
    createOpenCloseChart(window.stockData);
    createDetailVolumeChart(window.stockData);
    createHighLowChart(window.stockData);
     console.log("Detail charts initialized.");
}

// --- DOMContentLoaded Event Listener ---
document.addEventListener('DOMContentLoaded', function() {
    console.log("script.js: DOMContentLoaded event fired.");

    // Initialize charts only if valid stock data is present
    if (window.stockData && !window.stockData.error) {
        console.log("script.js: Initializing overview charts with loaded data.");
        // Initialize main price chart (default to candlestick)
        createOrUpdatePriceChart(window.stockData, 'candlestick');
        // Initialize small volume chart
        createOrUpdateVolumeChart(window.stockData);
        // Update header info
        updateStockInfo(window.stockData);

        // Check if the overview tab is initially active to load detail charts
        const overviewTab = document.getElementById('overview'); // Check the *content* div
        const overviewTabLink = document.getElementById('overview-tab'); // Check the *link*
        // Use the link's active state as Bootstrap manages that on load/refresh
        if (overviewTabLink && overviewTabLink.classList.contains('active')) {
            console.log("Overview tab is active on load, initializing detail charts.");
             initializeDetailCharts();
        } else {
             console.log("Overview tab not initially active, detail charts will load when tab is selected.");
        }
    } else if (window.stockData && window.stockData.error) {
         console.log("script.js: Initial stock data contains an error. Charts not initialized.");
         // Flash message is likely already shown by Flask
         // updateStockInfo might show N/A based on the partial data
         updateStockInfo(window.stockData);
    }
     else {
        console.log("script.js: No initial stock data found. Welcome screen likely shown.");
        // No charts to initialize
    }

    // Event listener for chart type selector (Candlestick/Line)
    const chartTypeRadios = document.querySelectorAll('.chart-type-selector input[name="chartType"]');
    chartTypeRadios.forEach(radio => {
        radio.addEventListener('change', function() {
            if (this.checked && window.stockData && !window.stockData.error) {
                 console.log(`Chart type changed to: ${this.dataset.chartType}`);
                createOrUpdatePriceChart(window.stockData, this.dataset.chartType);
            } else if (!window.stockData || window.stockData.error) {
                 console.warn("Cannot change chart type: Stock data is missing or invalid.");
            }
        });
    });

    // Event listener for the refresh button
    const refreshButton = document.getElementById('refreshStockBtn');
    if (refreshButton) {
        refreshButton.addEventListener('click', refreshStockData);
        console.log("Refresh button event listener attached.");
    } else {
         console.warn("Refresh button (#refreshStockBtn) not found.");
    }

    // Event listener for the Overview tab link to initialize detail charts when shown
    const overviewTabLinkForListener = document.getElementById('overview-tab'); // Get the tab link
    if (overviewTabLinkForListener) {
        overviewTabLinkForListener.addEventListener('shown.bs.tab', function(event) {
            console.log("Overview tab shown event triggered.");
            // Check if the charts need initialization (e.g., if they haven't been created yet)
            // We can check if the chart instances are null or simply call initialize again.
            // Calling initializeDetailCharts is idempotent due to the checks inside it.
            initializeDetailCharts();
        });
         console.log("Event listener for 'shown.bs.tab' on overview tab attached.");
    } else {
        console.warn("Overview tab link (#overview-tab) not found for attaching shown.bs.tab listener.");
    }

    // Initialize Bootstrap Tooltips (if any are used)
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    if (tooltipTriggerList.length > 0) {
        tooltipTriggerList.map(function (tooltipTriggerEl) {
             return new bootstrap.Tooltip(tooltipTriggerEl);
        });
         console.log("Bootstrap tooltips initialized.");
    }

    console.log("script.js: Initial setup and event listeners complete.");
}); // End of DOMContentLoaded