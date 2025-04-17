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
    if (typeof dateStr === 'number' && !isNaN(dateStr)) return dateStr;
    if (dateStr instanceof Date && !isNaN(dateStr.getTime())) return dateStr.getTime();
    if (typeof dateStr === 'string') {
        try {
            if (typeof luxon === 'undefined' || !luxon.DateTime) {
                 const dt = new Date(dateStr);
                 if (!isNaN(dt.getTime())) {
                     return dt.getTime();
                 } else {
                      console.error("formatDate: Luxon not available and native Date parsing failed for:", dateStr);
                      return null;
                 }
            }
            let dt = luxon.DateTime.fromISO(dateStr, { zone: 'utc' });
            if (dt.isValid) return dt.valueOf();
            dt = luxon.DateTime.fromSQL(dateStr);
            if (dt.isValid) return dt.valueOf();
            dt = luxon.DateTime.fromFormat(dateStr, 'yyyy-MM-dd HH:mm:ss');
            if (dt.isValid) return dt.valueOf();
             dt = luxon.DateTime.fromFormat(dateStr, 'yyyy-MM-dd');
             if (dt.isValid) return dt.startOf('day').valueOf();
            console.warn(`formatDate: Could not parse date string with Luxon: ${dateStr}`);
             const nativeDt = new Date(dateStr);
             if (!isNaN(nativeDt.getTime())) {
                 console.warn("formatDate: Luxon parsing failed, using native Date as fallback.");
                 return nativeDt.getTime();
             } else {
                 console.error("formatDate: Luxon and native Date parsing failed for:", dateStr);
                 return null;
             }
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
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
        mode: 'index',
        intersect: false,
    },
    scales: {
        x: {
            type: 'linear', // Using linear axis for timestamps
            grid: { display: false },
            ticks: {
                autoSkip: true, maxTicksLimit: 10, color: '#6c757d',
                callback: function(value, index, ticks) {
                    try {
                         if (typeof luxon !== 'undefined' && luxon.DateTime) {
                             const format = ticks.length > 15 ? 'dd MMM' : 'dd MMM yy';
                             return luxon.DateTime.fromMillis(value).toFormat(format);
                         } else {
                             const dt = new Date(value);
                             return `${dt.getDate()} ${dt.toLocaleString('default', { month: 'short' })}`;
                         }
                    } catch(e) { console.warn("Error formatting x-axis tick:", e); return value; }
                }
            }
        },
        y: {
            beginAtZero: false,
            grid: { color: 'rgba(200, 200, 200, 0.1)' },
            ticks: { color: '#6c757d', callback: function(value) { if (typeof value === 'number') { return value.toLocaleString(undefined, { style: 'currency', currency: window.stockData?.currency || 'USD', minimumFractionDigits: 2 }); } return value; } }
        }
    },
    plugins: {
        legend: { display: true, labels: { color: '#2c3e50' } },
        tooltip: {
            enabled: true, backgroundColor: 'rgba(0, 0, 0, 0.8)', titleColor: '#ffffff', bodyColor: '#ffffff', padding: 10,
            callbacks: {
                title: function(tooltipItems) {
                    const firstItem = tooltipItems[0];
                    if (firstItem && firstItem.parsed) {
                         try {
                              if (typeof luxon !== 'undefined' && luxon.DateTime) { return luxon.DateTime.fromMillis(firstItem.parsed.x).toFormat('DD MMM YYYY HH:mm'); }
                              else { const dt = new Date(firstItem.parsed.x); return dt.toLocaleDateString() + ' ' + dt.toLocaleTimeString(); }
                         } catch(e) { console.warn("Error formatting tooltip title:", e); return firstItem.parsed.x; }
                    } return '';
                },
                label: function(context) {
                    let label = context.dataset.label || '';
                    if (label) { label += ': '; }
                    if (context.parsed.y !== null) { label += context.parsed.y.toLocaleString(undefined, { style: 'currency', currency: window.stockData?.currency || 'USD', minimumFractionDigits: 2 }); }
                    else { label += 'N/A'; }
                    return label;
                }
            }
        }
    }
};


// --- Chart Creation/Update Functions ---
function createOrUpdateChart(canvasId, chartInstanceVar, config) {
    if (typeof Chart === 'undefined' || (typeof luxon === 'undefined' && config?.scales?.x?.type !== 'linear')) { // Check if luxon needed
         console.error(`createOrUpdateChart: Attempted to create chart '#${canvasId}' but libraries are not ready.`);
         const container = document.getElementById(canvasId)?.closest('.chart-container');
         if (container) container.innerHTML = `<div class="alert alert-warning text-center small p-2 m-0" role="alert">Chart libraries failed to load. Please refresh.</div>`;
         return null;
    }
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

function createOrUpdatePriceChart(stockData, chartType = 'candlestick') {
    const canvasId = 'priceChart';
    if (!stockData || typeof stockData !== 'object') {
        console.warn(`PriceChart: Invalid or missing stockData for #${canvasId}.`);
        createOrUpdateChart(canvasId, 'priceChartInstance', { type: 'line', data: { datasets: [] }, options: { plugins: { title: { display: true, text: 'No data available' } } } });
        return;
    }
    console.log(`Updating Price Chart (#${canvasId})`);
    let data, options = JSON.parse(JSON.stringify(commonChartOptions));

    // *** TEST: Force line chart ***
    // let configType = 'line';
   //  console.warn("Price chart type forced to 'line' for testing."); // Add warning
    let configType = chartType; // Original line

    options.plugins.title = { display: false };
    const currencyCode = stockData.currency || 'USD';
    options.scales.y.ticks.callback = function(value) { return typeof value === 'number' ? value.toLocaleString(undefined, { style: 'currency', currency: currencyCode, minimumFractionDigits: 2 }) : value; };
    options.plugins.tooltip.callbacks.label = function(context) { /* ... (currency formatting) ... */ let label = context.dataset.label || ''; if (label) { label += ': '; } if (context.parsed.y !== null) { label += context.parsed.y.toLocaleString(undefined, { style: 'currency', currency: currencyCode, minimumFractionDigits: 2 }); } else { label += 'N/A'; } return label;};

    if (configType === 'candlestick') {
        configType = 'candlestick';
        const candleData = parseCandlestickData(stockData.candlestick_data);
        // *** DEBUG LOG FOR CANDLESTICK DATA ***
        console.log('Price Chart - Candlestick Data:', JSON.stringify(candleData.slice(0, 5)));
        if (candleData.length === 0) { console.warn(`PriceChart (#${canvasId}): No valid candlestick data. Falling back to line chart.`); createOrUpdatePriceChart(stockData, 'line'); return; }
        data = { datasets: [{ label: stockData.company_name || 'Price', data: candleData, color: { up: 'rgba(25, 135, 84, 1)', down: 'rgba(220, 53, 69, 1)', unchanged: 'rgba(108, 117, 125, 1)', }, borderColor: 'rgba(0,0,0,0.5)', borderWidth: 1 }] };
        options.plugins.legend.display = false;
        options.plugins.tooltip.callbacks.label = function(context) { /* ... (candlestick tooltip) ... */ const raw = context.raw; if (raw && typeof raw.o === 'number') { const o = raw.o.toLocaleString(undefined, { style: 'currency', currency: currencyCode, minimumFractionDigits: 2 }); const h = raw.h.toLocaleString(undefined, { style: 'currency', currency: currencyCode, minimumFractionDigits: 2 }); const l = raw.l.toLocaleString(undefined, { style: 'currency', currency: currencyCode, minimumFractionDigits: 2 }); const c = raw.c.toLocaleString(undefined, { style: 'currency', currency: currencyCode, minimumFractionDigits: 2 }); return [` Open: ${o}`, ` High: ${h}`, ` Low: ${l}`, ` Close: ${c}`]; } return ''; };
        // Note: Check if financial plugin works with linear axis; might need override here if not.
         console.log("Attempting to render Candlestick chart (check compatibility with linear axis).");
    } else { // Line chart
        configType = 'line';
        const lineData = parseChartData(stockData.labels, stockData.values, 'Close Price');
        // *** DEBUG LOG FOR LINE DATA ***
        console.log('Price Chart - Line Data:', JSON.stringify(lineData.slice(0, 5)));
        if (lineData.length === 0) { console.warn(`PriceChart (#${canvasId}): No valid line data points.`); createOrUpdateChart(canvasId, 'priceChartInstance', { type: 'line', data: { datasets: [] }, options: { plugins: { title: { display: true, text: 'No price data available' } } } }); return; }
        data = { datasets: [{ label: 'Close Price', data: lineData, borderColor: '#0d6efd', backgroundColor: 'rgba(13, 110, 253, 0.1)', borderWidth: 1.5, tension: 0.1, pointRadius: 0, fill: true }] };
        options.plugins.legend.display = true;
    }
    createOrUpdateChart(canvasId, 'priceChartInstance', { type: configType, data: data, options: options });
}

function createOrUpdateVolumeChart(stockData) {
    const canvasId = 'volumeChart';
    if (!stockData || !stockData.labels || !stockData.volume_values) { console.warn(`VolumeChart (#${canvasId}): No volume data.`); createOrUpdateChart(canvasId, 'volumeChartInstance', { type: 'bar', data: { datasets: [] }, options: { plugins: { title: { display: true, text: 'No volume data' } } } }); return; }
    console.log(`Updating Volume Chart (#${canvasId})`);
    const volumeData = parseChartData(stockData.labels, stockData.volume_values, 'Volume');
    // *** DEBUG LOG FOR VOLUME DATA ***
    console.log('Volume Chart Data:', JSON.stringify(volumeData.slice(0, 5)));
    if (volumeData.length === 0) { console.warn(`VolumeChart (#${canvasId}): No valid volume data points.`); createOrUpdateChart(canvasId, 'volumeChartInstance', { type: 'bar', data: { datasets: [] }, options: { plugins: { title: { display: true, text: 'No volume data' } } } }); return; }
    const prices = stockData.values || [];
    const barColors = volumeData.map((dp, i) => { if (i > 0 && prices[i] !== null && prices[i-1] !== null) { return prices[i] >= prices[i-1] ? 'rgba(25, 135, 84, 0.6)' : 'rgba(220, 53, 69, 0.6)'; } return 'rgba(108, 117, 125, 0.6)'; });
    const options = { responsive: true, maintainAspectRatio: false, scales: { x: { type: 'linear', display: false }, y: { beginAtZero: true, display: false } }, plugins: { legend: { display: false }, tooltip: { enabled: false } } };
    // Override x-axis type for volume chart to match price chart
    options.scales.x.type = commonChartOptions.scales.x.type;
    createOrUpdateChart(canvasId, 'volumeChartInstance', { type: 'bar', data: { datasets: [{ label: 'Volume', data: volumeData, backgroundColor: barColors, borderWidth: 0 }] }, options: options });
}

function createOpenCloseChart(stockData) {
    const canvasId = 'openCloseChart';
     if (!stockData || !stockData.labels || !stockData.open_values || !stockData.values) { console.warn(`OpenCloseChart (#${canvasId}): Missing required data.`); createOrUpdateChart(canvasId, 'openCloseChartInstance', {type:'line', data:{datasets:[]}, options:{plugins: { title: { display: true, text: 'Open/Close data unavailable' }}}}); return; }
    console.log(`Creating/Updating Open/Close Chart (#${canvasId})`);
    const openData = parseChartData(stockData.labels, stockData.open_values, 'Open');
    const closeData = parseChartData(stockData.labels, stockData.values, 'Close');
     // *** DEBUG LOG FOR OPEN/CLOSE DATA ***
     console.log('Open/Close Chart - Open Data:', JSON.stringify(openData.slice(0, 5)));
     console.log('Open/Close Chart - Close Data:', JSON.stringify(closeData.slice(0, 5)));
    if (openData.length === 0 && closeData.length === 0) { console.warn(`OpenCloseChart (#${canvasId}): No valid data points.`); createOrUpdateChart(canvasId, 'openCloseChartInstance', {type:'line', data:{datasets:[]}, options:{plugins: { title: { display: true, text: 'Open/Close data unavailable' }}}}); return; }
    const options = JSON.parse(JSON.stringify(commonChartOptions));
    const currencyCode = stockData.currency || 'USD';
    options.scales.y.ticks.callback = function(value) { return typeof value === 'number' ? value.toLocaleString(undefined, { style: 'currency', currency: currencyCode, minimumFractionDigits: 2 }) : value; };
     options.plugins.tooltip.callbacks.label = function(context) { let label = context.dataset.label || ''; if (label) { label += ': '; } if (context.parsed.y !== null) { label += context.parsed.y.toLocaleString(undefined, { style: 'currency', currency: currencyCode, minimumFractionDigits: 2 }); } else { label += 'N/A'; } return label; };
    options.plugins.legend.position = 'top'; options.plugins.title = { display: false };
    createOrUpdateChart(canvasId, 'openCloseChartInstance', { type: 'line', data: { datasets: [ { label: 'Open', data: openData, borderColor: 'rgba(255, 159, 64, 1)', borderWidth: 1.5, pointRadius: 0, tension: 0.1, fill: false }, { label: 'Close', data: closeData, borderColor: 'rgba(54, 162, 235, 1)', borderWidth: 1.5, pointRadius: 0, tension: 0.1, fill: false } ] }, options: options });
}

function createDetailVolumeChart(stockData) {
    const canvasId = 'detailVolumeChart';
    if (!stockData || !stockData.labels || !stockData.volume_values) { console.warn(`DetailVolumeChart (#${canvasId}): Missing required data.`); createOrUpdateChart(canvasId, 'detailVolumeChartInstance', {type:'bar', data:{datasets:[]}, options:{plugins: { title: { display: true, text: 'Volume data unavailable' }}}}); return; }
     console.log(`Creating/Updating Detail Volume Chart (#${canvasId})`);
    const volumeData = parseChartData(stockData.labels, stockData.volume_values, 'Detail Volume');
     // *** DEBUG LOG FOR DETAIL VOLUME DATA ***
     console.log('Detail Volume Chart Data:', JSON.stringify(volumeData.slice(0, 5)));
    if (volumeData.length === 0) { console.warn(`DetailVolumeChart (#${canvasId}): No valid volume data points.`); createOrUpdateChart(canvasId, 'detailVolumeChartInstance', {type:'bar', data:{datasets:[]}, options:{plugins: { title: { display: true, text: 'Volume data unavailable' }}}}); return; }
    const prices = stockData.values || [];
    const barColors = volumeData.map((dp, i) => { if (i > 0 && prices[i] !== null && prices[i-1] !== null) { return prices[i] >= prices[i-1] ? 'rgba(25, 135, 84, 0.6)' : 'rgba(220, 53, 69, 0.6)'; } return 'rgba(108, 117, 125, 0.6)'; });
    const options = JSON.parse(JSON.stringify(commonChartOptions));
    options.scales.y.ticks.callback = function(value) { if (typeof value === 'number') { if (value >= 1e9) return (value / 1e9).toFixed(1) + 'B'; if (value >= 1e6) return (value / 1e6).toFixed(1) + 'M'; if (value >= 1e3) return (value / 1e3).toFixed(0) + 'K'; return value.toString(); } return value; };
    options.plugins.legend.display = false; options.plugins.title = { display: false };
     options.plugins.tooltip.callbacks.label = function(context) { let label=context.dataset.label||''; if(label) l+=': '; if (context.parsed.y !== null) { const v=context.parsed.y; if(v>=1e9) label+=(v/1e9).toFixed(2)+'B'; else if(v>=1e6) label+=(v/1e6).toFixed(2)+'M'; else if(v>=1e3) label+=Math.round(v/1e3)+'K'; else label+=v.toLocaleString(); } return label; };
    createOrUpdateChart(canvasId, 'detailVolumeChartInstance', { type: 'bar', data: { datasets: [{ label: 'Volume', data: volumeData, backgroundColor: barColors, borderWidth: 0 }] }, options: options });
}

function createHighLowChart(stockData) {
    const canvasId = 'highLowChart';
    if (!stockData || !stockData.labels || !stockData.high_values || !stockData.low_values) { console.warn(`HighLowChart (#${canvasId}): Missing required data.`); createOrUpdateChart(canvasId, 'highLowChartInstance', {type:'line', data:{datasets:[]}, options:{plugins: { title: { display: true, text: 'High/Low data unavailable' }}}}); return; }
    console.log(`Creating/Updating High/Low Chart (#${canvasId})`);
    const highData = parseChartData(stockData.labels, stockData.high_values, 'High');
    const lowData = parseChartData(stockData.labels, stockData.low_values, 'Low');
     // *** DEBUG LOG FOR HIGH/LOW DATA ***
     console.log('High/Low Chart - High Data:', JSON.stringify(highData.slice(0, 5)));
     console.log('High/Low Chart - Low Data:', JSON.stringify(lowData.slice(0, 5)));
    if (highData.length === 0 && lowData.length === 0) { console.warn(`HighLowChart (#${canvasId}): No valid data points.`); createOrUpdateChart(canvasId, 'highLowChartInstance', {type:'line', data:{datasets:[]}, options:{plugins: { title: { display: true, text: 'High/Low data unavailable' }}}}); return; }
    const options = JSON.parse(JSON.stringify(commonChartOptions));
    const currencyCode = stockData.currency || 'USD';
    options.scales.y.ticks.callback = function(value) { return typeof value === 'number' ? value.toLocaleString(undefined, { style: 'currency', currency: currencyCode, minimumFractionDigits: 2 }) : value; };
     options.plugins.tooltip.callbacks.label = function(context) { let label=context.dataset.label||''; if(label) l+=': '; l += (context.parsed.y !== null) ? context.parsed.y.toLocaleString(undefined, { style: 'currency', currency: currencyCode, minimumFractionDigits: 2 }) : 'N/A'; return label; };
    options.plugins.legend.position = 'top'; options.plugins.title = { display: false }; options.plugins.filler = { propagate: false }; options.interaction = { mode: 'nearest', axis: 'x', intersect: false };
    createOrUpdateChart(canvasId, 'highLowChartInstance', { type: 'line', data: { datasets: [ { label: 'Low', data: lowData, borderColor: 'rgba(220, 53, 69, 0.8)', backgroundColor: 'rgba(220, 53, 69, 0.1)', borderWidth: 1.5, pointRadius: 0, tension: 0.1, fill: false }, { label: 'High', data: highData, borderColor: 'rgba(25, 135, 84, 0.8)', backgroundColor: 'rgba(35, 195, 105, 0.2)', borderWidth: 1.5, pointRadius: 0, tension: 0.1, fill: '-1' } ] }, options: options });
}


// --- AJAX Data Refresh ---
function refreshStockData() {
    const stockSymbolInput = document.querySelector('#stock');
    const refreshButton = document.getElementById('refreshStockBtn');
    const refreshIcon = refreshButton ? refreshButton.querySelector('i') : null;
    if (!stockSymbolInput || !stockSymbolInput.value) { console.warn("refreshStockData: Stock symbol not found."); return; }
    const stockSymbol = stockSymbolInput.value.trim().toUpperCase();
    if (refreshIcon) { refreshIcon.classList.remove('fa-sync'); refreshIcon.classList.add('fa-spinner', 'fa-spin'); if(refreshButton) refreshButton.disabled = true; }
     console.log(`Refreshing data for ${stockSymbol}...`);
     showFlashMessage(`Refreshing data for ${stockSymbol}...`, "info", 2000);
    fetch(`/refresh_stock?stock=${stockSymbol}`)
        .then(response => { if (!response.ok) { return response.json().then(errData => { throw new Error(errData.message || `Network error ${response.status}`); }).catch(() => { throw new Error(`Network error ${response.status}`); }); } return response.json(); })
        .then(data => {
            if (data.status === 'success' && data.stock_data) {
                console.log("Data refreshed successfully via AJAX.");
                window.stockData = data.stock_data;
                updateStockInfo(window.stockData);
                const activeChartType = document.querySelector('.chart-type-selector input[name="chartType"]:checked')?.dataset.chartType || 'candlestick';
                // *** TEST: Force line chart on refresh too ***
                //createOrUpdatePriceChart(window.stockData, 'line');
                 createOrUpdatePriceChart(window.stockData, activeChartType); // Original line
                createOrUpdateVolumeChart(window.stockData);
                 const overviewTab = document.getElementById('overview');
                 if (overviewTab && overviewTab.classList.contains('active')) { console.log("Refreshing detail charts as overview tab is active."); initializeDetailCharts(); }
                 else { console.log("Overview tab not active, detail charts will initialize when tab is shown."); }
                const timestampEl = document.querySelector('.last-updated');
                 if (timestampEl && window.stockData.timestamp) {
                     try {
                         if (typeof luxon !== 'undefined') { const dt = luxon.DateTime.fromISO(window.stockData.timestamp).setZone('local'); timestampEl.textContent = dt.toFormat('dd MMM yyyy HH:mm:ss'); }
                         else { const dt = new Date(window.stockData.timestamp); timestampEl.textContent = dt.toLocaleString(); } // Fallback format
                     } catch (e) { console.error("Error formatting timestamp:", e); timestampEl.textContent = window.stockData.timestamp; }
                 }
                showFlashMessage('Stock data updated successfully.', 'success');
            } else { console.error('Backend Error:', data.message); showFlashMessage(data.message || 'Failed to update data.', 'danger'); }
        })
        .catch(error => { console.error('AJAX Fetch Error:', error); showFlashMessage(`Error updating data: ${error.message}`, 'danger'); })
        .finally(() => { if (refreshIcon) { refreshIcon.classList.remove('fa-spinner', 'fa-spin'); refreshIcon.classList.add('fa-sync'); if(refreshButton) refreshButton.disabled = false; } console.log(`Refresh action for ${stockSymbol} completed.`); });
}

// --- Update Header Info ---
function updateStockInfo(stockData) {
    console.log("Updating stock header info...");
    if (!stockData || typeof stockData !== 'object') { console.warn("updateStockInfo: Invalid or missing stockData."); document.querySelector('.current-price').textContent = 'N/A'; document.querySelector('.change-percent').textContent = 'N/A'; document.querySelector('.change-percent').className = 'change-percent small text-muted d-block'; const marketStatusBadge = document.querySelector('.market-status-badge'); if(marketStatusBadge) { marketStatusBadge.className = 'market-status-badge badge rounded-pill px-3 py-1 bg-secondary text-white'; marketStatusBadge.querySelector('i').className = 'fas fa-question-circle me-1'; marketStatusBadge.querySelector('span').textContent = 'Market Unknown'; } return; }
    const priceEl = document.querySelector('.current-price'); const changeEl = document.querySelector('.change-percent'); const marketStatusBadge = document.querySelector('.market-status-badge'); const marketIcon = marketStatusBadge ? marketStatusBadge.querySelector('i') : null; const marketSpan = marketStatusBadge ? marketStatusBadge.querySelector('span') : null;
    const currencyCode = stockData.currency || 'USD';
    if (priceEl) { priceEl.textContent = (stockData.current_price !== null && stockData.current_price !== undefined) ? stockData.current_price.toLocaleString(undefined, { style: 'currency', currency: currencyCode, minimumFractionDigits: 2 }) : 'N/A'; }
    if (changeEl) { const changePercent = stockData.change_percent; if (changePercent !== null && changePercent !== undefined) { const isPositive = changePercent >= 0; changeEl.classList.remove('text-success', 'text-danger', 'text-muted'); changeEl.classList.add(isPositive ? 'text-success' : 'text-danger'); changeEl.innerHTML = `<i class="fas ${isPositive ? 'fa-arrow-up' : 'fa-arrow-down'} me-1"></i>${changePercent.toFixed(2)}%`; } else { changeEl.innerHTML = 'N/A'; changeEl.className = 'change-percent small text-muted d-block'; } }
    if (marketStatusBadge && stockData.market_status) { const status = stockData.market_status.toUpperCase(); const status_lower = status.toLowerCase(); let badge_bg = 'bg-info'; let badge_text = 'text-white'; let badge_icon = 'fa-question-circle'; let displayStatus = status.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()); if (status_lower.includes('reg') || status_lower.includes('open')) { badge_bg = 'bg-success'; badge_text = 'text-white'; badge_icon = 'fa-check-circle'; displayStatus = 'Open'; } else if (status_lower.includes('pre')) { badge_bg = 'bg-warning'; badge_text = 'text-dark'; badge_icon = 'fa-hourglass-start'; displayStatus = 'Pre-Market'; } else if (status_lower.includes('post') || status_lower.includes('extended') || status_lower.includes('after')) { badge_bg = 'bg-warning'; badge_text = 'text-dark'; badge_icon = 'fa-hourglass-end'; displayStatus = 'After Hours'; } else if (status_lower.includes('closed')) { badge_bg = 'bg-secondary'; badge_text = 'text-white'; badge_icon = 'fa-times-circle'; displayStatus = 'Closed'; } marketStatusBadge.className = `market-status-badge badge rounded-pill px-3 py-1 ${badge_bg} ${badge_text}`; if (marketIcon) marketIcon.className = `fas ${badge_icon} me-1`; if (marketSpan) marketSpan.textContent = `Market ${displayStatus}`; }
    else if (marketStatusBadge) { marketStatusBadge.className = 'market-status-badge badge rounded-pill px-3 py-1 bg-secondary text-white'; if (marketIcon) marketIcon.className = 'fas fa-question-circle me-1'; if (marketSpan) marketSpan.textContent = 'Market Unknown'; }
     console.log("Stock header info updated.");
}

// --- Flash Messages Utility ---
function showFlashMessage(message, category = 'info', duration = 5000) { console.log(`Flash (${category}): ${message}`); const container = document.querySelector('.flash-messages-container') || createFlashContainer(); if (!container) { console.error("Flash message container not found."); return; } const alertDiv = document.createElement('div'); const alertClass = `alert-${category === 'error' ? 'danger' : category}`; const iconClass = category === 'danger' || category === 'error' ? 'fa-exclamation-triangle' : category === 'warning' ? 'fa-exclamation-circle' : category === 'success' ? 'fa-check-circle' : 'fa-info-circle'; alertDiv.className = `alert ${alertClass} alert-dismissible fade show shadow-sm`; alertDiv.setAttribute('role', 'alert'); alertDiv.innerHTML = `<i class="fas ${iconClass} me-2"></i>${message}<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>`; container.appendChild(alertDiv); setTimeout(() => { try { const bsAlert = bootstrap.Alert.getInstance(alertDiv); if (bsAlert) bsAlert.close(); else fadeOutAndRemove(alertDiv); } catch(e){ fadeOutAndRemove(alertDiv); } }, duration); }
function createFlashContainer() { let c=document.querySelector('.flash-messages-container'); if(!c){c=document.createElement('div'); c.className='flash-messages-container position-fixed top-0 end-0 p-3'; c.style.zIndex='1060'; document.body.appendChild(c);} return c; }
function fadeOutAndRemove(element) { if (!element || !element.parentNode) return; element.style.transition = 'opacity 0.5s ease'; element.style.opacity = '0'; setTimeout(() => { if (element.parentNode) element.parentNode.removeChild(element); }, 500); }

// --- Initialize Detail Charts ---
function initializeDetailCharts() {
    if (typeof Chart === 'undefined' || typeof luxon === 'undefined') { console.warn("initializeDetailCharts: Libraries not ready."); const detailContainers = document.querySelectorAll('.details-chart-container'); detailContainers.forEach(container => { container.innerHTML = `<div class="alert alert-light text-center small p-2 m-0">Charts cannot be displayed.</div>`; }); return; } // Simple message
    if (!window.stockData || window.stockData.error) { console.warn("Cannot initialize detail charts: window.stockData unavailable."); const detailContainers = document.querySelectorAll('.details-chart-container'); detailContainers.forEach(container => { container.innerHTML = `<div class="alert alert-light text-center small p-2 m-0">Data not available.</div>`; }); return; }
    console.log("Initializing detail charts...");
    createOpenCloseChart(window.stockData);
    createDetailVolumeChart(window.stockData);
    createHighLowChart(window.stockData);
     console.log("Detail charts initialized.");
}

// --- DOMContentLoaded Event Listener ---
document.addEventListener('DOMContentLoaded', function() {
    console.log("script.js: DOMContentLoaded event fired.");

    const maxWaitTime = 5000;
    const checkInterval = 100;
    let elapsedTime = 0;

    const libraryCheckInterval = setInterval(function() {
        elapsedTime += checkInterval;
        const chartJsReady = typeof Chart !== 'undefined';
        const luxonReady = typeof luxon !== 'undefined'; // Keep checking Luxon for formatting

        if (chartJsReady) { // Only strictly require Chart.js
            clearInterval(libraryCheckInterval);
            console.log("script.js: Chart.js library confirmed loaded.");
             if (!luxonReady) { console.warn("script.js: Luxon library not loaded, date formatting will use native Date object."); }
            initializePageCharts();
        } else if (elapsedTime >= maxWaitTime) {
            clearInterval(libraryCheckInterval);
            console.error("script.js: Timeout waiting for Chart.js library to load.");
            displayLibraryLoadError();
        }
    }, checkInterval);

    function initializePageCharts() {
        if (window.stockData && !window.stockData.error) {
            console.log("script.js: Initializing overview charts with loaded data.");
             // *** TEST: Force line chart initially ***
             createOrUpdatePriceChart(window.stockData, 'line');
            // createOrUpdatePriceChart(window.stockData, 'candlestick'); // Original line
            createOrUpdateVolumeChart(window.stockData);
            updateStockInfo(window.stockData);
            const overviewTabLink = document.getElementById('overview-tab');
            if (overviewTabLink && overviewTabLink.classList.contains('active')) { console.log("Overview tab is active on load, initializing detail charts."); initializeDetailCharts(); }
            else { console.log("Overview tab not initially active, detail charts will load when tab is selected."); }
        } else { console.log("script.js: No initial valid stock data found. Charts not initialized."); updateStockInfo(window.stockData); }
        attachEventListeners();
    }

    function displayLibraryLoadError() { showFlashMessage("Error: Charting libraries failed to load.", "danger", 10000); const chartContainers = document.querySelectorAll('.chart-container'); chartContainers.forEach(container => { container.innerHTML = `<div class="alert alert-danger text-center small p-2 m-0" role="alert">Charting libraries failed to load. Please refresh.</div>`; }); }

    function attachEventListeners() {
         const chartTypeRadios = document.querySelectorAll('.chart-type-selector input[name="chartType"]');
         chartTypeRadios.forEach(radio => {
             radio.addEventListener('change', function() {
                 if (this.checked && window.stockData && !window.stockData.error) {
                      console.log(`Chart type changed to: ${this.dataset.chartType}`);
                     // *** TEST: Force line chart on change too ***
                      createOrUpdatePriceChart(window.stockData, 'line');
                     // createOrUpdatePriceChart(window.stockData, this.dataset.chartType); // Original line
                 }
             });
         });
         const refreshButton = document.getElementById('refreshStockBtn');
         if (refreshButton) { refreshButton.addEventListener('click', refreshStockData); }
         const overviewTabLinkForListener = document.getElementById('overview-tab');
         if (overviewTabLinkForListener) { overviewTabLinkForListener.addEventListener('shown.bs.tab', initializeDetailCharts); }
         const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
         if (tooltipTriggerList.length > 0) { tooltipTriggerList.map(function (tooltipTriggerEl) { return new bootstrap.Tooltip(tooltipTriggerEl); }); }
         console.log("script.js: Event listeners attached.");
    }

}); // End DOMContentLoaded