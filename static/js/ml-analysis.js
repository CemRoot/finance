// static/js/ml-analysis.js
console.log("ML analysis module loaded");

document.addEventListener('DOMContentLoaded', function() {
    // Handle any ML analysis specific interactions
    
    // Add event listener to resize charts when ML analysis tab is shown
    const mlAnalysisTab = document.getElementById('ml-analysis-tab');
    if (mlAnalysisTab) {
        mlAnalysisTab.addEventListener('shown.bs.tab', function() {
            console.log("ML Analysis tab shown, resizing charts if needed");
            // If there are any Plotly charts that need resizing, handle them here
            if (typeof Plotly !== 'undefined') {
                const chartElements = document.querySelectorAll('.chart-container');
                chartElements.forEach(function(chartElement) {
                    try {
                        Plotly.Plots.resize(chartElement);
                    } catch (e) {
                        console.error("Error resizing Plotly chart:", e);
                    }
                });
            }
        });
    }
    
    // Add click handler for "Compare Models" button if it exists
    const compareModelsBtn = document.getElementById('compareModelsBtn');
    if (compareModelsBtn) {
        compareModelsBtn.addEventListener('click', function(e) {
            e.preventDefault();
            const stockSymbol = compareModelsBtn.dataset.stock || '';
            if (stockSymbol) {
                window.location.href = `/model_comparison?stock=${stockSymbol}`;
            } else {
                console.error("No stock symbol found for model comparison");
            }
        });
    }
    
    // Add click handler for "Random Forest Analysis" button if it exists
    const rfAnalysisBtn = document.getElementById('rfAnalysisBtn');
    if (rfAnalysisBtn) {
        rfAnalysisBtn.addEventListener('click', function(e) {
            e.preventDefault();
            const stockSymbol = rfAnalysisBtn.dataset.stock || '';
            if (stockSymbol) {
                window.location.href = `/random_forest_analysis?stock=${stockSymbol}`;
            } else {
                console.error("No stock symbol found for Random Forest analysis");
            }
        });
    }
});