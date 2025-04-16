// static/js/forecast.js
console.log("forecast.js yüklendi.");

// Global değişkenler
let forecastChartInstance = null; // Plotly grafik nesnesi
const MAX_RETRIES = 2; // Tekrar deneme sayısı
const RETRY_DELAY = 3000; // Denemeler arası bekleme (ms)

// --- Plotly Grafik Fonksiyonları ---

// Tahmin grafiğini başlatır veya yükleniyor durumunu gösterir
function initOrShowLoadingForecastChart() {
    const ctx = document.getElementById('forecastChart');
    if (!ctx) {
        console.error("initOrShowLoadingForecastChart: 'forecastChart' elementi bulunamadı.");
        return;
    }

    // Eğer zaten bir grafik varsa temizle
    if (forecastChartInstance) {
        try {
            Plotly.purge(ctx);
            forecastChartInstance = null; // Referansı temizle
            console.log("Previous forecast chart purged.");
        } catch (e) {
            console.error("Error purging forecast chart:", e);
        }
    }

    // Yükleniyor mesajı ile boş grafik oluştur
    const layout = {
        title: 'Fiyat Tahmini Yükleniyor...',
        xaxis: { visible: false }, // Eksenleri gizle
        yaxis: { visible: false },
        plot_bgcolor: '#FFFFFF',
        paper_bgcolor: '#FFFFFF',
        margin: { l: 40, r: 20, t: 60, b: 40 },
        annotations: [{
            text: "Tahmin verileri alınıyor...",
            xref: "paper", yref: "paper",
            x: 0.5, y: 0.5,
            showarrow: false,
            font: { size: 14, color: '#6c757d' }
        }]
    };

    try {
         Plotly.newPlot(ctx, [], layout, { responsive: true });
         console.log("Forecast chart initialized with loading state.");
    } catch (e) {
         console.error("Error creating initial forecast chart:", e);
    }
}

// Plotly tahmin grafiğini gelen veriyle günceller
function updateForecastChart(data, stockSymbol = 'Hisse') {
    const ctx = document.getElementById('forecastChart');
    if (!ctx) {
         console.error('updateForecastChart: Chart element not found.');
         return;
    }
    // Veri veya gerekli alt anahtarlar eksikse hata göster
    if (!data || !data.forecast_values || !data.forecast_values.dates || !data.forecast_values.values || !data.forecast_values.lower_bound || !data.forecast_values.upper_bound) {
        console.error('updateForecastChart: Geçersiz veya eksik tahmin verisi.', data);
         handleForecastError({ message: "Alınan tahmin verisi formatı geçersiz veya eksik." });
        return;
    }

    const forecastData = data.forecast_values;
    // Tahminlerde None/null değer var mı kontrol et ve filtrele/logla
    const validIndices = forecastData.values.map((v, i) => v !== null && forecastData.lower_bound[i] !== null && forecastData.upper_bound[i] !== null).reduce((acc, val, i) => val ? [...acc, i] : acc, []);

    if (validIndices.length === 0) {
        console.error('updateForecastChart: Tahmin verilerinde hiç geçerli nokta bulunamadı.');
        handleForecastError({ message: "Tahmin verisi tamamen geçersiz." });
        return;
    }
     if (validIndices.length < forecastData.dates.length) {
          console.warn(`updateForecastChart: Tahmin verilerinde ${forecastData.dates.length - validIndices.length} adet geçersiz (null) nokta bulundu ve atlandı.`);
     }

     // Sadece geçerli verileri al
     const validDates = validIndices.map(i => forecastData.dates[i]);
     const validValues = validIndices.map(i => forecastData.values[i]);
     const validLower = validIndices.map(i => forecastData.lower_bound[i]);
     const validUpper = validIndices.map(i => forecastData.upper_bound[i]);


    // Veri izlerini (traces) oluştur
    const traceForecast = {
        x: validDates,
        y: validValues,
        name: 'Tahmin',
        mode: 'lines',
        type: 'scatter',
        line: { color: '#3498db', width: 2.5 } // Ana tahmin çizgisi (biraz kalın)
    };

    const traceUpperBound = {
        x: validDates,
        y: validUpper,
        name: 'Güven Aralığı', // Tek legend girdisi
        mode: 'lines',
        type: 'scatter',
        line: { color: 'rgba(52, 152, 219, 0.3)', width: 1, dash: 'dot' }, // Şeffaf çizgi
        fill: 'none', // Dolgu yok
        showlegend: false // Üst sınır için legend gösterme
    };

    const traceLowerBound = {
        x: validDates,
        y: validLower,
        name: 'Alt Sınır', // Bu legend'da görünmeyecek
        mode: 'lines',
        type: 'scatter',
        line: { color: 'rgba(52, 152, 219, 0.3)', width: 1, dash: 'dot' }, // Şeffaf çizgi
        fill: 'tonexty', // Üst sınıra kadar doldur (traceUpperBound'a kadar)
        fillcolor: 'rgba(52, 152, 219, 0.1)', // Çok şeffaf mavi dolgu
        showlegend: true, // Sadece alt sınır için legend göster (Güven Aralığı olarak görünecek)
        legendgroup: 'confidence', // Üst ile grupla (gerekirse)
        name: 'Güven Aralığı' // Legend metnini ayarla
    };

    // Grafik düzenini (layout) ayarla
    const layout = {
        title: `${stockSymbol} Fiyat Tahmini (${validDates.length} Günlük)`,
        xaxis: {
            title: 'Tarih',
            showgrid: true, gridcolor: '#ecf0f1', type: 'date', tickformat: '%d %b %Y', // Tarih formatı
            range: [validDates[0], validDates[validDates.length - 1]] // Tarih aralığını ayarla
        },
        yaxis: {
            title: 'Fiyat ($)', // Para birimi eklendi
            showgrid: true, gridcolor: '#ecf0f1',
            // Fiyat aralığını verilere göre otomatik ayarla (Plotly varsayılanı)
            // range: [Math.min(...validLower) * 0.98, Math.max(...validUpper) * 1.02] // Manuel aralık (opsiyonel)
        },
        plot_bgcolor: '#FFFFFF',
        paper_bgcolor: '#FFFFFF',
        showlegend: true,
        legend: {
            x: 0.01, y: 0.99, // Sol üst köşe
            bgcolor: 'rgba(255,255,255,0.7)', bordercolor: '#CCCCCC', borderwidth: 1,
            orientation: 'h' // Yatay legend
        },
        margin: { l: 60, r: 30, t: 80, b: 50 }
    };

    // Mevcut grafiği yeni veri ve layout ile güncelle (purge yerine react kullanmak daha iyi)
    try {
        Plotly.react(ctx, [traceLowerBound, traceUpperBound, traceForecast], layout, { responsive: true });
        forecastChartInstance = ctx; // Referansı güncelle
        console.log("Forecast chart updated with new data.");
    } catch (e) {
         console.error("Error updating forecast chart with Plotly.react:", e);
         handleForecastError({ message: "Tahmin grafiği güncellenirken hata oluştu." });
    }
}

// --- Metrik Güncelleme Fonksiyonları ---

// Tahmin metriklerini HTML'deki ilgili alanlara yazar
function updateForecastMetrics(data) {
    const metricsContainer = document.querySelector('.metrics-container');
    if (!metricsContainer) return; // Metrik alanı yoksa çık

    // Metrik alanlarını bul
    const trendDirEl = metricsContainer.querySelector('#trendDirection');
    const trendStrEl = metricsContainer.querySelector('#trendStrength');
    const volEl = metricsContainer.querySelector('#historicalVolatility');
    const seasonEl = metricsContainer.querySelector('#seasonalityStrength');
    const confIntEl = metricsContainer.querySelector('#confidenceInterval');

    // Veri veya metrik yoksa alanları temizle/sıfırla
    if (!data || !data.metrics) {
        console.warn("updateForecastMetrics: Gerekli metrik verisi bulunamadı.");
        [trendDirEl, trendStrEl, volEl, seasonEl, confIntEl].forEach(el => {
            if (el) el.textContent = 'N/A';
        });
        updateForecastSummary(null); // Özeti de temizle
        return;
    }

    const metrics = data.metrics;

    try {
        // Trend Yönü
        if (trendDirEl) trendDirEl.textContent = metrics.trend_direction || 'Belirsiz';

        // Trend Gücü (Yüzde olarak)
        if (trendStrEl) trendStrEl.textContent = metrics.trend_strength !== null ? `${(metrics.trend_strength * 100).toFixed(1)}%` : 'N/A';

        // Tarihsel Volatilite (Yıllık Yüzde)
        if (volEl) {
            const volatility = metrics.historical_volatility;
            if (volatility !== null) {
                volEl.textContent = `${(volatility * 100).toFixed(1)}%`;
                 // Renklendirme (opsiyonel)
                 volEl.classList.remove('text-success', 'text-warning', 'text-danger');
                 if (volatility * 100 < 25) volEl.classList.add('text-success'); // Düşük volatilite
                 else if (volatility * 100 < 50) volEl.classList.add('text-warning'); // Orta
                 else volEl.classList.add('text-danger'); // Yüksek
            } else {
                volEl.textContent = 'N/A';
                volEl.classList.remove('text-success', 'text-warning', 'text-danger');
            }
        }

        // Mevsimsellik Etkisi (Yüzde olarak)
        if (seasonEl) seasonEl.textContent = metrics.seasonality_strength !== null ? `${(metrics.seasonality_strength * 100).toFixed(1)}%` : 'N/A';

        // Güven Aralığı (Son Tahmin, Fiyata Göre Yüzde Genişlik)
        if (confIntEl) {
             const intervalRelative = metrics.confidence_interval; // Artık göreceli genişlik
             if (intervalRelative !== null && intervalRelative >= 0) {
                  // Göreceli genişliği yüzde olarak gösterelim
                 confIntEl.textContent = `±${(intervalRelative / 2 * 100).toFixed(1)}%`; // Genişliğin yarısı +/- %
             } else {
                  confIntEl.textContent = 'N/A';
             }
         }

        // Özeti güncelle
        updateForecastSummary(data);

    } catch (e) {
        console.error("Metrik güncelleme hatası:", e);
        // Hata durumunda alanları temizle
        [trendDirEl, trendStrEl, volEl, seasonEl, confIntEl].forEach(el => { if (el) el.textContent = 'Hata'; });
         updateForecastSummary(null); // Özeti de temizle
    }
}

// Tahmin özetini oluşturur ve HTML'e yazar
function updateForecastSummary(data) {
    const summaryElement = document.getElementById('forecastSummary');
    if (!summaryElement) return;

    // Gerekli verilerin kontrolü
    if (!data || !data.forecast_values || !data.forecast_values.values || !data.metrics) {
         summaryElement.textContent = "Özet oluşturmak için yeterli tahmin verisi bulunamadı.";
         summaryElement.classList.add('text-danger');
         return;
    }

    try {
        const values = data.forecast_values.values.filter(v => v !== null); // Null olmayanları al
        const metrics = data.metrics;
        const lastActualPrice = data.last_actual_price; // Son gerçek fiyat

        if (values.length === 0) {
             summaryElement.textContent = "Geçerli tahmin değeri bulunamadı.";
             return;
        }

        // İlk ve son tahmin değerleri
        const firstForecastValue = values[0];
        const lastForecastValue = values[values.length - 1];

        // Yüzdelik değişim (son gerçek fiyata göre veya ilk tahmine göre)
        let changeBase = lastActualPrice !== null ? lastActualPrice : firstForecastValue;
        let percentChangeText = "belirsiz bir değişim";
        if (changeBase !== null && changeBase !== 0 && lastForecastValue !== null) {
            const percentChange = ((lastForecastValue - changeBase) / Math.abs(changeBase) * 100);
             const changeDirection = percentChange >= 0 ? 'artış' : 'düşüş';
             percentChangeText = `%${Math.abs(percentChange).toFixed(1)} ${changeDirection}`;
        }

        // Güven seviyesi (Volatiliteye göre)
        let confidenceLevel = 'orta';
        let volatilityText = metrics.historical_volatility !== null ? `%${(metrics.historical_volatility * 100).toFixed(1)}` : "bilinmiyor";
         if (metrics.historical_volatility !== null) {
            if (metrics.historical_volatility < 0.25) confidenceLevel = 'yüksek'; // Eşikler ayarlanabilir
             else if (metrics.historical_volatility > 0.50) confidenceLevel = 'düşük';
         } else {
             confidenceLevel = 'belirsiz';
         }

        // Özet metnini oluştur
         let summary = `Model, genel olarak <strong>${metrics.trend_direction || 'belirsiz'}</strong> bir trend öngörüyor. `;
         summary += `Tahmin dönemi sonunda fiyatın yaklaşık <strong>${percentChangeText}</strong> göstermesi bekleniyor. `;
         summary += `Tarihsel volatilite (${volatilityText}) göz önüne alındığında, tahmin güvenilirliği <strong>${confidenceLevel}</strong> seviyesindedir. `;
         if (metrics.seasonality_strength !== null && metrics.seasonality_strength > 0.1) { // Sadece belirginse göster
             summary += `Mevsimsel etkilerin belirgin olduğu (%${(metrics.seasonality_strength * 100).toFixed(0)}) gözlemlenmiştir.`;
         }

        summaryElement.innerHTML = summary; // HTML olarak ekle (strong için)
        summaryElement.classList.remove('text-danger'); // Hata varsa kaldır

    } catch (e) {
        console.error("Tahmin özeti oluşturma hatası:", e);
        summaryElement.textContent = "Tahmin özeti oluşturulurken bir hata oluştu.";
         summaryElement.classList.add('text-danger');
    }
}


// --- Veri Yükleme ve Hata Yönetimi ---

// Ana veri yükleme fonksiyonu
async function loadForecastData(stockSymbol) {
    if (!stockSymbol) {
        console.error('loadForecastData: Hisse senedi sembolü gerekli');
        return;
    }
    console.log(`Loading forecast data for: ${stockSymbol}`);

    // Yükleniyor durumunu göster
    initOrShowLoadingForecastChart();
    updateForecastMetrics(null); // Metrikleri temizle

    try {
        // Fetch işlemini retry ile yap
        const response = await fetchWithRetry(`/get_forecast_data?symbol=${stockSymbol}`, {}, MAX_RETRIES);

        // Yanıtı işle
        if (!response.ok) {
             // Fetch başarılı ama HTTP hatası var (4xx, 5xx)
             const errorData = await response.json().catch(() => ({ error: `Sunucu hatası: ${response.status}` })); // JSON parse edilemezse
             throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
         }

        const data = await response.json();

        // API kendi içinde hata döndürdüyse (örn. forecasting.py içinden)
        if (data.error) {
            throw new Error(data.error);
        }

        // Veri başarıyla alındı, grafikleri ve metrikleri güncelle
        console.log("Forecast data received:", data);
        updateForecastChart(data, stockSymbol);
        updateForecastMetrics(data);
        // Özet zaten updateForecastMetrics içinden çağrılıyor

    } catch (error) {
        console.error('Tahmin verisi çekme/işleme hatası:', error);
        handleForecastError(error); // Hata durumunu kullanıcıya göster
    }
}

// Fetch işlemini tekrar deneme mekanizması ile yapar
async function fetchWithRetry(url, options = {}, retries = 1) {
     console.log(`Attempting fetch (try ${retries}): ${url}`);
     try {
         const response = await fetch(url, options);
          // 429 (Rate Limit) hatasını özel olarak ele alıp tekrar deneyebiliriz
          if (response.status === 429 && retries < MAX_RETRIES) {
              const retryAfter = RETRY_DELAY * retries; // Artan bekleme
              console.warn(`Rate limit hit (429). Retrying after ${retryAfter}ms...`);
              await new Promise(resolve => setTimeout(resolve, retryAfter));
              return fetchWithRetry(url, options, retries + 1); // Tekrar dene
          }
         return response; // Diğer durumları doğrudan döndür (başarılı veya diğer hatalar)
     } catch (error) {
         // Ağ hatası gibi durumlarda tekrar deneme (opsiyonel)
         if (retries < MAX_RETRIES) {
             const retryAfter = RETRY_DELAY * retries;
             console.warn(`Fetch failed (${error.message}). Retrying after ${retryAfter}ms...`);
             await new Promise(resolve => setTimeout(resolve, retryAfter));
             return fetchWithRetry(url, options, retries + 1);
         }
         console.error(`Fetch failed after ${MAX_RETRIES} attempts: ${error.message}`);
         throw error; // Son denemeden sonra hatayı fırlat
     }
 }


// Tahmin yükleme sırasında oluşan hataları kullanıcıya gösterir
function handleForecastError(error) {
    const ctx = document.getElementById('forecastChart');
    if (ctx) {
        try {
             Plotly.purge(ctx); // Eski grafiği temizle (varsa)
             const layout = {
                 title: 'Tahmin Yüklenemedi',
                 xaxis: { visible: false }, yaxis: { visible: false },
                 plot_bgcolor: '#FFFFFF', paper_bgcolor: '#FFFFFF',
                 margin: { l: 40, r: 20, t: 60, b: 40 },
                 annotations: [{
                     text: `Hata: ${error.message || 'Bilinmeyen bir sorun oluştu.'}`,
                     xref: "paper", yref: "paper", x: 0.5, y: 0.5,
                     showarrow: false, font: { size: 12, color: '#dc3545' } // Kırmızı renk
                 }]
             };
             Plotly.newPlot(ctx, [], layout, { responsive: true });
             console.log("Forecast chart updated with error state.");
         } catch(e) { console.error("Error displaying forecast error state:", e); }
    }

    // Metrik alanlarını hata durumuna ayarla
    updateForecastMetrics(null); // Bu fonksiyon zaten N/A veya Hata yazar
    // Özeti hata mesajı ile güncelle
    const summaryElement = document.getElementById('forecastSummary');
    if (summaryElement) {
        summaryElement.textContent = `Tahmin verisi alınamadı: ${error.message || 'Bilinmeyen hata.'}`;
        summaryElement.classList.add('text-danger');
    }
}

// --- Olay Dinleyicileri ---
document.addEventListener('DOMContentLoaded', function() {
    console.log("forecast.js: DOMContentLoaded event fired.");

    // Tahmin sekmesi için grafik alanını hazırla
    const forecastChartElement = document.getElementById('forecastChart');
    if (forecastChartElement) {
        initOrShowLoadingForecastChart(); // Başlangıçta yükleniyor göster
    } else {
        console.warn("Forecast chart element (#forecastChart) not found on page load.");
    }

    // Sayfada başlangıçta bir hisse senedi varsa, tahmin verilerini yükle
    // Input'un ID'si 'stock' varsayılıyor (partials/_sidebar.html'deki gibi)
    const stockInput = document.getElementById('stock');
    const initialStockSymbol = stockInput ? stockInput.value : null;

    if (initialStockSymbol) {
        console.log(`Initial stock symbol found ('${initialStockSymbol}'), triggering forecast load.`);
        loadForecastData(initialStockSymbol);
    } else {
         console.log("No initial stock symbol found. Forecast will load when a stock is selected.");
          // Eğer hisse seçili değilse, grafikte "Hisse seçin" mesajı gösterebiliriz
          if (forecastChartElement) {
                try {
                     Plotly.purge(forecastChartElement);
                     Plotly.newPlot(forecastChartElement, [], {
                          title: 'Fiyat Tahmini',
                          xaxis: { visible: false }, yaxis: { visible: false },
                          annotations: [{ text: "Lütfen bir hisse senedi seçin.", xref: "paper", yref: "paper", x: 0.5, y: 0.5, showarrow: false, font: { size: 14, color: '#6c757d' } }]
                     });
                } catch(e) { console.error("Error showing initial 'select stock' message:", e); }
          }
          updateForecastMetrics(null); // Metrikleri temizle
          const summaryEl = document.getElementById('forecastSummary');
          if(summaryEl) summaryEl.textContent = "Tahminleri görmek için bir hisse senedi arayın.";

    }

    // Tahmin sekmesine tıklandığında grafiği yeniden boyutlandır (yeniden çizmek daha iyi olabilir)
    // Sidebar'daki link ID'si 'v-pills-forecast-tab' varsayılıyor
    const forecastTabLink = document.getElementById('v-pills-forecast-tab');
    if (forecastTabLink) {
        forecastTabLink.addEventListener('shown.bs.tab', function () {
            const forecastChartElement = document.getElementById('forecastChart');
            if (forecastChartElement && typeof Plotly !== 'undefined') {
                 // Plotly.Plots.resize() bazen yetersiz kalabilir, yeniden çizelim
                 // Ancak bu, veri tekrar çekilmeden sadece mevcut boyutlara göre ayarlar.
                 // Eğer boyutlandırma sorunu devam ederse, layout'u güncelleyip Plotly.react çağırmak gerekebilir.
                try {
                    console.log("Forecast tab shown, resizing Plotly chart...");
                    setTimeout(() => Plotly.Plots.resize(forecastChartElement), 50); // Kısa gecikme
                } catch(e) {
                     console.error("Plotly resize error on tab show:", e);
                }
            }
        });
    }

}); // DOMContentLoaded Sonu