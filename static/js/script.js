// static/js/script.js
console.log("Script.js yüklendi.");

// --- Global Değişkenler ---
let priceChartInstance = null; // Ana fiyat grafiği nesnesi
let volumeChartInstance = null; // Genel bakış hacim grafiği nesnesi
// Detay grafik nesneleri (ilgili sekmeler açıldığında oluşturulacak)
let openCloseChartInstance = null;
let detailVolumeChartInstance = null;
let highLowChartInstance = null;

// --- Yardımcı Fonksiyonlar ---

// Grafik verisini Luxon DateTime'a çevirir
function parseChartData(labels, values) {
    if (!labels || !values || labels.length !== values.length) {
        console.warn("parseChartData: Etiket ve değer sayısı uyuşmuyor veya eksik.");
        return [];
    }
    return labels.map((label, index) => {
        try {
            const timestamp = luxon.DateTime.fromISO(label).valueOf();
            const value = values[index];
            // Değerin null veya undefined olmadığından emin ol
            if (value !== null && value !== undefined && !isNaN(timestamp)) {
                return { x: timestamp, y: value };
            }
            console.warn(`parseChartData: Geçersiz değer veya zaman damgası atlandı: label=${label}, value=${value}, timestamp=${timestamp}`);
            return null; // Geçersiz veri için null döndür
        } catch (e) {
            console.error(`Geçersiz tarih formatı veya değer: ${label}`, e);
            return null;
        }
    }).filter(d => d !== null); // Null değerleri filtrele
}

// Mum grafiği verisini hazırlar
function parseCandlestickData(candlestickData) {
    if (!candlestickData) return [];
    return candlestickData.map(d => {
        try {
            const timestamp = luxon.DateTime.fromISO(d.t).valueOf();
            // Tüm ohlc değerlerinin sayı olduğundan emin ol
            if ([d.o, d.h, d.l, d.c].every(val => typeof val === 'number') && !isNaN(timestamp)) {
                 return { x: timestamp, o: d.o, h: d.h, l: d.l, c: d.c };
            }
            console.warn(`parseCandlestickData: Geçersiz OHLC değeri veya zaman damgası atlandı: t=${d.t}`);
             return null;
        } catch (e) {
            console.error(`Geçersiz mum verisi: ${d.t}`, e);
            return null;
        }
    }).filter(d => d !== null);
}


// Genel Chart.js Seçenekleri
const commonChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
        mode: 'index',
        intersect: false,
    },
    scales: {
        x: {
            type: 'time',
            time: {
                unit: 'day',
                tooltipFormat: 'DD MMM YYYY', // Luxon formatı
                displayFormats: {
                    day: 'DD MMM' // Luxon formatı
                }
            },
            grid: { display: false },
            ticks: { autoSkip: true, maxTicksLimit: 10, source: 'auto', color: '#6c757d' } // Renk eklendi
        },
        y: {
            beginAtZero: false,
            grid: { color: 'rgba(200, 200, 200, 0.1)' },
            ticks: {
                 color: '#6c757d', // Renk eklendi
                callback: function(value) {
                    if (typeof value === 'number') return value.toFixed(2) + '$'; // Para birimi ekle
                    return value;
                }
            }
        }
    },
    plugins: {
        legend: {
             display: true,
             labels: { color: '#2c3e50' } // Renk eklendi
        },
        tooltip: {
            enabled: true,
            backgroundColor: 'rgba(0, 0, 0, 0.7)', // Tooltip arkaplanı
            titleColor: '#ffffff',
            bodyColor: '#ffffff',
            // Tooltip içeriği grafik özelinde ayarlanacak
        }
    }
};

// --- Grafik Oluşturma Fonksiyonları ---

// Ana Fiyat Grafiği (Mum veya Çizgi)
function createOrUpdatePriceChart(stockData, chartType = 'candlestick') {
    const ctx = document.getElementById('priceChart');
    if (!ctx || !stockData) {
         console.warn("Fiyat grafiği oluşturulamadı: Element veya veri eksik.");
         const container = ctx ? ctx.closest('.chart-container') : document.querySelector('.price-chart-container');
         if (container) container.innerHTML = "<p class='text-muted text-center small p-3'>Fiyat grafiği için veri yüklenemedi.</p>";
         return;
    }

    // Önceki grafiği yok et
    if (priceChartInstance) {
        priceChartInstance.destroy();
        priceChartInstance = null;
    }

    let data, options = JSON.parse(JSON.stringify(commonChartOptions)); // Derin kopya
    let configType = chartType;

    options.plugins.title = { // Grafik başlığı (opsiyonel)
        display: false, // Başlığı HTML'den alıyoruz zaten
        // text: `${stockData.company_name || stockData.symbol || 'Hisse'} Fiyat Grafiği`
    };


    if (chartType === 'candlestick') {
        const candleData = parseCandlestickData(stockData.candlestick_data);
        if (candleData.length === 0) {
            console.warn("Mum grafiği için geçerli veri yok.");
            ctx.closest('.chart-container').innerHTML = "<p class='text-muted text-center small p-3'>Mum grafiği için veri yok.</p>";
            return;
        }
        data = { datasets: [{
            label: stockData.company_name || 'Fiyat',
            data: candleData,
            color: { // Renkler (chartjs-chart-financial)
                 up: 'rgba(38, 166, 154, 1)', // Yeşil (artış)
                 down: 'rgba(239, 83, 80, 1)', // Kırmızı (düşüş)
                 unchanged: 'rgba(158, 158, 158, 1)' // Gri (değişim yok)
             }
            }]
        };
        configType = 'candlestick'; // Chart.js financial için tip
         options.plugins.legend.display = false; // Mum grafiğinde legend gereksiz
         options.plugins.tooltip.callbacks = { // Özel tooltip
             label: function(context) {
                 const d = context.raw;
                 if (d && typeof d.o === 'number') { // Check if d.o exists and is a number
                     return [
                         ` Açılış: ${d.o.toFixed(2)}$`,
                         ` Yüksek: ${d.h.toFixed(2)}$`,
                         ` Düşük: ${d.l.toFixed(2)}$`,
                         ` Kapanış: ${d.c.toFixed(2)}$`
                     ];
                 } return '';
             }
         };
    } else { // Line chart
        const lineData = parseChartData(stockData.labels, stockData.values);
         if (lineData.length === 0) {
            console.warn("Çizgi grafiği için geçerli veri yok.");
             ctx.closest('.chart-container').innerHTML = "<p class='text-muted text-center small p-3'>Çizgi grafiği için veri yok.</p>";
            return;
        }
        data = {
            // labels: lineData.map(d => d.x), // time scale kullanıldığı için labels gereksiz
            datasets: [{
                label: 'Kapanış Fiyatı',
                data: lineData,
                borderColor: 'rgba(54, 162, 235, 1)',
                backgroundColor: 'rgba(54, 162, 235, 0.1)',
                borderWidth: 1.5,
                tension: 0.1, // Daha yumuşak çizgiler
                pointRadius: 0, // Noktaları gizle
                fill: true
            }]
        };
        configType = 'line';
        options.plugins.legend.display = true;
        options.plugins.tooltip.callbacks = { // Standart tooltip
             label: function(context) {
                 let label = context.dataset.label || '';
                 if (label) label += ': ';
                 // context.parsed.y null veya undefined olabilir, kontrol et
                 if (context.parsed && context.parsed.y !== null && context.parsed.y !== undefined) {
                      label += context.parsed.y.toFixed(2) + '$';
                 } else {
                      label += 'N/A';
                 }
                 return label;
             }
         };
    }

    try {
        priceChartInstance = new Chart(ctx.getContext('2d'), { type: configType, data: data, options: options });
        console.log(`${chartType} price chart created/updated.`);
    } catch (e) {
        console.error(`Error creating/updating ${chartType} price chart:`, e);
        ctx.closest('.chart-container').innerHTML = `<p class='text-danger text-center small p-3'>${chartType === 'candlestick' ? 'Mum' : 'Çizgi'} grafiği yüklenemedi.</p>`;
    }
}

// Genel Bakış Hacim Grafiği
function createOrUpdateVolumeChart(stockData) {
    const ctx = document.getElementById('volumeChart');
    const container = ctx ? ctx.closest('.chart-container') : null;
    if (!ctx || !container || !stockData || !stockData.labels || !stockData.volume_values) {
         console.warn("Hacim grafiği oluşturulamadı: Element veya veri eksik.");
         if(container) container.innerHTML = ""; // Alanı temizle
         return;
    }

    if (volumeChartInstance) {
        volumeChartInstance.destroy();
        volumeChartInstance = null;
    }

    const volumeData = parseChartData(stockData.labels, stockData.volume_values);
    if (volumeData.length === 0) {
        console.warn("Hacim grafiği için geçerli veri yok.");
        if(container) container.innerHTML = "";
        return;
    }

    // Son kapanış ve bir önceki kapanış fiyatlarını alarak renkleri belirle
    const prices = stockData.values || [];
    const barColors = volumeData.map((dataPoint, index) => {
         if (index > 0 && prices[index] !== null && prices[index-1] !== null) {
              return prices[index] >= prices[index-1] ? 'rgba(38, 166, 154, 0.6)' : 'rgba(239, 83, 80, 0.6)'; // Yeşil veya Kırmızı
         }
          return 'rgba(153, 102, 255, 0.6)'; // Varsayılan renk (Mor)
    });


    const options = {
        responsive: true, maintainAspectRatio: false,
        scales: {
            x: { type: 'time', display: false }, // X eksenini gizle
            y: { beginAtZero: true, display: false } // Y eksenini gizle
        },
        plugins: { legend: { display: false }, tooltip: { enabled: false } } // Legend ve tooltip'i kapat
    };

    try {
        volumeChartInstance = new Chart(ctx.getContext('2d'), {
            type: 'bar',
            data: {
                 // labels: volumeData.map(d => d.x), // Gerekli değil
                 datasets: [{
                    label: 'İşlem Hacmi',
                    data: volumeData,
                    backgroundColor: barColors, // Hesaplanan renkleri kullan
                    // borderColor: 'rgba(153, 102, 255, 1)',
                    borderWidth: 0 // Kenarlık istemiyoruz
                }]
            },
            options: options
        });
        console.log("Overview volume chart created/updated.");
    } catch (e) {
        console.error("Error creating/updating overview volume chart:", e);
        if(container) container.innerHTML = "<p class='text-danger text-center small'>Hacim grafiği yüklenemedi.</p>";
    }
}


// Detay Sekmesi Grafikleri (İstek üzerine oluşturulacak)
function createOpenCloseChart(stockData) {
    const ctx = document.getElementById('openCloseChart');
    const container = ctx ? ctx.closest('.chart-container') : null;
     if (!ctx || !container || !stockData || !stockData.labels || !stockData.open_values || !stockData.values) {
          console.warn("Açılış/Kapanış grafiği oluşturulamadı: Element veya veri eksik.");
          if(container) container.innerHTML = "<p class='text-muted text-center small p-3'>Grafik için veri yüklenemedi.</p>";
          return;
     }
     if (openCloseChartInstance) openCloseChartInstance.destroy();

     const openData = parseChartData(stockData.labels, stockData.open_values);
     const closeData = parseChartData(stockData.labels, stockData.values);

     if (openData.length === 0 || closeData.length === 0) {
          console.warn("Açılış/Kapanış grafiği için geçerli veri yok.");
          if(container) container.innerHTML = "<p class='text-muted text-center small p-3'>Grafik için veri yok.</p>";
          return;
     }

     const options = JSON.parse(JSON.stringify(commonChartOptions));
     options.plugins.tooltip.callbacks = { // Tooltip özelleştirme
          label: function(context) {
              let label = context.dataset.label || '';
              if (label) label += ': ';
              if (context.parsed.y !== null) label += context.parsed.y.toFixed(2) + '$';
              return label;
          }
      };
      options.plugins.legend.position = 'top';


     try {
          openCloseChartInstance = new Chart(ctx.getContext('2d'), {
               type: 'line',
               data: {
                    datasets: [
                         { label: 'Açılış', data: openData, borderColor: 'rgba(255, 159, 64, 1)', borderWidth: 1.5, pointRadius: 0, tension: 0.1, fill: false },
                         { label: 'Kapanış', data: closeData, borderColor: 'rgba(54, 162, 235, 1)', borderWidth: 1.5, pointRadius: 0, tension: 0.1, fill: false }
                    ]
               },
               options: options
          });
          console.log("Open/Close chart created.");
     }
     catch(e) {
          console.error("Error creating Open/Close chart:", e);
          if(container) container.innerHTML = "<p class='text-danger text-center small p-3'>Açılış/Kapanış grafiği yüklenemedi.</p>";
      }
}

function createDetailVolumeChart(stockData) {
    const ctx = document.getElementById('detailVolumeChart');
    const container = ctx ? ctx.closest('.chart-container') : null;
     if (!ctx || !container ||!stockData || !stockData.labels || !stockData.volume_values) {
         console.warn("Detay Hacim grafiği oluşturulamadı: Element veya veri eksik.");
         if(container) container.innerHTML = "<p class='text-muted text-center small p-3'>Grafik için veri yüklenemedi.</p>";
         return;
      }
     if (detailVolumeChartInstance) detailVolumeChartInstance.destroy();

     const volumeData = parseChartData(stockData.labels, stockData.volume_values);
     if (volumeData.length === 0) {
          console.warn("Detay Hacim grafiği için geçerli veri yok.");
          if(container) container.innerHTML = "<p class='text-muted text-center small p-3'>Grafik için veri yok.</p>";
          return;
     }

      // Renkleri hesapla
      const prices = stockData.values || [];
      const barColors = volumeData.map((dataPoint, index) => {
           if (index > 0 && prices[index] !== null && prices[index-1] !== null) {
                return prices[index] >= prices[index-1] ? 'rgba(38, 166, 154, 0.6)' : 'rgba(239, 83, 80, 0.6)'; // Yeşil veya Kırmızı
           }
            return 'rgba(153, 102, 255, 0.6)'; // Varsayılan renk (Mor)
      });


     const options = JSON.parse(JSON.stringify(commonChartOptions));
     options.scales.y.ticks.callback = function(value) { // Hacim formatlama
          if (typeof value === 'number') {
              if (value >= 1e9) return (value / 1e9).toFixed(1) + 'B'; // Milyar
              if (value >= 1e6) return (value / 1e6).toFixed(1) + 'M'; // Milyon
              if (value >= 1e3) return (value / 1e3).toFixed(0) + 'K'; // Bin
              return value;
          } return value;
      };
     options.plugins.legend.display = false;
     options.plugins.tooltip.callbacks = { // Tooltip formatlama
          label: function(context) {
               let label = context.dataset.label || '';
               if (label) label += ': ';
               if (context.parsed.y !== null) {
                   const value = context.parsed.y;
                    if (value >= 1e9) label += (value / 1e9).toFixed(2) + 'B';
                    else if (value >= 1e6) label += (value / 1e6).toFixed(2) + 'M';
                    else if (value >= 1e3) label += (value / 1e3).toFixed(0) + 'K';
                    else label += value;
               }
               return label;
           }
       };

     try {
          detailVolumeChartInstance = new Chart(ctx.getContext('2d'), {
                type: 'bar',
                data: {
                     // labels: volumeData.map(d => d.x),
                     datasets: [{
                          label: 'İşlem Hacmi',
                          data: volumeData,
                          backgroundColor: barColors,
                          borderWidth: 0
                     }]
                },
                options: options
           });
          console.log("Detail Volume chart created.");
     }
     catch(e) {
          console.error("Error creating Detail Volume chart:", e);
          if(container) container.innerHTML = "<p class='text-danger text-center small p-3'>Hacim grafiği yüklenemedi.</p>";
      }
}

function createHighLowChart(stockData) {
    const ctx = document.getElementById('highLowChart');
    const container = ctx ? ctx.closest('.chart-container') : null;
     if (!ctx || !container ||!stockData || !stockData.labels || !stockData.high_values || !stockData.low_values) {
          console.warn("Yüksek/Düşük grafiği oluşturulamadı: Element veya veri eksik.");
          if(container) container.innerHTML = "<p class='text-muted text-center small p-3'>Grafik için veri yüklenemedi.</p>";
          return;
      }
     if (highLowChartInstance) highLowChartInstance.destroy();

     const highData = parseChartData(stockData.labels, stockData.high_values);
     const lowData = parseChartData(stockData.labels, stockData.low_values);

     if (highData.length === 0 || lowData.length === 0) {
          console.warn("Yüksek/Düşük grafiği için geçerli veri yok.");
          if(container) container.innerHTML = "<p class='text-muted text-center small p-3'>Grafik için veri yok.</p>";
          return;
     }

     const options = JSON.parse(JSON.stringify(commonChartOptions));
     options.plugins.tooltip.callbacks = {
          label: function(context) {
               let label = context.dataset.label || '';
               if (label) label += ': ';
               if (context.parsed.y !== null) label += context.parsed.y.toFixed(2) + '$';
               return label;
           }
       };
     options.plugins.legend.position = 'top';
     // Dolgu plugin'ini etkinleştir
     options.plugins.filler = { propagate: false };
     options.interaction.mode = 'nearest'; // Yakın modu daha iyi olabilir
     options.interaction.axis = 'x';
     options.interaction.intersect = false;

     try {
          highLowChartInstance = new Chart(ctx.getContext('2d'), {
               type: 'line',
               data: {
                   // labels: highData.map(d => d.x),
                    datasets: [
                         {
                              label: 'En Düşük', // Önce düşüğü çiz
                              data: lowData,
                              borderColor: 'rgba(239, 83, 80, 1)', // Kırmızı
                              backgroundColor: 'rgba(239, 83, 80, 0.1)', // Kırmızı dolgu
                              borderWidth: 1.5, pointRadius: 0, tension: 0.1,
                              fill: false // Dolguyu alta yap
                          },
                         {
                             label: 'En Yüksek', // Sonra yükseği çiz
                             data: highData,
                             borderColor: 'rgba(38, 166, 154, 1)', // Yeşil
                             backgroundColor: 'rgba(38, 166, 154, 0.1)', // Yeşil dolgu
                             borderWidth: 1.5, pointRadius: 0, tension: 0.1,
                             fill: '-1' // Önceki datasete (En Düşük) kadar doldur
                         }
                    ]
               },
               options: options
          });
          console.log("High/Low chart created.");
     }
     catch(e) {
          console.error("Error creating High/Low chart:", e);
          if(container) container.innerHTML = "<p class='text-danger text-center small p-3'>Yüksek/Düşük grafiği yüklenemedi.</p>";
     }
}


// --- AJAX Veri Yenileme ---
function refreshStockData() {
    const stockSymbolInput = document.querySelector('#stock');
    const refreshButton = document.getElementById('refreshStockBtn'); // Butonu ID ile al
    const refreshIcon = refreshButton ? refreshButton.querySelector('i') : null;

    if (!stockSymbolInput || !stockSymbolInput.value) {
        console.warn("refreshStockData: Hisse sembolü bulunamadı.");
        return;
    }
    const stockSymbol = stockSymbolInput.value;

    if (refreshIcon) {
        refreshIcon.classList.remove('fa-sync');
        refreshIcon.classList.add('fa-spinner', 'fa-spin');
        if(refreshButton) refreshButton.disabled = true; // Butonu devre dışı bırak
    }
     console.log(`Refreshing data for ${stockSymbol}...`);

    fetch(`/refresh_stock?stock=${stockSymbol}`)
        .then(response => {
            if (!response.ok) {
                 // HTTP hata durumunu yakala (4xx, 5xx)
                 return response.json().then(errData => {
                     throw new Error(errData.message || `HTTP error ${response.status}`);
                 });
            }
            return response.json();
        })
        .then(data => {
            if (data.status === 'success' && data.stock_data) {
                console.log("Data refreshed successfully:", data.stock_data);
                // Gelen veriyi global window.stockData'ya ata (tam veri geliyorsa)
                // VEYA sadece güncellenen alanları kullan
                const refreshedData = data.stock_data; // Sadece gerekli alanlar geliyor

                // Başlık bilgilerini güncelle
                updateStockInfo(refreshedData);

                // Grafik verilerini güncelle (eğer grafikler varsa)
                // Not: AJAX yanıtı tüm grafik verilerini içeriyorsa güncelleyebiliriz
                // Şimdilik sadece fiyat ve hacim güncelleniyor varsayalım
                if (priceChartInstance) {
                    // Mevcut grafik tipini al
                    const activeChartTypeButton = document.querySelector('.chart-type-selector input[name="chartType"]:checked');
                    const chartType = activeChartTypeButton ? activeChartTypeButton.dataset.chartType : 'candlestick';
                    // Güncellenmiş veriyle ana grafiği yeniden oluştur/güncelle
                    // AJAX yanıtının 'candlestick_data' ve 'labels' içerdiğinden emin olmalıyız
                    if (refreshedData.labels && refreshedData.values && refreshedData.candlestick_data) {
                        // Güncellenmiş veriyi global stockData'ya yansıtabiliriz (dikkatli)
                         window.stockData = { ...window.stockData, ...refreshedData };
                         createOrUpdatePriceChart(window.stockData, chartType);
                    } else {
                         console.warn("Refreshed data missing required fields for price chart update.");
                    }
                }
                 if (volumeChartInstance && refreshedData.labels && refreshedData.volume_values) {
                     window.stockData = { ...window.stockData, ...refreshedData };
                     createOrUpdateVolumeChart(window.stockData);
                 }

                // Zaman damgasını güncelle
                const timestampEl = document.querySelector('.last-updated');
                 if (timestampEl && refreshedData.timestamp) {
                    // format_date filtresini JS'de yeniden uygulamak yerine basit formatlama
                     try {
                         const dt = luxon.DateTime.fromSQL(refreshedData.timestamp); // YYYY-MM-DD HH:MM:SS formatını parse et
                         timestampEl.textContent = dt.toFormat('dd.MM.yyyy HH:mm:ss');
                     } catch (e) {
                          console.error("Error formatting refreshed timestamp:", e);
                          timestampEl.textContent = refreshedData.timestamp; // Ham veriyi göster
                     }
                 }

                // Flash mesajı gösterebiliriz (başarı)
                showFlashMessage('Hisse senedi verileri güncellendi.', 'success');

            } else {
                // API'den 'error' durumu geldiyse
                console.error('Veri güncelleme hatası (API):', data.message);
                showFlashMessage(data.message || 'Veri güncellenemedi.', 'danger');
            }
        })
        .catch(error => {
            console.error('AJAX hatası:', error);
            showFlashMessage(`Veri güncelleme başarısız: ${error.message}`, 'danger');
        })
        .finally(() => {
            // Spinner'ı kaldır ve butonu etkinleştir
            if (refreshIcon) {
                refreshIcon.classList.remove('fa-spinner', 'fa-spin');
                refreshIcon.classList.add('fa-sync');
                 if(refreshButton) refreshButton.disabled = false;
            }
        });
}

// Başlık Alanındaki Stok Bilgilerini Güncelle
function updateStockInfo(stockData) {
    const priceEl = document.querySelector('.current-price');
    const changeEl = document.querySelector('.change-percent');
    const marketStatusEl = document.querySelector('.market-status-badge'); // Dashboard header'daki
    const marketIcon = marketStatusEl ? marketStatusEl.querySelector('i') : null;

    if (priceEl && stockData.current_price !== undefined && stockData.current_price !== null) {
         priceEl.textContent = `${stockData.current_price.toFixed(2)}$`;
     } else if (priceEl) {
          priceEl.textContent = 'N/A';
     }

    if (changeEl && stockData.change_percent !== undefined && stockData.change_percent !== null) {
        const changePercent = stockData.change_percent;
        const isPositive = changePercent >= 0;
        // Önceki sınıfları kaldır (text-success/text-danger)
        changeEl.classList.remove('text-success', 'text-danger');
        changeEl.classList.add(isPositive ? 'text-success' : 'text-danger');
        changeEl.innerHTML = `
            <i class="fas ${isPositive ? 'fa-arrow-up' : 'fa-arrow-down'} me-1"></i>
            ${changePercent.toFixed(2)}%
        `;
     } else if (changeEl) {
         changeEl.textContent = 'N/A';
         changeEl.classList.remove('text-success', 'text-danger');
     }

    if (marketStatusEl && stockData.market_status) {
        const isOpen = stockData.market_status === 'Açık';
        // Önceki sınıfları kaldır (bg-success/bg-secondary)
        marketStatusEl.classList.remove('bg-success', 'bg-secondary', 'text-white');
        marketStatusEl.classList.add(isOpen ? 'bg-success' : 'bg-secondary');
        if (!isOpen) marketStatusEl.classList.add('text-white'); // Kapalıyken beyaz yazı

        if (marketIcon) {
            marketIcon.classList.remove('fa-check-circle', 'fa-clock', 'fa-question-circle');
            marketIcon.classList.add(isOpen ? 'fa-check-circle' : 'fa-clock');
        }
         // Metni güncelle (ikon dahil etmeden, zaten var)
         marketStatusEl.childNodes[marketStatusEl.childNodes.length - 1].nodeValue = ` Piyasa ${stockData.market_status}`;

     } else if (marketStatusEl) {
          marketStatusEl.classList.remove('bg-success', 'bg-secondary');
          marketStatusEl.classList.add('bg-secondary', 'text-white');
          if (marketIcon) {
               marketIcon.classList.remove('fa-check-circle', 'fa-clock');
               marketIcon.classList.add('fa-question-circle');
          }
          marketStatusEl.childNodes[marketStatusEl.childNodes.length - 1].nodeValue = ' Piyasa Bilinmiyor';
     }
}

// Dinamik Flash Mesaj Gösterme
function showFlashMessage(message, category = 'info', duration = 4000) {
     const container = document.querySelector('.flash-messages-container') || createFlashContainer();
     if (!container) return;

     const alertDiv = document.createElement('div');
     const alertClass = `alert-${category}`;
     const iconClass = category === 'danger' ? 'fa-exclamation-triangle' :
                       category === 'warning' ? 'fa-exclamation-circle' :
                       category === 'success' ? 'fa-check-circle' : 'fa-info-circle';

     alertDiv.className = `alert ${alertClass} alert-dismissible fade show shadow-sm`;
     alertDiv.setAttribute('role', 'alert');
     alertDiv.innerHTML = `
         <i class="fas ${iconClass} me-2"></i>
         ${message}
         <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
     `;

     container.appendChild(alertDiv);

     // Otomatik kapatma
     setTimeout(() => {
         const bsAlert = bootstrap.Alert.getInstance(alertDiv);
         if (bsAlert) {
             bsAlert.close();
         } else {
              // Bootstrap instance yoksa manuel olarak kaldır
              alertDiv.classList.remove('show');
              setTimeout(() => alertDiv.remove(), 150); // Geçiş efekti için bekle
         }
     }, duration);
 }

 // Flash mesaj container'ı yoksa oluşturur
 function createFlashContainer() {
     let container = document.querySelector('.flash-messages-container');
     if (!container) {
         container = document.createElement('div');
         container.className = 'flash-messages-container position-fixed top-0 end-0 p-3';
         container.style.zIndex = '1055';
         document.body.appendChild(container);
     }
     return container;
 }


// --- DOMContentLoaded Olayları ---
document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM fully loaded and parsed.");

    // Eğer global `window.stockData` varsa ve hata yoksa ilk grafikleri yükle
    if (window.stockData && !window.stockData.error) {
        console.log("Initial stock data found, creating overview charts.");
        createOrUpdatePriceChart(window.stockData, 'candlestick'); // Varsayılan tip
        createOrUpdateVolumeChart(window.stockData);
    } else if (window.stockData && window.stockData.error) {
        console.warn("Initial stock data contains an error:", window.stockData.error);
        // Hata mesajı zaten Flask tarafından flash ile gösterilmiş olmalı
    } else {
        console.log("No initial stock data found or available for charts.");
         // Hoşgeldin ekranı gösteriliyor olmalı
    }

    // Grafik tipi değiştirme butonları
    const chartTypeRadios = document.querySelectorAll('.chart-type-selector input[name="chartType"]');
    chartTypeRadios.forEach(radio => {
        radio.addEventListener('change', function() {
            if (this.checked && window.stockData) {
                const chartType = this.dataset.chartType;
                console.log("Chart type changed to:", chartType);
                createOrUpdatePriceChart(window.stockData, chartType);
            }
        });
    });

    // Yenileme butonu
    const refreshButton = document.getElementById('refreshStockBtn');
    if (refreshButton) {
        refreshButton.addEventListener('click', function() {
            console.log("Refresh button clicked.");
            refreshStockData();
        });
    }

    // Detaylar Sekmesi Gösterildiğinde Grafik Oluşturma
    const detailsTab = document.getElementById('v-pills-details-tab');
    if (detailsTab) {
        detailsTab.addEventListener('shown.bs.tab', function (event) {
            console.log("Details tab shown - initializing detail charts if needed...");
            // Sadece grafikler daha önce oluşturulmadıysa veya veri güncellendiyse oluştur
            if (window.stockData && !window.stockData.error) {
                 // Bu grafikler her sekme açıldığında yeniden oluşturulabilir veya
                 // sadece ilk açılışta oluşturulup sonra güncellenebilir. Şimdilik yeniden oluşturalım.
                 createOpenCloseChart(window.stockData);
                 createDetailVolumeChart(window.stockData);
                 createHighLowChart(window.stockData);
            } else {
                console.warn("Cannot create detail charts: stockData not available or contains error.");
                 // Sekme içeriğinde hata mesajı gösterilebilir
                 const detailsContent = document.getElementById('details');
                 if(detailsContent && !detailsContent.querySelector('.alert')) { // Zaten mesaj yoksa ekle
                      detailsContent.innerHTML = `
                           <div class="alert alert-light text-center mt-4">
                               <i class="fas fa-magnifying-glass-chart fa-2x text-muted mb-3"></i><br>
                               Detaylı analiz grafikleri için veri yüklenemedi.
                           </div>`;
                 }
            }
        });
    }

    // Tahmin Sekmesi Gösterildiğinde Plotly Grafiğini Yeniden Boyutlandırma
    // (Bu olay dinleyici forecast.js içinde de olabilir, çift olmamasına dikkat edin)
    const forecastTab = document.getElementById('v-pills-forecast-tab');
    if (forecastTab && typeof Plotly !== 'undefined') { // Plotly yüklü mü kontrolü
        forecastTab.addEventListener('shown.bs.tab', function () {
            const forecastChartElement = document.getElementById('forecastChart');
            if (forecastChartElement && Plotly.Plots && Plotly.Plots.getPlot(forecastChartElement)) {
                 console.log("Forecast tab shown, resizing Plotly chart.");
                try {
                     // Kısa bir gecikme ile yeniden boyutlandırma daha stabil olabilir
                     setTimeout(() => Plotly.Plots.resize(forecastChartElement), 100);
                } catch(e) {
                     console.error("Plotly resize error:", e);
                }
            } else {
                 console.warn("Forecast tab shown, but Plotly chart element not found or ready.");
            }
        });
    }

    // Sayfa ilk yüklendiğinde tahmin verilerini yükle (eğer hisse seçiliyse)
    const initialStockSymbol = document.getElementById('stock')?.value;
    if (initialStockSymbol && typeof loadForecastData === 'function') {
        console.log(`Initial stock symbol found (${initialStockSymbol}), loading forecast data.`);
        // loadForecastData zaten forecast.js içinde çağrılıyor olmalı (DOMContentLoaded'de)
        // Bu yüzden burada tekrar çağırmaya gerek yok, çift yüklemeyi önleyelim.
        // loadForecastData(initialStockSymbol);
    } else if (initialStockSymbol) {
        console.warn("Initial stock symbol found, but 'loadForecastData' function is not available. Forecast might not load automatically.");
    }


}); // DOMContentLoaded Sonu