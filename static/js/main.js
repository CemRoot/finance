// static/js/main.js
console.log("main.js yüklendi.");

// Popüler hisse senedi butonlarına tıklama olayını yönetir
function loadStock(symbol) {
    const stockInput = document.getElementById('stock');
    const searchForm = document.getElementById('searchForm');
    const submitButton = searchForm ? searchForm.querySelector('button[type="submit"]') : null;
    const buttonIcon = submitButton ? submitButton.querySelector('i') : null;

    if (stockInput && searchForm) {
        stockInput.value = symbol; // Input alanını güncelle

        // Yükleme durumunu göster
        if (submitButton && buttonIcon) {
            buttonIcon.classList.remove('fa-search');
            buttonIcon.classList.add('fa-spinner', 'fa-spin');
            submitButton.disabled = true;
        } else {
            console.warn("Arama butonu veya ikonu bulunamadı.");
        }

        // Kısa bir gecikmeyle formu gönder (spinner'ın görünmesi için)
        setTimeout(() => {
            searchForm.submit(); // Formu POST et (app.py'deki PRG modeline göre GET'e yönlenecek)
        }, 100); // 100ms bekle

    } else {
        console.error("loadStock: Gerekli elementler bulunamadı (stockInput veya searchForm).");
    }
}

// Piyasa durumu rozetini periyodik olarak güncellemek için (Opsiyonel - Backend zaten veriyor)
// Bu fonksiyon frontend tarafında saati kontrol ederek anlık durumu gösterir,
// ancak backend'den gelen veri daha güvenilirdir.
// function updateMarketStatusBadgeLocally() {
//     const marketStatusBadge = document.querySelector('.market-status-badge');
//     if (!marketStatusBadge) return;
//     const marketIcon = marketStatusBadge.querySelector('i');

//     try {
//         // New York saatini al (Basit DST varsayımı)
//         const now = new Date();
//         const utcHour = now.getUTCHours();
//         const month = now.getMonth(); // 0-11
//         // Basit DST: Mart'ın 2. Pazarından Kasım'ın 1. Pazarına kadar (ABD kuralı)
//         // Bu çok kaba bir tahmin, moment-timezone daha doğru olurdu.
//         const isDST = (month > 1 && month < 10); // Mart-Ekim arası yaklaşık
//         const nyOffset = isDST ? -4 : -5;
//         const nyHour = (utcHour + nyOffset + 24) % 24;
//         const nyMinutes = now.getUTCMinutes();
//         const nyDay = now.getUTCDay(); // 0=Pazar, 6=Cumartesi

//         // Piyasa açık mı? (Pazartesi-Cuma, 9:30 - 16:00 NY saati)
//         const isOpen = nyDay >= 1 && nyDay <= 5 &&
//                       (nyHour > 9 || (nyHour === 9 && nyMinutes >= 30)) &&
//                       nyHour < 16;

//         // Rozeti güncelle
//         marketStatusBadge.classList.remove('bg-success', 'bg-secondary', 'text-white');
//         marketStatusBadge.classList.add(isOpen ? 'bg-success' : 'bg-secondary');
//         if (!isOpen) marketStatusBadge.classList.add('text-white');

//         if (marketIcon) {
//             marketIcon.classList.remove('fa-check-circle', 'fa-clock', 'fa-question-circle');
//             marketIcon.classList.add(isOpen ? 'fa-check-circle' : 'fa-clock');
//         }
//         // İkonun yanındaki metni güncelle (varsa)
//         const textNode = Array.from(marketStatusBadge.childNodes).find(node => node.nodeType === Node.TEXT_NODE);
//         if (textNode) {
//             textNode.nodeValue = ` Piyasa ${isOpen ? 'Açık' : 'Kapalı'}`;
//         }


//     } catch (e) {
//         console.error("Error updating market status locally:", e);
//          // Hata durumunda varsayılanı göster
//          marketStatusBadge.classList.remove('bg-success', 'bg-secondary');
//          marketStatusBadge.classList.add('bg-secondary', 'text-white');
//           if (marketIcon) {
//                marketIcon.classList.remove('fa-check-circle', 'fa-clock');
//                marketIcon.classList.add('fa-question-circle');
//           }
//            const textNode = Array.from(marketStatusBadge.childNodes).find(node => node.nodeType === Node.TEXT_NODE);
//            if (textNode) textNode.nodeValue = ' Piyasa Bilinmiyor';
//     }
// }

// Belge yüklendiğinde olay dinleyicilerini başlat
document.addEventListener('DOMContentLoaded', function() {
    console.log("main.js: DOMContentLoaded event fired.");

    // Arama formu gönderildiğinde yükleme durumunu göster
    const searchForm = document.getElementById('searchForm');
    if (searchForm) {
        searchForm.addEventListener('submit', function() {
             const submitButton = searchForm.querySelector('button[type="submit"]');
             const buttonIcon = submitButton ? submitButton.querySelector('i') : null;
             if (submitButton && buttonIcon) {
                 buttonIcon.classList.remove('fa-search');
                 buttonIcon.classList.add('fa-spinner', 'fa-spin');
                 submitButton.disabled = true;
                 console.log("Search form submitted, showing spinner.");
             }
        });
    }

    // URL'deki 'stock' parametresine göre inputu doldur (sayfa yenilendiğinde)
     const urlParams = new URLSearchParams(window.location.search);
     const stockParam = urlParams.get('stock');
     const stockInput = document.getElementById('stock');
     if (stockParam && stockInput && !stockInput.value) { // Sadece input boşsa doldur
         stockInput.value = stockParam;
         console.log(`Stock input populated from URL param: ${stockParam}`);
     }

     // Piyasa durumunu lokal olarak güncelle (opsiyonel, her dakika)
     // Backend'den gelen veri daha güvenilir olduğu için bunu devredışı bırakabiliriz.
     // updateMarketStatusBadgeLocally();
     // setInterval(updateMarketStatusBadgeLocally, 60000); // Her 60 saniyede bir

     // Bootstrap Tooltip'lerini etkinleştir (varsa)
     const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
     tooltipTriggerList.map(function (tooltipTriggerEl) {
       return new bootstrap.Tooltip(tooltipTriggerEl);
     });

     // Sidebar aktif linkini ayarla (Bootstrap bunu nav-pills ile otomatik yapar)
     // Eğer sayfa yenilendiğinde aktif sekme kayboluyorsa, localStorage ile saklanabilir.
     const sidebarLinks = document.querySelectorAll('#v-pills-tab .nav-link');
     sidebarLinks.forEach(link => {
         link.addEventListener('click', function() {
              // Aktif sekmenin ID'sini localStorage'a kaydet
              // localStorage.setItem('activeFinTab', this.id);
              console.log(`Sidebar tab clicked: ${this.id}`);
         });
     });

      // Sayfa yüklendiğinde localStorage'dan aktif sekmeyi yükle (opsiyonel)
     // const activeTabId = localStorage.getItem('activeFinTab');
     // if (activeTabId) {
     //     const activeTab = document.getElementById(activeTabId);
     //     if (activeTab) {
     //         const tab = new bootstrap.Tab(activeTab);
     //         tab.show();
     //         console.log(`Restored active tab: ${activeTabId}`);
     //     } else {
     //          // Eğer kaydedilen ID yoksa veya geçersizse varsayılanı (overview) aktif et
     //          const defaultTab = document.getElementById('v-pills-overview-tab');
     //          if (defaultTab) new bootstrap.Tab(defaultTab).show();
     //     }
     // }


});