// static/js/main.js
console.log("main.js loaded.");

/**
 * Populates the search input with the given stock symbol and submits the search form.
 * Shows a loading indicator on the search button during submission.
 *
 * @param {string} symbol The stock symbol to load (e.g., 'AAPL').
 */
function loadStock(symbol) {
    console.log(`loadStock called with symbol: ${symbol}`);
    const stockInput = document.getElementById('stock'); // Target the search input
    const searchForm = document.getElementById('searchForm'); // Target the search form
    // Find the submit button within the form specifically
    const submitButton = searchForm ? searchForm.querySelector('button[type="submit"]') : null;
    const buttonIcon = submitButton ? submitButton.querySelector('i') : null; // Find the icon within the button

    // Ensure all required elements exist
    if (stockInput && searchForm && submitButton && buttonIcon) {
        stockInput.value = symbol; // Update the input field value

        // Visually indicate loading state on the button
        buttonIcon.classList.remove('fa-search'); // Remove search icon
        buttonIcon.classList.add('fa-spinner', 'fa-spin'); // Add spinner icon
        submitButton.disabled = true; // Disable button to prevent multiple clicks
        console.log(`Set loading state for symbol: ${symbol}`);

        // Submit the form programmatically after a short delay
        // This allows the browser to render the loading state before navigation starts.
        // The form uses POST, which redirects to GET via app.py's PRG pattern.
        setTimeout(() => {
            try {
                console.log("Submitting search form...");
                searchForm.submit();
            } catch (e) {
                // In case submit fails unexpectedly
                console.error("Error submitting search form:", e);
                // Reset button state on error
                buttonIcon.classList.remove('fa-spinner', 'fa-spin');
                buttonIcon.classList.add('fa-search');
                submitButton.disabled = false;
            }
        }, 150); // 150ms delay

    } else {
        // Log an error if any required element is missing
        console.error("loadStock: Could not find required elements (stock input, search form, submit button, or button icon).");
        // Attempt to log which elements were found/missing
        if (!stockInput) console.error("Missing: #stock input");
        if (!searchForm) console.error("Missing: #searchForm");
        if (!submitButton) console.error("Missing: submit button in #searchForm");
        if (!buttonIcon) console.error("Missing: icon (<i>) in submit button");
    }
}

// --- DOMContentLoaded Event Listener ---
// Executes code once the basic HTML structure is loaded.
document.addEventListener('DOMContentLoaded', function() {
    console.log("main.js: DOMContentLoaded event fired.");

    // Add submit listener to the search form to show loading state
    // (This handles cases where the user types and presses Enter/clicks Search manually)
    const searchForm = document.getElementById('searchForm');
    if (searchForm) {
        searchForm.addEventListener('submit', function(event) {
             // Check if the input is actually filled (basic validation)
             const stockInput = document.getElementById('stock');
             if (!stockInput || !stockInput.value.trim()) {
                  console.log("Search form submitted with empty input. Preventing submission.");
                  event.preventDefault(); // Prevent empty submission
                  // Optionally show a message to the user
                  stockInput?.focus(); // Focus the input field
                  showFlashMessage("Please enter a stock symbol.", "warning"); // Use flash message utility
                  return; // Stop processing
             }

             // If input is valid, show loading state
             console.log("Search form submitted manually, showing loading state.");
             const submitButton = searchForm.querySelector('button[type="submit"]');
             const buttonIcon = submitButton ? submitButton.querySelector('i') : null;
             if (submitButton && buttonIcon) {
                 buttonIcon.classList.remove('fa-search');
                 buttonIcon.classList.add('fa-spinner', 'fa-spin');
                 submitButton.disabled = true;
             }
             // Allow the form submission to proceed naturally
        });
        console.log("Submit listener added to #searchForm.");
    } else {
         console.warn("main.js: Search form (#searchForm) not found. Submit listener not added.");
    }

     // Optional: Populate search input from URL parameter on page load/refresh
     // This helps maintain state if the page is reloaded with a stock parameter
     try {
         const urlParams = new URLSearchParams(window.location.search);
         const stockParam = urlParams.get('stock');
         const stockInput = document.getElementById('stock');
         if (stockParam && stockInput && !stockInput.value) { // Only fill if input is currently empty
             console.log(`Populating stock input from URL parameter: ${stockParam}`);
             stockInput.value = stockParam;
         }
     } catch (e) {
         console.error("Error processing URL parameters:", e);
     }

     // Initialize Bootstrap Tooltips (can be done here or in script.js, ensure it's done once)
     // If script.js already does this, remove it from here.
     /*
     const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
     if (tooltipTriggerList.length > 0) {
         tooltipTriggerList.map(function (tooltipTriggerEl) {
           return new bootstrap.Tooltip(tooltipTriggerEl);
         });
         console.log("main.js: Bootstrap tooltips initialized.");
     }
     */

     console.log("main.js: Initial setup complete.");

}); // End DOMContentLoaded