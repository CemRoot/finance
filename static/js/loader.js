let statusInterval;
let progressInterval;
let currentStatusIndex = 0;
let loaderElement;
let statusTextElement;
let neuralNetworkElement;
let progressBarElement;
let backgroundNumbersElement;

const statusMessages = [
    "Initializing machine learning models...",
    "Fetching & preparing historical data...",
    "Analyzing market patterns...",
    "Training models (RF, SVM, XGB)...",
    "Running deep learning analysis (LSTM)...",
    "Comparing model performances...",
    "Generating visualizations...",
    "Finalizing analysis..."
];

function createNeuralNetwork(nodeCount = 9, layerCount = 3, spacing = 45) {
    if (!neuralNetworkElement) return;
    neuralNetworkElement.innerHTML = '';
    const networkWidth = (layerCount - 1) * spacing;
    const networkHeight = 130;
    const layerNodes = [];

    for (let l = 0; l < layerCount; l++) {
        const currentLayerNodes = [];
        const layerX = l * spacing + (neuralNetworkElement.offsetWidth / 2 - networkWidth / 2);
        const nodesInLayer = nodeCount - Math.abs(l - Math.floor(layerCount / 2)) * 2.5;

        for (let n = 0; n < nodesInLayer; n++) {
            const node = document.createElement('div');
            node.className = 'node';
            const nodeY = (n * (networkHeight / (nodesInLayer - 1 || 1))) + (neuralNetworkElement.offsetHeight / 2 - networkHeight / 2);
            node.style.left = `${layerX - 5}px`;
            node.style.top = `${nodeY - 5}px`;
            node.style.animationDelay = `${Math.random() * 1.2}s`;
            neuralNetworkElement.appendChild(node);
            currentLayerNodes.push({ x: layerX, y: nodeY });
        }
        layerNodes.push(currentLayerNodes);
    }

    for (let l = 0; l < layerCount - 1; l++) {
        for (const node1 of layerNodes[l]) {
            for (const node2 of layerNodes[l + 1]) {
                const connection = document.createElement('div');
                connection.className = 'connection';
                const angle = Math.atan2(node2.y - node1.y, node2.x - node1.x);
                const length = Math.hypot(node2.x - node1.x, node2.y - node1.y);
                connection.style.width = `${length}px`;
                connection.style.left = `${node1.x}px`;
                connection.style.top = `${node1.y}px`;
                connection.style.transform = `rotate(${angle}rad)`;
                connection.style.animationDelay = `${Math.random() * 1.5 + 0.2}s`;
                neuralNetworkElement.appendChild(connection);
            }
        }
    }
}

function createBackgroundNumbers(count = 60) {
    if (!backgroundNumbersElement) return;
    backgroundNumbersElement.innerHTML = '';
    for (let i = 0; i < count; i++) {
        const span = document.createElement('span');
        span.textContent = (Math.random() > 0.5 ? '1' : '0');
        span.style.left = `${Math.random() * 100}vw`;
        span.style.top = `${Math.random() * 110 - 10}vh`;
        span.style.fontSize = `${Math.random() * 0.6 + 0.7}rem`;
        const duration = Math.random() * 8 + 12;
        const delay = Math.random() * 8;
        span.style.animation = `moveNumbers ${duration}s linear ${delay}s infinite`;
        backgroundNumbersElement.appendChild(span);
    }
}

function updateStatusText() {
    if (!statusTextElement) return;
    statusTextElement.classList.remove('visible');
    setTimeout(() => {
         currentStatusIndex = (currentStatusIndex + 1 < statusMessages.length ? currentStatusIndex + 1 : currentStatusIndex);
         statusTextElement.textContent = statusMessages[currentStatusIndex];
         statusTextElement.classList.add('visible');
    }, 400);
}

function showLoader(estimatedDuration = 10000) {
    loaderElement = document.getElementById('loaderContainer');
    statusTextElement = document.getElementById('statusText');
    neuralNetworkElement = document.getElementById('neuralNetwork');
    progressBarElement = document.getElementById('progressBar');
    backgroundNumbersElement = document.getElementById('backgroundNumbers');

    if (!loaderElement || !statusTextElement || !neuralNetworkElement || !progressBarElement || !backgroundNumbersElement) {
         console.error("Loader elements not found!"); return;
    }
    if (loaderElement.classList.contains('visible')) return;

    loaderElement.classList.add('visible');
    neuralNetworkElement.innerHTML = '';
    createNeuralNetwork();
    createBackgroundNumbers();
    currentStatusIndex = -1;
    updateStatusText();

    const statusUpdateIntervalTime = Math.max(1500, estimatedDuration / (statusMessages.length || 1));
    clearInterval(statusInterval);
    statusInterval = setInterval(updateStatusText, statusUpdateIntervalTime);

    let progress = 0;
    progressBarElement.style.width = '0%';
    clearInterval(progressInterval);
    progressInterval = setInterval(() => {
        progress += 100 / (estimatedDuration / 100);
        progressBarElement.style.width = `${Math.min(progress, 100)}%`;
        if (progress >= 100) {
            clearInterval(progressInterval);
            progressInterval = null;
        }
    }, 100);
    console.log("Loader shown");
}

function hideLoader() {
    clearInterval(statusInterval);
    clearInterval(progressInterval);
    statusInterval = null;
    progressInterval = null;
    if (loaderElement) loaderElement.classList.remove('visible');
    if (backgroundNumbersElement) backgroundNumbersElement.innerHTML = '';
    if (progressBarElement) progressBarElement.style.width = '0%';
    console.log("Loader hidden");
}

// Initialize elements on DOM load
document.addEventListener('DOMContentLoaded', function() {
    loaderElement = document.getElementById('loaderContainer');
    statusTextElement = document.getElementById('statusText');
    neuralNetworkElement = document.getElementById('neuralNetwork');
    progressBarElement = document.getElementById('progressBar');
    backgroundNumbersElement = document.getElementById('backgroundNumbers');
});
