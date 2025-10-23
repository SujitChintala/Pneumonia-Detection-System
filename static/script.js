// JavaScript for Pneumonia Detection System Frontend

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewArea = document.getElementById('previewArea');
const previewImage = document.getElementById('previewImage');
const removeBtn = document.getElementById('removeBtn');
const actionButtons = document.getElementById('actionButtons');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('resultsSection');
const analyzeAnotherBtn = document.getElementById('analyzeAnotherBtn');

let selectedFile = null;

// Event Listeners
uploadArea.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
removeBtn.addEventListener('click', clearImage);
clearBtn.addEventListener('click', clearImage);
analyzeBtn.addEventListener('click', analyzeImage);
analyzeAnotherBtn.addEventListener('click', resetAll);

// Drag and Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// Handle file selection
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

// Handle file
function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file (JPG, JPEG, PNG)');
        return;
    }

    selectedFile = file;

    // Preview image
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadArea.style.display = 'none';
        previewArea.style.display = 'block';
        actionButtons.style.display = 'flex';
        resultsSection.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// Clear image
function clearImage() {
    selectedFile = null;
    previewImage.src = '';
    fileInput.value = '';
    uploadArea.style.display = 'block';
    previewArea.style.display = 'none';
    actionButtons.style.display = 'none';
    resultsSection.style.display = 'none';
}

// Reset all
function resetAll() {
    clearImage();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Analyze image
async function analyzeImage() {
    if (!selectedFile) {
        alert('Please select an image first');
        return;
    }

    // Hide elements and show loading
    actionButtons.style.display = 'none';
    resultsSection.style.display = 'none';
    loading.style.display = 'block';

    // Create form data
    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
        // Send request to API
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        // Hide loading
        loading.style.display = 'none';

        if (data.success) {
            displayResults(data);
        } else {
            alert('Error: ' + (data.error || 'Unknown error occurred'));
            actionButtons.style.display = 'flex';
        }
    } catch (error) {
        console.error('Error:', error);
        loading.style.display = 'none';
        alert('Error connecting to server. Please make sure the server is running.');
        actionButtons.style.display = 'flex';
    }
}

// Display results
function displayResults(data) {
    // Show results section
    resultsSection.style.display = 'block';

    // Set prediction
    const resultPrediction = document.getElementById('resultPrediction');
    const resultConfidence = document.getElementById('resultConfidence');
    const resultIcon = document.getElementById('resultIcon');

    resultPrediction.textContent = data.prediction;
    resultConfidence.textContent = data.confidence + '%';

    // Set color based on prediction
    if (data.prediction === 'NORMAL') {
        resultPrediction.className = 'result-prediction normal';
        resultIcon.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="#10b981">
                <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" stroke-width="2"/>
            </svg>
        `;
    } else {
        resultPrediction.className = 'result-prediction pneumonia';
        resultIcon.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="#ef4444">
                <path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" stroke-width="2"/>
            </svg>
        `;
    }

    // Set probabilities
    const normalProb = data.probabilities.NORMAL;
    const pneumoniaProb = data.probabilities.PNEUMONIA;

    document.getElementById('probNormal').textContent = normalProb + '%';
    document.getElementById('probPneumonia').textContent = pneumoniaProb + '%';
    document.getElementById('probBarNormal').style.width = normalProb + '%';
    document.getElementById('probBarPneumonia').style.width = pneumoniaProb + '%';

    // Smooth scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 100);
}

// Health check on page load
async function checkHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        if (!data.model_loaded) {
            console.warn('Model not loaded on server');
        }
    } catch (error) {
        console.error('Health check failed:', error);
    }
}

// Initialize
checkHealth();
