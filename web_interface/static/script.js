document.addEventListener('DOMContentLoaded', () => {
    init();
});

let eegChart = null;

// State
const state = {
    subject: 1,
    loso: false,
    modelType: 'original',
    currentTrialIndex: 0,
    totalTrials: 0
};

async function init() {
    // Add event listeners
    document.getElementById('strategy-select').addEventListener('change', (e) => {
        state.loso = e.target.value === 'loso';
        updateSubjectUI();
    });

    document.getElementById('model-select').addEventListener('change', (e) => {
        state.modelType = e.target.value;
    });

    document.getElementById('load-btn').addEventListener('click', loadModel);
    document.getElementById('predict-btn').addEventListener('click', predictTrial);
    document.getElementById('run-all-btn').addEventListener('click', runBatch);

    // Initial UI state check
    updateSubjectUI();

    // Fetch config
    try {
        const response = await fetch('/api/init');
        const data = await response.json();

        // Populate Model Select
        const modelSelect = document.getElementById('model-select');
        modelSelect.innerHTML = data.model_types.map(t =>
            `<option value="${t.id}">${t.name}</option>`
        ).join('');

    } catch (e) {
        console.error("Init failed", e);
    }
}

function updateSubjectUI() {
    const prevBtn = document.querySelector('.selector-btn.prev');
    const nextBtn = document.querySelector('.selector-btn.next');
    const subjectDisplay = document.getElementById('subject-display');
    const modelSelect = document.getElementById('model-select');

    if (state.loso) {
        // Lock to Subject 1 for Independent
        state.subject = 1;
        subjectDisplay.innerText = '1';
        prevBtn.disabled = true;
        nextBtn.disabled = true;
        prevBtn.style.opacity = '0.3';
        nextBtn.style.opacity = '0.3';

        // Filter Model Options: Show only Original (Keras)
        // Adjust this if you have other types, but user asked to remove "quantized (tflite)"
        Array.from(modelSelect.options).forEach(opt => {
            if (opt.value === 'quantized') {
                opt.style.display = 'none';
            } else {
                opt.style.display = 'block';
            }
        });
        // Force selection to original if hidden option was selected
        if (modelSelect.value === 'quantized') {
            modelSelect.value = 'original';
            state.modelType = 'original';
        }

    } else {
        prevBtn.disabled = false;
        nextBtn.disabled = false;
        prevBtn.style.opacity = '1';
        nextBtn.style.opacity = '1';

        // Show all options
        Array.from(modelSelect.options).forEach(opt => {
            opt.style.display = 'block';
        });
    }
}

function adjustSubject(delta) {
    if (state.loso) return; // Prevent change if locked

    let newSub = state.subject + delta;
    if (newSub < 1) newSub = 9;
    if (newSub > 9) newSub = 1;
    state.subject = newSub;
    document.getElementById('subject-display').innerText = newSub;
}

async function loadModel() {
    const btn = document.getElementById('load-btn');
    const status = document.getElementById('system-status');
    const modelName = document.getElementById('current-model-name');
    const timeDisplay = document.getElementById('load-time');

    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Loading...';
    status.className = 'value';
    status.innerText = 'Loading...';

    try {
        const response = await fetch('/api/load_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                subject: state.subject,
                loso: state.loso,
                model_type: document.getElementById('model-select').value
            })
        });

        const data = await response.json();

        if (response.ok) {
            status.innerText = 'Ready';
            status.className = 'value ready';
            modelName.innerText = data.message.replace('Model selected: ', '');
            timeDisplay.innerText = data.data_load_time.toFixed(2) + 's';

            // Enable Predict Button
            document.getElementById('predict-btn').disabled = false;

            // Load a random trial
            getTrial('random');

        } else {
            status.innerText = 'Error';
            status.className = 'value incorrect';
            alert('Error: ' + data.error);
        }

    } catch (e) {
        console.error(e);
        status.innerText = 'Error';
        alert('Failed to connect to server');
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-download"></i> Load Model';
    }
}

async function getTrial(mode) {
    if (mode === 'index') {
        state.currentTrialIndex = document.getElementById('trial-index-input').value;
    } else if (mode === 'next') {
        state.currentTrialIndex++;
    } else if (mode === 'prev') {
        state.currentTrialIndex--;
        if (state.currentTrialIndex < 0) state.currentTrialIndex = 0;
    }

    try {
        const response = await fetch('/api/get_trial', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                mode: mode,
                index: state.currentTrialIndex
            })
        });

        const data = await response.json();

        if (response.ok) {
            // Update UI
            state.currentTrialIndex = data.trial_index;
            state.totalTrials = data.total_trials;

            document.getElementById('current-trial-id').innerText = data.trial_index;
            document.getElementById('total-trials').innerText = data.total_trials;
            document.getElementById('true-label').innerText = data.true_label_name;
            document.getElementById('trial-index-input').value = data.trial_index;

            // Reset prediction result
            resetPredictionUI();

            // Render Chart
            renderChart(data.eeg_data);
        } else {
            alert(data.error);
        }

    } catch (e) {
        console.error(e);
    }
}

async function predictTrial() {
    const btn = document.getElementById('predict-btn');
    btn.disabled = true;

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ index: state.currentTrialIndex })
        });

        const result = await response.json();

        if (response.ok) {
            document.getElementById('pred-class').innerText = result.predicted_label;
            document.getElementById('pred-conf').innerText = (result.confidence * 100).toFixed(1) + '%';
            document.getElementById('pred-time').innerText = result.inference_time_ms.toFixed(1) + ' ms';

            const icon = document.getElementById('pred-icon');
            const box = document.getElementById('pred-result-box');

            if (result.correct) {
                icon.innerHTML = '<i class="fa-solid fa-check-circle correct"></i>';
                box.style.borderBottom = '4px solid var(--accent-green)';
            } else {
                icon.innerHTML = '<i class="fa-solid fa-circle-xmark incorrect"></i>';
                box.style.borderBottom = '4px solid var(--accent-red)';
            }
        }

    } catch (e) {
        console.error(e);
    } finally {
        btn.disabled = false;
    }
}

async function runBatch() {
    const btn = document.getElementById('run-all-btn');
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing...';

    // Clear table
    const tbody = document.querySelector('#results-table tbody');
    tbody.innerHTML = '<tr><td colspan="5" style="text-align:center">Processing... this may take a while</td></tr>';

    try {
        const response = await fetch('/api/predict_all', { method: 'POST' });
        const data = await response.json();

        if (response.ok) {
            document.getElementById('batch-acc').innerText = data.accuracy.toFixed(2) + '%';
            document.getElementById('batch-time').innerText = data.avg_time_ms.toFixed(2) + ' ms';

            // Populate Table
            tbody.innerHTML = '';
            data.results.forEach(row => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${row.id}</td>
                    <td>${row.true}</td>
                    <td>${row.pred}</td>
                    <td class="${row.correct ? 'correct' : 'incorrect'}">${row.correct ? 'Correct' : 'Wrong'}</td>
                    <td>${row.time_ms.toFixed(1)}</td>
                `;
                tbody.appendChild(tr);
            });
        }
    } catch (e) {
        console.error(e);
        tbody.innerHTML = '<tr><td colspan="5">Error running batch</td></tr>';
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-play"></i> Run All Trials';
    }
}

function renderChart(eegData) {
    const ctx = document.getElementById('eegChart').getContext('2d');

    // BCI Competition IV 2a Standard Channel Order
    const channelNames = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
        'P1', 'Pz', 'P2', 'POz'
    ];

    // Construct labels (time points)
    const labels = Array.from({ length: eegData[0].length }, (_, i) => i);

    const datasets = eegData.map((channelData, index) => {
        // Generate a distinct color for each channel
        // Using HSL to spread colors evenly
        const hue = (index * 360 / 22) % 360;
        const color = `hsl(${hue}, 70%, 50%)`;

        return {
            label: channelNames[index] || `Ch ${index + 1}`,
            data: channelData,
            borderColor: color,
            borderWidth: 1,
            pointRadius: 0,
            hidden: false // Show all by default
        };
    });

    if (eegChart) {
        eegChart.destroy();
    }

    eegChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'nearest',
            },
            scales: {
                x: {
                    display: false,
                    grid: { color: 'rgba(255,255,255,0.05)' }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.05)' }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#94a3b8',
                        boxWidth: 10,
                        font: { size: 10 }
                    },
                    position: 'right', // Move legend to right sidebar style to avoid clutter
                    maxHeight: 400
                },
                tooltip: {
                    enabled: true,
                    mode: 'index',
                    intersect: false
                }
            },
            elements: {
                line: { tension: 0.1 } // slight smoothing
            }
        }
    });
}

function switchTab(tab) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.view-panel').forEach(p => p.classList.remove('active'));

    // Simple naive check based on text
    const buttons = document.querySelectorAll('.tab-btn');
    if (tab === 'single') {
        buttons[0].classList.add('active');
        document.getElementById('single-view').classList.add('active');
    } else {
        buttons[1].classList.add('active');
        document.getElementById('batch-view').classList.add('active');
    }
}

function resetPredictionUI() {
    document.getElementById('pred-class').innerText = '--';
    document.getElementById('pred-conf').innerText = '--';
    document.getElementById('pred-time').innerText = '-- ms';
    document.getElementById('pred-result-box').style.borderBottom = 'none';
    document.getElementById('pred-icon').innerHTML = '';
}
