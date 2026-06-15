const API_BASE = 'http://localhost:8000/api';

// DOM Elements
const els = {
    uptime: document.getElementById('uptime'),
    frameCount: document.getElementById('frame-count'),
    riskScore: document.getElementById('risk-score'),
    scoreRingFill: document.getElementById('score-ring-fill'),
    severityBadge: document.getElementById('severity-badge'),
    facePresent: document.getElementById('face-present'),
    personCount: document.getElementById('person-count'),
    phoneDetected: document.getElementById('phone-detected'),
    gazeStatus: document.getElementById('gaze-status'),
    totalAlerts: document.getElementById('total-alerts'),
    alertsList: document.getElementById('alerts-list'),
    pulseIndicator: document.querySelector('.pulse-indicator'),
    fps: document.getElementById('fps'),
    infTime: document.getElementById('inf-time'),
    cpuUsage: document.getElementById('cpu-usage'),
    ramUsage: document.getElementById('ram-usage'),
    endSessionBtn: document.getElementById('end-session-btn'),
    reportModal: document.getElementById('report-modal'),
    reportContent: document.getElementById('report-content'),
    closeModalBtn: document.getElementById('close-modal-btn')
};

// State
let lastAlertCount = 0;

// Utility functions
const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}m ${secs}s`;
};

const getSeverityClass = (severity) => {
    const map = {
        'SAFE': 'safe',
        'LOW': 'low',
        'MEDIUM': 'medium',
        'HIGH': 'high',
        'CRITICAL': 'critical'
    };
    return map[severity] || 'safe';
};

const updateSeverityStyling = (el, severity, isBg = false) => {
    el.className = isBg ? `severity-badge bg-${getSeverityClass(severity)}` : `metric-value text-${getSeverityClass(severity)}`;
};

const updateScoreRing = (score, severity) => {
    const maxOffset = 283;
    const offset = maxOffset - (score / 100) * maxOffset;
    els.scoreRingFill.style.strokeDashoffset = offset;
    
    // Update color based on severity
    const cssVarMap = {
        'SAFE': 'var(--color-safe)',
        'LOW': 'var(--color-low)',
        'MEDIUM': 'var(--color-medium)',
        'HIGH': 'var(--color-high)',
        'CRITICAL': 'var(--color-critical)'
    };
    els.scoreRingFill.style.stroke = cssVarMap[severity] || 'var(--color-safe)';
    
    // Update pulse indicator color
    els.pulseIndicator.style.backgroundColor = cssVarMap[severity] || 'var(--color-safe)';
    els.pulseIndicator.style.boxShadow = `0 0 10px ${cssVarMap[severity] || 'var(--color-safe)'}`;
};

// Fetch and update status
const updateStatus = async () => {
    try {
        const response = await fetch(`${API_BASE}/status`);
        if (!response.ok) throw new Error('API Error');
        const data = await response.json();
        
        // Header info
        els.uptime.innerText = formatTime(data.uptime_seconds);
        els.frameCount.innerText = data.total_frames_processed.toLocaleString();
        els.totalAlerts.innerText = data.total_alerts;

        // Update alerts if count changed
        if (data.total_alerts > lastAlertCount) {
            updateAlerts();
            lastAlertCount = data.total_alerts;
        }

        // Decision data
        if (data.current_decision) {
            const dec = data.current_decision;
            
            // Risk score & Severity
            els.riskScore.innerText = dec.risk_score;
            els.severityBadge.innerText = dec.severity;
            updateSeverityStyling(els.severityBadge, dec.severity, true);
            updateScoreRing(dec.risk_score, dec.severity);
            
            // Face
            els.facePresent.innerText = dec.face_present ? 'Detected' : 'Missing';
            updateSeverityStyling(els.facePresent, dec.face_present ? 'SAFE' : 'HIGH');
            
            // Persons
            els.personCount.innerText = dec.person_count;
            updateSeverityStyling(els.personCount, dec.person_count === 1 ? 'SAFE' : (dec.person_count === 0 ? 'LOW' : 'HIGH'));
            
            // Phone
            els.phoneDetected.innerText = dec.phone_detected ? 'Detected!' : 'None';
            updateSeverityStyling(els.phoneDetected, dec.phone_detected ? 'HIGH' : 'SAFE');
            
            // Gaze
            els.gazeStatus.innerText = dec.gaze_status;
            let gazeSev = 'SAFE';
            if (dec.gaze_status === 'AWAY') gazeSev = 'MEDIUM';
            else if (dec.gaze_status === 'UNKNOWN') gazeSev = 'LOW';
            updateSeverityStyling(els.gazeStatus, gazeSev);
        } else {
            // Waiting for frames
            els.severityBadge.innerText = "WAITING";
            els.severityBadge.className = "severity-badge bg-low";
            els.facePresent.innerText = "Waiting...";
            els.personCount.innerText = "-";
            els.phoneDetected.innerText = "Waiting...";
            els.gazeStatus.innerText = "Waiting...";
            els.riskScore.innerText = "0";
            updateScoreRing(0, 'SAFE');
        }
    } catch (e) {
        console.error("Failed to fetch status", e);
        els.severityBadge.innerText = "DISCONNECTED";
        els.severityBadge.className = "severity-badge bg-critical";
        els.pulseIndicator.style.animation = 'none';
        els.pulseIndicator.style.backgroundColor = 'var(--color-critical)';
    }
};

// Fetch and update alerts
const updateAlerts = async () => {
    try {
        const response = await fetch(`${API_BASE}/alerts`);
        if (!response.ok) return;
        const alerts = await response.json();
        
        if (alerts.length === 0) {
            els.alertsList.innerHTML = '<div class="empty-state">No alerts detected yet.</div>';
            return;
        }
        
        // Show 5 most recent alerts, reversed so newest is on top
        const recent = alerts.slice(-5).reverse();
        
        els.alertsList.innerHTML = recent.map(alert => {
            const timeStr = new Date(alert.timestamp).toLocaleTimeString();
            const sevClass = getSeverityClass(alert.severity);
            
            return `
                <div class="alert-item" style="border-left-color: var(--color-${sevClass})">
                    <div class="alert-time">${timeStr}</div>
                    <div class="alert-rule">${alert.rule_code}</div>
                    <div class="alert-score text-${sevClass}">Risk: ${alert.risk_score}</div>
                    <div class="alert-severity bg-${sevClass}">${alert.severity}</div>
                </div>
            `;
        }).join('');
    } catch (e) {
        console.error("Failed to fetch alerts", e);
    }
};

// Fetch and update performance
const updatePerformance = async () => {
    try {
        const response = await fetch(`${API_BASE}/performance`);
        if (!response.ok) return;
        const data = await response.json();
        
        if (!data.error) {
            els.fps.innerText = data.fps.toFixed(1);
            els.infTime.innerText = Math.round(data.inference_time_ms);
            els.cpuUsage.innerText = data.cpu_percent.toFixed(1);
            els.ramUsage.innerText = Math.round(data.memory_mb);
        }
    } catch (e) {
        console.error("Failed to fetch performance", e);
    }
};

// End Session and Generate Report
const endSession = async () => {
    try {
        els.endSessionBtn.disabled = true;
        els.endSessionBtn.innerText = "Generating Report...";
        
        const response = await fetch(`${API_BASE}/session/report`);
        if (!response.ok) throw new Error('Failed to generate report');
        const report = await response.json();
        
        // Build report HTML
        let html = `
            <div style="margin-bottom: 2rem;">
                <p><strong>Session ID:</strong> <span class="mono">${report.session_id}</span></p>
                <p><strong>Candidate ID:</strong> <span class="mono">${report.candidate_id}</span></p>
                <p><strong>Duration:</strong> ${new Date(report.start_time).toLocaleTimeString()} to ${new Date(report.end_time).toLocaleTimeString()}</p>
                <p><strong>Final Status:</strong> <span class="severity-badge bg-${getSeverityClass(report.final_status)}">${report.final_status}</span></p>
                <p><strong>Final Risk Score:</strong> ${report.session_risk_score}/100</p>
                <p><strong>Total Violations:</strong> ${report.violations}</p>
            </div>
            <h3>Violation Timeline</h3>
            <div style="margin-top: 1rem;">
        `;
        
        if (report.timeline && report.timeline.length > 0) {
            html += report.timeline.map(item => `
                <div class="timeline-item">
                    <span class="mono text-secondary">${new Date(item.time).toLocaleTimeString()}</span>
                    <strong>${item.event}</strong>
                    <span class="severity-badge bg-${getSeverityClass(item.severity)}">${item.severity}</span>
                    <span style="font-size: 0.8em; padding: 0.2rem 0.5rem; background: rgba(255,255,255,0.1); border-radius: 4px;">${item.priority}</span>
                </div>
            `).join('');
        } else {
            html += `<p class="empty-state">No violations recorded.</p>`;
        }
        
        html += `</div>`;
        
        els.reportContent.innerHTML = html;
        els.reportModal.classList.remove('hidden');
        
    } catch (e) {
        alert("Error generating report: " + e.message);
        els.endSessionBtn.disabled = false;
        els.endSessionBtn.innerText = "End Session & Generate Report";
    }
};

els.endSessionBtn.addEventListener('click', endSession);
els.closeModalBtn.addEventListener('click', () => {
    els.reportModal.classList.add('hidden');
    // Refresh page to reset dashboard
    window.location.reload();
});

// Polling loop
const POLLING_INTERVAL = 250; // 4Hz refresh
const PERF_INTERVAL = 1000;   // 1Hz refresh

setInterval(updateStatus, POLLING_INTERVAL);
setInterval(updatePerformance, PERF_INTERVAL);

// Initial fetches
updateStatus();
updateAlerts();
updatePerformance();
