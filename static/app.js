// app.js — Multimodal Clinical Decision Support Client Logic

let currentReport = null;
let chatHistory = [];
let kgData = null;

// Canvas & Camera State for Knowledge Graph
let kgCanvas, kgCtx;
let kgModalCanvas, kgModalCtx;
let kgNodes = [];
let kgEdges = [];
let activeNodeSet = new Set();
let activeEdgeSet = new Set();

let camera = { x: 0, y: 0, scale: 1.0 };
let modalCamera = { x: 0, y: 0, scale: 1.0 };
let hoveredNode = null;
let modalHoveredNode = null;
let isModalOpen = false;

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  initHealthCheck();
  initFormControls();
  initKnowledgeGraph();
  initChatbot();
});

// ============================================================================
// 1. Health Status Check (No Gemini Mention)
// ============================================================================
async function initHealthCheck() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();

    const setStatus = (id, online, label) => {
      const el = document.getElementById(id);
      if (!el) return;
      if (online) {
        el.classList.add('online');
        el.innerHTML = `<span class="status-dot"></span> ${label}`;
      } else {
        el.classList.remove('online');
        el.innerHTML = `<span class="status-dot"></span> ${label} (Offline)`;
      }
    };

    setStatus('badge-diabetes', data.models.diabetes.available, 'Diabetes Model');
    setStatus('badge-dr', data.models.diabetic_retinopathy.available, 'DR EfficientNet');
    setStatus('badge-skin', data.models.skin_disease.available, 'Skin EfficientNet');
    setStatus('badge-kg', data.knowledge_graph.available, `Medical KG (${data.knowledge_graph.nodes} Nodes)`);
    setStatus('badge-llm', data.llm.configured, 'AI Reasoning Engine');
  } catch (err) {
    console.warn('Health check failed:', err);
  }
}

// ============================================================================
// 2. Form & Image Upload Controls (User Uploads Their Own Images)
// ============================================================================
function initFormControls() {
  const form = document.getElementById('analysisForm');
  const btnLoadSample = document.getElementById('btnLoadSample');
  const jsonFileInput = document.getElementById('jsonFileInput');
  const btnResetForm = document.getElementById('btnResetForm');
  const btnToggleJsonGuide = document.getElementById('btnToggleJsonGuide');
  const jsonGuidePanel = document.getElementById('jsonGuidePanel');
  const btnCopyJsonTemplate = document.getElementById('btnCopyJsonTemplate');

  // Toggle JSON format guide
  btnToggleJsonGuide.addEventListener('click', () => {
    const isHidden = jsonGuidePanel.style.display === 'none';
    jsonGuidePanel.style.display = isHidden ? 'block' : 'none';
    btnToggleJsonGuide.classList.toggle('active-sample', isHidden);
  });

  // Copy JSON Template
  btnCopyJsonTemplate.addEventListener('click', () => {
    const code = document.getElementById('jsonTemplateCodeBlock').textContent;
    navigator.clipboard.writeText(code).then(() => {
      btnCopyJsonTemplate.textContent = 'Copied!';
      setTimeout(() => { btnCopyJsonTemplate.textContent = 'Copy Template'; }, 2000);
    });
  });

  // Load sample patient demographics & biomarkers (Does NOT preload sample images)
  btnLoadSample.addEventListener('click', async () => {
    btnLoadSample.classList.add('active-sample');
    try {
      const res = await fetch('/api/sample');
      const data = await res.json();
      populateForm(data.patient_data);
    } catch (e) {
      console.error('Failed to load sample:', e);
    }
  });

  // Import JSON file
  jsonFileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const parsed = JSON.parse(event.target.result);
        populateForm(parsed);
      } catch (err) {
        alert('Invalid JSON format: ' + err.message);
      }
    };
    reader.readAsText(file);
  });

  // Reset form
  btnResetForm.addEventListener('click', () => {
    form.reset();
    resetImageDropzone('eye');
    resetImageDropzone('skin');
    btnLoadSample.classList.remove('active-sample');
    document.getElementById('glucoseWarning').textContent = 'Biomarkers Normal';
    document.getElementById('glucoseWarning').className = 'input-badge';
    currentReport = null;
    updateChatPromptChips(null, null);
  });

  // Dynamic Glucose biomarker alert badge
  const glucoseInput = document.getElementById('glucose');
  const glucoseWarning = document.getElementById('glucoseWarning');
  glucoseInput.addEventListener('input', () => {
    const val = parseFloat(glucoseInput.value) || 0;
    if (val >= 140) {
      glucoseWarning.textContent = `Glucose: ${val} mg/dL (High Alert)`;
      glucoseWarning.className = 'input-badge warn';
    } else if (val < 70) {
      glucoseWarning.textContent = `Glucose: ${val} mg/dL (Low)`;
      glucoseWarning.className = 'input-badge warn';
    } else {
      glucoseWarning.textContent = `Glucose: ${val} mg/dL (Normal)`;
      glucoseWarning.className = 'input-badge';
    }
  });

  // Setup user image dropzones
  setupUserImageDropzone('eye', 'eyeDropzone', 'eyeFileInput', 'eyePreview', 'eyeBadge', 'btnClearEye');
  setupUserImageDropzone('skin', 'skinDropzone', 'skinFileInput', 'skinPreview', 'skinBadge', 'btnClearSkin');

  // Handle Form Submit
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    await executeAnalysis();
  });

  // Export handlers
  document.getElementById('btnDownloadJson').addEventListener('click', downloadJsonReport);
  document.getElementById('btnPrintReport').addEventListener('click', () => window.print());
}

function populateForm(data) {
  const p = data.patient || {};
  const t = data.tabular_data || {};

  document.getElementById('age').value = p.age || 50;
  if (p.sex) document.getElementById('sex').value = p.sex.toLowerCase();
  document.getElementById('clinical_notes').value = p.clinical_notes || '';
  document.getElementById('known_conditions').value = (p.known_conditions || []).join(', ');
  document.getElementById('medications').value = (p.medications || []).join(', ');

  document.getElementById('glucose').value = t.Glucose ?? 120;
  document.getElementById('blood_pressure').value = t.BloodPressure ?? 75;
  document.getElementById('skin_thickness').value = t.SkinThickness ?? 25;
  document.getElementById('insulin').value = t.Insulin ?? 0;
  document.getElementById('bmi').value = t.BMI ?? 28.0;
  document.getElementById('dpf').value = t.DiabetesPedigreeFunction ?? 0.5;
  document.getElementById('pregnancies').value = t.Pregnancies ?? 0;

  document.getElementById('glucose').dispatchEvent(new Event('input'));
}

function setupUserImageDropzone(type, cardId, inputId, previewId, badgeId, clearBtnId) {
  const card = document.getElementById(cardId);
  const input = document.getElementById(inputId);
  const preview = document.getElementById(previewId);
  const badge = document.getElementById(badgeId);
  const clearBtn = document.getElementById(clearBtnId);

  card.addEventListener('click', (e) => {
    if (e.target === clearBtn) return;
    input.click();
  });

  input.addEventListener('change', () => {
    const file = input.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        preview.src = e.target.result;
        preview.classList.add('show');
        card.classList.add('has-image');
        badge.style.display = 'block';
        badge.textContent = file.name.length > 18 ? file.name.substring(0, 15) + '...' : file.name;
        clearBtn.style.display = 'inline-block';
      };
      reader.readAsDataURL(file);
    }
  });

  clearBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetImageDropzone(type);
  });

  // Drag and drop
  card.addEventListener('dragover', (e) => {
    e.preventDefault();
    card.style.borderColor = 'var(--accent-cyan)';
  });
  card.addEventListener('dragleave', () => {
    card.style.borderColor = '';
  });
  card.addEventListener('drop', (e) => {
    e.preventDefault();
    card.style.borderColor = '';
    if (e.dataTransfer.files.length) {
      input.files = e.dataTransfer.files;
      input.dispatchEvent(new Event('change'));
    }
  });
}

function resetImageDropzone(type) {
  document.getElementById(`${type}Preview`).classList.remove('show');
  document.getElementById(`${type}Dropzone`).classList.remove('has-image');
  document.getElementById(`${type}Badge`).style.display = 'none';
  document.getElementById(`btnClear${type.charAt(0).toUpperCase() + type.slice(1)}`).style.display = 'none';
  document.getElementById(`${type}FileInput`).value = '';
}

// ============================================================================
// 3. Multimodal Analysis Execution
// ============================================================================
async function executeAnalysis() {
  const btn = document.getElementById('btnAnalyze');
  const btnText = document.getElementById('btnText');

  btn.disabled = true;
  btnText.textContent = 'Synthesizing Multimodal Evidence...';

  try {
    const form = document.getElementById('analysisForm');
    const formData = new FormData(form);

    // Append user-uploaded images if selected
    const eyeFile = document.getElementById('eyeFileInput').files[0];
    if (eyeFile) {
      formData.append('eye_image', eyeFile);
    }

    const skinFile = document.getElementById('skinFileInput').files[0];
    if (skinFile) {
      formData.append('skin_image', skinFile);
    }

    const res = await fetch('/api/analyze', {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Analysis request failed');
    }

    const report = await res.json();
    currentReport = report;

    renderReport(report);

  } catch (err) {
    alert('Error executing analysis: ' + err.message);
    console.error(err);
  } finally {
    btn.disabled = false;
    btnText.textContent = 'Run Multimodal Clinical Analysis';
  }
}

function renderReport(report) {
  document.getElementById('emptyState').style.display = 'none';
  const resContainer = document.getElementById('resultsContainer');
  resContainer.style.display = 'flex';

  // 1. Triage Alert Banner
  const alertBanner = document.getElementById('alertBanner');
  const alertList = document.getElementById('alertBannerList');
  alertList.innerHTML = '';

  if (report.alerts && report.alerts.length > 0) {
    alertBanner.className = 'alert-banner';
    document.getElementById('alertBannerTag').textContent = 'CRITICAL TRIAGE ALERT';
    document.getElementById('alertBannerTitle').textContent = `Clinical Triage Alerts Identified (${report.alerts.length})`;
    document.getElementById('alertBannerSubtitle').textContent = 'Actionable risks requiring clinical specialist evaluation.';
    report.alerts.forEach(alertText => {
      const li = document.createElement('li');
      li.textContent = alertText;
      alertList.appendChild(li);
    });
  } else {
    alertBanner.className = 'alert-banner safe';
    document.getElementById('alertBannerTag').textContent = 'TRIAGE SCREENING NORMAL';
    document.getElementById('alertBannerTitle').textContent = 'No Immediate High-Risk Alerts Identified';
    document.getElementById('alertBannerSubtitle').textContent = 'All multimodal findings within baseline or routine monitoring thresholds.';
  }

  // 2. AI Synthesis
  document.getElementById('confidenceBadge').textContent = `Confidence: ${(report.confidence_level || 'Moderate').toUpperCase()}`;
  document.getElementById('clinicalSummaryText').textContent = report.clinical_summary || 'No summary available.';
  document.getElementById('clinicalReasoningNotes').innerHTML = `<strong>Clinical Rationale:</strong> ${report.reasoning || 'Diagnostic synthesis completed.'}`;

  const recsContainer = document.getElementById('recommendationsList');
  recsContainer.innerHTML = '';
  (report.recommendations || []).forEach(rec => {
    const item = document.createElement('div');
    item.className = 'rec-item';
    item.innerHTML = `<span class="rec-bullet">&bull;</span> <span>${rec}</span>`;
    recsContainer.appendChild(item);
  });

  // 3. Model Findings Cards
  const findingsGrid = document.getElementById('findingsGrid');
  findingsGrid.innerHTML = '';

  (report.findings || []).forEach(f => {
    const card = document.createElement('div');
    card.className = `finding-card ${f.alert ? 'alert-card' : ''}`;

    const confPct = (f.confidence * 100).toFixed(1);
    const sev = (f.severity || 'normal').toLowerCase();

    let detailsHtml = '';
    if (f.raw_probabilities && Object.keys(f.raw_probabilities).length > 1) {
      const rows = Object.entries(f.raw_probabilities)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
        .map(([name, prob]) => `
          <div class="prob-row">
            <span class="prob-name" title="${name}">${name}</span>
            <div class="prob-line">
              <div class="prob-line-fill" style="width: ${(prob*100).toFixed(1)}%;"></div>
            </div>
            <span class="prob-val">${(prob*100).toFixed(1)}%</span>
          </div>
        `).join('');

      detailsHtml = `
        <details class="prob-details">
          <summary>Probability Breakdown</summary>
          <div class="prob-bars">${rows}</div>
        </details>
      `;
    }

    card.innerHTML = `
      <div>
        <div class="finding-header">
          <span class="finding-type">${f.type.replace('_', ' ')}</span>
          <span class="severity-chip ${sev}">${sev}</span>
        </div>
        <div class="finding-main">
          <div class="finding-label">${f.label}</div>
          <div class="confidence-bar-container">
            <div class="confidence-header">
              <span>Confidence</span>
              <strong>${confPct}%</strong>
            </div>
            <div class="confidence-bar">
              <div class="confidence-fill" style="width: ${confPct}%;"></div>
            </div>
          </div>
        </div>
        <div class="finding-notes">${f.notes}</div>
      </div>
      ${detailsHtml}
    `;
    findingsGrid.appendChild(card);
  });

  // 4. Update Knowledge Graph Active Subgraph
  if (report.active_kg) {
    updateKgActiveHighlights(report.active_kg);
  }

  // Update dynamic chat prompt chips based on constructed KG and active findings
  updateChatPromptChips(report, report.active_kg);

  // 5. Update Chatbot with Welcome
  chatHistory = [];
  const chatMessages = document.getElementById('chatMessages');
  chatMessages.innerHTML = `
    <div class="chat-bubble assistant">
      I have analyzed the multimodal clinical report for this patient. 
      <br><br>
      &bull; <strong>Diabetes Risk:</strong> ${report.findings.find(f => f.type === 'diabetes_risk')?.label || 'N/A'}<br>
      &bull; <strong>Retinopathy:</strong> ${report.findings.find(f => f.type === 'diabetic_retinopathy')?.label || 'Not uploaded'}<br>
      &bull; <strong>Skin Lesion:</strong> ${report.findings.find(f => f.type === 'skin_lesion')?.label || 'Not uploaded'}<br><br>
      How can I assist you with clinical interpretations, guidelines, or referral questions?
    </div>
  `;
}

// ============================================================================
// 4. Knowledge Graph Visualizer (Dual-Canvas Clean View, Zoom, Pan, Modal Popup)
// ============================================================================
function initKnowledgeGraph() {
  kgCanvas = document.getElementById('kgCanvas');
  if (kgCanvas) kgCtx = kgCanvas.getContext('2d');

  kgModalCanvas = document.getElementById('kgModalCanvas');
  if (kgModalCanvas) kgModalCtx = kgModalCanvas.getContext('2d');

  window.addEventListener('resize', onWindowResize);

  // Inline Controls
  const btnKgZoomIn = document.getElementById('btnKgZoomIn');
  const btnKgZoomOut = document.getElementById('btnKgZoomOut');
  const btnResetKgView = document.getElementById('btnResetKgView');
  const btnKgView = document.getElementById('btnKgView');

  if (btnKgZoomIn) btnKgZoomIn.addEventListener('click', () => zoomInline(1.2));
  if (btnKgZoomOut) btnKgZoomOut.addEventListener('click', () => zoomInline(0.8));
  if (btnResetKgView) btnResetKgView.addEventListener('click', resetInlineCamera);
  if (btnKgView) btnKgView.addEventListener('click', openKgModal);

  // Modal Controls
  const btnCloseKgModal = document.getElementById('btnCloseKgModal');
  const btnModalZoomIn = document.getElementById('btnModalZoomIn');
  const btnModalZoomOut = document.getElementById('btnModalZoomOut');
  const btnModalReset = document.getElementById('btnModalReset');

  if (btnCloseKgModal) btnCloseKgModal.addEventListener('click', closeKgModal);
  if (btnModalZoomIn) btnModalZoomIn.addEventListener('click', () => zoomModal(1.2));
  if (btnModalZoomOut) btnModalZoomOut.addEventListener('click', () => zoomModal(0.8));
  if (btnModalReset) btnModalReset.addEventListener('click', resetModalCamera);

  // Close on Escape or click outside modal content
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && isModalOpen) closeKgModal();
  });

  const modalOverlay = document.getElementById('kgFullscreenModal');
  if (modalOverlay) {
    modalOverlay.addEventListener('click', (e) => {
      if (e.target.id === 'kgFullscreenModal') closeKgModal();
    });
  }

  // Attach interactive drag, pan, zoom, and tooltips
  if (kgCanvas) {
    attachCanvasInteractions({
      canvas: kgCanvas,
      getCamera: () => camera,
      getHovered: () => hoveredNode,
      setHovered: (n) => { hoveredNode = n; },
      redraw: drawInlineGraph,
      tooltipId: 'kgTooltip',
    });
  }

  if (kgModalCanvas) {
    attachCanvasInteractions({
      canvas: kgModalCanvas,
      getCamera: () => modalCamera,
      getHovered: () => modalHoveredNode,
      setHovered: (n) => { modalHoveredNode = n; },
      redraw: drawModalGraph,
      tooltipId: 'kgModalTooltip',
    });
  }

  // Fetch KG data asynchronously
  loadKgData();
}

async function loadKgData() {
  try {
    const res = await fetch('/api/kg');
    kgData = await res.json();
    setupGraphLayout(kgData.nodes, kgData.edges);
    resizeInlineCanvas();
    resetInlineCamera();
    if (isModalOpen) {
      resizeModalCanvas();
      resetModalCamera();
    }
    updateChatPromptChips(currentReport, currentReport ? currentReport.active_kg : null);
  } catch (err) {
    console.warn('Could not load KG data:', err);
  }
}

function onWindowResize() {
  resizeInlineCanvas();
  if (isModalOpen) {
    resizeModalCanvas();
  }
}

function openKgModal() {
  const modal = document.getElementById('kgFullscreenModal');
  if (!modal) {
    console.warn('kgFullscreenModal element not found');
    return;
  }
  modal.classList.add('open');
  modal.style.display = 'flex';
  document.body.style.overflow = 'hidden';
  isModalOpen = true;

  if (!kgModalCanvas) {
    kgModalCanvas = document.getElementById('kgModalCanvas');
  }
  if (kgModalCanvas && !kgModalCtx) {
    kgModalCtx = kgModalCanvas.getContext('2d');
  }

  if (!kgData) {
    loadKgData();
  } else {
    resizeModalCanvas();
    resetModalCamera();
  }

  setTimeout(() => {
    resizeModalCanvas();
    resetModalCamera();
  }, 40);

  setTimeout(() => {
    resizeModalCanvas();
    resetModalCamera();
  }, 200);
}

function closeKgModal() {
  const modal = document.getElementById('kgFullscreenModal');
  if (!modal) return;
  modal.classList.remove('open');
  modal.style.display = 'none';
  document.body.style.overflow = '';
  isModalOpen = false;

  const tooltip = document.getElementById('kgModalTooltip');
  if (tooltip) tooltip.style.display = 'none';
}

// Expose globally for inline onclick handlers
window.openKgModal = openKgModal;
window.closeKgModal = closeKgModal;

function resizeInlineCanvas() {
  if (!kgCanvas || !kgCtx) return;
  const container = document.getElementById('kgCanvasContainer');
  if (!container) return;
  const rect = container.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return;
  const dpr = window.devicePixelRatio || 1;

  kgCanvas.width = rect.width * dpr;
  kgCanvas.height = rect.height * dpr;
  kgCanvas.style.width = rect.width + 'px';
  kgCanvas.style.height = rect.height + 'px';

  kgCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  drawInlineGraph();
}

function resizeModalCanvas() {
  if (!kgModalCanvas || !kgModalCtx) return;
  const container = document.getElementById('kgModalBody');
  if (!container) return;
  const rect = container.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return;
  const dpr = window.devicePixelRatio || 1;

  kgModalCanvas.width = rect.width * dpr;
  kgModalCanvas.height = rect.height * dpr;
  kgModalCanvas.style.width = rect.width + 'px';
  kgModalCanvas.style.height = rect.height + 'px';

  kgModalCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  drawModalGraph();
}

function resetInlineCamera() {
  const container = document.getElementById('kgCanvasContainer');
  if (!container) return;
  const rect = container.getBoundingClientRect();
  if (rect.width === 0) return;
  camera.scale = Math.min(rect.width / 1300, rect.height / 800) * 0.95;
  camera.x = (rect.width - 1200 * camera.scale) / 2;
  camera.y = (rect.height - 700 * camera.scale) / 2;
  drawInlineGraph();
}

function resetModalCamera() {
  const container = document.getElementById('kgModalBody');
  if (!container) return;
  const rect = container.getBoundingClientRect();
  if (rect.width === 0) return;
  modalCamera.scale = Math.min(rect.width / 1300, rect.height / 800) * 0.95;
  modalCamera.x = (rect.width - 1200 * modalCamera.scale) / 2;
  modalCamera.y = (rect.height - 700 * modalCamera.scale) / 2;
  drawModalGraph();
}

function zoomInline(factor) {
  const container = document.getElementById('kgCanvasContainer');
  if (!container) return;
  const rect = container.getBoundingClientRect();
  const cx = rect.width / 2;
  const cy = rect.height / 2;
  const newScale = Math.max(0.25, Math.min(camera.scale * factor, 3.5));
  camera.x = cx - (cx - camera.x) * (newScale / camera.scale);
  camera.y = cy - (cy - camera.y) * (newScale / camera.scale);
  camera.scale = newScale;
  drawInlineGraph();
}

function zoomModal(factor) {
  const container = document.getElementById('kgModalBody');
  if (!container) return;
  const rect = container.getBoundingClientRect();
  const cx = rect.width / 2;
  const cy = rect.height / 2;
  const newScale = Math.max(0.25, Math.min(modalCamera.scale * factor, 3.5));
  modalCamera.x = cx - (cx - modalCamera.x) * (newScale / modalCamera.scale);
  modalCamera.y = cy - (cy - modalCamera.y) * (newScale / modalCamera.scale);
  modalCamera.scale = newScale;
  drawModalGraph();
}

function setupGraphLayout(nodes, edges) {
  const width = 1200;
  const height = 700;
  const cx = width / 2;
  const cy = height / 2;

  const ringDistances = {
    'disease': 90,
    'risk_factor': 180,
    'finding': 270,
    'complication': 350,
    'drug': 420,
    'recommendation': 490,
    'monitoring': 520,
  };

  const typeGroups = {};
  nodes.forEach(n => {
    if (!typeGroups[n.type]) typeGroups[n.type] = [];
    typeGroups[n.type].push(n);
  });

  kgNodes = [];
  Object.keys(typeGroups).forEach(type => {
    const list = typeGroups[type];
    const r = ringDistances[type] || 300;
    const total = list.length;

    list.forEach((n, idx) => {
      const angleOffset = (type.length * 0.4);
      const angle = (idx / total) * 2 * Math.PI + angleOffset;

      kgNodes.push({
        id: n.id,
        name: n.name,
        type: n.type,
        color: n.color || '#38bdf8',
        x: cx + Math.cos(angle) * r,
        y: cy + Math.sin(angle) * r,
        radius: 13,
        attrs: n.attrs || {},
      });
    });
  });

  kgEdges = edges.map(e => ({
    source: e.source,
    target: e.target,
    relation: e.relation,
  }));
}

function updateKgActiveHighlights(activeKg) {
  activeNodeSet = new Set(activeKg.active_nodes || []);
  activeEdgeSet = new Set((activeKg.active_edges || []).map(e => `${e.source}->${e.target}`));

  const activeSummary = document.getElementById('kgActivePathSummary');
  if (activeSummary && activeKg.active_nodes && activeKg.active_nodes.length > 0) {
    const activeNames = activeKg.active_nodes.map(id => {
      const n = kgNodes.find(node => node.id === id);
      return n ? n.name : id;
    });
    activeSummary.innerHTML = `<strong>Active Clinical Decision Pathways:</strong> ${activeNames.slice(0, 7).join(' &rarr; ')}`;
  }

  drawAllGraphs();
}

function drawInlineGraph() {
  drawCanvasGraph(kgCanvas, kgCtx, camera, hoveredNode);
}

function drawModalGraph() {
  drawCanvasGraph(kgModalCanvas, kgModalCtx, modalCamera, modalHoveredNode);
}

function drawAllGraphs() {
  drawInlineGraph();
  if (isModalOpen) {
    drawModalGraph();
  }
}

function drawGraph() {
  drawAllGraphs();
}

function drawCanvasGraph(canvas, ctx, cam, currentHoveredNode) {
  if (!canvas || !ctx) return;
  const rect = canvas.getBoundingClientRect();
  const w = rect.width;
  const h = rect.height;
  if (w === 0 || h === 0) return;

  ctx.save();
  ctx.clearRect(0, 0, w, h);

  // Apply camera translation & scale
  ctx.translate(cam.x, cam.y);
  ctx.scale(cam.scale, cam.scale);

  const nodeMap = new Map(kgNodes.map(n => [n.id, n]));
  const hasActiveNodes = activeNodeSet.size > 0;

  // 1. Draw Edges
  kgEdges.forEach(e => {
    const u = nodeMap.get(e.source);
    const v = nodeMap.get(e.target);
    if (!u || !v) return;

    const isActive = activeEdgeSet.has(`${e.source}->${e.target}`) || (activeNodeSet.has(e.source) && activeNodeSet.has(e.target));

    ctx.beginPath();
    ctx.moveTo(u.x, u.y);
    ctx.lineTo(v.x, v.y);

    if (isActive) {
      ctx.strokeStyle = '#06b6d4';
      ctx.lineWidth = 3.0;
      ctx.shadowColor = '#06b6d4';
      ctx.shadowBlur = 12;
    } else {
      ctx.strokeStyle = hasActiveNodes ? 'rgba(255, 255, 255, 0.04)' : 'rgba(255, 255, 255, 0.09)';
      ctx.lineWidth = 1;
      ctx.shadowBlur = 0;
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Draw relation arrow on active edges
    if (isActive) {
      drawArrow(ctx, u.x, u.y, v.x, v.y, '#06b6d4');
    }
  });

  // 2. Draw Nodes
  kgNodes.forEach(n => {
    const isActive = activeNodeSet.has(n.id);
    const isHovered = currentHoveredNode === n;

    const r = isActive ? n.radius + 5 : n.radius;

    // Active Halo
    if (isActive) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, r + 8, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(6, 182, 212, 0.22)';
      ctx.fill();
    }

    ctx.beginPath();
    ctx.arc(n.x, n.y, r, 0, 2 * Math.PI);

    if (isActive) {
      ctx.fillStyle = n.color;
      ctx.shadowColor = n.color;
      ctx.shadowBlur = 18;
    } else if (hasActiveNodes) {
      ctx.fillStyle = isHovered ? '#f1f5f9' : 'rgba(71, 85, 105, 0.35)';
      ctx.shadowBlur = 0;
    } else {
      ctx.fillStyle = isHovered ? '#ffffff' : n.color;
      ctx.shadowBlur = isHovered ? 12 : 0;
      ctx.shadowColor = n.color;
    }
    ctx.fill();

    ctx.strokeStyle = isActive ? '#ffffff' : 'rgba(255, 255, 255, 0.25)';
    ctx.lineWidth = isActive ? 2.5 : 1;
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Node Label
    const showLabel = isActive || isHovered || cam.scale > 0.65;
    if (showLabel) {
      ctx.font = `${isActive ? '700 12px' : '500 10.5px'} Inter, sans-serif`;
      ctx.fillStyle = isActive ? '#ffffff' : (hasActiveNodes && !isHovered ? 'rgba(148, 163, 184, 0.45)' : '#cbd5e1');
      ctx.textAlign = 'center';
      ctx.fillText(n.name, n.x, n.y - r - 6);
    }
  });

  ctx.restore();
}

function drawArrow(ctx, fromX, fromY, toX, toY, color) {
  const midX = (fromX + toX) / 2;
  const midY = (fromY + toY) / 2;
  const angle = Math.atan2(toY - fromY, toX - fromX);
  const headLen = 8;

  ctx.beginPath();
  ctx.moveTo(midX, midY);
  ctx.lineTo(midX - headLen * Math.cos(angle - Math.PI / 6), midY - headLen * Math.sin(angle - Math.PI / 6));
  ctx.moveTo(midX, midY);
  ctx.lineTo(midX - headLen * Math.cos(angle + Math.PI / 6), midY - headLen * Math.sin(angle + Math.PI / 6));
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.stroke();
}

function attachCanvasInteractions({ canvas, getCamera, getHovered, setHovered, redraw, tooltipId }) {
  let isDraggingNode = false;
  let activeDraggedNode = null;
  let isPanningCanvas = false;
  let panStartPoint = { x: 0, y: 0 };

  const tooltip = document.getElementById(tooltipId);

  function getCoords(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const cam = getCamera();
    return {
      x: ((clientX - rect.left) - cam.x) / cam.scale,
      y: ((clientY - rect.top) - cam.y) / cam.scale,
    };
  }

  function getNodeAt(x, y) {
    return kgNodes.find(n => Math.hypot(n.x - x, n.y - y) <= n.radius + 8);
  }

  canvas.addEventListener('mousedown', (e) => {
    const coords = getCoords(e.clientX, e.clientY);
    const node = getNodeAt(coords.x, coords.y);
    const cam = getCamera();

    if (node) {
      isDraggingNode = true;
      activeDraggedNode = node;
    } else {
      isPanningCanvas = true;
      panStartPoint = { x: e.clientX - cam.x, y: e.clientY - cam.y };
      canvas.classList.add('panning');
    }
  });

  window.addEventListener('mousemove', (e) => {
    const cam = getCamera();

    if (isDraggingNode && activeDraggedNode) {
      const coords = getCoords(e.clientX, e.clientY);
      activeDraggedNode.x = coords.x;
      activeDraggedNode.y = coords.y;
      drawAllGraphs();
      return;
    }

    if (isPanningCanvas) {
      cam.x = e.clientX - panStartPoint.x;
      cam.y = e.clientY - panStartPoint.y;
      redraw();
      return;
    }

    const rect = canvas.getBoundingClientRect();
    const isOverCanvas = (
      e.clientX >= rect.left &&
      e.clientX <= rect.right &&
      e.clientY >= rect.top &&
      e.clientY <= rect.bottom
    );

    if (!isOverCanvas) {
      if (getHovered()) {
        setHovered(null);
        if (tooltip) tooltip.style.display = 'none';
        canvas.style.cursor = 'grab';
        redraw();
      }
      return;
    }

    const coords = getCoords(e.clientX, e.clientY);
    const node = getNodeAt(coords.x, coords.y);

    if (node !== getHovered()) {
      setHovered(node);
      canvas.style.cursor = node ? 'pointer' : 'grab';
      redraw();

      if (node && tooltip) {
        tooltip.style.display = 'block';
        tooltip.style.left = (e.clientX - rect.left + 14) + 'px';
        tooltip.style.top = (e.clientY - rect.top + 14) + 'px';

        const attrsList = Object.entries(node.attrs || {})
          .map(([k, v]) => `<div><span style="color: var(--text-dim);">${k}:</span> ${v}</div>`)
          .join('');

        tooltip.innerHTML = `
          <div style="font-weight: 700; color: #fff; margin-bottom: 2px;">${node.name}</div>
          <div style="color: ${node.color}; font-size: 0.7rem; text-transform: uppercase; font-weight: 600; margin-bottom: 4px;">${node.type}</div>
          ${attrsList ? `<div style="border-top: 1px solid rgba(255,255,255,0.1); padding-top: 4px; margin-top: 4px;">${attrsList}</div>` : ''}
        `;
      } else if (tooltip) {
        tooltip.style.display = 'none';
      }
    } else if (node && tooltip && tooltip.style.display === 'block') {
      tooltip.style.left = (e.clientX - rect.left + 14) + 'px';
      tooltip.style.top = (e.clientY - rect.top + 14) + 'px';
    }
  });

  window.addEventListener('mouseup', () => {
    isDraggingNode = false;
    activeDraggedNode = null;
    if (isPanningCanvas) {
      isPanningCanvas = false;
      canvas.classList.remove('panning');
    }
  });

  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const cam = getCamera();

    const zoomFactor = e.deltaY < 0 ? 1.12 : 0.89;
    const newScale = Math.max(0.25, Math.min(cam.scale * zoomFactor, 3.5));

    cam.x = mouseX - (mouseX - cam.x) * (newScale / cam.scale);
    cam.y = mouseY - (mouseY - cam.y) * (newScale / cam.scale);
    cam.scale = newScale;

    redraw();
  }, { passive: false });

  canvas.addEventListener('mouseleave', () => {
    if (tooltip) tooltip.style.display = 'none';
  });
}

// ============================================================================
// 5. Clinical AI Follow-up Assistant Chatbot (No Gemini Mention)
// ============================================================================
function initChatbot() {
  const chatInput = document.getElementById('chatInput');
  const btnChatSend = document.getElementById('btnChatSend');
  const quickPromptsContainer = document.getElementById('chatQuickPrompts');

  btnChatSend.addEventListener('click', sendChatMessage);
  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendChatMessage();
  });

  // Delegated click handler for dynamic chips
  if (quickPromptsContainer) {
    quickPromptsContainer.addEventListener('click', (e) => {
      const chip = e.target.closest('.chip-prompt');
      if (chip && chip.dataset.q) {
        chatInput.value = chip.dataset.q;
        sendChatMessage();
      }
    });
  }

  // Initial population of prompt chips from Knowledge Graph
  updateChatPromptChips(currentReport, null);
}

function updateChatPromptChips(report, activeKg) {
  const container = document.getElementById('chatQuickPrompts');
  if (!container) return;

  const chips = [];

  if (report && report.findings && report.findings.length > 0) {
    // 1. Diabetic Retinopathy Finding from KG
    const drFinding = report.findings.find(f => f.type === 'diabetic_retinopathy');
    if (drFinding && drFinding.label !== 'Not uploaded') {
      const label = drFinding.label;
      const isSevere = drFinding.severity === 'critical' || drFinding.severity === 'high' || label.includes('Grade 2') || label.includes('Grade 3') || label.includes('Grade 4');
      if (isSevere) {
        chips.push({
          label: `Ophthalmology: ${label.split('(')[0].trim()}`,
          query: `In the Knowledge Graph, why does ${label} trigger an urgent ophthalmology referral and what are the intervention protocols?`
        });
      } else {
        chips.push({
          label: `Monitoring: ${label.split('(')[0].trim()}`,
          query: `According to the Knowledge Graph decision pathways, what is the follow-up monitoring timeline for ${label}?`
        });
      }
    }

    // 2. Skin Lesion Finding from KG
    const skinFinding = report.findings.find(f => f.type === 'skin_lesion');
    if (skinFinding && skinFinding.label !== 'Not uploaded') {
      const label = skinFinding.label;
      const isUrgent = skinFinding.severity === 'critical' || skinFinding.severity === 'high' || label.toLowerCase().includes('melanoma') || label.toLowerCase().includes('carcinoma');
      if (isUrgent) {
        chips.push({
          label: `Biopsy: ${label}`,
          query: `Explain the malignancy risk, Knowledge Graph correlation, and biopsy protocol for ${label}.`
        });
      } else {
        chips.push({
          label: `Skin evaluation: ${label}`,
          query: `What does the Knowledge Graph recommend for benign presentations like ${label}?`
        });
      }
    }

    // 3. Diabetes Risk & Biomarkers from KG
    const diabetesFinding = report.findings.find(f => f.type === 'diabetes_risk');
    if (diabetesFinding) {
      const isDiabetic = diabetesFinding.label.toLowerCase().includes('diabetic') || diabetesFinding.severity === 'critical' || diabetesFinding.severity === 'high';
      if (isDiabetic) {
        chips.push({
          label: 'Diabetes & Complications',
          query: 'How does high glycemic risk link to microvascular and macrovascular complications in the constructed Knowledge Graph?'
        });
      } else {
        chips.push({
          label: 'Metabolic Prevention',
          query: 'What preventative lifestyle and glycemic targets are codified in the Knowledge Graph for this metabolic profile?'
        });
      }
    }

    // 4. Active KG Decision Pathway / Traverse Edge
    if (activeKg && activeKg.active_edges && activeKg.active_edges.length > 0) {
      const nodeMap = new Map(kgNodes.map(n => [n.id, n]));
      for (const edge of activeKg.active_edges) {
        const u = nodeMap.get(edge.source);
        const v = nodeMap.get(edge.target);
        if (u && v && u.name !== v.name) {
          chips.push({
            label: `${u.name} → ${v.name}`,
            query: `Explain the clinical pathway traversed in the Knowledge Graph between ${u.name} and ${v.name}.`
          });
          break;
        }
      }
    }

    // 5. Active Clinical Recommendations from KG
    if (report.recommendations && report.recommendations.length > 0) {
      const rec = report.recommendations[0];
      const recText = rec.title || rec.name || 'Next Steps';
      chips.push({
        label: `Guideline: ${recText.slice(0, 24)}${recText.length > 24 ? '...' : ''}`,
        query: `What clinical evidence and Knowledge Graph rules prioritize the recommendation: "${recText}"?`
      });
    }

    // 6. Active Triage Alerts
    if (report.alerts && report.alerts.length > 0) {
      const alert = report.alerts[0];
      chips.push({
        label: `Alert: ${alert.title.slice(0, 22)}...`,
        query: `What immediate clinical actions are required to address the alert: "${alert.title}"?`
      });
    }
  } else if (kgData && kgData.nodes && kgData.nodes.length > 0) {
    // Dynamic prompts generated directly from the constructed Knowledge Graph ontology
    chips.push({
      label: 'Explore: Diabetes & Retinopathy',
      query: 'Explain how Type 2 Diabetes connects to Diabetic Retinopathy grades in the medical knowledge graph.'
    });
    chips.push({
      label: 'Explore: Microvascular Complications',
      query: 'What microvascular complications (nephropathy, neuropathy, retinopathy) are encoded in the knowledge graph?'
    });
    chips.push({
      label: 'Explore: Skin Malignancy Pathways',
      query: 'How does the knowledge graph classify skin lesions between benign and malignant triage pathways?'
    });
    chips.push({
      label: 'Explore: Medication Indications',
      query: 'What medications (e.g., Metformin, Insulin, ACE inhibitors) and clinical monitoring rules are defined in the knowledge graph?'
    });
  } else {
    // Fallback prompt chips
    chips.push({
      label: 'Knowledge Graph Pathways',
      query: 'Explain the core disease pathways in the medical knowledge graph.'
    });
    chips.push({
      label: 'Clinical Decision Rules',
      query: 'How does the multimodal decision support system evaluate risk scores?'
    });
  }

  container.innerHTML = chips.map(c => `
    <span class="chip-prompt" data-q="${c.query.replace(/"/g, '&quot;')}">${c.label}</span>
  `).join('');
}

async function sendChatMessage() {
  const input = document.getElementById('chatInput');
  const message = input.value.trim();
  if (!message) return;

  const messagesContainer = document.getElementById('chatMessages');

  // User bubble
  const userBubble = document.createElement('div');
  userBubble.className = 'chat-bubble user';
  userBubble.textContent = message;
  messagesContainer.appendChild(userBubble);
  input.value = '';

  // Thinking bubble
  const thinkingBubble = document.createElement('div');
  thinkingBubble.className = 'chat-bubble assistant';
  thinkingBubble.innerHTML = 'Consulting clinical knowledge base...';
  messagesContainer.appendChild(thinkingBubble);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;

  chatHistory.push({ role: 'user', content: message });

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: message,
        report: currentReport || {},
        history: chatHistory,
      }),
    });

    const data = await res.json();
    const reply = data.reply || 'No clinical response generated.';

    thinkingBubble.innerHTML = formatMarkdownReply(reply);
    chatHistory.push({ role: 'assistant', content: reply });

  } catch (err) {
    thinkingBubble.innerHTML = 'The clinical assistant encountered a communication issue. Please retry.';
    console.error(err);
  } finally {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }
}

function formatMarkdownReply(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/^• (.*$)/gim, '<div style="margin-left: 0.85rem;">&bull; $1</div>')
    .replace(/\n\n/g, '<br><br>')
    .replace(/\n/g, '<br>');
}

// ============================================================================
// 6. JSON Export
// ============================================================================
function downloadJsonReport() {
  if (!currentReport) {
    alert('Please run a clinical analysis first before downloading.');
    return;
  }
  const filename = 'Clinical_Report.json';
  const blob = new Blob([JSON.stringify(currentReport, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
