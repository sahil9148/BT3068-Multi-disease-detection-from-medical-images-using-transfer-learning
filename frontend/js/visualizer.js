/**
 * BT3068 — Heatmap Visualizer
 * Renders Grad-CAM overlays with opacity control
 */

const Visualizer = {
  currentOriginal: null,
  currentOverlay: null,

  displayHeatmap(originalBase64, heatmapData) {
    this.currentOriginal = originalBase64;
    this.currentOverlay = heatmapData;

    const origImg = document.getElementById('heatmap-original');
    const overlayImg = document.getElementById('heatmap-overlay');

    if (origImg) {
      origImg.src = originalBase64.startsWith('data:') ? originalBase64 : `data:image/png;base64,${originalBase64}`;
    }
    if (overlayImg && heatmapData.heatmap_overlay) {
      overlayImg.src = `data:image/png;base64,${heatmapData.heatmap_overlay}`;
    }

    // Activation regions
    const regionsEl = document.getElementById('activation-regions');
    if (regionsEl && heatmapData.activation_regions) {
      regionsEl.innerHTML = heatmapData.activation_regions.map((r, i) => `
        <div class="result-card" style="animation-delay: ${i * 0.1}s">
          <span class="result-risk-icon">🎯</span>
          <div class="result-info">
            <div class="result-disease">REGION ${i + 1}</div>
            <div class="result-meta">
              <span class="result-probability">Activation: ${(r.max_activation * 100).toFixed(1)}%</span>
              <span class="result-probability">${r.width}×${r.height}px</span>
            </div>
          </div>
        </div>
      `).join('');
    }
  },

  updateOpacity(value) {
    const label = document.getElementById('opacity-value');
    if (label) label.textContent = `${Math.round(value * 100)}%`;

    // Re-request heatmap with new opacity if we have original data
    if (this.currentOriginal && Upload.currentBase64) {
      fetch('/api/heatmap', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: Upload.currentBase64,
          prediction_class: '',
          opacity: parseFloat(value)
        })
      }).then(r => r.json()).then(data => {
        const overlayImg = document.getElementById('heatmap-overlay');
        if (overlayImg && data.heatmap_overlay) {
          overlayImg.src = `data:image/png;base64,${data.heatmap_overlay}`;
        }
      }).catch(() => {});
    }
  },

  downloadOverlay() {
    const img = document.getElementById('heatmap-overlay');
    if (!img || !img.src) return;

    const link = document.createElement('a');
    link.download = `BT3068_GradCAM_${Date.now()}.png`;
    link.href = img.src;
    link.click();
  }
};
