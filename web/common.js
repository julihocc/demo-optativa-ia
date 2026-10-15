// Inferencia, carga del modelo, barras de probabilidad y lienzo de dibujo (modulo ES).

// --- Inferencia: MLP (ReLU) + softmax ---
export function forward(layers, x) {
  let a = x;
  layers.forEach((L, li) => {
    const n = L.b.length, z = new Float32Array(n);
    for (let j = 0; j < n; j++) z[j] = L.b[j];
    for (let i = 0; i < a.length; i++) {
      const ai = a[i]; if (ai === 0) continue;
      const w = L.W[i];
      for (let j = 0; j < n; j++) z[j] += ai * w[j];
    }
    if (li < layers.length - 1) for (let j = 0; j < n; j++) z[j] = Math.max(0, z[j]);
    a = z;
  });
  const mx = Math.max(...a);
  const e = Array.from(a, v => Math.exp(v - mx));
  const s = e.reduce((p, c) => p + c, 0);
  return e.map(v => v / s);
}

// --- Carga del modelo (pesos y etiquetas); null si falla ---
export async function loadModel(url) {
  try {
    const r = await fetch(url);
    if (!r.ok) throw new Error(r.statusText);
    return await r.json();
  } catch {
    document.querySelector('aside').insertAdjacentHTML('beforeend', `<p>No se pudo cargar ${url} (¿sirves la carpeta web por HTTP?)</p>`);
    return null;
  }
}

// --- Barra lateral: una fila por clase; `labelsHtml` es el HTML de la etiqueta de cada fila ---
export function createBars(list) {
  const rows = [];
  return {
    build(labelsHtml) {
      labelsHtml.forEach(label => {
        const li = document.createElement('li');
        li.innerHTML = `${label}<div class="track"><div class="fill"></div></div><span class="pct">0.0%</span>`;
        list.appendChild(li);
        rows.push({ li, fill: li.querySelector('.fill'), pct: li.querySelector('.pct') });
      });
    },
    show(p) {
      const best = p ? p.indexOf(Math.max(...p)) : -1;
      rows.forEach((r, i) => {
        const v = p ? p[i] : 0;
        r.fill.style.width = (v * 100) + '%';
        r.pct.textContent = (v * 100).toFixed(1) + '%';
        r.li.classList.toggle('top', i === best);
      });
    },
  };
}

// --- Lienzo de dibujo: trazo blanco sobre negro, boton #clear y callbacks por evento ---
export function setupCanvas(canvas, { lineWidth, onStrokeStart, onStrokePoint, onClear, onChange }) {
  const ctx = canvas.getContext('2d');
  function clear() {
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    onClear?.();
  }
  ctx.lineWidth = lineWidth; ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.strokeStyle = '#fff';
  let drawing = false;
  function pos(e) {
    const r = canvas.getBoundingClientRect();
    return [(e.clientX - r.left) * canvas.width / r.width, (e.clientY - r.top) * canvas.height / r.height];
  }
  canvas.addEventListener('pointerdown', e => {
    drawing = true; canvas.setPointerCapture(e.pointerId);
    const [x, y] = pos(e);
    onStrokeStart?.(x, y);
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + 0.01, y); ctx.stroke();
    onChange?.();
  });
  canvas.addEventListener('pointermove', e => {
    if (!drawing) return;
    const [x, y] = pos(e);
    onStrokePoint?.(x, y);
    ctx.lineTo(x, y); ctx.stroke(); ctx.beginPath(); ctx.moveTo(x, y);
    onChange?.();
  });
  ['pointerup', 'pointercancel'].forEach(t => canvas.addEventListener(t, () => { drawing = false; }));
  document.getElementById('clear').addEventListener('click', clear);
  clear();
  return { ctx, clear };
}
