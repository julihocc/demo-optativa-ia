// Modo juego: en 1 minuto, dibuja tantos emojis pedidos como puedas. Los emojis se agrupan por
// complejidad (nivel 1..3, campo `nivel` de las etiquetas) y subes de nivel al alcanzar cierto puntaje.
const GAME_SECONDS = 60;
const WIN_PROB = 0.6;                 // probabilidad minima de la clase pedida para acertar
const POINTS_PER_LEVEL = 10;          // puntos por emoji = 10 x nivel
const LEVEL_MIN_SCORE = [0, 40, 120]; // puntaje minimo para estar en el nivel 1, 2 y 3
const NEXT_DELAY = 800;               // ms entre un acierto y el siguiente emoji
const BEST_KEY = 'emoji-game-best';
const IDLE_TARGET = '🎨';

function readBest() {
  try { return Number(localStorage.getItem(BEST_KEY)) || 0; } catch { return 0; }
}
function saveBest(v) {
  try { localStorage.setItem(BEST_KEY, v); } catch { /* sin almacenamiento: el record dura la sesion */ }
}

// `labels`: [{emoji, nombre, nivel}] en el orden de las clases; `clear`: borra el lienzo.
export function createGame({ labels, clear }) {
  const $ = id => document.getElementById(id);
  const el = {
    time: $('time'), score: $('score'), best: $('best'), level: $('level'), target: $('target'),
    fill: $('time-fill'), msg: $('msg'), play: $('play'), skip: $('skip'),
  };
  let state = 'idle';   // idle | playing | between
  let score = 0, solved = 0, level = 1, best = readBest(), target = -1, deadline = 0, tick = null;

  const levelFor = s => LEVEL_MIN_SCORE.filter(m => s >= m).length;

  el.best.textContent = best;
  el.play.disabled = false;

  function pickTarget() {
    let pool = labels.map((_, i) => i).filter(i => labels[i].nivel === level);
    if (!pool.length) pool = labels.map((_, i) => i);
    if (pool.length > 1) pool = pool.filter(i => i !== target);
    return pool[Math.floor(Math.random() * pool.length)];
  }

  function update() {
    const left = deadline - performance.now();
    if (left <= 0) return end('⏰ ¡Tiempo!');
    el.time.textContent = Math.ceil(left / 1000);
    el.fill.style.width = (left / (GAME_SECONDS * 1000) * 100) + '%';
  }

  function nextRound() {
    target = pickTarget();
    clear();
    state = 'playing';
    el.skip.disabled = false;
    el.target.innerHTML = `<span class="big">${labels[target].emoji}</span> ${labels[target].nombre}`;
  }

  function end(message) {
    clearInterval(tick);
    state = 'idle';
    el.play.textContent = '▶️ Jugar';
    el.skip.disabled = true;
    el.target.textContent = IDLE_TARGET;
    el.time.textContent = 0;
    el.fill.style.width = '0';
    message += ` ${score} puntos · ${solved} emojis · nivel ${level}.`;
    if (score > best) {
      best = score;
      saveBest(best);
      el.best.textContent = best;
      message += ' ¡Nuevo récord! 🏆';
    }
    el.msg.textContent = message;
  }

  el.play.addEventListener('click', () => {
    if (state !== 'idle') return end('Partida terminada.');
    score = 0; solved = 0; level = 1; target = -1;
    el.score.textContent = 0;
    el.level.textContent = 1;
    el.msg.textContent = '';
    el.play.textContent = '⏹️ Parar';
    deadline = performance.now() + GAME_SECONDS * 1000;
    clearInterval(tick);
    tick = setInterval(update, 100);
    update();
    nextRound();
  });

  el.skip.addEventListener('click', () => { if (state === 'playing') nextRound(); });

  return {
    // Se llama con las probabilidades tras cada trazo.
    check(p) {
      if (state !== 'playing' || p[target] < WIN_PROB) return;
      state = 'between';
      el.skip.disabled = true;
      const gain = POINTS_PER_LEVEL * level;
      score += gain;
      solved++;
      el.score.textContent = score;
      let msg = `¡Bien! ${labels[target].emoji} ${(p[target] * 100).toFixed(0)}% · +${gain}`;
      const newLevel = levelFor(score);
      if (newLevel > level) {
        level = newLevel;
        el.level.textContent = level;
        msg += ` · 🎉 ¡Nivel ${level}!`;
      }
      el.msg.textContent = msg;
      setTimeout(() => { if (state === 'between') nextRound(); }, NEXT_DELAY);
    },
  };
}
