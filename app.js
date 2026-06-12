// ── CONFIG ─────────────────────────────────────────────────────────────────
// TMDB_API_KEY is defined in config.js, loaded before this script in index.html.
const CONFIG = {
  API_URL:      'http://localhost:5000/search',
  TMDB_API_KEY,                                         // references the const above
  TMDB_IMG:     'https://image.tmdb.org/t/p/w342',
  TMDB_DETAILS: 'https://api.themoviedb.org/3/movie',
  DEMO_URL:     'demo_results.json',
  DEFAULT_K:    5,
};

// ── STATE ──────────────────────────────────────────────────────────────────
const posterCache = new Map();
let   demoData    = null;

// ── DOM REFS ───────────────────────────────────────────────────────────────
const queryInput     = document.getElementById('query-input');
const searchBtn      = document.getElementById('search-btn');
const kSlider        = document.getElementById('k-slider');
const kValLabel      = document.getElementById('k-val');
const spinner        = document.getElementById('spinner');
const resultsSection = document.getElementById('results-section');
const resultsGrid    = document.getElementById('results-grid');
const resultsQuery   = document.getElementById('results-query');
const modeBadge      = document.getElementById('mode-badge');
const errorSection   = document.getElementById('error-section');
const errorMsg       = document.getElementById('error-msg');

// ── INIT ───────────────────────────────────────────────────────────────────
(function init() {
  // Sync k slider label and gradient fill
  kSlider.addEventListener('input', () => {
    const pct = ((kSlider.value - kSlider.min) / (kSlider.max - kSlider.min)) * 100;
    kValLabel.textContent = kSlider.value;
    kSlider.style.setProperty('--pct', pct + '%');
  });
  // Trigger once to set initial gradient
  kSlider.dispatchEvent(new Event('input'));

  searchBtn.addEventListener('click', triggerSearch);
  queryInput.addEventListener('keydown', e => { if (e.key === 'Enter') triggerSearch(); });

  document.querySelectorAll('.demo-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      queryInput.value = pill.textContent;
      triggerSearch();
    });
  });
})();

function triggerSearch() {
  const query = queryInput.value.trim();
  if (!query) { queryInput.focus(); return; }
  handleSearch(query, parseInt(kSlider.value, 10));
}

// ── SEARCH FLOW ────────────────────────────────────────────────────────────
async function handleSearch(query, k) {
  showSpinner();
  hideError();
  hideResults();

  try {
    const results = await tryApiSearch(query, k);
    await renderResults(results, query, false);
  } catch {
    // API unavailable — fall back to precomputed demo results
    try {
      const { results, matchedQuery } = await loadDemoResults(query, k);
      await renderResults(results, query, true, matchedQuery);
    } catch {
      showError(
        'Could not reach the local server and demo_results.json failed to load. ' +
        'Run "python server.py" to enable live search.'
      );
    }
  } finally {
    hideSpinner();
  }
}

async function tryApiSearch(query, k) {
  const res = await fetch(CONFIG.API_URL, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ query, k }),
    signal:  AbortSignal.timeout(8000),
  });
  if (!res.ok) throw new Error('API error ' + res.status);
  const data = await res.json();
  return data.results;
}

async function loadDemoResults(query, k) {
  if (!demoData) {
    const res = await fetch(CONFIG.DEMO_URL);
    if (!res.ok) throw new Error('demo_results.json not found');
    demoData = await res.json();
  }
  const entry = pickBestDemo(query, demoData);
  return { results: entry.results.slice(0, k), matchedQuery: entry.query };
}

function pickBestDemo(query, demos) {
  const q = query.toLowerCase();
  const qWords = q.split(/\W+/).filter(w => w.length > 2);

  // Pass 1: word-level substring overlap (e.g. "sci" matches "science").
  let best = null, bestScore = 0;
  for (const demo of demos) {
    const demoText = demo.query.toLowerCase();
    const score = qWords.filter(w => demoText.includes(w)).length;
    if (score > bestScore) { bestScore = score; best = demo; }
  }
  if (best) return best;

  // Pass 2 (no words matched): character bigram overlap against each demo.
  const qBigrams = bigrams(q);
  let bigramBest = null, bigramBestScore = -1;
  for (const demo of demos) {
    const score = bigramOverlap(qBigrams, bigrams(demo.query.toLowerCase()));
    if (score > bigramBestScore) { bigramBestScore = score; bigramBest = demo; }
  }
  if (bigramBestScore > 0) return bigramBest;

  // Pass 3 (complete miss): round-robin so results at least vary.
  return demos[query.length % demos.length];
}

function bigrams(str) {
  const out = [];
  for (let i = 0; i < str.length - 1; i++) out.push(str.slice(i, i + 2));
  return out;
}

function bigramOverlap(aBigrams, bBigrams) {
  const bSet = new Set(bBigrams);
  return aBigrams.filter(bg => bSet.has(bg)).length;
}

// ── TMDB POSTER ────────────────────────────────────────────────────────────
async function fetchPoster(movieId) {
  if (!CONFIG.TMDB_API_KEY) return null;
  if (posterCache.has(movieId)) return posterCache.get(movieId);

  try {
    const res = await fetch(
      `${CONFIG.TMDB_DETAILS}/${movieId}?api_key=${CONFIG.TMDB_API_KEY}`
    );
    if (!res.ok) { posterCache.set(movieId, null); return null; }
    const data = await res.json();
    const url  = data.poster_path ? CONFIG.TMDB_IMG + data.poster_path : null;
    posterCache.set(movieId, url);
    return url;
  } catch {
    posterCache.set(movieId, null);
    return null;
  }
}

// ── RENDER ─────────────────────────────────────────────────────────────────
async function renderResults(results, query, isDemo, matchedQuery = '') {
  resultsQuery.textContent = `"${query}"`;

  if (isDemo) {
    modeBadge.innerHTML =
      `<span class="badge demo">Demo · closest: "${escapeHtml(matchedQuery)}"</span>`;
  } else {
    modeBadge.innerHTML = `<span class="badge live">Live API</span>`;
  }

  resultsGrid.innerHTML = '';

  const cards = results.map((movie, i) => {
    const card = buildCard(movie, i + 1, i * 0.07);
    resultsGrid.appendChild(card);
    return { card, movie };
  });

  showResults();
  const gridTop      = resultsGrid.getBoundingClientRect().top + window.scrollY;
  const centerOffset = (window.innerHeight - resultsGrid.offsetHeight) / 2;
  const target       = Math.max(0, gridTop - centerOffset);
  window.scrollTo({ top: target, behavior: 'smooth' });

  // Animate score bars after paint
  requestAnimationFrame(() => {
    cards.forEach(({ card, movie }, i) => {
      const pct = Math.max(0, Math.min(100, Math.round(movie.score * 100)));
      setTimeout(() => {
        const fill = card.querySelector('.score-bar-fill');
        if (fill) fill.style.width = pct + '%';
      }, i * 70 + 80);
    });
  });

  // Load TMDB posters asynchronously
  if (CONFIG.TMDB_API_KEY) {
    cards.forEach(async ({ card, movie }) => {
      const url = await fetchPoster(movie.movie_id);
      if (!url) return;
      const img         = card.querySelector('.poster-img');
      const placeholder = card.querySelector('.poster-placeholder');
      if (!img) return;
      const showPoster = () => {
        img.classList.add('loaded');
        if (placeholder) placeholder.classList.add('hidden-by-img');
      };
      img.onload  = showPoster;
      img.onerror = () => { img.removeAttribute('src'); };
      img.src = url;
      // Handle already-cached images that fire load before onload is set
      if (img.complete && img.naturalWidth) showPoster();
      img.alt = movie.title;
    });
  }
}

function buildCard(movie, rank, animDelay) {
  const pct      = Math.max(0, Math.min(100, Math.round(movie.score * 100)));
  const rankCls  = rank <= 3 ? `rank-${rank}` : 'rank-other';
  const genres   = formatGenres(movie.genres);
  const overview = movie.overview.length > 200
    ? movie.overview.slice(0, 197) + '…'
    : movie.overview;

  const article = document.createElement('article');
  article.className = 'movie-card';
  article.style.setProperty('--anim-delay', animDelay + 's');

  article.innerHTML = `
    <div class="poster-wrap">
      <div class="poster-placeholder">
        <span class="poster-initial" aria-hidden="true">${escapeHtml(movie.title.charAt(0).toUpperCase())}</span>
      </div>
      <img class="poster-img" src="" alt="" />
      <div class="rank-badge ${rankCls}" aria-label="Rank ${rank}">${ordinal(rank)}</div>
      <div class="score-overlay" aria-label="${pct}% similarity">
        <div class="score-bar-track" role="progressbar" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100">
          <div class="score-bar-fill"></div>
        </div>
        <span class="score-text">${pct}% match</span>
      </div>
    </div>
    <div class="card-body">
      <h3 class="card-title">${escapeHtml(movie.title)}</h3>
      <div class="card-meta">
        <span class="card-year">${escapeHtml(movie.year)}</span>
        <span class="card-genres" title="${escapeHtml(movie.genres)}">${escapeHtml(genres)}</span>
      </div>
      <p class="card-overview">${escapeHtml(overview)}</p>
    </div>
  `;

  return article;
}

// ── HELPERS ────────────────────────────────────────────────────────────────
function formatGenres(str) {
  // Handle "Science Fiction" and "TV Movie" as compound tokens
  const normalised = str
    .replace('Science Fiction', 'Science·Fiction')
    .replace('TV Movie', 'TV·Movie');
  return normalised
    .split(' ')
    .slice(0, 4)
    .map(g => g.replace('·', ' '))
    .join(' · ');
}

function ordinal(n) {
  const s = ['th', 'st', 'nd', 'rd'];
  const v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g,  '&amp;')
    .replace(/</g,  '&lt;')
    .replace(/>/g,  '&gt;')
    .replace(/"/g,  '&quot;')
    .replace(/'/g,  '&#39;');
}

// ── UI STATE ───────────────────────────────────────────────────────────────
function showSpinner()  { spinner.classList.remove('hidden'); }
function hideSpinner()  { spinner.classList.add('hidden'); }
function showResults()  { resultsSection.classList.remove('hidden'); }
function hideResults()  { resultsSection.classList.add('hidden'); }
function showError(msg) { errorMsg.textContent = msg; errorSection.classList.remove('hidden'); }
function hideError()    { errorSection.classList.add('hidden'); }
