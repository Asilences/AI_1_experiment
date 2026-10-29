'use strict';

const $ = (selector, scope = document) => scope.querySelector(selector);
const relationNames = {P57: '导演', P136: '电影类型', P495: '制作国家', P364: '原始语言', UNKNOWN: '未支持的关系'};
const linkNames = {exact: '名称 / 别名匹配', fuzzy: '模糊匹配', ambiguous: '实体存在歧义', not_found: '未找到电影'};
const storageKey = 'film-kgqa-questions-v1';
let appStatus = null;
let latestResult = null;
let busy = false;
let searchVersion = 0;
let searchTimer;
let toastTimer;
let offset = 0;
let selectedMovie = null;
let history = [];
try { history = JSON.parse(localStorage.getItem(storageKey) || '[]'); } catch { /* Storage is optional. */ }
if (!Array.isArray(history)) history = [];
history = history.filter(item => typeof item === 'string').slice(0, 6);

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function formatTime(ms) {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`;
}

function toast(message, error = false) {
  const box = $('#toast');
  box.textContent = message;
  box.classList.toggle('error', error);
  box.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { box.hidden = true; }, 2600);
}

async function request(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || '本地服务暂时无法处理请求。');
  return payload;
}

function setQuestion(question, focus = true) {
  $('#question').value = question;
  updateCount();
  if (focus) {
    $('#question').focus();
    $('#question').scrollIntoView({behavior: 'smooth', block: 'center'});
  }
}

function updateCount() { $('#question-count').textContent = `${$('#question').value.length} / 1000`; }

function renderHistory() {
  const list = $('#history-list');
  list.replaceChildren();
  $('#clear-history').disabled = !history.length;
  if (!history.length) list.append(node('p', 'history-empty', '还没有提问。最近的问题会保存在当前浏览器中。'));
  for (const question of history) {
    const button = node('button', 'history-item', question);
    button.type = 'button';
    button.title = question;
    button.addEventListener('click', () => setQuestion(question));
    list.append(button);
  }
}

function remember(question) {
  history = [question, ...history.filter(item => item !== question)].slice(0, 6);
  try { localStorage.setItem(storageKey, JSON.stringify(history)); } catch { /* Private browsing can disable storage. */ }
  renderHistory();
}

async function loadMovies(append = false) {
  const version = ++searchVersion;
  const query = $('#movie-search').value.trim();
  const nextOffset = append ? offset : 0;
  const more = $('#load-more');
  more.disabled = true;
  try {
    const data = await request(`/api/movies?q=${encodeURIComponent(query)}&offset=${nextOffset}&limit=30`);
    if (version !== searchVersion) return;
    const list = $('#movie-list');
    if (!append) list.replaceChildren();
    offset = nextOffset + data.movies.length;
    $('#search-summary').textContent = query ? `找到 ${data.total} 部电影` : '已收录的电影';
    if (!data.movies.length && !append) {
      list.append(node('p', 'history-empty', query ? '没有找到这部电影。试试英文片名或别名。' : '知识库暂不可用，请查看页面提示。'));
    }
    data.movies.forEach((movie, index) => {
      const button = node('button', 'movie-item');
      button.type = 'button';
      button.dataset.movie = movie.id;
      button.title = `${movie.display_label || movie.label} · ${movie.id}`;
      button.classList.toggle('selected', selectedMovie === movie.id);
      button.append(node('span', 'movie-number', String(nextOffset + index + 1).padStart(2, '0')));
      const text = node('span', 'movie-text');
      text.append(node('span', 'movie-title', movie.display_label || movie.label), node('span', 'movie-subtitle', `${movie.id} · ${movie.fact_count} 条事实`));
      button.append(text, node('span', 'movie-arrow', '↗'));
      button.addEventListener('click', () => {
        selectedMovie = movie.id;
        document.querySelectorAll('.movie-item').forEach(item => item.classList.toggle('selected', item.dataset.movie === movie.id));
        const templates = {P57: 'Who directed {title}?', P136: 'What genre is {title}?', P495: 'Which country did {title} originate in?', P364: 'What is the original language of {title}?'};
        const relation = movie.relations[0] || 'P57';
        setQuestion(templates[relation].replace('{title}', movie.question_title || movie.label));
      });
      list.append(button);
    });
    more.hidden = offset >= data.total;
  } catch {
    if (version === searchVersion) $('#search-summary').textContent = '电影列表加载失败，请刷新重试';
  } finally {
    if (version === searchVersion) more.disabled = false;
  }
}

function renderResult(result) {
  const card = $('#answer-template').content.firstElementChild.cloneNode(true);
  card.classList.toggle('neural', result.system === 'best');
  card.classList.toggle('refused', !!result.refused);
  card.classList.toggle('failed', !result.ok);
  $('.card-eyebrow', card).textContent = result.system === 'a0' ? 'PIPELINE / FACT QUERY' : 'NEURAL / ANSWER GENERATION';
  $('.system-name', card).textContent = result.name;
  $('.result-badge', card).textContent = !result.ok ? '暂不可用' : result.refused ? '信息不足 · 拒答' : '回答完成';
  $('.system-timing', card).textContent = result.ok ? `处理耗时 ${formatTime(result.elapsed_ms)}${result.cold_start ? ' · 含首次加载' : ''}` : '';
  const copy = $('.copy-button', card);
  if (!result.ok) {
    $('.answer-text', card).hidden = true;
    $('.card-error', card).hidden = false;
    $('.card-error', card).textContent = result.error;
    $('.link-info', card).hidden = true;
    $('.evidence-details', card).hidden = true;
    copy.hidden = true;
    return card;
  }
  $('.answer-text', card).textContent = result.answer;
  copy.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(result.answer);
      toast('答案已复制');
    } catch { toast('复制失败，请直接选中答案复制。', true); }
  });
  $('.movie-name', card).textContent = result.movie ? `${result.movie.label} (${result.movie.id})` : '未能确定电影';
  $('.link-method', card).textContent = `${linkNames[result.link_status] || result.link_status}${result.movie && result.link_status === 'fuzzy' ? ` · 相似度 ${result.link_score.toFixed(3)}` : ''}`;
  const relation = $('.relation-name', card);
  relation.textContent = result.relation ? `${relationNames[result.relation] || result.relation}${result.confidence != null ? ` · 置信度 ${(result.confidence * 100).toFixed(1)}%` : ''}` : '';
  $('.evidence-label', card).textContent = result.system === 'a0' ? '回答命中的事实' : '输入给模型的候选事实';
  $('.evidence-count', card).textContent = String(result.evidence.length);
  const evidence = $('.evidence-list', card);
  for (const fact of result.evidence) evidence.append(node('li', '', fact));
  if (!result.evidence.length) evidence.append(node('li', 'empty-evidence', '没有可用事实。'));
  return card;
}

function setBusy(value) {
  busy = value;
  $('#ask-button').disabled = value || !appStatus?.a0.available;
  $('#ask-label').textContent = value ? '正在处理' : '开始提问';
  $('#question-form').setAttribute('aria-busy', String(value));
  $('#loading').hidden = !value;
  $('#results').hidden = value;
  $('#empty-state').hidden = value || !!latestResult;
  $('#export-result').disabled = value || !latestResult;
  $('#answered-question').hidden = value || !latestResult;
}

$('#question-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (busy) return;
  const question = $('#question').value.trim();
  if (!question) { $('#question').focus(); return; }
  const system = $('input[name="system"]:checked').value;
  const errorBox = $('#request-error');
  errorBox.hidden = true;
  $('#request-timing').textContent = '';
  $('#timing-note').hidden = true;
  setBusy(true);
  try {
    const data = await request('/api/answer', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({question, system})});
    latestResult = {...data, saved_at: new Date().toISOString()};
    $('#answered-question').textContent = `已回答的问题：${data.question}`;
    const results = $('#results');
    results.replaceChildren(...data.results.map(renderResult));
    results.classList.toggle('single', data.results.length === 1);
    $('#request-timing').textContent = `总耗时 ${formatTime(data.elapsed_ms)}`;
    $('#timing-note').hidden = false;
    remember(question);
  } catch (error) {
    errorBox.textContent = error instanceof TypeError ? '无法连接本地服务，请确认启动终端仍在运行。' : error.message;
    errorBox.hidden = false;
  } finally { setBusy(false); }
});

$('#question').addEventListener('input', updateCount);
$('#question').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!$('#ask-button').disabled) $('#question-form').requestSubmit();
  }
});
$('#movie-search').addEventListener('input', () => {
  clearTimeout(searchTimer);
  ++searchVersion;
  searchTimer = setTimeout(() => loadMovies(), 180);
});
$('#load-more').addEventListener('click', () => loadMovies(true));
$('#clear-history').addEventListener('click', () => {
  history = [];
  try { localStorage.removeItem(storageKey); } catch { /* Storage is optional. */ }
  renderHistory();
});
$('#export-result').addEventListener('click', () => {
  if (!latestResult) return;
  const blob = new Blob([JSON.stringify(latestResult, null, 2)], {type: 'application/json;charset=utf-8'});
  const url = URL.createObjectURL(blob);
  const link = node('a');
  link.href = url;
  link.download = `film-qa-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

async function init() {
  renderHistory();
  $('#ask-button').disabled = true;
  try {
    appStatus = await request('/api/status');
    const connection = $('#connection');
    connection.classList.add('connected');
    connection.lastChild.textContent = '本地服务已连接';
    $('#stat-movies').textContent = appStatus.movie_count.toLocaleString();
    $('#stat-facts').textContent = appStatus.fact_count.toLocaleString();
    $('#stat-relations').textContent = String(appStatus.relation_count);
    $('#library-count').textContent = appStatus.movie_count.toLocaleString();
    $('#dataset-name').textContent = appStatus.dataset_name || '本地快照';
    $('#neural-option').textContent = appStatus.neural.name;
    $('input[value="best"]').disabled = !appStatus.neural.available;
    $('input[value="both"]').disabled = !appStatus.neural.available;
    if (!appStatus.neural.available) $('input[value="a0"]').checked = true;
    if (!appStatus.a0.available || !appStatus.neural.available) {
      const notice = $('#setup-notice');
      notice.textContent = !appStatus.a0.available ? appStatus.a0.detail : `${appStatus.neural.detail} 你仍可使用 A0 管道问答。`;
      notice.hidden = false;
    }
    const examples = $('#example-list');
    appStatus.examples.forEach(example => {
      const chip = node('button', 'example-chip');
      chip.type = 'button';
      chip.append(node('span', '', relationNames[example.relation]), document.createTextNode(example.question));
      chip.addEventListener('click', () => setQuestion(example.question));
      examples.append(chip);
    });
    if (appStatus.examples.length) $('#question').placeholder = appStatus.examples[0].question;
    await loadMovies();
  } catch {
    $('#connection').lastChild.textContent = '本地服务未连接';
    $('#setup-notice').textContent = '无法读取本地服务。请确认启动终端仍在运行，然后刷新页面。';
    $('#setup-notice').hidden = false;
  } finally { setBusy(false); }
}

init();
