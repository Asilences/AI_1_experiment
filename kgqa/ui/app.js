'use strict';

const $ = (selector, scope = document) => scope.querySelector(selector);
const relationNames = {P57: '导演', P136: '电影类型', P495: '制作国家', P364: '原始语言', UNKNOWN: '未支持的关系'};
const linkNames = {exact: '名称 / 别名匹配', alias: '中文名称 / 别名匹配', qid: 'QID 明确指定', fuzzy: '模糊匹配', ambiguous: '片名存在歧义', not_found: '未找到电影', unsupported: '请按提示调整问题'};
const storageKey = 'film-kgqa-questions-v1';
const questionTemplates = {
  zh: {P57: '《{title}》的导演是谁？', P136: '《{title}》是什么类型的电影？', P495: '《{title}》是哪个国家制作的？', P364: '《{title}》的原始语言是什么？'},
  en: {P57: 'Who directed {title}?', P136: 'What genre is {title}?', P495: 'Which country did {title} originate in?', P364: 'What is the original language of {title}?'}
};
let appStatus = null;
let latestResult = null;
let busy = false;
let searchVersion = 0;
let searchTimer;
let toastTimer;
let offset = 0;
let selectedMovie = null;
let selectedRelation = 'P57';
let generatedQuestion = '';
let visibleMovies = [];
let history = [];
let exploreRequest = null;
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

function questionLanguage() { return $('#question-language').value; }

function movieTitle(movie, language = questionLanguage()) {
  return language === 'zh' ? movie.label_zh || movie.chinese_title || movie.display_label || movie.label : movie.display_label || movie.label;
}

function movieQuestion(movie, relation) {
  const language = questionLanguage();
  const title = language === 'zh' ? movie.label_zh || movie.chinese_title || movie.question_title || movie.label : movie.question_title || movie.label;
  return questionTemplates[language][relation].replace('{title}', title);
}

function renderMovieTemplates(fillQuestion = false) {
  const picker = $('#movie-templates');
  picker.hidden = !selectedMovie;
  if (!selectedMovie) return;
  const relations = selectedMovie.relations || [];
  if (!relations.includes(selectedRelation)) selectedRelation = relations[0] || 'P57';
  $('#selected-movie-name').textContent = movieTitle(selectedMovie);
  const options = $('#relation-options');
  options.replaceChildren();
  for (const relation of Object.keys(questionTemplates.zh)) {
    const button = node('button', 'relation-option', relationNames[relation]);
    button.type = 'button';
    button.disabled = !relations.includes(relation);
    button.title = button.disabled ? '这部电影暂无此类事实' : `询问${relationNames[relation]}`;
    button.setAttribute('aria-pressed', String(selectedRelation === relation));
    button.addEventListener('click', () => {
      selectedRelation = relation;
      renderMovieTemplates(true);
    });
    options.append(button);
  }
  if (fillQuestion) {
    generatedQuestion = movieQuestion(selectedMovie, selectedRelation);
    setQuestion(generatedQuestion);
  }
}

function renderExamples() {
  const examples = $('#example-list');
  examples.replaceChildren();
  let items = questionLanguage() === 'zh' ? appStatus?.examples_zh || [] : appStatus?.examples || [];
  if ($('#semantic-mode').checked) {
    items = (questionLanguage() === 'zh' ? [
      '《盗梦空间》是谁执导的，剧本又是谁写的？',
      '《盗梦空间》和《星际穿越》里，有谁两部都出演过？',
      '我看完《盗梦空间》了，同一位导演还拍过哪些片？'
    ] : [
      'Who directed Inception, and who wrote the screenplay?',
      'Who appeared in both Inception and Interstellar?',
      'I enjoyed Inception. What other films in this catalogue share its director?'
    ]).map(question => ({question}));
  }
  for (const example of items) {
    const chip = node('button', 'example-chip');
    chip.type = 'button';
    if (example.relation) chip.append(node('span', '', relationNames[example.relation]));
    chip.append(document.createTextNode(example.question));
    chip.addEventListener('click', () => setQuestion(example.question));
    examples.append(chip);
  }
  $('.example-row').hidden = !items.length;
  $('#question').placeholder = items[0]?.question || (questionLanguage() === 'zh' ? '《禁闭岛》的导演是谁？' : 'Who directed Shutter Island?');
}

function renderMovieList(movies) {
  const list = $('#movie-list');
  list.replaceChildren();
  if (!movies.length) {
    list.append(node('p', 'history-empty', $('#movie-search').value.trim() ? '没有找到这部电影。试试中文片名、英文片名或别名。' : '知识库暂不可用，请查看页面提示。'));
  }
  movies.forEach((movie, index) => {
    const button = node('button', 'movie-item');
    button.type = 'button';
    button.dataset.movie = movie.id;
    button.title = `${movieTitle(movie)} · ${movie.question_title || movie.label} · ${movie.id}`;
    button.classList.toggle('selected', selectedMovie?.id === movie.id);
    button.append(node('span', 'movie-number', String(index + 1).padStart(2, '0')));
    const text = node('span', 'movie-text');
    const englishTitle = movie.question_title || movie.label;
    const subtitle = questionLanguage() === 'zh' && (movie.label_zh || movie.chinese_title) ? `${englishTitle} · ${movie.fact_count} 条事实` : `${movie.id} · ${movie.fact_count} 条事实`;
    text.append(node('span', 'movie-title', movieTitle(movie)), node('span', 'movie-subtitle', subtitle));
    button.append(text, node('span', 'movie-arrow', '↗'));
    button.addEventListener('click', () => {
      selectedMovie = movie;
      document.querySelectorAll('.movie-item').forEach(item => item.classList.toggle('selected', item.dataset.movie === movie.id));
      renderMovieTemplates(true);
    });
    list.append(button);
  });
}

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
    visibleMovies = append ? [...visibleMovies, ...data.movies] : data.movies;
    offset = nextOffset + data.movies.length;
    $('#search-summary').textContent = query ? `找到 ${data.total} 部电影` : '已收录的电影';
    renderMovieList(visibleMovies);
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
  $('.result-badge', card).textContent = !result.ok ? '暂不可用'
    : result.guard?.action === 'clarify' ? '请确认问题'
    : result.refused ? '信息不足 · 拒答' : result.fallback_used ? '工具回退' : '回答完成';
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
  const answer = $('.answer-text', card);
  answer.textContent = result.answer;
  answer.lang = result.language === 'zh' || /[\u3400-\u9fff]/.test(result.answer) ? 'zh-CN' : 'en';
  copy.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(result.answer);
      toast('答案已复制');
    } catch { toast('复制失败，请直接选中答案复制。', true); }
  });
  $('.movie-name', card).textContent = result.movie ? `${movieTitle(result.movie, answer.lang === 'zh-CN' ? 'zh' : 'en')} (${result.movie.id})` : '未能确定电影';
  $('.link-method', card).textContent = `${linkNames[result.link_status] || result.link_status}${result.movie && result.link_status === 'fuzzy' ? ` · 相似度 ${result.link_score.toFixed(3)}` : ''}`;
  const relation = $('.relation-name', card);
  relation.textContent = result.relation ? `${relationNames[result.relation] || result.relation}${result.confidence != null ? ` · 置信度 ${(result.confidence * 100).toFixed(1)}%` : ''}` : '';
  $('.evidence-label', card).textContent = result.guard ? '最终回答所依据的事实'
    : result.system === 'a0' ? '回答命中的事实' : '输入给模型的候选事实';
  $('.evidence-count', card).textContent = String(result.evidence.length);
  const evidence = $('.evidence-list', card);
  for (const fact of result.evidence) evidence.append(node('li', '', fact));
  if (!result.evidence.length) evidence.append(node('li', 'empty-evidence', '没有可用事实。'));
  const details = $('.processing-details', card);
  const content = $('.processing-content', card);
  const addDetail = (label, text, language) => {
    content.append(node('span', 'processing-label', label));
    const paragraph = node('p', 'processing-text', text);
    if (language) paragraph.lang = language;
    content.append(paragraph);
  };
  if (result.model_question) addDetail(result.language === 'zh' ? '中文问题 → 英文模型输入' : '英文模型输入', result.model_question, 'en');
  if (result.language_note) addDetail('语言处理说明', result.language_note);
  if (result.guard) {
    const actions = {accepted: '回答通过核验', rejected: '生成未通过核验', tool_fallback: '使用图谱查询与模板回退',
      clarify: '需要用户确认', blocked: '范围或证据不足'};
    addDetail('回答处理', `${actions[result.guard.action] || result.guard.action} · ${result.guard.reason}`);
    addDetail('工具介入', result.fallback_used ? '最终答案由知识库查询与模板给出，模型原始输出保留如下。' : '未使用工具模板替换。');
    if (result.model_evidence?.length) addDetail('实际提供给生成器的证据', result.model_evidence.join('\n'));
  }
  if (typeof result.original_answer === 'string' && result.original_answer !== result.answer) {
    addDetail('中文呈现前的原答案', result.original_answer, 'en');
  }
  if (typeof result.raw_answer === 'string') {
    addDetail(result.guard ? '核验与回退前的原始输出' : '名称恢复前的神经原始生成', result.raw_answer, 'en');
    const replacements = Array.isArray(result.replacements) ? result.replacements : [];
    if (!result.guard) addDetail('名称恢复', `${replacements.length} 处${result.recovery_ms != null ? ` · ${formatTime(result.recovery_ms)}` : ''}`);
    if (replacements.length) {
      const list = node('ul', 'recovery-list');
      for (const item of replacements) list.append(node('li', '', `${item.before || '原名称'} → ${item.after || item.entity_id}${item.after && item.entity_id ? ` (${item.entity_id})` : ''}`));
      content.append(list);
    }
  }
  details.hidden = !content.childElementCount;
  return card;
}

function setBusy(value) {
  busy = value;
  const semantic = $('#semantic-mode').checked;
  const extended = $('#explore-mode').checked;
  const available = semantic ? appStatus?.semantic?.available : extended ? appStatus?.explorer?.available : appStatus?.a0.available;
  $('#ask-button').disabled = value || !available;
  $('#ask-label').textContent = value ? '正在处理' : '开始提问';
  $('#question-form').setAttribute('aria-busy', String(value));
  $('#loading').hidden = !value;
  $('#loading strong').textContent = semantic ? '正在理解问题并查询图谱' : extended ? '正在查询扩展事实' : '正在查询与生成答案';
  $('#loading p').textContent = semantic ? '正在生成查询计划、核对事实并组织回答，请稍候。' : extended ? '正在核对电影、属性与本地证据。' : '首次使用神经系统需要加载模型，请稍候。';
  $('#results').hidden = value;
  $('#explore-results').hidden = value || !(latestResult?.plan || latestResult?.ui_mode === 'semantic');
  $('#empty-state').hidden = value || !!latestResult;
  $('#export-result').disabled = value || !latestResult;
  $('#answered-question').hidden = value || !latestResult;
}

$('#question-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (busy) return;
  const question = $('#question').value.trim();
  if (!question) { $('#question').focus(); return; }
  if ($('#semantic-mode').checked) {
    await submitExtended({question}, true);
    return;
  }
  if ($('#explore-mode').checked && appStatus?.explorer?.available) {
    await submitExplore({question});
    return;
  }
  $('#explore-results').hidden = true;
  const system = $('input[name="system"]:checked').value;
  const errorBox = $('#request-error');
  errorBox.hidden = true;
  $('#request-timing').textContent = '';
  $('#timing-note').hidden = true;
  setBusy(true);
  try {
    const mode = appStatus?.guarded?.available && $('#freeform-mode').checked ? 'guarded' : 'standard';
    const data = await request('/api/answer', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({question, system, mode})});
    latestResult = {...data, saved_at: new Date().toISOString()};
    $('#answered-question').textContent = `已回答的问题：${data.question}`;
    const parsed = data.interpretation;
    $('#interpretation').hidden = !parsed;
    if (parsed) {
      $('#interpretation-text').textContent = parsed.canonical_question && parsed.status === 'ready'
        ? `理解为：${parsed.canonical_question} ${parsed.message}` : parsed.message;
      const options = $('#clarification-options');
      options.replaceChildren();
      for (const suggestion of parsed.suggestions || []) {
        const button = node('button', 'example-chip', suggestion);
        button.type = 'button';
        button.addEventListener('click', () => setQuestion(suggestion));
        options.append(button);
      }
      for (const candidate of parsed.candidates || []) {
        const button = node('button', 'example-chip', `${candidate.label} (${candidate.id})`);
        button.type = 'button';
        button.addEventListener('click', () => {
          if (candidate.question) setQuestion(candidate.question);
          else {
            $('#movie-search').value = candidate.id;
            loadMovies();
            toast('请在左侧选择电影，再选择问题类型；原问题的额外条件需要另行确认。');
          }
        });
        options.append(button);
      }
    }
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

async function submitExplore(payload) {
  return submitExtended(payload, false);
}

async function submitExtended(payload, semantic = false) {
  if (busy) return;
  $('#request-error').hidden = true;
  $('#interpretation').hidden = true;
  setBusy(true);
  try {
    const data = await request(semantic ? '/api/semantic' : '/api/explore', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
    exploreRequest = payload;
    latestResult = {...data, ui_mode: semantic ? 'semantic' : 'explore', saved_at: new Date().toISOString()};
    $('#answered-question').textContent = `当前问题：${data.question}`;
    $('#results').replaceChildren();
    renderExplore(data, semantic);
    $('#request-timing').textContent = `${semantic ? 'P0 总耗时' : '工具耗时'} ${formatTime(data.elapsed_ms || 0)}`;
    if (semantic) $('#semantic-summary').textContent = data.status === 'unavailable' || data.status === 'service_unavailable' ? '模型服务暂不可用' : '冻结模型服务已响应';
    $('#timing-note').hidden = true;
    remember(data.question);
  } catch (error) {
    if (semantic) $('#semantic-summary').textContent = '本次请求未完成，请检查提示';
    $('#request-error').textContent = error.message;
    $('#request-error').hidden = false;
  } finally { setBusy(false); }
}

function renderExplore(data, semantic = false) {
  const target = $('#explore-results');
  target.replaceChildren();
  const card = node('article', 'answer-card explore-card');
  const top = node('div', 'card-top');
  const heading = node('div');
  heading.append(node('span', 'card-eyebrow', semantic ? 'FROZEN MODEL / KNOWLEDGE GRAPH' : 'EXTENDED CATALOGUE / EVIDENCE TOOLS'), node('h3', 'system-name', semantic ? 'P0 自然语言图谱问答' : '扩展事实与查询工具'));
  const labels = {answered: '查询完成', partial: '部分可回答', needs_confirmation: '请选择电影', not_found: '未找到电影',
    needs_clarification: '需要补充信息', unsupported: '暂未支持', not_comparable: '口径不足以比较', unanswerable: '缺少事实',
    no_recorded_overlap: '未发现已记录共同项', no_recorded_other_works: '片库未记录其他作品',
    unavailable: '模型服务不可用', service_unavailable: '模型服务不可用', invalid_plan: '查询计划未通过核验'};
  top.append(heading, node('span', 'result-badge', labels[data.status] || data.status));
  const body = node('div', 'card-body');
  body.append(node('p', 'answer-text', data.answer || data.message || '本次没有产生可显示的回答。'));
  if (semantic && data.fallback_used) body.append(node('p', 'semantic-note', '最终回答使用了图谱结果与完整句模板。模型原始输出保留在下方。'));
  const queryPlan = data.plan || {};
  const candidates = data.candidates || queryPlan.candidates || [];
  if (candidates.length) {
    const choices = node('div', 'candidate-cards');
    for (const movie of candidates) {
      const button = node('button', 'candidate-card');
      button.type = 'button';
      button.append(node('strong', '', movie.label_zh || movie.label), node('span', '', `${movie.label} · ${movie.year || '年份未知'} · ${movie.id}`),
        node('span', '', '选择这部电影 →'));
      button.addEventListener('click', () => submitExtended({question: data.question, movie_ids: [...new Set([...(exploreRequest?.movie_ids || []), movie.id])]}, semantic));
      choices.append(button);
    }
    body.append(choices);
  }
  const plan = node('details', 'evidence-details');
  plan.open = semantic;
  plan.append(node('summary', '', semantic ? '问题理解与查询计划' : '查看理解与查询范围'), node('pre', 'query-plan', JSON.stringify({
    ...queryPlan, scope: '当前冻结片库',
    total: data.total, offset: data.offset, limit: data.limit
  }, null, 2)));
  body.append(plan);
  if (semantic && (data.raw_answer != null || data.raw_plan != null || data.guard || data.model)) {
    const raw = node('details', 'processing-details');
    raw.append(node('summary', '', '查看模型输出与核验记录'));
    for (const [label, value] of [['原始规划', data.raw_plan], ['原始回答', data.raw_answer], ['核验结果', data.guard], ['模型信息', data.model]]) {
      if (value == null) continue;
      raw.append(node('span', 'processing-label', label), node('pre', 'query-plan', typeof value === 'string' ? value : JSON.stringify(value, null, 2)));
    }
    body.append(raw);
  }
  if ((data.evidence || []).length) {
    const details = node('details', 'evidence-details');
    details.open = true;
    details.append(node('summary', '', `来源与限定条件 · ${data.evidence.length} 条`));
    const list = node('ul', 'evidence-list');
    for (const fact of data.evidence) {
      const item = node('li');
      item.append(node('span', '', fact.text));
      const citation = node('a', 'source-link', `Wikidata · 修订 ${fact.revision || '未知'} · 提取 ${fact.retrieved_at || '未知'}`);
      // Source URLs are supplied by the importer, not generated prose.
      if (/^https:\/\/www\.wikidata\.org\/wiki\/Q[1-9]\d*$/.test(fact.source_url)) {
        citation.href = `${fact.source_url}?oldid=${encodeURIComponent(fact.revision || '')}`;
        citation.target = '_blank';
        citation.rel = 'noopener noreferrer';
      }
      item.append(citation);
      if ((fact.references || []).length) {
        const refs = node('details', 'reference-details');
        refs.append(node('summary', '', `原始参考来源 ${fact.references.length} 组`), node('pre', 'query-plan', JSON.stringify(fact.references, null, 2)));
        item.append(refs);
      }
      list.append(item);
    }
    details.append(list);
    body.append(details);
  }
  if (data.next_offset != null) {
    const next = node('button', 'example-chip', `下一页 · 共 ${data.total} 项`);
    next.type = 'button';
    next.addEventListener('click', () => submitExtended({...exploreRequest, offset: data.next_offset}, semantic));
    body.append(next);
  }
  card.append(top, body);
  target.append(card);
}

function updateMode() {
  const semantic = $('#semantic-mode').checked;
  const active = $('#explore-mode').checked || semantic;
  $('#stat-facts').textContent = (active ? appStatus?.explorer?.statement_count || 0 : appStatus?.fact_count || 0).toLocaleString();
  $('#stat-relations').textContent = String(active ? 12 : appStatus?.relation_count || 4);
  $('.system-switch').hidden = active;
  $('#freeform-control').hidden = active || !appStatus?.freeform?.available;
  $('#movie-templates').hidden = active || !selectedMovie;
  renderExamples();
  if (active && !semantic) $('#question').placeholder = '《盗梦空间》的导演和演员是谁？ / Who wrote Inception?';
  $('#question-hint').textContent = semantic ? '中英文自然提问；有歧义时请确认，未支持的条件会说明' : active ? '扩展事实按问题语言回答；近似片名先确认，长列表分页显示' : 'Enter 提问，Shift + Enter 换行';
  $('#question-scope').textContent = semantic ? 'P0 查询当前十二属性片库，支持多属性、共同演员和导演其他作品；剧情、推荐及未实现的条件仍有边界。'
    : active ? '扩展目录支持十二种属性；金额和日期保留原始口径与限定条件。'
    : questionLanguage() === 'zh' ? '支持中文或英文提问：导演、电影类型、制作国家、原始语言。' : '英文提问支持：导演、电影类型、制作国家、原始语言；界面与证据说明保留中文。';
  setBusy(busy);
}

$('#explore-mode').addEventListener('change', () => {
  if ($('#explore-mode').checked) $('#semantic-mode').checked = false;
  updateMode();
});
$('#semantic-mode').addEventListener('change', () => {
  if ($('#semantic-mode').checked) $('#explore-mode').checked = false;
  updateMode();
});

$('#question').addEventListener('input', updateCount);
$('#question-language').addEventListener('change', () => {
  const useTemplate = selectedMovie && (!$('#question').value.trim() || $('#question').value === generatedQuestion);
  renderExamples();
  renderMovieList(visibleMovies);
  renderMovieTemplates(useTemplate);
  $('#question-scope').textContent = questionLanguage() === 'zh' ? '支持中文或英文提问：导演、电影类型、制作国家、原始语言。' : '英文提问支持：导演、电影类型、制作国家、原始语言；界面与证据说明保留中文。';
  $('#empty-description').replaceChildren(document.createTextNode(questionLanguage() === 'zh' ? '输入中文问题，或选择电影与问题类型。' : '输入英文问题，或选择电影与问题类型。'), node('br'), document.createTextNode('这里将展示回答、实体链接与图谱事实。'));
  updateMode();
});
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
    $('#semantic-control').hidden = !appStatus.semantic?.available;
    if (appStatus.semantic?.available) $('#semantic-summary').textContent = appStatus.semantic.model_status === 'reachable' ? '冻结模型服务已响应' : '模型服务首次提问时验证';
    $('#explore-control').hidden = !appStatus.explorer?.available;
    if (appStatus.explorer?.available) {
      $('#explore-summary').textContent = `${appStatus.explorer.movie_count.toLocaleString()} 部电影 · ${appStatus.explorer.statement_count.toLocaleString()} 条扩展声明`;
      if (appStatus.explorer.unresolved_entity_label_count) $('#explore-control p').textContent += ` 部分实体名称待补齐，暂以QID显示，可点击来源核对。`;
    }
    $('#freeform-control').hidden = !appStatus.freeform?.available;
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
    renderExamples();
    await loadMovies();
  } catch {
    $('#connection').lastChild.textContent = '本地服务未连接';
    $('#setup-notice').textContent = '无法读取本地服务。请确认启动终端仍在运行，然后刷新页面。';
    $('#setup-notice').hidden = false;
  } finally { setBusy(false); }
}

init();
