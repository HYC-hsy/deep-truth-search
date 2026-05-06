/**
 * Deep Truth Search — 前端交互（对话流版）
 *
 * 核心特性：
 * - 对话流布局（消息堆叠）
 * - 每条 Agent 回复有独立的 MessageController
 * - 会话持久化 + 侧边栏历史
 * - 智能滚动（用户上滚时不强制拉回）
 * - SVG 头像 + 角色标签
 * - Header 压缩（对话开始后收起副标题）
 */

(function () {
  "use strict";

  // ── DOM 引用 ────────────────────────────────────────────────

  var queryInput    = document.getElementById("query-input");
  var sendBtn       = document.getElementById("send-btn");
  var conversation  = document.getElementById("conversation");
  var emptyState    = document.getElementById("empty-state");
  var header        = document.getElementById("header");
  var sidebar       = document.getElementById("sidebar");
  var sidebarBackdrop = document.getElementById("sidebar-backdrop");
  var sidebarToggle = document.getElementById("sidebar-toggle");
  var newChatBtn    = document.getElementById("new-chat-btn");
  var sessionList   = document.getElementById("session-list");
  var themeToggle   = document.getElementById("theme-toggle");

  // ── Dark Mode 主题切换 ─────────────────────────────────────

  function getPreferredTheme() {
    var stored = localStorage.getItem("theme");
    if (stored === "dark" || stored === "light") return stored;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }

  applyTheme(getPreferredTheme());

  themeToggle.addEventListener("click", function () {
    var current = document.documentElement.getAttribute("data-theme");
    var next = current === "dark" ? "light" : "dark";
    applyTheme(next);
    themeToggle.setAttribute("aria-label", next === "dark" ? "切换浅色模式" : "切换深色模式");
  });

  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function (e) {
    if (!localStorage.getItem("theme")) {
      applyTheme(e.matches ? "dark" : "light");
    }
  });

  // ── SVG 图标 ──────────────────────────────────────────────

  var SVG_USER =
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
      '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>' +
      '<circle cx="12" cy="7" r="4"/>' +
    '</svg>';

  var SVG_AGENT =
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
      '<circle cx="11" cy="11" r="8"/>' +
      '<line x1="21" y1="21" x2="16.65" y2="16.65"/>' +
    '</svg>';

  var SVG_CHECK =
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">' +
      '<polyline points="20 6 9 17 4 12"/>' +
    '</svg>';

  // ── 会话状态 ────────────────────────────────────────────────

  var currentSessionId = null;
  var isSearching = false;

  // ── 搜索日志面板状态 ─────────────────────────────────────────

  var searchLogs = {};         // index → [{event, content}]
  var searchDoneMap = {};      // index → boolean
  var activePanelIndex = null; // 当前打开的面板对应的 index，null 表示关闭

  // 延迟获取面板 DOM（确保 DOM 就绪）
  function getLogPanel() {
    var el = document.getElementById("searchLogPanel");
    if (!el) return null;
    return {
      el: el,
      body: el.querySelector(".search-log-panel__body"),
      title: el.querySelector(".search-log-panel__title"),
      close: el.querySelector(".search-log-panel__close"),
      backdrop: document.getElementById("searchLogBackdrop"),
    };
  }

  // 绑定关闭事件（延迟到首次使用）
  var _panelBound = false;
  function ensurePanelBound() {
    if (_panelBound) return;
    _panelBound = true;
    var p = getLogPanel();
    if (!p) return;
    if (p.close) p.close.addEventListener("click", closeLogPanel);
    if (p.backdrop) p.backdrop.addEventListener("click", closeLogPanel);
  }

  function openLogPanel(idx, title) {
    ensurePanelBound();
    var p = getLogPanel();
    if (!p || !p.body) { console.warn("search-log-panel not found"); return; }

    activePanelIndex = idx;
    p.title.textContent = title || ("子观点 " + (idx + 1));
    p.body.innerHTML = "";

    // 渲染已有日志
    var logs = searchLogs[idx] || [];
    logs.forEach(function (entry) {
      p.body.appendChild(createLogEntryEl(entry));
    });

    // 设置搜索中/已完成样式
    p.el.classList.remove("search-log-panel--active", "search-log-panel--done");
    p.el.classList.add(searchDoneMap[idx] ? "search-log-panel--done" : "search-log-panel--active");

    p.el.style.display = "flex";
    if (p.backdrop) p.backdrop.style.display = "block";

    // 滚到底部
    p.body.scrollTop = p.body.scrollHeight;
  }

  function closeLogPanel() {
    activePanelIndex = null;
    var p = getLogPanel();
    if (p) {
      p.el.style.display = "none";
      if (p.backdrop) p.backdrop.style.display = "none";
    }
  }

  function appendLiveLogEntry(entry) {
    var p = getLogPanel();
    if (!p || !p.body) return;
    p.body.appendChild(createLogEntryEl(entry));
    p.body.scrollTop = p.body.scrollHeight;
  }

  function createLogEntryEl(entry) {
    var el = document.createElement("div");
    var evtClass = "search-log-entry--" + (entry.event || "think");
    el.className = "search-log-entry " + evtClass;

    var formatted = formatLogEntry(entry);
    el.innerHTML =
      '<span class="search-log-entry__label">' + escapeHtml(formatted.label) + '</span>' +
      '<div class="search-log-entry__content">' + escapeHtml(formatted.content) + '</div>' +
      (formatted.preview ? '<div class="search-log-entry__preview">' + escapeHtml(formatted.preview) + '</div>' : '');
    return el;
  }

  function formatLogEntry(entry) {
    var event = entry.event || "";
    var content = entry.content || "";

    if (event === "think") {
      return { label: "思考", content: content };
    }

    if (event === "tool_call") {
      try {
        var call = JSON.parse(content);
        var tool = call.tool || "";
        var args = call.args || {};
        if (tool === "web_search") {
          var queries = args.queries || [args.query || ""];
          return { label: "搜索", content: queries.join("\n") };
        }
        if (tool === "visit_page") {
          var urls = args.urls || (args.url ? [args.url] : []);
          var lines = urls.map(function (u) { return u.length > 60 ? u.substring(0, 60) + "..." : u; });
          return { label: "访问", content: lines.join("\n") };
        }
        if (tool === "submit_evidence") {
          return { label: "提交", content: "提交证据..." };
        }
        return { label: "工具", content: tool + " " + JSON.stringify(args).substring(0, 100) };
      } catch (_) {
        return { label: "工具", content: content.substring(0, 200) };
      }
    }

    if (event === "tool_result") {
      try {
        var res = JSON.parse(content);
        var tool = res.tool || "";
        var data = res.data || {};

        if (tool === "web_search") {
          var cnt = data.result_count || 0;
          return { label: "搜索结果", content: "找到 " + cnt + " 条结果" };
        }

        if (tool === "visit_page") {
          // 可能是单个结果或批量结果
          var results = data.results || (data.url ? [data] : []);
          var lines = [];
          var preview = "";
          results.forEach(function (r) {
            var title = (r.title || "").substring(0, 30);
            var domain = r.domain || "";
            var score = r.score != null ? r.score.toFixed(0) + "/100" : "";
            var status = r.status || "";

            if (status === "accepted") {
              var evCount = r.evidence_extracted || 0;
              lines.push(domain + " — " + title + " | " + score + " | " + evCount + " 条证据");
              if (r.body_preview) preview = r.body_preview;
            } else if (status === "rejected") {
              lines.push(domain + " — " + title + " | " + score + " 未通过");
            } else if (status === "skipped") {
              lines.push((r.url || "").substring(0, 50) + " 已访问过");
            }
          });
          return { label: "访问结果", content: lines.join("\n"), preview: preview };
        }

        if (tool === "submit_evidence") {
          var n = Array.isArray(data) ? data.length : 0;
          return { label: "提交完成", content: "提交了 " + n + " 条证据" };
        }

        return { label: "结果", content: JSON.stringify(data).substring(0, 200) };
      } catch (_) {
        return { label: "结果", content: content.substring(0, 200) };
      }
    }

    return { label: event, content: content.substring(0, 200) };
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  // ── 三阶段常量 ─────────────────────────────────────────────

  var STAGES = ["analyzing", "searching", "generating"];

  // ── 智能滚动 ──────────────────────────────────────────────

  var userScrolledUp = false;

  conversation.addEventListener("scroll", function () {
    var threshold = 60;
    var atBottom = conversation.scrollHeight - conversation.scrollTop - conversation.clientHeight < threshold;
    userScrolledUp = !atBottom;
  });

  function scrollToBottom() {
    if (!userScrolledUp) {
      setTimeout(function () {
        conversation.scrollTop = conversation.scrollHeight;
      }, 50);
    }
  }

  // ── 初始化 ──────────────────────────────────────────────────

  loadSessionList();
  createNewSession();

  // ── Textarea 自动增高 ───────────────────────────────────────

  queryInput.addEventListener("input", function () {
    this.style.height = "auto";
    this.style.height = Math.min(this.scrollHeight, 120) + "px";
  });

  // ── 发送事件（含 IME 兼容）────────────────────────────────

  sendBtn.addEventListener("click", handleSubmit);

  queryInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      handleSubmit();
    }
  });

  // ── 侧边栏切换 ─────────────────────────────────────────────

  sidebarToggle.addEventListener("click", function () {
    sidebar.classList.toggle("sidebar--open");
    sidebarBackdrop.classList.toggle("sidebar-backdrop--visible");
  });

  sidebarBackdrop.addEventListener("click", function () {
    sidebar.classList.remove("sidebar--open");
    sidebarBackdrop.classList.remove("sidebar-backdrop--visible");
  });

  newChatBtn.addEventListener("click", function () {
    createNewSession();
    sidebar.classList.remove("sidebar--open");
    sidebarBackdrop.classList.remove("sidebar-backdrop--visible");
  });

  // ── 提交处理 ────────────────────────────────────────────────

  async function handleSubmit() {
    var query = queryInput.value.trim();
    if (!query || isSearching) return;

    isSearching = true;

    // 隐藏空状态，压缩 header
    emptyState.style.display = "none";
    header.classList.add("header--compact");

    // 追加用户消息气泡
    appendUserMessage(query);

    // 清空输入
    queryInput.value = "";
    queryInput.style.height = "auto";
    sendBtn.disabled = true;

    // 创建 Agent 回复容器
    var agentEl = appendAgentMessage();
    var ctrl = createMessageController(agentEl);
    ctrl.showProgress("正在分析观点...");

    try {
      var res = await fetch("/api/research", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query, session_id: currentSessionId }),
      });

      if (!res.ok) throw new Error("服务器错误: " + res.status);

      var data = await res.json();
      var taskId = data.task_id;
      agentEl.dataset.taskId = taskId;

      // SSE 优先，自动降级为轮询
      await streamResult(taskId, ctrl);

      // 搜索完成后刷新侧边栏
      loadSessionList();

    } catch (err) {
      ctrl.showError(err.message || "网络或数据访问异常，请稍后重试。");
    } finally {
      sendBtn.disabled = false;
      isSearching = false;
    }
  }

  // ── 消息 DOM 创建 ──────────────────────────────────────────

  function appendUserMessage(text) {
    var msg = document.createElement("div");
    msg.className = "message message--user";
    msg.innerHTML =
      '<div class="message__avatar">' + SVG_USER + '</div>' +
      '<div class="message__body">' +
        '<div class="message__role">User</div>' +
        '<div class="message__content"><p class="message__text"></p></div>' +
      '</div>';
    msg.querySelector(".message__text").textContent = text;
    conversation.appendChild(msg);
    scrollToBottom();
  }

  function appendAgentMessage() {
    var msg = document.createElement("div");
    msg.className = "message message--agent";
    msg.innerHTML =
      '<div class="message__avatar">' + SVG_AGENT + '</div>' +
      '<div class="message__body">' +
        '<div class="message__role">Deep Truth Search</div>' +
        '<div class="message__content">' +
          '<div class="agent-progress" style="display:none;">' +
            '<div class="progress-steps" role="progressbar" aria-label="搜索进度">' +
              '<div class="progress-step active" data-step="analyzing">' +
                '<div class="step-circle">1</div><span class="step-label">分析观点</span>' +
              '</div>' +
              '<div class="progress-connector"><div class="connector-fill"></div></div>' +
              '<div class="progress-step pending" data-step="searching">' +
                '<div class="step-circle">2</div><span class="step-label">搜索证据</span>' +
              '</div>' +
              '<div class="progress-connector"><div class="connector-fill"></div></div>' +
              '<div class="progress-step pending" data-step="generating">' +
                '<div class="step-circle">3</div><span class="step-label">生成结果</span>' +
              '</div>' +
            '</div>' +
            '<div class="status-panel">' +
              '<div class="status-panel__indicator">' +
                '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>' +
              '</div>' +
              '<p class="status-panel__message">正在分析观点...</p>' +
              '<p class="status-panel__detail"></p>' +
            '</div>' +
            '<ul class="search-status__log"></ul>' +
          '</div>' +
          '<div class="agent-results" style="display:none;"></div>' +
          '<div class="agent-error" style="display:none;"></div>' +
        '</div>' +
      '</div>';
    conversation.appendChild(msg);
    scrollToBottom();
    return msg;
  }

  // ── MessageController（每条 Agent 消息独立的控制器）────────

  function createMessageController(agentEl) {
    var contentEl   = agentEl.querySelector(".message__content");
    var progressEl  = contentEl.querySelector(".agent-progress");
    var resultsEl   = contentEl.querySelector(".agent-results");
    var errorEl     = contentEl.querySelector(".agent-error");

    var statusMsg   = progressEl.querySelector(".status-panel__message");
    var statusDet   = progressEl.querySelector(".status-panel__detail");
    var logEl       = progressEl.querySelector(".search-status__log");
    var steps       = progressEl.querySelectorAll(".progress-step");
    var connectors  = progressEl.querySelectorAll(".progress-connector");

    var lastStatusMsg = "";
    var lastStage     = "";
    var progressStartTime = 0;
    var MIN_ANALYZING_MS  = 1500;  // "分析观点"阶段最短显示时间
    var pendingUpdate     = null;  // 延迟的状态更新

    function showProgress(msg) {
      progressStartTime = Date.now();
      progressEl.style.display = "block";
      resultsEl.style.display  = "none";
      errorEl.style.display    = "none";
      statusMsg.textContent    = msg;
      statusDet.textContent    = "";
      logEl.innerHTML          = "";
      lastStatusMsg            = msg;
      lastStage                = "analyzing";
      steps.forEach(function (s, i) {
        s.classList.remove("active", "completed", "pending");
        s.querySelector(".step-circle").textContent = String(i + 1);
        if (i === 0) s.classList.add("active");
        else s.classList.add("pending");
      });
      connectors.forEach(function (c) { c.classList.remove("filled"); });
    }

    function setStage(stage) {
      var idx = STAGES.indexOf(stage);
      if (idx === -1 && stage !== "done") return;
      var targetIdx = (stage === "done") ? STAGES.length : idx;

      steps.forEach(function (step, i) {
        step.classList.remove("active", "completed", "pending");
        if (i < targetIdx) {
          step.classList.add("completed");
          step.querySelector(".step-circle").innerHTML = "&#10003;";
        } else if (i === targetIdx && stage !== "done") {
          step.classList.add("active");
        } else {
          step.classList.add("pending");
        }
      });
      connectors.forEach(function (conn, i) {
        if (i < targetIdx) conn.classList.add("filled");
        else conn.classList.remove("filled");
      });
    }

    // ── 并行搜索进度跟踪 ──────────────────────────────────────

    var parallelContainer = null;  // .parallel-searches 容器
    var parallelSubclaims = [];    // batch_start 时记录的子观点文本列表
    var parallelItems = {};        // index → DOM element
    var parallelTotal = 0;
    var parallelDone = 0;

    function handleBatchStart(data) {
      var subclaims = data.subclaims || [];
      parallelSubclaims = subclaims;
      parallelTotal = subclaims.length;
      parallelDone = 0;
      parallelItems = {};
      searchLogs = {};
      searchDoneMap = {};
      closeLogPanel();

      // 切换到搜索阶段
      setStage("searching");

      // 更新主状态消息
      statusMsg.style.opacity = "0";
      setTimeout(function () {
        statusMsg.textContent = "正在并行搜索 " + parallelTotal + " 个方向...";
        statusMsg.style.opacity = "1";
      }, 150);
      statusDet.textContent = "";

      // 如果之前有分析阶段的消息，移入日志
      if (lastStatusMsg && lastStage === "analyzing") {
        appendStepLog(lastStatusMsg);
      }

      // 只创建容器，不创建条目 — 条目在 search_start 时才浮现
      parallelContainer = document.createElement("div");
      parallelContainer.className = "parallel-searches";

      var statusPanel = progressEl.querySelector(".status-panel");
      statusPanel.parentNode.insertBefore(parallelContainer, logEl);

      lastStatusMsg = "正在并行搜索 " + parallelTotal + " 个方向...";
      lastStage = "searching";
      scrollToBottom();
    }

    function handleSearchStart(data) {
      var idx = data.index;
      // 用 batch_start 记录的文本，search_start 的文本作为 fallback
      var text = (parallelSubclaims[idx] || data.subclaim || "");

      // 初始化该 index 的日志缓冲
      if (!searchLogs[idx]) searchLogs[idx] = [];

      // 动态创建条目并浮现
      var item = document.createElement("div");
      item.className = "parallel-search-item";
      item.setAttribute("data-index", idx);
      item.style.cursor = "pointer";
      item.innerHTML =
        '<span class="parallel-search-item__spinner"></span>' +
        '<span class="parallel-search-item__icon" style="display:none;"></span>' +
        '<span class="parallel-search-item__text"></span>' +
        '<span class="parallel-search-item__badge">搜索中...</span>';
      item.querySelector(".parallel-search-item__text").textContent = text;

      // 点击打开日志面板
      (function (capturedIdx, capturedText) {
        item.addEventListener("click", function () {
          openLogPanel(capturedIdx, capturedText);
        });
      })(idx, text);

      if (parallelContainer) {
        parallelContainer.appendChild(item);
      }
      parallelItems[idx] = item;
      scrollToBottom();
    }

    function handleSearchDone(data) {
      var idx = data.index;
      var item = parallelItems[idx];

      // 如果 search_start 未到但 search_done 先到（理论上不会），创建条目
      if (!item) {
        handleSearchStart(data);
        item = parallelItems[idx];
        if (!item) return;
      }

      parallelDone++;
      searchDoneMap[idx] = true;
      var hasError = !!data.error;

      // 如果面板正在显示这个 index，切换为已完成样式
      if (activePanelIndex === idx) {
        var _p = getLogPanel();
        if (_p) {
          _p.el.classList.remove("search-log-panel--active");
          _p.el.classList.add("search-log-panel--done");
        }
      }

      // 隐藏 spinner，显示图标
      item.querySelector(".parallel-search-item__spinner").style.display = "none";
      var iconEl = item.querySelector(".parallel-search-item__icon");
      iconEl.style.display = "";

      if (hasError) {
        item.classList.add("parallel-search-item--error");
        iconEl.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
        item.querySelector(".parallel-search-item__badge").textContent = "失败";
      } else {
        item.classList.add("parallel-search-item--done");
        iconEl.innerHTML = SVG_CHECK;
        var count = data.evidence_found || 0;
        item.querySelector(".parallel-search-item__badge").textContent = count + " 条证据";
      }

      // 更新主状态
      statusMsg.textContent = "已完成 " + parallelDone + "/" + parallelTotal + " 个方向";

      if (parallelDone >= parallelTotal) {
        statusMsg.textContent = "所有方向搜索完成，正在整理结果...";
        lastStage = "generating";
        setStage("generating");
      }
      scrollToBottom();
    }

    function handleSearchLog(data) {
      var idx = data.index;
      if (idx == null) return;
      if (!searchLogs[idx]) searchLogs[idx] = [];
      var entry = { event: data.event || "", content: data.content || "" };
      searchLogs[idx].push(entry);

      // 如果面板正在显示这个 index，实时追加
      if (activePanelIndex === idx) {
        appendLiveLogEntry(entry);
      }
    }

    // ── 通用状态处理 ──────────────────────────────────────────

    function handleStatusUpdate(data) {
      // 分发结构化事件
      var eventType = data.type || "status";
      if (eventType === "batch_start") { handleBatchStart(data); return; }
      if (eventType === "search_start") { handleSearchStart(data); return; }
      if (eventType === "search_done") { handleSearchDone(data); return; }
      if (eventType === "search_log") { handleSearchLog(data); return; }

      var msg = data.status_message || data.message || "";
      if (!msg || msg === lastStatusMsg) return;
      if (/^Agent\s+第/.test(msg)) return;

      var stage = inferStage(msg, data.status);

      // 如果还在 analyzing 最短显示期内，且新阶段要跳到 searching，延迟执行
      if (lastStage === "analyzing" && stage !== "analyzing" && progressStartTime > 0) {
        var elapsed = Date.now() - progressStartTime;
        if (elapsed < MIN_ANALYZING_MS) {
          if (pendingUpdate) clearTimeout(pendingUpdate);
          var delayData = data;
          pendingUpdate = setTimeout(function () {
            pendingUpdate = null;
            handleStatusUpdate(delayData);
          }, MIN_ANALYZING_MS - elapsed);
          return;
        }
      }

      var stageChanged = stage !== lastStage;
      var searchStepChanged = (stage === "searching" && lastStage === "searching"
        && msg.indexOf("方向") !== -1 && lastStatusMsg.indexOf("方向") !== -1);
      if (lastStatusMsg && (stageChanged || searchStepChanged)) {
        appendStepLog(lastStatusMsg);
      }

      statusMsg.style.opacity = "0";
      setTimeout(function () {
        statusMsg.textContent = msg;
        statusMsg.style.opacity = "1";
      }, 150);

      var detail = extractDetail(msg);
      statusDet.textContent = detail;
      setStage(stage);
      lastStatusMsg = msg;
      lastStage     = stage;
      scrollToBottom();
    }

    function appendStepLog(msg) {
      var step = document.createElement("li");
      step.className = "search-status__step";
      step.innerHTML =
        '<span class="search-status__step-icon">' + SVG_CHECK + '</span>' +
        '<span class="search-status__step-text"></span>';
      step.querySelector(".search-status__step-text").textContent = msg;
      logEl.appendChild(step);
    }

    function renderResults(result) {
      progressEl.style.display = "none";
      errorEl.style.display    = "none";
      resultsEl.style.display  = "flex";
      resultsEl.innerHTML      = "";

      if (!result || !result.claims || result.claims.length === 0) {
        resultsEl.innerHTML = '<p class="agent-empty">当前主题暂无足够证据，稍后可重试或扩展搜索范围。</p>';
        return;
      }

      var summaryEl = document.createElement("div");
      summaryEl.className = "results-summary";
      summaryEl.textContent = "共找到 " + result.total_evidences + " 条证据，覆盖 " + result.claims.length + " 个论点";
      resultsEl.appendChild(summaryEl);

      result.claims.forEach(function (claim) {
        resultsEl.appendChild(createClaimCard(claim));
      });

      scrollToBottom();
    }

    function showError(msg) {
      progressEl.style.display = "none";
      resultsEl.style.display  = "none";
      errorEl.style.display    = "block";
      errorEl.innerHTML        = '<p class="error-state__text"></p>';
      errorEl.querySelector(".error-state__text").textContent = msg;
    }

    return {
      showProgress: showProgress,
      setStage: setStage,
      handleStatusUpdate: handleStatusUpdate,
      renderResults: renderResults,
      showError: showError,
    };
  }

  // ── SSE 实时状态流（作用域化到 ctrl）──────────────────────

  function streamResult(taskId, ctrl) {
    return new Promise(function (resolve, reject) {
      if (typeof EventSource === "undefined") {
        pollResult(taskId, ctrl).then(resolve).catch(reject);
        return;
      }

      var url = "/api/research/" + taskId + "/stream";
      var source;
      var settled = false;
      var sseWorking = false;
      var errorCount = 0;

      function finish(fn, val) {
        if (settled) return;
        settled = true;
        try { source.close(); } catch (_) {}
        fn(val);
      }

      function fallbackToPoll() {
        if (settled) return;
        settled = true;
        try { source.close(); } catch (_) {}
        pollResult(taskId, ctrl).then(resolve).catch(reject);
      }

      try {
        source = new EventSource(url);
      } catch (_) {
        pollResult(taskId, ctrl).then(resolve).catch(reject);
        return;
      }

      source.addEventListener("status", function (e) {
        sseWorking = true;
        errorCount = 0;
        try { ctrl.handleStatusUpdate(JSON.parse(e.data)); } catch (_) {}
      });

      source.addEventListener("done", function (e) {
        sseWorking = true;
        try {
          var d = JSON.parse(e.data);
          ctrl.setStage("done");
          setTimeout(function () {
            ctrl.renderResults(d.result);
            finish(resolve);
          }, 400);
        } catch (err) {
          finish(reject, err);
        }
      });

      source.addEventListener("error", function (e) {
        if (e.data) {
          try {
            var d = JSON.parse(e.data);
            finish(reject, new Error(d.message || "搜索过程中发生错误"));
          } catch (_) {
            finish(reject, new Error("搜索过程中发生错误"));
          }
          return;
        }
        errorCount++;
        if (!sseWorking) { fallbackToPoll(); return; }
        if (errorCount >= 3) { fallbackToPoll(); return; }
        if (source.readyState === EventSource.CLOSED) { fallbackToPoll(); return; }
      });

      setTimeout(function () {
        if (!sseWorking && !settled) fallbackToPoll();
      }, 5000);

      setTimeout(function () {
        finish(reject, new Error("搜索超时，请稍后重试。"));
      }, 1800000);
    });
  }

  // ── 轮询备用方案（作用域化到 ctrl）────────────────────────

  async function pollResult(taskId, ctrl) {
    var interval = 2000;
    var inactiveTimeout = 360000;
    var lastActiveAt = null;
    var maxAttempts = 600;

    for (var i = 0; i < maxAttempts; i++) {
      var res = await fetch("/api/research/" + taskId + "/status");
      if (!res.ok) throw new Error("查询状态失败: " + res.status);

      var data = await res.json();

      if (data.status_message) ctrl.handleStatusUpdate(data);

      if (data.status === "done") {
        ctrl.setStage("done");
        ctrl.renderResults(data.result);
        return;
      }

      if (data.status === "error") {
        throw new Error(data.error || "搜索过程中发生错误");
      }

      var serverActive = data.last_active_at ? new Date(data.last_active_at) : null;
      if (serverActive) {
        if (!lastActiveAt || serverActive.getTime() !== lastActiveAt.getTime()) {
          lastActiveAt = serverActive;
        } else {
          var silentMs = Date.now() - serverActive.getTime();
          if (silentMs > inactiveTimeout) {
            throw new Error("系统长时间无响应，请稍后重试。");
          }
        }
      }

      await sleep(interval);
    }

    throw new Error("搜索超时，请稍后重试。");
  }

  // ── 共享辅助函数 ──────────────────────────────────────────

  function inferStage(msg, status) {
    if (!msg) return "analyzing";
    if (/^Agent\s+第/.test(msg)) return "analyzing";
    if (msg.indexOf("分析") !== -1 || msg.indexOf("规划") !== -1) return "analyzing";
    if (msg.indexOf("搜索") !== -1) return "searching";
    if (msg.indexOf("生成") !== -1 || msg.indexOf("整理") !== -1 || msg.indexOf("组装") !== -1) return "generating";
    return "analyzing";
  }

  function extractDetail(msg) {
    var match = msg.match(/方向[：:]\s*(.+)/);
    return match ? match[1] : "";
  }

  function formatTime(isoStr) {
    if (!isoStr) return "";
    var d = new Date(isoStr);
    var now = new Date();
    var diff = now - d;

    if (diff < 60000) return "刚刚";
    if (diff < 3600000) return Math.floor(diff / 60000) + " 分钟前";
    if (diff < 86400000) return Math.floor(diff / 3600000) + " 小时前";
    if (diff < 604800000) return Math.floor(diff / 86400000) + " 天前";

    var month = d.getMonth() + 1;
    var day = d.getDate();
    return month + "月" + day + "日";
  }

  // ── 创建论点卡片（复用原有逻辑）──────────────────────────

  function createClaimCard(claim) {
    var card = document.createElement("div");
    card.className = "claim-card";

    var title = document.createElement("h2");
    title.className = "claim-card__title";
    title.textContent = claim.claim_title;
    card.appendChild(title);

    var count = document.createElement("p");
    count.className = "claim-card__count";
    count.textContent = claim.evidences.length + " 条证据";
    card.appendChild(count);

    var list = document.createElement("ul");
    list.className = "evidence-list";

    var defaultShow = 2;
    claim.evidences.forEach(function (ev, idx) {
      var item = createEvidenceItem(ev);
      if (idx >= defaultShow) {
        item.style.display = "none";
        item.dataset.hidden = "true";
      }
      list.appendChild(item);
    });

    card.appendChild(list);

    if (claim.evidences.length > defaultShow) {
      var toggleBtn = document.createElement("button");
      toggleBtn.className = "claim-card__toggle";
      toggleBtn.textContent = "展开全部 (" + claim.evidences.length + ")";
      toggleBtn.setAttribute("aria-expanded", "false");

      toggleBtn.addEventListener("click", function () {
        var expanded = toggleBtn.getAttribute("aria-expanded") === "true";
        var hiddenItems = list.querySelectorAll("[data-hidden]");

        hiddenItems.forEach(function (el) {
          el.style.display = expanded ? "none" : "block";
        });

        toggleBtn.setAttribute("aria-expanded", String(!expanded));
        toggleBtn.textContent = expanded
          ? "展开全部 (" + claim.evidences.length + ")"
          : "收起";
      });

      card.appendChild(toggleBtn);
    }

    return card;
  }

  // ── 创建证据条目（含评分徽章，复用原有逻辑）──────────────

  function createEvidenceItem(ev) {
    var item = document.createElement("li");
    item.className = "evidence-item";

    var text = document.createElement("p");
    text.className = "evidence-item__text";
    text.textContent = ev.evidence_text;
    item.appendChild(text);

    var source = document.createElement("div");
    source.className = "evidence-item__source";

    var link = document.createElement("a");
    link.className = "evidence-item__link";
    link.href = ev.source_url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = ev.source_title || ev.source_domain || ev.source_url;
    source.appendChild(link);

    if (ev.score && ev.score.total_score > 0) {
      var scoreWrap = document.createElement("div");
      scoreWrap.className = "evidence-item__score-wrap";

      var scoreBadge = document.createElement("button");
      var scoreVal = Math.round(ev.score.total_score);
      scoreBadge.className = "evidence-item__score" + (scoreVal >= 80 ? " evidence-item__score--high" : "");
      scoreBadge.textContent = scoreVal + " 分";
      scoreBadge.title = "点击查看评分详情";
      scoreBadge.setAttribute("aria-expanded", "false");
      scoreBadge.setAttribute("aria-label", "评分 " + Math.round(ev.score.total_score) + " 分，点击查看详情");
      scoreWrap.appendChild(scoreBadge);

      var details = document.createElement("div");
      details.className = "score-details";
      details.style.display = "none";

      if (ev.score.dimensions && ev.score.dimensions.length > 0) {
        ev.score.dimensions.forEach(function (dim) {
          var row = document.createElement("div");
          row.className = "score-details__row";

          var label = document.createElement("span");
          label.className = "score-details__label";
          label.textContent = dim.label || dim.name;

          var bar = document.createElement("div");
          bar.className = "score-details__bar";
          var fill = document.createElement("div");
          fill.className = "score-details__fill";
          var pct = Math.round((dim.score / dim.max_score) * 100);
          fill.style.width = pct + "%";
          if (pct >= 70) fill.classList.add("score-details__fill--high");
          else if (pct >= 40) fill.classList.add("score-details__fill--mid");
          else fill.classList.add("score-details__fill--low");
          bar.appendChild(fill);

          var val = document.createElement("span");
          val.className = "score-details__value";
          val.textContent = Math.round(dim.score) + "/" + Math.round(dim.max_score);

          row.appendChild(label);
          row.appendChild(bar);
          row.appendChild(val);

          if (dim.deduction_reason) row.title = dim.deduction_reason;

          details.appendChild(row);
        });
      }

      scoreWrap.appendChild(details);
      source.appendChild(scoreWrap);

      scoreBadge.addEventListener("click", function (e) {
        e.stopPropagation();
        var isVisible = details.style.display !== "none";
        details.style.display = isVisible ? "none" : "block";
        scoreBadge.classList.toggle("evidence-item__score--active", !isVisible);
        scoreBadge.setAttribute("aria-expanded", String(!isVisible));
      });
    }

    item.appendChild(source);
    return item;
  }

  // ── 会话管理 ──────────────────────────────────────────────

  async function createNewSession() {
    clearConversation();
    header.classList.remove("header--compact");
    try {
      var res = await fetch("/api/sessions", { method: "POST" });
      var data = await res.json();
      currentSessionId = data.session_id;
    } catch (err) {
      console.error("创建会话失败:", err);
      currentSessionId = null;
    }
  }

  async function loadSession(sessionId) {
    try {
      var res = await fetch("/api/sessions/" + sessionId);
      var data = await res.json();
      if (data.error) return;

      currentSessionId = sessionId;
      clearConversation();

      if (data.messages && data.messages.length > 0) {
        header.classList.add("header--compact");
      }

      data.messages.forEach(function (msg) {
        if (msg.role === "user") {
          appendUserMessage(msg.content);
        } else if (msg.role === "agent") {
          renderHistoryAgentMessage(msg.content);
        }
      });

      if (!data.messages || data.messages.length === 0) {
        emptyState.style.display = "flex";
      }

    } catch (err) {
      console.error("加载会话失败:", err);
    }
  }

  function renderHistoryAgentMessage(contentJson) {
    var agentEl = appendAgentMessage();
    var ctrl = createMessageController(agentEl);

    try {
      var result = JSON.parse(contentJson);
      ctrl.renderResults(result);
    } catch (_) {
      ctrl.showError("无法解析历史结果");
    }
  }

  function getDateGroup(isoStr) {
    if (!isoStr) return "更早";
    var d = new Date(isoStr);
    var now = new Date();
    var today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    var yesterday = new Date(today.getTime() - 86400000);
    if (d >= today) return "今天";
    if (d >= yesterday) return "昨天";
    return "更早";
  }

  function createSessionItem(s) {
    var li = document.createElement("li");
    li.className = "sidebar__item";
    li.setAttribute("tabindex", "0");
    li.setAttribute("role", "button");
    li.setAttribute("aria-label", "会话: " + s.title);
    if (s.session_id === currentSessionId) {
      li.classList.add("sidebar__item--active");
      li.setAttribute("aria-current", "true");
    }

    var titleSpan = document.createElement("span");
    titleSpan.className = "sidebar__item-title";
    titleSpan.textContent = s.title;
    li.appendChild(titleSpan);

    var timeSpan = document.createElement("span");
    timeSpan.className = "sidebar__item-time";
    timeSpan.textContent = formatTime(s.updated_at);
    li.appendChild(timeSpan);

    var delBtn = document.createElement("button");
    delBtn.className = "sidebar__item-del";
    delBtn.innerHTML = '&times;';
    delBtn.title = "删除会话";
    delBtn.setAttribute("aria-label", "删除会话: " + s.title);
    delBtn.addEventListener("click", function (e) {
      e.stopPropagation();
      deleteSession(s.session_id);
    });
    li.appendChild(delBtn);

    function activateSession() {
      loadSession(s.session_id);
      sidebar.classList.remove("sidebar--open");
      sidebarBackdrop.classList.remove("sidebar-backdrop--visible");
      loadSessionList();
    }

    li.addEventListener("click", activateSession);
    li.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        activateSession();
      }
    });

    return li;
  }

  async function loadSessionList() {
    try {
      var res = await fetch("/api/sessions");
      var sessions = await res.json();
      sessionList.innerHTML = "";

      var lastGroup = "";
      sessions.forEach(function (s) {
        if (!s.title) return;

        var group = getDateGroup(s.updated_at);
        if (group !== lastGroup) {
          var label = document.createElement("li");
          label.className = "sidebar__group-label";
          label.textContent = group;
          label.setAttribute("role", "presentation");
          sessionList.appendChild(label);
          lastGroup = group;
        }

        sessionList.appendChild(createSessionItem(s));
      });
    } catch (err) {
      console.error("加载会话列表失败:", err);
    }
  }

  async function deleteSession(sessionId) {
    try {
      await fetch("/api/sessions/" + sessionId, { method: "DELETE" });
      if (sessionId === currentSessionId) {
        createNewSession();
      }
      loadSessionList();
    } catch (err) {
      console.error("删除会话失败:", err);
    }
  }

  function clearConversation() {
    var children = Array.from(conversation.children);
    children.forEach(function (child) {
      if (child.id !== "empty-state") {
        conversation.removeChild(child);
      }
    });
    emptyState.style.display = "flex";
  }

  // ── 工具函数 ──────────────────────────────────────────────

  function sleep(ms) {
    return new Promise(function (resolve) { setTimeout(resolve, ms); });
  }

})();
