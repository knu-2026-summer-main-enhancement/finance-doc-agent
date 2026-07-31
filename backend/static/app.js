"use strict";

// 구조화된 API 응답을 채팅 본문, 인물 카드, 금액 상세로 렌더링하는 UI 진입점이다.
// 서버의 한국어 answer 문자열을 다시 파싱하지 말고 entities·segments·details를 사용한다.

const elements = {
  sidebar: document.getElementById("sidebar"),
  mainPanel: document.querySelector(".main-panel"),
  closeSidebar: document.getElementById("closeSidebar"),
  openSidebar: document.getElementById("openSidebar"),
  refreshDocuments: document.getElementById("refreshDocuments"),
  documentSearch: document.getElementById("documentSearch"),
  allDocuments: document.getElementById("allDocuments"),
  documentList: document.getElementById("documentList"),
  uploadToggle: document.getElementById("uploadToggle"),
  uploadForm: document.getElementById("uploadForm"),
  filePicker: document.getElementById("filePicker"),
  uploadFile: document.getElementById("uploadFile"),
  uploadFileName: document.getElementById("uploadFileName"),
  filenameOverride: document.getElementById("filenameOverride"),
  uploadButton: document.getElementById("uploadButton"),
  uploadProgress: document.getElementById("uploadProgress"),
  uploadProgressText: document.getElementById("uploadProgressText"),
  scopeSummary: document.getElementById("scopeSummary"),
  clearChat: document.getElementById("clearChat"),
  chatArea: document.getElementById("chatArea"),
  selectedFiles: document.getElementById("selectedFiles"),
  queryModeRow: document.getElementById("queryModeRow"),
  naturalMode: document.getElementById("naturalMode"),
  modeHelpWrap: document.getElementById("modeHelpWrap"),
  modeHelpButton: document.getElementById("modeHelpButton"),
  modeHelpPopover: document.getElementById("modeHelpPopover"),
  chatForm: document.getElementById("chatForm"),
  questionInput: document.getElementById("questionInput"),
  questionAutocomplete: document.getElementById("questionAutocomplete"),
  quickAttach: document.getElementById("quickAttach"),
  sendButton: document.getElementById("sendButton"),
  renameModal: document.getElementById("renameModal"),
  renameForm: document.getElementById("renameForm"),
  renameCurrentName: document.getElementById("renameCurrentName"),
  renameInput: document.getElementById("renameInput"),
  renameCancel: document.getElementById("renameCancel"),
  renameSubmit: document.getElementById("renameSubmit"),
  deleteModal: document.getElementById("deleteModal"),
  deleteCurrentName: document.getElementById("deleteCurrentName"),
  deleteCancel: document.getElementById("deleteCancel"),
  deleteSubmit: document.getElementById("deleteSubmit"),
  detailDialog: document.getElementById("detailDialog"),
  detailTitle: document.getElementById("detailTitle"),
  detailBody: document.getElementById("detailBody"),
  closeDetail: document.getElementById("closeDetail"),
  detailMore: document.getElementById("detailMore"),
  toast: document.getElementById("toast"),
};

const state = {
  documents: [],
  selected: new Set(),
  busy: false,
  chatController: null,
  chatRequestId: null,
  renameSource: "",
  deleteSource: "",
  suggestionCatalogController: null,
  suggestionIndex: -1,
  suggestionCatalogs: new Map(),
  suggestionCatalog: [],
  personAutocomplete: { names: [], actions: [], mode: "local" },
  remotePersonCandidates: [],
  personSuggestionController: null,
  personSuggestionTimer: null,
  personSuggestionCache: new Map(),
  suggestionRenderTimer: null,
  dateAutocomplete: { actions: [] },
  suggestionUsage: new Map(),
  documentsLoaded: false,
  ingestPolls: new Map(),
  sidebarSwipe: null,
  pendingVectorSeed: "",
};

const initialChat = elements.chatArea.innerHTML;
const SUGGESTION_USAGE_STORAGE_KEY = "finance-doc-agent.suggestion-usage.v1";

function apiHeaders(json = false) {
  const headers = {};
  if (json) headers["Content-Type"] = "application/json";
  return headers;
}

function isMobileChatUi() {
  return window.innerWidth <= 820
    && document.documentElement.classList.contains("ui-v3");
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("show");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => elements.toast.classList.remove("show"), 2200);
}

function fileType(document) {
  return String(document.file_type || document.source?.split(".").pop() || "DOC").toUpperCase().slice(0, 5);
}

function renderDocumentIcon(icon, source, item) {
  const extension = String(source).split(".").pop().toLocaleLowerCase("ko-KR");
  const icons = {
    xlsx: { className: "excel", markup: '<path d="M8 3.5h7.2L19 7.3v13.2H8z" fill="currentColor" opacity=".95"/><path d="M15 3.7v3.8h3.8" fill="none" stroke="#bce6c9" stroke-width="1.3" stroke-linejoin="round"/><path d="M4.5 7h7v10h-7z" fill="#fff"/><path d="m6.3 9 3.4 6M9.7 9l-3.4 6" stroke="#207245" stroke-width="1.45" stroke-linecap="round"/>' },
    pdf: { className: "pdf", markup: '<path d="M5.5 3.5h8.8L18.5 7.7v12.8h-13z" fill="currentColor"/><path d="M14 3.7v4h4" fill="none" stroke="#ffd1d1" stroke-width="1.3" stroke-linejoin="round"/><text x="7.2" y="16" fill="#fff" font-size="5.1" font-family="Arial, sans-serif" font-weight="700">PDF</text>' },
    hwp: { className: "hwp", markup: '<path d="M5.5 3.5h8.8L18.5 7.7v12.8h-13z" fill="currentColor"/><path d="M14 3.7v4h4" fill="none" stroke="#cddfff" stroke-width="1.3" stroke-linejoin="round"/><text x="6.8" y="16" fill="#fff" font-size="5.1" font-family="Arial, sans-serif" font-weight="700">HWP</text>' },
    hwpx: { className: "hwp", markup: '<path d="M5.5 3.5h8.8L18.5 7.7v12.8h-13z" fill="currentColor"/><path d="M14 3.7v4h4" fill="none" stroke="#cddfff" stroke-width="1.3" stroke-linejoin="round"/><text x="6" y="16" fill="#fff" font-size="4.2" font-family="Arial, sans-serif" font-weight="700">HWPX</text>' },
    jpg: { className: "image", markup: '<rect x="4" y="5" width="16" height="14" rx="2" fill="currentColor"/><circle cx="9" cy="9.3" r="1.6" fill="#e2dfff"/><path d="m5.8 17 4.5-4.8 2.8 2.8 2.2-2.3 3 4.3" fill="none" stroke="#fff" stroke-width="1.45" stroke-linecap="round" stroke-linejoin="round"/>' },
    jpeg: { className: "image", markup: '<rect x="4" y="5" width="16" height="14" rx="2" fill="currentColor"/><circle cx="9" cy="9.3" r="1.6" fill="#e2dfff"/><path d="m5.8 17 4.5-4.8 2.8 2.8 2.2-2.3 3 4.3" fill="none" stroke="#fff" stroke-width="1.45" stroke-linecap="round" stroke-linejoin="round"/>' },
    png: { className: "image", markup: '<rect x="4" y="5" width="16" height="14" rx="2" fill="currentColor"/><circle cx="9" cy="9.3" r="1.6" fill="#e2dfff"/><path d="m5.8 17 4.5-4.8 2.8 2.8 2.2-2.3 3 4.3" fill="none" stroke="#fff" stroke-width="1.45" stroke-linecap="round" stroke-linejoin="round"/>' },
  };
  ["webp", "bmp", "tif", "tiff"].forEach((imageExtension) => {
    icons[imageExtension] = icons.png;
  });
  const definition = icons[extension];
  if (!definition) {
    icon.textContent = fileType(item);
    return;
  }
  icon.classList.add(definition.className);
  icon.setAttribute("aria-label", `${extension.toUpperCase()} 파일`);
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  svg.innerHTML = definition.markup;
  icon.append(svg);
}

function renderDocuments() {
  const keyword = elements.documentSearch.value.trim().toLocaleLowerCase("ko-KR");
  const filtered = state.documents.filter((document) =>
    String(document.source || "").toLocaleLowerCase("ko-KR").includes(keyword)
  );
  elements.documentList.replaceChildren();

  if (!filtered.length) {
    const empty = document.createElement("div");
    empty.className = "document-empty";
    empty.textContent = state.documents.length ? "검색 결과가 없습니다." : "적재된 문서가 없습니다.";
    elements.documentList.append(empty);
    return;
  }

  filtered.forEach((item) => {
    const source = String(item.source || "");
    const ready = item.status === "SUCCESS";
    const row = document.createElement("div");
    row.className = "document-row";
    const button = document.createElement("button");
    button.type = "button";
    button.className = `document-item${state.selected.has(source) ? " selected" : ""}`;
    button.disabled = !ready;
    button.title = ready ? source : `${source} (${item.status || "상태 미확인"})`;

    const icon = document.createElement("span");
    icon.className = "document-icon";
    renderDocumentIcon(icon, source, item);
    const label = document.createElement("span");
    const strong = document.createElement("strong");
    strong.textContent = source;
    const small = document.createElement("small");
    small.textContent = ready
      ? `색인 ${Number(item.chroma_doc_count || 0).toLocaleString()}건`
      : item.status || "처리 상태 미확인";
    label.append(strong, small);
    const dot = document.createElement("span");
    dot.className = "selection-dot";
    button.append(icon, label, dot);
    button.addEventListener("click", () => toggleDocument(source));
    const renameButton = document.createElement("button");
    renameButton.type = "button";
    renameButton.className = "rename-document";
    renameButton.textContent = "✎";
    renameButton.title = `${source} 이름 수정`;
    renameButton.setAttribute("aria-label", `${source} 이름 수정`);
    renameButton.addEventListener("click", () => renameDocument(source));
    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "delete-document";
    deleteButton.textContent = "×";
    deleteButton.title = `${source} 삭제`;
    deleteButton.setAttribute("aria-label", `${source} 삭제`);
    deleteButton.addEventListener("click", () => deleteDocument(source));
    const supportsSections = (item.metadata?.capabilities || []).includes("list_sections");
    const sectionButton = document.createElement("button");
    sectionButton.type = "button";
    sectionButton.className = "sections-document";
    sectionButton.textContent = "목차";
    sectionButton.title = `${source} 섹션 보기`;
    sectionButton.setAttribute("aria-label", `${source} 섹션 보기`);
    sectionButton.hidden = !supportsSections;
    sectionButton.addEventListener("click", () => openDocumentSections(source));
    row.classList.toggle("has-sections", supportsSections);
    row.append(button, sectionButton, renameButton, deleteButton);
    elements.documentList.append(row);
  });
}

function toggleDocument(source) {
  if (state.selected.has(source)) state.selected.delete(source);
  else state.selected.add(source);
  updateScope();
}

function updateScope() {
  hideQuestionSuggestions();
  restoreSuggestionUsage();
  state.remotePersonCandidates = [];
  state.personSuggestionCache.clear();
  const selected = [...state.selected];
  elements.allDocuments.classList.toggle("selected", selected.length === 0);
  elements.scopeSummary.textContent = selected.length === 0
    ? "전체 문서"
    : selected.length === 1 ? selected[0] : `${selected.length}개 문서 선택`;
  elements.selectedFiles.replaceChildren();
  selected.forEach((source) => {
    const chip = document.createElement("span");
    chip.className = "selected-chip";
    chip.textContent = source;
    chip.title = "클릭하여 선택 해제";
    chip.addEventListener("click", () => toggleDocument(source));
    elements.selectedFiles.append(chip);
  });
  renderDocuments();
  primeQuestionCatalog();
}

async function loadDocuments() {
  state.suggestionCatalogs.clear();
  state.documentsLoaded = false;
  elements.documentList.innerHTML = '<div class="document-loading">문서 목록을 불러오는 중입니다.</div>';
  try {
    const response = await fetch("/documents", {
      headers: apiHeaders(),
      cache: "no-store",
    });
    if (!response.ok) throw new Error(await errorMessage(response));
    const data = await response.json();
    state.documents = Array.isArray(data.files) ? data.files : [];
    const available = new Set(state.documents.map((item) => item.source));
    state.selected.forEach((source) => { if (!available.has(source)) state.selected.delete(source); });
    state.documentsLoaded = true;
    updateScope();
  } catch (error) {
    state.documents = [];
    state.documentsLoaded = true;
    elements.documentList.innerHTML = `<div class="document-empty"></div>`;
    elements.documentList.firstElementChild.textContent = `목록 조회 실패: ${error.message}`;
  }
}

function deleteDocument(source) {
  state.deleteSource = source;
  elements.deleteCurrentName.textContent = source;
  elements.deleteModal.hidden = false;
  window.setTimeout(() => elements.deleteCancel.focus(), 0);
}

function closeDeleteModal() {
  elements.deleteModal.hidden = true;
  state.deleteSource = "";
  elements.deleteSubmit.disabled = false;
}

async function submitDeleteDocument() {
  const source = state.deleteSource;
  if (!source) return;

  try {
    elements.deleteSubmit.disabled = true;
    const response = await fetch(`/documents/${encodeURIComponent(source)}`, {
      method: "DELETE",
      headers: apiHeaders(),
    });
    if (!response.ok) throw new Error(await errorMessage(response));
    state.selected.delete(source);
    closeDeleteModal();
    await loadDocuments();
    showToast(`'${source}' 문서를 삭제했습니다.`);
  } catch (error) {
    showToast(`삭제 실패: ${error.message}`);
    elements.deleteSubmit.disabled = false;
  }
}

function renameDocument(source) {
  state.renameSource = source;
  elements.renameCurrentName.textContent = source;
  elements.renameInput.value = source;
  elements.renameModal.hidden = false;
  window.setTimeout(() => {
    elements.renameInput.focus();
    elements.renameInput.select();
  }, 0);
}

function closeRenameModal() {
  elements.renameModal.hidden = true;
  state.renameSource = "";
  elements.renameSubmit.disabled = false;
}

async function submitRenameDocument(event) {
  event.preventDefault();
  const source = state.renameSource;
  const newName = elements.renameInput.value.trim();
  if (!source || !newName || newName === source) {
    closeRenameModal();
    return;
  }

  try {
    elements.renameSubmit.disabled = true;
    const response = await fetch(`/documents/${encodeURIComponent(source)}`, {
      method: "PATCH",
      headers: apiHeaders(true),
      body: JSON.stringify({ new_name: newName }),
    });
    if (!response.ok) throw new Error(await errorMessage(response));
    const data = await response.json();
    const filename = data.filename || newName;
    state.selected.delete(source);
    closeRenameModal();
    await loadDocuments();
    showToast(data.message || "파일 이름을 변경했습니다.");
    if (data.status === "accepted") pollIngestStatus(filename);
  } catch (error) {
    showToast(`이름 수정 실패: ${error.message}`);
    elements.renameSubmit.disabled = false;
  }
}

function selectUploadFile(file) {
  if (!file) return;
  const allowed = new Set(["xlsx", "pdf", "hwp", "hwpx", "png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"]);
  const extension = file.name.includes(".") ? file.name.split(".").pop().toLocaleLowerCase() : "";
  if (!allowed.has(extension)) {
    elements.uploadFile.value = "";
    elements.uploadFileName.textContent = "지원하지 않는 파일 형식";
    showToast(`.${extension || "(확장자 없음)"} 파일은 업로드할 수 없습니다.`);
    return;
  }
  elements.uploadFileName.textContent = file.name;
}

async function pollIngestStatus(filename) {
  const pollToken = Symbol(filename);
  state.ingestPolls.set(filename, pollToken);
  const deadline = Date.now() + (30 * 60 * 1000);
  let transientErrors = 0;
  while (Date.now() < deadline && state.ingestPolls.get(filename) === pollToken) {
    const elapsed = (30 * 60 * 1000) - (deadline - Date.now());
    const delay = elapsed < 2 * 60 * 1000 ? 2000 : 5000;
    await new Promise((resolve) => window.setTimeout(resolve, delay));
    try {
      const response = await fetch(`/status?source=${encodeURIComponent(filename)}`, {
        headers: apiHeaders(),
        cache: "no-store",
      });
      if (response.status === 404) continue;
      if (!response.ok) throw new Error(await errorMessage(response));
      const status = await response.json();
      transientErrors = 0;
      if (status.status === "SUCCESS") {
        elements.uploadProgressText.textContent = "적재 완료";
        await loadDocuments();
        showToast(`'${filename}' 적재가 완료됐습니다.`);
        state.ingestPolls.delete(filename);
        return;
      }
      if (status.status === "FAILED") {
        const message = status.error_message || "문서 적재에 실패했습니다.";
        elements.uploadProgressText.textContent = "적재 실패";
        showToast(message);
        state.ingestPolls.delete(filename);
        await loadDocuments();
        return;
      }
      elements.uploadProgressText.textContent = "문서 분석 및 적재 중";
    } catch (error) {
      transientErrors += 1;
      if (transientErrors < 5) continue;
      elements.uploadProgressText.textContent = "상태 확인 재시도 필요";
      showToast(error.message);
      state.ingestPolls.delete(filename);
      return;
    }
  }
  if (state.ingestPolls.get(filename) === pollToken) {
    state.ingestPolls.delete(filename);
    elements.uploadProgressText.textContent = "적재가 오래 걸리고 있습니다";
    await loadDocuments();
  }
}

async function uploadDocument(event) {
  event.preventDefault();
  const file = elements.uploadFile.files[0];
  if (!file) {
    showToast("업로드할 파일을 먼저 선택하세요.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  const override = elements.filenameOverride.value.trim();
  const query = override ? `?filename_override=${encodeURIComponent(override)}` : "";
  elements.uploadButton.disabled = true;
  elements.uploadProgress.hidden = false;
  elements.uploadProgressText.textContent = "파일 업로드 중";

  try {
    const response = await fetch(`/ingest/upload${query}`, {
      method: "POST",
      headers: apiHeaders(),
      body: formData,
    });
    if (!response.ok) throw new Error(await errorMessage(response));
    const data = await response.json();
    const filename = data.filename || override || file.name;
    elements.uploadProgressText.textContent = "문서 분석 및 적재 중";
    showToast(data.message || "업로드를 시작했습니다.");
    elements.uploadFile.value = "";
    elements.filenameOverride.value = "";
    elements.uploadFileName.textContent = "파일 선택";
    await loadDocuments();
    pollIngestStatus(filename);
  } catch (error) {
    elements.uploadProgress.hidden = true;
    showToast(`업로드 실패: ${error.message}`);
  } finally {
    elements.uploadButton.disabled = false;
  }
}

function appendMessage(
  role,
  text,
  route = "",
  sources = [],
  retryRequest = null,
  actionsHidden = false,
  evidence = [],
) {
  const message = document.createElement("article");
  message.className = `message ${role}`;

  if (role === "assistant") {
    const head = document.createElement("div");
    head.className = "message-head";
    head.textContent = "Finance Doc";
    if (route) {
      const badge = document.createElement("span");
      badge.className = `route-badge ${route.toLocaleLowerCase()}`;
      badge.textContent = route === "natural" ? "AI 문서 검색" : route.toUpperCase();
      head.append(badge);
    }
    message.append(head);
  }

  const body = document.createElement("div");
  body.className = "message-body";
  body.textContent = text;
  message.append(body);

  if (sources.length) {
    const sourceRow = document.createElement("div");
    sourceRow.className = "source-row";
    sources.forEach((source) => {
      const chip = document.createElement("span");
      chip.className = "source-chip";
      chip.textContent = source;
      sourceRow.append(chip);
    });
    message.append(sourceRow);
  }
  if (role === "assistant" && evidence.length) {
    const details = document.createElement("details");
    details.className = "document-evidence";
    const summary = document.createElement("summary");
    summary.textContent = `근거 ${evidence.length}개`;
    const list = document.createElement("div");
    list.className = "document-evidence-list";
    evidence.forEach((item) => {
      const row = document.createElement("div");
      row.className = "document-evidence-item";
      const location = [
        item.source,
        item.page ? `p.${item.page}` : "",
      ].filter(Boolean).join(" · ");
      const title = item.section_title ? ` — ${item.section_title}` : "";
      row.textContent = `${location}${title}`;
      list.append(row);
    });
    details.append(summary, list);
    message.append(details);
  }
  if (role === "assistant" && retryRequest) {
    const actions = document.createElement("div");
    actions.className = "message-actions";
    actions.hidden = actionsHidden;

    const copyButton = document.createElement("button");
    copyButton.className = "message-action";
    copyButton.type = "button";
    copyButton.textContent = "⧉";
    copyButton.setAttribute("aria-label", "답변 복사");
    copyButton.title = "답변 복사";
    copyButton.addEventListener("click", () => copyAnswer(body.textContent));

    const retryButton = document.createElement("button");
    retryButton.className = "message-action";
    retryButton.type = "button";
    retryButton.textContent = "↻";
    retryButton.setAttribute("aria-label", "답변 다시 시도");
    retryButton.title = "답변 다시 시도";
    retryButton.addEventListener("click", () => {
      if (state.busy) {
        showToast("현재 답변을 생성 중입니다.");
        return;
      }
      sendQuestion(retryRequest.question);
    });
    actions.append(copyButton, retryButton);
    message.append(actions);
  }
  elements.chatArea.append(message);
  elements.chatArea.scrollTop = elements.chatArea.scrollHeight;
  return message;
}

function appendLoading() {
  const message = document.createElement("article");
  message.className = "message assistant";
  const head = document.createElement("div");
  head.className = "message-head";
  head.textContent = "Finance Doc · 답변 생성 중";
  const dots = document.createElement("div");
  dots.className = "loading-dots";
  dots.innerHTML = "<i></i><i></i><i></i>";
  message.append(head, dots);
  elements.chatArea.append(message);
  elements.chatArea.scrollTop = elements.chatArea.scrollHeight;
  return message;
}

async function errorMessage(response) {
  try {
    const data = await response.json();
    return data.detail || `요청 실패 (${response.status})`;
  } catch (_) {
    return `요청 실패 (${response.status})`;
  }
}

function evidenceMatch(text) {
  const evidenceTypes = [
    { marker: "계산 근거:", label: "계산 근거" },
    { marker: "조회 근거:", label: "조회 근거" },
  ];
  return evidenceTypes
    .map((type) => ({ ...type, index: text.indexOf(type.marker) }))
    .filter((type) => type.index >= 0)
    .sort((left, right) => left.index - right.index)[0];
}

function renderExpandableNameList(body, names, resultReference, fullAnswer = "") {
  const initialVisibleCount = 200;
  if (!Array.isArray(names) || !names.length || !resultReference) return false;
  const evidence = evidenceMatch(fullAnswer);
  const answerEnd = evidence ? evidence.index : fullAnswer.length;
  const firstNameIndex = fullAnswer.indexOf(names[0].display_name);
  const heading = (firstNameIndex >= 0
    ? fullAnswer.slice(0, firstNameIndex)
    : fullAnswer.slice(0, answerEnd)
  ).trimEnd();
  body.replaceChildren(document.createTextNode(heading));
  const list = document.createElement("div");
  list.className = "expandable-name-list";
  const extraRows = document.createElement("div");
  extraRows.className = "expandable-name-extra";
  extraRows.hidden = true;
  names.forEach((entry, index) => {
    const row = document.createElement("div");
    row.className = "expandable-name-row";
    row.append(document.createTextNode("- "));
    const button = document.createElement("button");
    button.type = "button";
    button.className = "inline-detail-link entity";
    button.textContent = entry.display_name;
    button.title = "인물 정보와 납부 기록 보기";
    button.addEventListener("click", () => openRecordEntity(resultReference, entry.row_index));
    row.append(button);
    if (index >= initialVisibleCount) extraRows.append(row);
    else list.append(row);
  });
  list.append(extraRows);
  body.append(list);
  if (names.length > initialVisibleCount) {
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "records-more-button";
    toggle.textContent = `더 보기 (${names.length - initialVisibleCount}명)`;
    let expanded = false;
    toggle.addEventListener("click", () => {
      expanded = !expanded;
      if (expanded) {
        extraRows.hidden = false;
        extraRows.animate(
          [{ opacity: 0, transform: "translateY(-6px)" }, { opacity: 1, transform: "translateY(0)" }],
          { duration: 220, easing: "ease-out" },
        );
      } else {
        const animation = extraRows.animate(
          [{ opacity: 1, transform: "translateY(0)" }, { opacity: 0, transform: "translateY(-6px)" }],
          { duration: 160, easing: "ease-in" },
        );
        animation.onfinish = () => { extraRows.hidden = true; };
      }
      toggle.textContent = expanded ? "접기" : `더 보기 (${names.length - initialVisibleCount}명)`;
    });
    body.append(toggle);
  }
  if (evidence) body.append(document.createTextNode(`\n\n${fullAnswer.slice(evidence.index)}`));
  collapseEvidence(body);
  return true;
}

function renderInlineSegments(body, segments, fullAnswer = "") {
  if (!Array.isArray(segments) || !segments.length) {
    collapseEvidence(body);
    return;
  }
  body.replaceChildren();
  segments.forEach((segment) => {
    if (!segment.detail_ref && !segment.result_ref) {
      body.append(document.createTextNode(segment.text || ""));
      return;
    }
    const button = document.createElement("button");
    button.type = "button";
    button.className = `inline-detail-link ${segment.kind === "record_entity" ? "entity" : (segment.kind || "detail")}`;
    button.textContent = segment.text || "상세 보기";
    button.title = (segment.kind === "entity" || segment.kind === "record_entity") ? "인물 정보와 납부 기록 보기" : "금액 계산 근거 보기";
    button.addEventListener("click", () => {
      if (segment.result_ref) {
        openRecordEntity(segment.result_ref, segment.row_index);
      } else {
        openDetail(segment.detail_ref);
      }
    });
    body.append(button);
  });
  // Interactive segments omit calculation evidence, so restore the original
  // text before turning the evidence into the shared collapsible section.
  const originalEvidence = evidenceMatch(fullAnswer);
  if (originalEvidence && !evidenceMatch(body.textContent)) {
    body.append(document.createTextNode(`\n\n${fullAnswer.slice(originalEvidence.index)}`));
  }
  collapseEvidence(body);
}

function appendDetailFields(container, fields) {
  Object.entries(fields || {}).forEach(([label, value]) => {
    const field = document.createElement("div");
    const name = document.createElement("span");
    const content = document.createElement("strong");
    name.textContent = label;
    content.textContent = value ?? "-";
    field.append(name, content);
    container.append(field);
  });
}

function renderDetail(detail) {
  elements.detailBody.replaceChildren();
  elements.detailTitle.textContent = detail.kind === "entity_detail"
    ? `${detail.display_name || "인물"} 정보`
    : detail.kind === "entity_collection_detail" ? `${detail.display_name || "동명이인"} 선택`
      : detail.kind === "records_detail" ? (detail.title || "조회 결과 전체 목록") : "금액 계산 근거";

  if (detail.kind === "entity_detail") {
    (detail.attributes || []).forEach((item) => {
      const row = document.createElement("div");
      row.className = "detail-row";
      const label = document.createElement("strong");
      const value = document.createElement("span");
      label.textContent = item.column;
      value.textContent = item.value ?? "-";
      row.append(label, value);
      elements.detailBody.append(row);
    });
    if ((detail.payment_history || []).length) {
      const title = document.createElement("h3");
      title.className = "detail-section-title";
      title.textContent = `납부 기록 ${detail.payment_history.length}건`;
      elements.detailBody.append(title);
      detail.payment_history.forEach((record, index) => {
        const card = document.createElement("div");
        card.className = "detail-record-card payment-history-card";
        const number = document.createElement("span");
        number.className = "detail-record-number";
        number.textContent = index + 1;
        const fields = document.createElement("div");
        fields.className = "detail-record-fields";
        (record.fields || []).forEach((item) => appendDetailFields(fields, { [item.column]: item.value }));
        card.append(number, fields);
        elements.detailBody.append(card);
      });
    }
  } else if (detail.kind === "entity_collection_detail") {
    (detail.candidates || []).forEach((candidate, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "detail-candidate";
      button.textContent = `${detail.display_name} ${index + 1} · 상세 보기`;
      button.addEventListener("click", () => openDetail(candidate.detail_ref));
      elements.detailBody.append(button);
    });
  } else if (detail.kind === "records_detail") {
    (detail.records || []).forEach((record, index) => {
      const card = document.createElement("div");
      card.className = "detail-record-card";
      const number = document.createElement("span");
      number.className = "detail-record-number";
      number.textContent = (detail.page?.offset || 0) + index + 1;
      const fields = document.createElement("div");
      fields.className = "detail-record-fields";
      appendDetailFields(fields, record);
      card.append(number, fields);
      elements.detailBody.append(card);
    });
  } else {
    const summary = document.createElement("div");
    summary.className = "calculation-summary";
    appendDetailFields(summary, { 계산: detail.operation, 대상: detail.target, 결과: detail.value, "유효/제외": `${detail.valid_rows ?? 0} / ${detail.excluded_rows ?? 0}` });
    elements.detailBody.append(summary);
    (detail.contributors || []).forEach((record, index) => {
      const card = document.createElement("div");
      card.className = "detail-record-card";
      const number = document.createElement("span");
      number.className = "detail-record-number";
      number.textContent = (detail.page?.offset || 0) + index + 1;
      const fields = document.createElement("div");
      fields.className = "detail-record-fields";
      appendDetailFields(fields, record);
      card.append(number, fields);
      elements.detailBody.append(card);
    });
  }
  const hasMoreDetails = Boolean(
    detail.page
    && detail.page.has_more
    && Number.isFinite(Number(detail.page.offset))
    && Number.isFinite(Number(detail.page.limit))
  );
  elements.detailMore.hidden = !hasMoreDetails;
  elements.detailMore.onclick = hasMoreDetails
    ? () => openDetail(detail._reference, detail.page.offset + detail.page.limit)
    : null;
  if (!elements.detailDialog.open) elements.detailDialog.showModal();
}

function sectionPageLabel(section) {
  if (!section.start_page) return "페이지 정보 없음";
  return section.start_page === section.end_page
    ? `${section.start_page}쪽`
    : `${section.start_page}~${section.end_page}쪽`;
}

async function openDocumentSections(source) {
  elements.detailTitle.textContent = `${source} · 목차`;
  elements.detailBody.innerHTML = '<div class="document-loading">섹션을 불러오는 중입니다.</div>';
  elements.detailMore.hidden = true;
  if (!elements.detailDialog.open) elements.detailDialog.showModal();
  try {
    const response = await fetch(
      `/documents/${encodeURIComponent(source)}/sections`,
      { headers: apiHeaders() }
    );
    if (!response.ok) throw new Error(await errorMessage(response));
    const data = await response.json();
    elements.detailBody.replaceChildren();
    const summary = document.createElement("p");
    summary.className = "section-browser-summary";
    summary.textContent = `${data.statistics?.section_count || 0}개 섹션 · ${data.statistics?.page_count || 0}쪽`;
    elements.detailBody.append(summary);
    (data.sections || []).forEach((section) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "section-browser-item";
      const title = document.createElement("strong");
      title.textContent = section.title;
      const meta = document.createElement("small");
      meta.textContent = `${sectionPageLabel(section)} · ${section.chunk_count}개 조각`;
      button.append(title, meta);
      button.addEventListener("click", () => openDocumentSection(source, section.section_id));
      elements.detailBody.append(button);
    });
  } catch (error) {
    elements.detailBody.textContent = `섹션 조회 실패: ${error.message}`;
  }
}

async function openDocumentSection(source, sectionId) {
  elements.detailTitle.textContent = "섹션 내용";
  elements.detailBody.innerHTML = '<div class="document-loading">섹션 내용을 불러오는 중입니다.</div>';
  // The section can be opened from either the document browser or a chat
  // overview.  The latter has no already-open detail dialog.
  if (!elements.detailDialog.open) elements.detailDialog.showModal();
  try {
    const response = await fetch(
      `/documents/${encodeURIComponent(source)}/sections/${encodeURIComponent(sectionId)}`,
      { headers: apiHeaders() }
    );
    if (!response.ok) throw new Error(await errorMessage(response));
    const section = await response.json();
    elements.detailTitle.textContent = section.title || "섹션 내용";
    elements.detailBody.replaceChildren();

    const actions = document.createElement("div");
    actions.className = "section-browser-actions";
    const back = document.createElement("button");
    back.type = "button";
    back.className = "section-browser-action secondary";
    back.textContent = "목차로";
    back.addEventListener("click", () => openDocumentSections(source));
    const ask = document.createElement("button");
    ask.type = "button";
    ask.className = "section-browser-action primary";
    ask.textContent = "이 섹션 질문하기";
    ask.addEventListener("click", () => {
      elements.detailDialog.close();
      elements.questionInput.value = `${section.title}에 대해 알려줘`;
      resizeTextarea();
      elements.questionInput.focus();
    });
    actions.append(back, ask);

    const meta = document.createElement("p");
    meta.className = "section-browser-summary";
    meta.textContent = `${sectionPageLabel(section)} · ${section.chunk_count}개 조각`;
    const content = document.createElement("div");
    content.className = "section-browser-content";
    content.textContent = section.content || "표시할 섹션 내용이 없습니다.";
    elements.detailBody.append(actions, meta, content);
  } catch (error) {
    elements.detailBody.textContent = `섹션 내용 조회 실패: ${error.message}`;
  }
}

async function openDetail(reference, offset = 0) {
  try {
    const response = await fetch(`/chat/details/${encodeURIComponent(reference)}?offset=${offset}&limit=50`, { headers: apiHeaders() });
    if (!response.ok) throw new Error(await errorMessage(response));
    const detail = await response.json();
    detail._reference = reference;
    renderDetail(detail);
  } catch (error) {
    showToast(error.message || "상세 정보를 불러오지 못했습니다.");
  }
}

async function openRecordEntity(resultReference, rowIndex) {
  try {
    const response = await fetch(
      `/chat/results/${encodeURIComponent(resultReference)}/person/${rowIndex}`,
      { headers: apiHeaders() },
    );
    if (!response.ok) throw new Error(await errorMessage(response));
    const detail = await response.json();
    renderDetail(detail);
  } catch (error) {
    showToast(error.message || "인물 정보를 불러오지 못했습니다.");
  }
}

function appendRecordRows(message, records, columns, recordEntities = [], bulletList = false, resultReference = "") {
  if (!records?.length) return;
  const body = message.querySelector(".message-body");
  const evidence = body.querySelector(".calculation-evidence");
  if (bulletList) {
    const continuation = document.createElement("span");
    continuation.className = "record-continuation";
    continuation.append(document.createTextNode("\n"));
    records.forEach((record, index) => {
      const entity = recordEntities[index];
      const name = entity?.display_name || String(record?.[columns[0]] ?? "-");
      continuation.append(document.createTextNode("- "));
      if (entity?.detail_ref) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "inline-detail-link entity";
        button.textContent = name;
        button.title = "인물 정보와 납부 기록 보기";
        button.addEventListener("click", () => openRecordEntity(resultReference, entity.row_index));
        continuation.append(button);
      } else {
        continuation.append(document.createTextNode(name));
      }
      if (index < records.length - 1) continuation.append(document.createTextNode("\n"));
    });
    body.insertBefore(continuation, evidence || null);
    return;
  }
  const values = records.map((record) => columns.map((column) => String(record?.[column] ?? "-")));
  const widths = columns.map((column, index) => Math.max(
    column.length,
    ...values.map((row) => row[index].length),
  ));
  const continuation = document.createElement("span");
  continuation.className = "record-continuation";
  continuation.append(document.createTextNode("\n"));
  values.forEach((row, rowIndex) => {
    const entity = recordEntities[rowIndex];
    let linked = false;
    row.forEach((value, columnIndex) => {
      if (!linked && entity?.detail_ref && value === entity.display_name) {
        const name = document.createElement("button");
        name.type = "button";
        name.className = "inline-detail-link entity";
        name.textContent = value;
        name.title = "인물 정보와 납부 기록 보기";
        name.addEventListener("click", () => openRecordEntity(resultReference, entity.row_index));
        continuation.append(name);
        linked = true;
      } else {
        continuation.append(document.createTextNode(value));
      }
      const gap = " ".repeat(Math.max(0, widths[columnIndex] - value.length))
        + (columnIndex === row.length - 1 ? "" : "  ");
      continuation.append(document.createTextNode(gap));
    });
    if (rowIndex < values.length - 1) continuation.append(document.createTextNode("\n"));
  });
  body.insertBefore(continuation, evidence || null);
}

function appendRecordsMoreAction(message, result) {
  const page = result?.page;
  if (!result?.records_detail_ref || !page?.has_more) return;
  const columns = [...new Set((result.records || []).flatMap((record) => Object.keys(record || {})))];
  const bulletList = /^\s*-\s+/m.test(message.querySelector(".message-body").textContent);
  const action = document.createElement("button");
  action.type = "button";
  action.className = "records-more-button";
  let offset = Number(page.limit) || 50;
  const limit = Number(page.limit) || 50;
  const updateLabel = (total) => {
    action.textContent = `더 보기 (다음 ${Math.min(limit, Math.max(0, total - offset))}건)`;
  };
  updateLabel(Number(page.total) || offset);
  action.addEventListener("click", async () => {
    action.disabled = true;
    action.textContent = "불러오는 중…";
    try {
      const response = await fetch(
        `/chat/details/${encodeURIComponent(result.records_detail_ref)}?offset=${offset}&limit=${limit}`,
        { headers: apiHeaders() },
      );
      if (!response.ok) throw new Error(await errorMessage(response));
      const detail = await response.json();
      appendRecordRows(
        message, detail.records, columns, detail.record_entities || [], bulletList,
        result.records_detail_ref,
      );
      offset = (Number(detail.page?.offset) || offset) + (detail.records?.length || 0);
      if (detail.page?.has_more) {
        updateLabel(Number(detail.page.total) || offset);
        action.disabled = false;
      } else {
        action.remove();
      }
      elements.chatArea.scrollTop = elements.chatArea.scrollHeight;
    } catch (error) {
      action.disabled = false;
      updateLabel(Number(page.total) || offset);
      showToast(error.message || "목록을 불러오지 못했습니다.");
    }
  });
  message.append(action);
}

async function sendQuestion(question, options = {}) {
  const value = question.trim();
  if (!value || state.busy) return;

  const controller = new AbortController();
  const requestId = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  state.chatController = controller;
  state.chatRequestId = requestId;
  setChatBusy(true);
  const request = {
    question: value,
    request_id: requestId,
    sources: options.sources ? [...options.sources] : [...state.selected],
    mode: options.mode
      || (state.pendingVectorSeed ? "natural" : "")
      || (elements.naturalMode.checked ? "natural" : "auto"),
  };
  state.pendingVectorSeed = "";
  elements.questionInput.value = "";
  hideQuestionSuggestions();
  resizeTextarea();
  if (isMobileChatUi()) elements.questionInput.blur();
  elements.chatArea.querySelector(".welcome-card")?.remove();
  appendMessage("user", value);
  const loading = appendLoading();

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: apiHeaders(true),
      body: JSON.stringify(request),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(await errorMessage(response));
    const data = await response.json();
    loading.remove();
    const message = appendMessage(
      "assistant",
      data.answer || "답변이 비어 있습니다.",
      data.source || "",
      data.sources || [],
      request,
      false,
      data.evidence || [],
    );
    const renderedNameList = renderExpandableNameList(
      message.querySelector(".message-body"), data.result?.name_list,
      data.result?.records_detail_ref, data.answer || "",
    );
    if (!renderedNameList) {
      renderInlineSegments(message.querySelector(".message-body"), data.result?.inline_segments, data.answer || "");
      appendRecordsMoreAction(message, data.result);
    }
  } catch (error) {
    loading.remove();
    appendMessage(
      "assistant",
      error.name === "AbortError" ? "답변 생성을 중단했습니다." : (error.message || "답변 처리 중 오류가 발생했습니다."),
      "error",
      [],
      request,
    );
  } finally {
    if (state.chatController === controller) {
      state.chatController = null;
      state.chatRequestId = null;
    }
    setChatBusy(false);
    hideQuestionSuggestions();
    if (!isMobileChatUi()) elements.questionInput.focus();
  }
}

async function copyAnswer(text) {
  if (!text.trim()) return;
  try {
    await navigator.clipboard.writeText(text);
    showToast("답변을 복사했습니다.");
  } catch (_) {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.append(textarea);
    textarea.select();
    const copied = document.execCommand("copy");
    textarea.remove();
    showToast(copied ? "답변을 복사했습니다." : "복사하지 못했습니다.");
  }
}

function collapseEvidence(body) {
  const text = body.textContent;
  const match = evidenceMatch(text);
  if (!match) return;

  const markerIndex = match.index;
  if (markerIndex <= 0) return;

  const evidence = text.slice(markerIndex + match.marker.length).trim();
  if (!evidence) return;

  // Keep interactive name/detail buttons that precede the evidence marker.
  const answerNodes = [];
  let consumed = 0;
  for (const node of [...body.childNodes]) {
    const nodeLength = node.textContent.length;
    if (consumed + nodeLength <= markerIndex) {
      answerNodes.push(node);
    } else if (consumed < markerIndex && node.nodeType === Node.TEXT_NODE) {
      const prefix = node.textContent.slice(0, markerIndex - consumed).trimEnd();
      if (prefix) answerNodes.push(document.createTextNode(prefix));
    }
    consumed += nodeLength;
    if (consumed >= markerIndex) break;
  }

  body.replaceChildren();
  body.append(...answerNodes);

  const details = document.createElement("details");
  details.className = "calculation-evidence";
  const summary = document.createElement("summary");
  summary.textContent = match.label;
  const evidenceText = document.createElement("div");
  evidenceText.className = "calculation-evidence-body";
  evidenceText.textContent = evidence;
  details.append(summary, evidenceText);
  body.append(details);
}

function setChatBusy(busy) {
  state.busy = busy;
  elements.clearChat.disabled = busy;
  elements.sendButton.classList.toggle("stop", busy);
  elements.sendButton.setAttribute("aria-label", busy ? "답변 생성 중단" : "질문 전송");
  elements.sendButton.querySelector("[data-send-label]").textContent = busy ? "중단" : "전송";
  elements.sendButton.querySelector("[data-send-icon]").textContent = busy ? "■" : "→";
  elements.naturalMode.disabled = busy;
}

function stopChat() {
  const requestId = state.chatRequestId;
  if (requestId) {
    fetch(`/chat/cancel/${encodeURIComponent(requestId)}`, {
      method: "POST",
      headers: apiHeaders(),
    }).catch(() => {});
  }
  state.chatController?.abort();
}

function resizeTextarea() {
  elements.questionInput.style.height = "auto";
  elements.questionInput.style.height = `${Math.min(elements.questionInput.scrollHeight, 150)}px`;
}

function syncMobileComposerInset() {
  if (!document.documentElement.classList.contains("ui-v3")) {
    document.documentElement.style.removeProperty("--composer-inset");
    return;
  }
  const inset = Math.ceil(document.querySelector(".composer-wrap").getBoundingClientRect().height);
  document.documentElement.style.setProperty("--composer-inset", `${inset}px`);
}

let viewportSyncFrame = 0;

function syncMobileKeyboardInset() {
  cancelAnimationFrame(viewportSyncFrame);
  viewportSyncFrame = requestAnimationFrame(() => {
    const isMobileV3 = window.innerWidth <= 820
      && document.documentElement.classList.contains("ui-v3");
    const viewport = window.visualViewport;
    if (!isMobileV3 || !viewport || document.activeElement !== elements.questionInput) {
      document.documentElement.style.setProperty("--keyboard-offset", "0px");
      return;
    }
    const occludedHeight = Math.max(
      0,
      window.innerHeight - viewport.height - viewport.offsetTop,
    );
    // Address-bar movement is much smaller than a keyboard and should not move the composer.
    const keyboardOffset = occludedHeight >= 120 ? Math.ceil(occludedHeight) : 0;
    document.documentElement.style.setProperty("--keyboard-offset", `${keyboardOffset}px`);
  });
}

function scheduleQuestionSuggestions(delay = 120) {
  window.clearTimeout(state.suggestionRenderTimer);
  state.suggestionRenderTimer = window.setTimeout(() => {
    state.suggestionRenderTimer = null;
    showLocalQuestionSuggestions();
  }, delay);
}

let suggestionHideTimer = null;

function hideQuestionSuggestions() {
  clearTimeout(suggestionHideTimer);
  elements.questionAutocomplete.classList.remove("visible");
  syncMobileComposerInset();
  suggestionHideTimer = window.setTimeout(() => {
    elements.questionAutocomplete.hidden = true;
    elements.questionAutocomplete.replaceChildren();
    syncMobileComposerInset();
  }, window.innerWidth <= 820 ? 220 : 0);
  elements.questionInput.setAttribute("aria-expanded", "false");
  elements.questionInput.removeAttribute("aria-activedescendant");
  state.suggestionIndex = -1;
}

function setSuggestionIndex(index) {
  const options = [...elements.questionAutocomplete.querySelectorAll(".autocomplete-option")];
  if (!options.length) return;
  state.suggestionIndex = (index + options.length) % options.length;
  options.forEach((option, optionIndex) => {
    const active = optionIndex === state.suggestionIndex;
    option.classList.toggle("active", active);
    option.setAttribute("aria-selected", String(active));
  });
  elements.questionInput.setAttribute("aria-activedescendant", options[state.suggestionIndex].id);
  options[state.suggestionIndex].scrollIntoView({ block: "nearest" });
}

async function showDocumentSectionsInChat(source, question) {
  if (!source || state.busy) return;
  elements.questionInput.value = "";
  hideQuestionSuggestions();
  resizeTextarea();
  if (isMobileChatUi()) elements.questionInput.blur();
  elements.chatArea.querySelector(".welcome-card")?.remove();
  appendMessage("user", question);
  const message = appendMessage("assistant", "섹션을 불러오는 중입니다.", "metadata", [source]);
  const body = message.querySelector(".message-body");

  try {
    const response = await fetch(
      `/documents/${encodeURIComponent(source)}/sections`,
      { headers: apiHeaders() }
    );
    if (!response.ok) throw new Error(await errorMessage(response));
    const data = await response.json();
    const sections = Array.isArray(data.sections) ? data.sections : [];
    body.replaceChildren();

    const intro = document.createElement("p");
    intro.className = "section-overview-intro";
    intro.textContent = `이 문서는 ${sections.length}개 항목으로 구성되어 있어요. 원하는 항목을 누르면 질문을 바로 만들 수 있어요.`;
    const list = document.createElement("div");
    list.className = "section-overview-list";
    sections.forEach((section) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "section-overview-item";
      button.textContent = section.title;
      button.addEventListener("click", () => prepareSectionQuestion(section.title));
      list.append(button);
    });
    body.append(intro, list);
  } catch (error) {
    body.textContent = `섹션을 불러오지 못했습니다: ${error.message}`;
  }
  elements.chatArea.scrollTop = elements.chatArea.scrollHeight;
}

function prepareSectionQuestion(title) {
  elements.questionInput.value = `${title} 알려줘`;
  state.pendingVectorSeed = title;
  resizeTextarea();
  elements.questionInput.focus();
}

function chooseQuestionSuggestion(text, operation = "") {
  recordSuggestionUsage(operation);
  if (operation === "list_document_sections" && state.selected.size === 1) {
    showDocumentSectionsInChat([...state.selected][0], text);
    return;
  }
  state.pendingVectorSeed = operation === "document_section_question"
    ? text.trim().split(/\s+/).slice(0, -1).join(" ")
    : "";
  elements.questionInput.value = text;
  resizeTextarea();
  hideQuestionSuggestions();
  elements.questionInput.focus();
}

function appendHighlightedText(container, text, query) {
  const needle = query.trim();
  const index = needle ? text.toLocaleLowerCase("ko-KR").indexOf(needle.toLocaleLowerCase("ko-KR")) : -1;
  if (index < 0) {
    container.textContent = text;
    return;
  }
  container.append(document.createTextNode(text.slice(0, index)));
  const mark = document.createElement("mark");
  mark.textContent = text.slice(index, index + needle.length);
  container.append(mark, document.createTextNode(text.slice(index + needle.length)));
}

function renderQuestionSuggestions(suggestions, query = elements.questionInput.value, hint = "") {
  const suggestionLimit = window.innerWidth <= 820
    && document.documentElement.classList.contains("ui-v3")
    ? 2
    : 3;
  suggestions = suggestions.slice(0, suggestionLimit);
  clearTimeout(suggestionHideTimer);
  const wasHidden = elements.questionAutocomplete.hidden;
  elements.questionAutocomplete.replaceChildren();
  if (suggestions.length || hint) {
    const header = document.createElement("div");
    header.className = "autocomplete-header";
    const title = document.createElement("strong");
    title.textContent = query.trim() ? "이어서 질문해 보세요" : "검증된 질문";
    const navigationHint = document.createElement("span");
    navigationHint.textContent = "↑↓ 이동 · Enter 선택";
    header.append(title, navigationHint);
    elements.questionAutocomplete.append(header);
  }
  suggestions.forEach((suggestion, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "autocomplete-option";
    button.id = `questionSuggestion${index}`;
    button.dataset.text = suggestion.text;
    button.dataset.operation = suggestion.operation || "";
    button.setAttribute("role", "option");
    button.setAttribute("aria-selected", "false");
    const main = document.createElement("span");
    main.className = "autocomplete-main";
    const icon = document.createElement("span");
    icon.className = `autocomplete-icon ${suggestion.path || "classified"}`;
    icon.setAttribute("aria-hidden", "true");
    icon.textContent = suggestion.path === "fast" ? "⚡" : suggestion.path === "vector" ? "AI" : "↗";
    const text = document.createElement("span");
    text.className = "autocomplete-text";
    appendHighlightedText(text, suggestion.text, query);
    main.append(icon, text);
    const label = document.createElement("small");
    label.textContent = `${suggestion.label} · ${suggestion.path_label || "추천 질문"}`;
    button.append(main, label);
    button.addEventListener("pointerdown", (event) => event.preventDefault());
    button.addEventListener("click", () => chooseQuestionSuggestion(suggestion.text, suggestion.operation));
    button.addEventListener("mouseenter", () => setSuggestionIndex(index));
    elements.questionAutocomplete.append(button);
  });
  if (hint) {
    const message = document.createElement("div");
    message.className = "autocomplete-hint";
    message.textContent = hint;
    elements.questionAutocomplete.append(message);
  }
  const shouldOpen = suggestions.length > 0 || Boolean(hint);
  if (shouldOpen) {
    elements.questionAutocomplete.hidden = false;
    if (wasHidden) {
      elements.questionAutocomplete.classList.remove("visible");
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          elements.questionAutocomplete.classList.add("visible");
          syncMobileComposerInset();
        });
      });
    } else {
      elements.questionAutocomplete.classList.add("visible");
      syncMobileComposerInset();
    }
  } else {
    hideQuestionSuggestions();
  }
  elements.questionInput.setAttribute("aria-expanded", String(suggestions.length > 0));
  state.suggestionIndex = -1;
}

function suggestionScopeKey() {
  return [...state.selected].sort((left, right) => left.localeCompare(right, "ko-KR")).join("\u001f");
}

async function primeQuestionCatalog() {
  if (!state.documentsLoaded) return [];
  const scopeKey = suggestionScopeKey();
  if (state.suggestionCatalogs.has(scopeKey)) {
    const cached = state.suggestionCatalogs.get(scopeKey);
    state.suggestionCatalog = cached.suggestions;
    state.personAutocomplete = cached.personAutocomplete;
    state.dateAutocomplete = cached.dateAutocomplete || { actions: [] };
    return state.suggestionCatalog;
  }
  state.suggestionCatalogController?.abort();
  const controller = new AbortController();
  state.suggestionCatalogController = controller;
  try {
    const response = await fetch("/chat/suggestions", {
      method: "POST",
      headers: apiHeaders(true),
      body: JSON.stringify({ query: "", sources: [...state.selected], limit: 50, catalog: true }),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(await errorMessage(response));
    const data = await response.json();
    if (scopeKey === suggestionScopeKey()) {
      // A sectioned document exposes its first shortcuts from server-side
      // metadata.  The browser does not guess section names from PDF text.
      state.suggestionCatalog = [
        ...(Array.isArray(data.section_suggestions) ? data.section_suggestions : []),
        ...(data.suggestions || []),
      ];
      state.personAutocomplete = {
        names: Array.isArray(data.person_names) ? data.person_names : [],
        actions: Array.isArray(data.person_actions) ? data.person_actions : [],
        mode: data.person_mode === "remote" ? "remote" : "local",
      };
      state.dateAutocomplete = {
        actions: Array.isArray(data.date_actions) ? data.date_actions : [],
      };
      state.suggestionCatalogs.set(scopeKey, {
        suggestions: state.suggestionCatalog,
        personAutocomplete: state.personAutocomplete,
        dateAutocomplete: state.dateAutocomplete,
      });
      if (document.activeElement === elements.questionInput && !elements.naturalMode.checked) {
        showLocalQuestionSuggestions();
      }
    }
  } catch (error) {
    if (error.name !== "AbortError") {
      state.suggestionCatalog = [];
      state.personAutocomplete = { names: [], actions: [], mode: "local" };
      state.dateAutocomplete = { actions: [] };
    }
  } finally {
    if (state.suggestionCatalogController === controller) state.suggestionCatalogController = null;
  }
  return state.suggestionCatalog;
}

function normalizedSuggestionText(value) {
  return String(value || "").normalize("NFKC").toLocaleLowerCase("ko-KR").replace(/[^\p{L}\p{N}*]+/gu, "");
}

function suggestionTerms(query) {
  return query.split(/\s+/).filter(Boolean).flatMap((token) => {
    const normalized = normalizedSuggestionText(token);
    const withoutParticle = normalized.replace(/(은|는|이|가|을|를|의|도|만)$/u, "");
    return withoutParticle && withoutParticle !== normalized ? [normalized, withoutParticle] : [normalized];
  }).filter(Boolean);
}

function personNameMatchesInput(name, query) {
  const normalizedName = normalizedSuggestionText(name);
  const normalizedQuery = normalizedSuggestionText(query);
  if (!normalizedName || !normalizedQuery) return false;
  if (normalizedQuery.includes(normalizedName)) return true;
  if (!/^[가-힣*]+$/u.test(query)) return false;
  if (normalizedName.startsWith(normalizedQuery)) return true;
  return normalizedName.length === normalizedQuery.length
    && [...normalizedName].every((character, index) => character === "*" || character === [...normalizedQuery][index]);
}

function suggestionUsageScopeKey() {
  const selected = [...state.selected].sort((left, right) => left.localeCompare(right, "ko-KR"));
  return selected.length ? selected.join("\u001f") : "__all_documents__";
}

function restoreSuggestionUsage() {
  try {
    const saved = JSON.parse(window.localStorage.getItem(SUGGESTION_USAGE_STORAGE_KEY) || "{}");
    const records = saved[suggestionUsageScopeKey()] || {};
    state.suggestionUsage = new Map(Object.entries(records).filter(([, value]) =>
      value && Number.isFinite(Number(value.count)) && Number.isFinite(Number(value.lastUsed))
    ));
  } catch (_) {
    state.suggestionUsage = new Map();
  }
}

function recordSuggestionUsage(operation) {
  if (!operation) return;
  const current = state.suggestionUsage.get(operation) || { count: 0, lastUsed: 0 };
  state.suggestionUsage.set(operation, {
    count: Math.min(Number(current.count || 0) + 1, 50),
    lastUsed: Date.now(),
  });
  try {
    const saved = JSON.parse(window.localStorage.getItem(SUGGESTION_USAGE_STORAGE_KEY) || "{}");
    saved[suggestionUsageScopeKey()] = Object.fromEntries(state.suggestionUsage);
    window.localStorage.setItem(SUGGESTION_USAGE_STORAGE_KEY, JSON.stringify(saved));
  } catch (_) {
    // Autocomplete remains usable when local storage is unavailable.
  }
}

function rankPersonCompletions(query) {
  if (!normalizedSuggestionText(query)) return [];
  const suggestions = [];
  const names = state.personAutocomplete.mode === "remote"
    ? state.remotePersonCandidates
    : state.personAutocomplete.names;
  for (const name of names) {
    if (!personNameMatchesInput(name, query)) continue;
    for (const action of state.personAutocomplete.actions) {
      const text = `${name} ${action.suffix}`;
      suggestions.push({ ...action, text, category: "person" });
      if (suggestions.length === 9) return suggestions;
    }
  }
  return suggestions;
}

function remotePersonPrefix(query) {
  const tokens = String(query || "").normalize("NFKC").match(/[가-힣*]{2,}/gu) || [];
  const particles = /(에게|한테|께|은|는|이|가|을|를|의|도|만)$/u;
  return tokens
    .map((token) => token.replace(particles, ""))
    .filter((token) => token.length >= 2)
    .sort((left, right) => right.length - left.length)[0] || "";
}

function scheduleRemotePersonSearch(query) {
  if (state.personAutocomplete.mode !== "remote" || elements.naturalMode.checked || state.busy) return;
  const prefix = remotePersonPrefix(query);
  if (!prefix) {
    state.personSuggestionController?.abort();
    window.clearTimeout(state.personSuggestionTimer);
    state.remotePersonCandidates = [];
    return;
  }
  const cacheKey = `${suggestionScopeKey()}\u001f${prefix}`;
  const cached = state.personSuggestionCache.get(cacheKey);
  if (cached) {
    state.remotePersonCandidates = cached;
    return;
  }
  state.personSuggestionController?.abort();
  window.clearTimeout(state.personSuggestionTimer);
  state.personSuggestionTimer = window.setTimeout(async () => {
    const controller = new AbortController();
    state.personSuggestionController = controller;
    try {
      const response = await fetch("/chat/person-suggestions", {
        method: "POST",
        headers: apiHeaders(true),
        body: JSON.stringify({ prefix, sources: [...state.selected], limit: 10 }),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(await errorMessage(response));
      const data = await response.json();
      if (prefix === remotePersonPrefix(elements.questionInput.value)) {
        state.remotePersonCandidates = Array.isArray(data.names) ? data.names : [];
        state.personSuggestionCache.set(cacheKey, state.remotePersonCandidates);
        showLocalQuestionSuggestions();
      }
    } catch (error) {
      if (error.name !== "AbortError") state.remotePersonCandidates = [];
    } finally {
      if (state.personSuggestionController === controller) state.personSuggestionController = null;
    }
  }, 200);
}

function dateExpressionState(query) {
  const value = String(query || "").normalize("NFKC").trim().replace(/\s+/g, " ");
  const range = value.match(/^((?:19|20)\d{2}\s*년\s*(?:1[0-2]|[1-9])\s*월\s*부터\s*(?:19|20)\d{2}\s*년\s*(?:1[0-2]|[1-9])\s*월\s*까지)/u);
  if (range) return { prefix: range[1].replace(/\s+/g, " "), rangePending: false };
  if (/(?:부터|에서|~|〜|-)\s*(?:(?:19|20)\d{0,4}(?:\s*년)?(?:\s*(?:1[0-2]|[1-9])?\s*월?)?)?$/u.test(value)
    && /(?:19|20)\d{2}\s*년\s*(?:1[0-2]|[1-9])\s*월/u.test(value)) {
    return { prefix: "", rangePending: true };
  }
  const yearMonth = value.match(/^((?:19|20)\d{2})\s*년\s*(1[0-2]|[1-9])\s*월/u);
  if (yearMonth) return { prefix: `${yearMonth[1]}년 ${yearMonth[2]}월`, rangePending: false };
  const year = value.match(/^((?:19|20)\d{2})\s*년?/u);
  if (year) return { prefix: `${year[1]}년`, rangePending: false };
  return { prefix: "", rangePending: false };
}

function rankDateCompletions(query) {
  const { prefix } = dateExpressionState(query);
  if (!prefix) return [];
  const candidates = state.dateAutocomplete.actions.map((action) => ({
    ...action,
    text: `${action.lead ? `${action.lead} ` : ""}${prefix} ${action.suffix}`,
    category: "date",
  }));
  const normalizedQuery = normalizedSuggestionText(query);
  return candidates.filter((candidate) => normalizedSuggestionText(candidate.text).startsWith(normalizedQuery));
}

function scoreSuggestionCandidate(candidate, query, index) {
  const normalizedQuery = normalizedSuggestionText(query);
  const searchable = normalizedSuggestionText(`${candidate.text} ${candidate.label}`);
  const terms = suggestionTerms(query);
  const usage = state.suggestionUsage.get(candidate.operation);
  const ageHours = usage ? (Date.now() - Number(usage.lastUsed || 0)) / 3_600_000 : Infinity;
  let score = usage ? Math.min(Number(usage.count || 0), 8) : 0;
  if (ageHours <= 24) score += 6;
  else if (ageHours <= 24 * 7) score += 3;
  if (!normalizedQuery) {
    score += candidate.path === "fast" ? 20 : 10;
    if (candidate.featured) score += 1000;
  } else if (searchable.startsWith(normalizedQuery)) score += 100;
  else if (searchable.includes(normalizedQuery)) score += 70;
  else {
    const matches = terms.filter((term) => searchable.includes(term)).length;
    if (!matches && candidate.category !== "person") return null;
    score += matches === terms.length && terms.length ? 50 : 15 * matches;
  }
  if (candidate.category === "person") score += 35;
  if (candidate.category === "date") score += 25;
  if (candidate.scope === "multi") score += 20;
  if (candidate.path === "fast") score += 5;
  return { candidate, score, index };
}

function rankUnifiedSuggestions(query) {
  const candidates = [
    ...rankPersonCompletions(query),
    ...rankDateCompletions(query),
    ...state.suggestionCatalog.map((suggestion) => ({ ...suggestion, category: "general" })),
  ];
  const seen = new Set();
  const ranked = candidates
    .map((candidate, index) => scoreSuggestionCandidate(candidate, query, index))
    .filter(Boolean)
    .filter(({ candidate }) => {
      const key = normalizedSuggestionText(candidate.text);
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    })
    .sort((left, right) => right.score - left.score || left.index - right.index);
  const selected = [];
  const categories = new Set();
  for (const item of ranked) {
    if (categories.has(item.candidate.category)) continue;
    selected.push(item.candidate);
    categories.add(item.candidate.category);
    if (selected.length === 3) return selected;
  }
  for (const item of ranked) {
    if (selected.includes(item.candidate)) continue;
    selected.push(item.candidate);
    if (selected.length === 3) break;
  }
  return selected;
}

function showLocalQuestionSuggestions() {
  if (elements.naturalMode.checked || state.busy) {
    hideQuestionSuggestions();
    return;
  }
  const query = elements.questionInput.value.trim();
  const dateState = dateExpressionState(query);
  const rangeHint = dateState.rangePending
    ? "종료 날짜를 입력하면 기간 목록·합계·인원 질문을 추천합니다."
    : "";
  renderQuestionSuggestions(rankUnifiedSuggestions(query), query, rangeHint);
}

function bindSuggestions() {
  elements.chatArea.querySelectorAll(".suggestion").forEach((button) => {
    button.addEventListener("click", () => {
      const label = button.querySelector(".suggestion-icon + span");
      sendQuestion(label?.textContent || button.textContent);
    });
  });
}

function setUploadPanelOpen(open) {
  elements.uploadForm.hidden = !open;
  elements.uploadToggle.classList.toggle("active", open);
  elements.uploadToggle.setAttribute("aria-expanded", String(open));
  elements.uploadToggle.setAttribute("aria-label", open ? "문서 업로드 닫기" : "문서 업로드 열기");
}

function setModeHelpOpen(open) {
  elements.modeHelpPopover.hidden = !open;
  elements.modeHelpButton.setAttribute("aria-expanded", String(open));
}

function updateNaturalMode() {
  const active = elements.naturalMode.checked;
  hideQuestionSuggestions();
  elements.queryModeRow.classList.toggle("active", active);
  elements.questionInput.placeholder = active
    ? "선택한 문서의 의미와 문맥으로 검색하세요."
    : (document.documentElement.classList.contains("ui-v3")
      ? "질문을 입력하세요..."
      : "문서에 대해 질문하세요.");
}

function setNaturalMode(active) {
  if (active && !elements.naturalMode.checked) {
    const confirmed = window.confirm(
      "주의: AI 문서 검색은 의미가 비슷한 내용을 바탕으로 답변하므로 중요한 내용을 누락하거나 틀린 답변을 만들 수 있습니다.\n\n"
      + "금액·인원·통계 계산이나 원본 확인이 필요한 질문에는 사용하지 마세요. 그래도 AI 문서 검색을 켤까요?",
    );
    if (!confirmed) return false;
  }
  elements.naturalMode.checked = active;
  updateNaturalMode();
  return true;
}

elements.chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  if (state.busy) {
    stopChat();
    return;
  }
  sendQuestion(elements.questionInput.value);
});
elements.questionInput.addEventListener("input", () => {
  if (
    state.pendingVectorSeed
    && !normalizedSuggestionText(elements.questionInput.value).includes(
      normalizedSuggestionText(state.pendingVectorSeed)
    )
  ) {
    state.pendingVectorSeed = "";
  }
  resizeTextarea();
  scheduleRemotePersonSearch(elements.questionInput.value);
  scheduleQuestionSuggestions();
});
elements.questionInput.addEventListener("focus", async () => {
  syncMobileKeyboardInset();
  await primeQuestionCatalog();
  showLocalQuestionSuggestions();
});
elements.questionInput.addEventListener("blur", syncMobileKeyboardInset);
elements.questionInput.addEventListener("keydown", (event) => {
  const suggestionsOpen = !elements.questionAutocomplete.hidden;
  if (suggestionsOpen && event.key === "ArrowDown") {
    event.preventDefault();
    setSuggestionIndex(state.suggestionIndex + 1);
    return;
  }
  if (suggestionsOpen && event.key === "ArrowUp") {
    event.preventDefault();
    setSuggestionIndex(state.suggestionIndex - 1);
    return;
  }
  if (suggestionsOpen && event.key === "Escape") {
    event.preventDefault();
    hideQuestionSuggestions();
    return;
  }
  if (suggestionsOpen && event.key === "Enter" && !event.shiftKey && state.suggestionIndex >= 0) {
    event.preventDefault();
    const active = elements.questionAutocomplete.querySelector(".autocomplete-option.active");
    if (active) chooseQuestionSuggestion(active.dataset.text, active.dataset.operation);
    return;
  }
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    if (state.busy) return;
    elements.chatForm.requestSubmit();
  }
});
elements.documentSearch.addEventListener("input", renderDocuments);
elements.refreshDocuments.addEventListener("click", loadDocuments);
elements.uploadToggle.addEventListener("click", () => {
  setUploadPanelOpen(elements.uploadForm.hidden);
});
document.addEventListener("pointerdown", (event) => {
  if (!elements.questionAutocomplete.contains(event.target) && event.target !== elements.questionInput) {
    hideQuestionSuggestions();
  }
  if (!elements.renameModal.hidden && event.target === elements.renameModal) {
    closeRenameModal();
  }
  if (!elements.deleteModal.hidden && event.target === elements.deleteModal) {
    closeDeleteModal();
  }
  if (
    !elements.uploadForm.hidden
    && !elements.uploadForm.contains(event.target)
    && !elements.uploadToggle.contains(event.target)
  ) {
    setUploadPanelOpen(false);
  }
  if (!elements.modeHelpPopover.hidden && !elements.modeHelpWrap.contains(event.target)) {
    setModeHelpOpen(false);
  }
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    if (!elements.deleteModal.hidden) {
      closeDeleteModal();
      return;
    }
    if (!elements.renameModal.hidden) {
      closeRenameModal();
      return;
    }
    if (!elements.uploadForm.hidden) {
      setUploadPanelOpen(false);
      elements.uploadToggle.focus();
    }
    if (!elements.modeHelpPopover.hidden) {
      setModeHelpOpen(false);
      elements.modeHelpButton.focus();
    }
  }
});
elements.naturalMode.addEventListener("change", () => {
  if (elements.naturalMode.checked) {
    elements.naturalMode.checked = false;
    setNaturalMode(true);
    return;
  }
  updateNaturalMode();
});
elements.modeHelpButton.addEventListener("click", () => {
  setModeHelpOpen(elements.modeHelpPopover.hidden);
});
elements.renameForm.addEventListener("submit", submitRenameDocument);
elements.renameCancel.addEventListener("click", closeRenameModal);
elements.deleteCancel.addEventListener("click", closeDeleteModal);
elements.deleteSubmit.addEventListener("click", submitDeleteDocument);
elements.closeDetail.addEventListener("click", () => elements.detailDialog.close());
elements.detailDialog.addEventListener("click", (event) => {
  if (event.target === elements.detailDialog) elements.detailDialog.close();
});
elements.uploadFile.addEventListener("change", () => selectUploadFile(elements.uploadFile.files[0]));
elements.uploadForm.addEventListener("submit", uploadDocument);
elements.filePicker.addEventListener("dragover", (event) => {
  event.preventDefault();
  elements.filePicker.classList.add("dragging");
});
elements.filePicker.addEventListener("dragleave", () => elements.filePicker.classList.remove("dragging"));
elements.filePicker.addEventListener("drop", (event) => {
  event.preventDefault();
  elements.filePicker.classList.remove("dragging");
  const file = event.dataTransfer.files[0];
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  elements.uploadFile.files = transfer.files;
  selectUploadFile(file);
});
elements.allDocuments.addEventListener("click", () => {
  state.selected.clear();
  updateScope();
});
elements.clearChat.addEventListener("click", () => {
  if (state.busy) return;
  elements.chatArea.innerHTML = initialChat;
  bindSuggestions();
});
elements.openSidebar.addEventListener("click", () => elements.sidebar.classList.add("open"));
elements.closeSidebar.addEventListener("click", () => elements.sidebar.classList.remove("open"));

function beginSidebarSwipe(event) {
  if (!isMobileChatUi() || event.touches.length !== 1) return;
  if (event.target.closest("input, textarea, button, a, dialog, .question-autocomplete")) return;
  const touch = event.touches[0];
  state.sidebarSwipe = {
    x: touch.clientX,
    y: touch.clientY,
    sidebarOpen: elements.sidebar.classList.contains("open"),
  };
}

function finishSidebarSwipe(event) {
  const swipe = state.sidebarSwipe;
  state.sidebarSwipe = null;
  if (!swipe || !isMobileChatUi() || !event.changedTouches.length) return;
  const touch = event.changedTouches[0];
  const deltaX = touch.clientX - swipe.x;
  const deltaY = touch.clientY - swipe.y;
  if (Math.abs(deltaX) < 72 || Math.abs(deltaX) < Math.abs(deltaY) * 1.35) return;
  event.preventDefault();
  if (!swipe.sidebarOpen && deltaX > 0) {
    elements.sidebar.classList.add("open");
  } else if (swipe.sidebarOpen && deltaX < 0) {
    elements.sidebar.classList.remove("open");
  }
}

elements.mainPanel.addEventListener("touchstart", beginSidebarSwipe, { passive: true });
elements.mainPanel.addEventListener("touchend", finishSidebarSwipe, { passive: false });
elements.mainPanel.addEventListener("touchcancel", () => { state.sidebarSwipe = null; }, { passive: true });
elements.sidebar.addEventListener("touchstart", beginSidebarSwipe, { passive: true });
elements.sidebar.addEventListener("touchend", finishSidebarSwipe, { passive: false });
elements.sidebar.addEventListener("touchcancel", () => { state.sidebarSwipe = null; }, { passive: true });

document.addEventListener("click", (event) => {
  if (window.innerWidth > 820 && !document.documentElement.classList.contains("ui-v3")) return;
  if (!elements.sidebar.classList.contains("open")) return;
  if (event.target.closest("#sidebar, #openSidebar")) return;
  elements.sidebar.classList.remove("open");
});
elements.quickAttach.addEventListener("click", () => {
  elements.sidebar.classList.add("open");
  setUploadPanelOpen(true);
});
bindSuggestions();
loadDocuments();
updateNaturalMode();
const composerResizeObserver = new ResizeObserver(syncMobileComposerInset);
composerResizeObserver.observe(document.querySelector(".composer-wrap"));
window.addEventListener("resize", () => {
  syncMobileComposerInset();
  syncMobileKeyboardInset();
});
window.visualViewport?.addEventListener("resize", syncMobileKeyboardInset);
window.visualViewport?.addEventListener("scroll", syncMobileKeyboardInset);
syncMobileComposerInset();
syncMobileKeyboardInset();
if (window.matchMedia("(min-width: 821px) and (pointer: fine)").matches) {
  elements.questionInput.focus();
}
