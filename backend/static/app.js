"use strict";

const elements = {
  sidebar: document.getElementById("sidebar"),
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
  quickAttach: document.getElementById("quickAttach"),
  quickModeToggle: document.getElementById("quickModeToggle"),
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
  contactNames: new Set(),
  contactNamesPromise: null,
  renameSource: "",
  deleteSource: "",
};

const initialChat = elements.chatArea.innerHTML;

function apiHeaders(json = false) {
  const headers = {};
  if (json) headers["Content-Type"] = "application/json";
  return headers;
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
    row.append(button, renameButton, deleteButton);
    elements.documentList.append(row);
  });
}

function toggleDocument(source) {
  if (state.selected.has(source)) state.selected.delete(source);
  else state.selected.add(source);
  updateScope();
}

function updateScope() {
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
}

async function loadDocuments() {
  elements.documentList.innerHTML = '<div class="document-loading">문서 목록을 불러오는 중입니다.</div>';
  try {
    const response = await fetch("/documents", { headers: apiHeaders() });
    if (!response.ok) throw new Error(await errorMessage(response));
    const data = await response.json();
    state.documents = Array.isArray(data.files) ? data.files : [];
    const available = new Set(state.documents.map((item) => item.source));
    state.selected.forEach((source) => { if (!available.has(source)) state.selected.delete(source); });
    updateScope();
  } catch (error) {
    state.documents = [];
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

async function loadContactNames() {
  if (state.contactNamesPromise) return state.contactNamesPromise;
  state.contactNamesPromise = fetch("/contacts/names", { headers: apiHeaders() })
    .then(async (response) => {
      if (!response.ok) throw new Error();
      const data = await response.json();
      state.contactNames = new Set((data.names || []).map((name) => String(name).trim()).filter(Boolean));
    })
    .catch(() => {
      state.contactNames = new Set();
    });
  return state.contactNamesPromise;
}

function linkContactNames(body) {
  if (!state.contactNames.size) return;
  const names = [...state.contactNames].sort((left, right) => right.length - left.length);
  const escaped = names.map((name) => name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const matcher = new RegExp(`(${escaped.join("|")})`, "g");
  const walker = document.createTreeWalker(body, NodeFilter.SHOW_TEXT);
  const textNodes = [];
  let node;
  while ((node = walker.nextNode())) {
    if (node.parentElement?.closest(".contact-name, button")) continue;
    if (matcher.test(node.textContent)) textNodes.push(node);
    matcher.lastIndex = 0;
  }
  textNodes.forEach((textNode) => {
    const text = textNode.textContent;
    const fragment = document.createDocumentFragment();
    let cursor = 0;
    text.replace(matcher, (matched, _group, offset) => {
      fragment.append(document.createTextNode(text.slice(cursor, offset)));
      const nameWrap = document.createElement("span");
      nameWrap.className = "contact-name-wrap";
      const name = document.createElement("button");
      name.type = "button";
      name.className = "contact-name";
      name.textContent = matched;
      name.dataset.contactName = matched;
      name.title = "연락처 보기";
      nameWrap.append(name);
      fragment.append(nameWrap);
      cursor = offset + matched.length;
      return matched;
    });
    fragment.append(document.createTextNode(text.slice(cursor)));
    textNode.replaceWith(fragment);
  });
}

async function toggleContactCard(button) {
  const existing = button.nextElementSibling;
  if (existing?.classList.contains("contact-card")) {
    existing.remove();
    return;
  }
  document.querySelectorAll(".contact-card").forEach((card) => card.remove());
  const card = document.createElement("span");
  card.className = "contact-card loading";
  card.textContent = "연락처 확인 중";
  button.after(card);
  try {
    const response = await fetch(`/contacts/${encodeURIComponent(button.dataset.contactName)}`, { headers: apiHeaders() });
    if (!response.ok) throw new Error(await errorMessage(response));
    const contact = await response.json();
    const details = [
      ...(contact.departments || []).map((department) => `학과 ${department}`),
      ...(contact.phones || []).map((phone) => `전화 ${phone}`),
      ...(contact.emails || []).map((email) => `이메일 ${email}`),
    ];
    card.classList.remove("loading");
    card.textContent = details.length ? details.join("\n") : "등록된 연락처가 없습니다.";
  } catch (_) {
    card.remove();
    showToast("연락처 정보를 불러오지 못했습니다.");
  }
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

async function pollIngestStatus(filename, attempts = 90) {
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    await new Promise((resolve) => window.setTimeout(resolve, 2000));
    try {
      const response = await fetch(`/status?source=${encodeURIComponent(filename)}`, { headers: apiHeaders() });
      if (response.status === 404) continue;
      if (!response.ok) throw new Error(await errorMessage(response));
      const status = await response.json();
      if (status.status === "SUCCESS") {
        elements.uploadProgressText.textContent = "적재 완료";
        await loadDocuments();
        showToast(`'${filename}' 적재가 완료됐습니다.`);
        return;
      }
      if (status.status === "FAILED") {
        throw new Error(status.error_message || "문서 적재에 실패했습니다.");
      }
      elements.uploadProgressText.textContent = "문서 분석 및 적재 중";
    } catch (error) {
      elements.uploadProgressText.textContent = "상태 확인 실패";
      showToast(error.message);
      return;
    }
  }
  elements.uploadProgressText.textContent = "적재 상태는 문서 목록에서 확인하세요";
  loadDocuments();
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

function renderInlineSegments(body, segments, fullAnswer = "") {
  if (!Array.isArray(segments) || !segments.length) {
    collapseEvidence(body);
    linkContactNames(body);
    return;
  }
  body.replaceChildren();
  segments.forEach((segment) => {
    if (!segment.detail_ref) {
      body.append(document.createTextNode(segment.text || ""));
      return;
    }
    const button = document.createElement("button");
    button.type = "button";
    button.className = `inline-detail-link ${segment.kind || "detail"}`;
    button.textContent = segment.text || "상세 보기";
    button.title = segment.kind === "entity" ? "인물 정보와 납부 기록 보기" : "금액 계산 근거 보기";
    button.addEventListener("click", () => openDetail(segment.detail_ref));
    body.append(button);
  });
  // Interactive segments omit calculation evidence, so restore the original
  // text before turning the evidence into the shared collapsible section.
  const originalEvidence = evidenceMatch(fullAnswer);
  if (originalEvidence && !evidenceMatch(body.textContent)) {
    body.append(document.createTextNode(`\n\n${fullAnswer.slice(originalEvidence.index)}`));
  }
  collapseEvidence(body);
  linkContactNames(body);
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
    : detail.kind === "entity_collection_detail" ? `${detail.display_name || "동명이인"} 선택` : "금액 계산 근거";

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
  elements.detailMore.hidden = !detail.page?.has_more;
  elements.detailMore.onclick = () => openDetail(detail._reference, detail.page.offset + detail.page.limit);
  if (!elements.detailDialog.open) elements.detailDialog.showModal();
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
async function sendQuestion(question, options = {}) {
  const value = question.trim();
  if (!value || state.busy) return;

  await loadContactNames();
  const controller = new AbortController();
  state.chatController = controller;
  setChatBusy(true);
  const request = {
    question: value,
    sources: options.sources ? [...options.sources] : [...state.selected],
    mode: options.mode || (elements.naturalMode.checked ? "natural" : "auto"),
  };
  elements.questionInput.value = "";
  resizeTextarea();
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
    );
    renderInlineSegments(message.querySelector(".message-body"), data.result?.inline_segments, data.answer || "");
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
    if (state.chatController === controller) state.chatController = null;
    setChatBusy(false);
    elements.questionInput.focus();
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
  state.chatController?.abort();
}

function resizeTextarea() {
  elements.questionInput.style.height = "auto";
  elements.questionInput.style.height = `${Math.min(elements.questionInput.scrollHeight, 150)}px`;
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
  elements.queryModeRow.classList.toggle("active", active);
  elements.questionInput.placeholder = active
    ? "질문의 의미와 문맥으로 검색하세요."
    : (document.documentElement.classList.contains("ui-v3")
      ? "질문을 입력하세요..."
      : "문서에 대해 질문하세요.");
}

elements.chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  if (state.busy) {
    stopChat();
    return;
  }
  sendQuestion(elements.questionInput.value);
});
elements.questionInput.addEventListener("input", resizeTextarea);
elements.questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.chatForm.requestSubmit();
  }
});
elements.documentSearch.addEventListener("input", renderDocuments);
elements.refreshDocuments.addEventListener("click", loadDocuments);
elements.uploadToggle.addEventListener("click", () => {
  setUploadPanelOpen(elements.uploadForm.hidden);
});
document.addEventListener("pointerdown", (event) => {
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
elements.naturalMode.addEventListener("change", updateNaturalMode);
elements.renameForm.addEventListener("submit", submitRenameDocument);
elements.renameCancel.addEventListener("click", closeRenameModal);
elements.deleteCancel.addEventListener("click", closeDeleteModal);
elements.deleteSubmit.addEventListener("click", submitDeleteDocument);
elements.closeDetail.addEventListener("click", () => elements.detailDialog.close());
elements.detailDialog.addEventListener("click", (event) => {
  if (event.target === elements.detailDialog) elements.detailDialog.close();
});
elements.modeHelpButton.addEventListener("click", () => {
  setModeHelpOpen(elements.modeHelpPopover.hidden);
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
elements.chatArea.addEventListener("click", (event) => {
  const contactName = event.target.closest(".contact-name");
  if (contactName) toggleContactCard(contactName);
});
elements.openSidebar.addEventListener("click", () => elements.sidebar.classList.add("open"));
elements.closeSidebar.addEventListener("click", () => elements.sidebar.classList.remove("open"));
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
elements.quickModeToggle.addEventListener("click", () => {
  elements.naturalMode.checked = !elements.naturalMode.checked;
  updateNaturalMode();
});

bindSuggestions();
loadDocuments();
updateNaturalMode();
elements.questionInput.focus();
