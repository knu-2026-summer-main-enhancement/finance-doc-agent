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
  apiKey: document.getElementById("apiKey"),
  saveApiKey: document.getElementById("saveApiKey"),
  statusDot: document.getElementById("statusDot"),
  serverStatus: document.getElementById("serverStatus"),
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
  sendButton: document.getElementById("sendButton"),
  toast: document.getElementById("toast"),
  detailDialog: document.getElementById("detailDialog"),
  detailTitle: document.getElementById("detailTitle"),
  detailBody: document.getElementById("detailBody"),
  closeDetail: document.getElementById("closeDetail"),
  detailMore: document.getElementById("detailMore"),
};

const state = {
  documents: [],
  selected: new Set(),
  busy: false,
  apiKey: sessionStorage.getItem("finance-doc-api-key") || "",
};

const initialChat = elements.chatArea.innerHTML;
elements.apiKey.value = state.apiKey;

function apiHeaders(json = false) {
  const headers = {};
  if (json) headers["Content-Type"] = "application/json";
  if (state.apiKey) headers["X-API-Key"] = state.apiKey;
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
    icon.textContent = fileType(item);
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
    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "delete-document";
    deleteButton.textContent = "×";
    deleteButton.title = `${source} 삭제`;
    deleteButton.setAttribute("aria-label", `${source} 삭제`);
    deleteButton.addEventListener("click", () => deleteDocument(source));
    row.append(button, deleteButton);
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

async function checkServer() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error();
    elements.statusDot.className = "status-dot ok";
    elements.serverStatus.textContent = "서버 연결됨";
  } catch (_) {
    elements.statusDot.className = "status-dot warn";
    elements.serverStatus.textContent = "서버 확인 필요";
  }
}

async function deleteDocument(source) {
  const confirmed = window.confirm(
    `'${source}' 문서를 삭제할까요?\n\n원본 파일, Parquet, ChromaDB 색인과 적재 기록이 함께 삭제됩니다.`
  );
  if (!confirmed) return;

  try {
    const response = await fetch(`/documents/${encodeURIComponent(source)}`, {
      method: "DELETE",
      headers: apiHeaders(),
    });
    if (!response.ok) throw new Error(await errorMessage(response));
    state.selected.delete(source);
    await loadDocuments();
    showToast(`'${source}' 문서를 삭제했습니다.`);
  } catch (error) {
    showToast(`삭제 실패: ${error.message}`);
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

function appendMessage(role, text, route = "", sources = [], result = null) {
  const message = document.createElement("article");
  message.className = `message ${role}`;

  if (role === "assistant") {
    const head = document.createElement("div");
    head.className = "message-head";
    head.textContent = "Finance Doc";
    if (route) {
      const badge = document.createElement("span");
      badge.className = `route-badge ${route.toLocaleLowerCase()}`;
      badge.textContent = route === "natural" ? "자연어 검색" : route.toUpperCase();
      head.append(badge);
    }
    message.append(head);
  }

  const body = document.createElement("div");
  body.className = "message-body";
  const inlineSegments = role === "assistant" ? result?.inline_segments : null;
  if (Array.isArray(inlineSegments) && inlineSegments.length) {
    inlineSegments.forEach((segment) => {
      if (!segment.kind) {
        body.append(document.createTextNode(segment.text));
        return;
      }
      const link = document.createElement("button");
      link.type = "button";
      link.className = `answer-link ${segment.kind}-link`;
      link.textContent = segment.text;
      link.addEventListener("click", () => openDetail(segment.detail_ref));
      body.append(link);
    });
  } else {
    body.textContent = text.split("\n\n계산 근거:", 1)[0];
  }
  message.append(body);

  elements.chatArea.append(message);
  elements.chatArea.scrollTop = elements.chatArea.scrollHeight;
  return message;
}

function renderDetail(detail) {
  elements.detailTitle.textContent = detail.kind === "entity_detail"
    ? `${detail.display_name || "사람"} 정보`
    : detail.kind === "entity_collection_detail"
      ? `${detail.display_name || "동명이인"} 정보 선택`
      : "계산 상세";
  elements.detailBody.replaceChildren();
  if (detail.kind === "entity_detail") {
    (detail.attributes || []).forEach((item) => {
      const row = document.createElement("div"); row.className = "detail-row";
      const label = document.createElement("strong"); label.textContent = item.column;
      const value = document.createElement("span"); value.textContent = item.missing ? "없음" : item.value;
      row.append(label, value); elements.detailBody.append(row);
    });
    if ((detail.payment_history || []).length) {
      const historyTitle = document.createElement("h3");
      historyTitle.className = "detail-section-title";
      historyTitle.textContent = `결제 이력 ${detail.payment_history.length}건`;
      elements.detailBody.append(historyTitle);
      (detail.payment_history || []).forEach((record, index) => {
        const card = document.createElement("section");
        card.className = "detail-record-card payment-history-card";
        const number = document.createElement("div");
        number.className = "detail-record-number";
        number.textContent = `${index + 1}`;
        const fields = document.createElement("div");
        fields.className = "detail-record-fields";
        (record.fields || []).forEach((historyField) => {
          const field = document.createElement("div");
          const label = document.createElement("span"); label.textContent = historyField.column;
          const value = document.createElement("strong");
          value.textContent = historyField.data_type === "money" && typeof historyField.value === "number"
            ? `${historyField.value.toLocaleString("ko-KR")}원`
            : historyField.value ?? "없음";
          field.append(label, value); fields.append(field);
        });
        card.append(number, fields);
        elements.detailBody.append(card);
      });
    }
  } else if (detail.kind === "entity_collection_detail") {
    const guide = document.createElement("p");
    guide.className = "contributor-caption";
    guide.textContent = `같은 이름의 정보가 ${detail.candidates?.length ?? 0}개 있습니다.`;
    elements.detailBody.append(guide);
    (detail.candidates || []).forEach((candidate, index) => {
      const card = document.createElement("button");
      card.type = "button";
      card.className = "entity-candidate-card";
      const title = document.createElement("strong"); title.textContent = `${detail.display_name} ${index + 1}`;
      const preview = document.createElement("span");
      preview.textContent = (candidate.attributes || []).slice(0, 3).map((item) => `${item.column}: ${item.value ?? "없음"}`).join(" · ");
      card.append(title, preview);
      card.addEventListener("click", () => openDetail(candidate.detail_ref));
      elements.detailBody.append(card);
    });
  } else {
    const summary = document.createElement("div");
    summary.className = "calculation-summary";
    const operationLabels = { sum: "합계", mean: "평균", median: "중앙값", min: "최솟값", max: "최댓값", mode: "최빈값", count: "개수", group_sum: "그룹 합계" };
    const summaryTitle = document.createElement("strong"); summaryTitle.textContent = operationLabels[detail.operation] || "계산";
    const summaryTarget = document.createElement("span"); summaryTarget.textContent = detail.target || "대상";
    const summaryValue = document.createElement("b"); summaryValue.textContent = detail.value ?? "-";
    const summaryCounts = document.createElement("small"); summaryCounts.textContent = `유효 ${detail.valid_rows ?? 0}건 · 제외 ${detail.excluded_rows ?? 0}건`;
    summary.append(summaryTitle, summaryTarget, summaryValue, summaryCounts);
    elements.detailBody.append(summary);
    const caption = document.createElement("p");
    caption.className = "contributor-caption";
    caption.textContent = `기여 기록 ${detail.page?.total ?? 0}건`;
    elements.detailBody.append(caption);
    (detail.contributors || []).forEach((record, index) => {
      const card = document.createElement("section");
      card.className = "detail-record-card";
      const number = document.createElement("div");
      number.className = "detail-record-number";
      number.textContent = `${(detail.page?.offset ?? 0) + index + 1}`;
      const fields = document.createElement("div");
      fields.className = "detail-record-fields";
      Object.entries(record).forEach(([key, rawValue]) => {
        const field = document.createElement("div");
        const label = document.createElement("span"); label.textContent = key;
        const value = document.createElement("strong"); value.textContent = rawValue ?? "없음";
        field.append(label, value); fields.append(field);
      });
      card.append(number, fields);
      elements.detailBody.append(card);
    });
  }
  const page = detail.page;
  elements.detailMore.hidden = !page?.has_more;
  elements.detailMore.onclick = () => openDetail(detail._reference, page.offset + page.limit);
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
    showToast(`상세 조회 실패: ${error.message}`);
  }
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

async function sendQuestion(question) {
  const value = question.trim();
  if (!value || state.busy) return;

  state.busy = true;
  const mode = elements.naturalMode.checked ? "natural" : "auto";
  elements.sendButton.disabled = true;
  elements.naturalMode.disabled = true;
  elements.questionInput.value = "";
  resizeTextarea();
  elements.chatArea.querySelector(".welcome-card")?.remove();
  appendMessage("user", value);
  const loading = appendLoading();

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: apiHeaders(true),
      body: JSON.stringify({ question: value, sources: [...state.selected], mode }),
    });
    if (!response.ok) throw new Error(await errorMessage(response));
    const data = await response.json();
    loading.remove();
    const responseMode = mode === "natural" ? "natural" : (data.source || "");
    appendMessage("assistant", data.answer || "답변이 비어 있습니다.", responseMode, data.sources || [], data.result || null);
  } catch (error) {
    loading.remove();
    appendMessage("assistant", error.message, "error");
  } finally {
    state.busy = false;
    elements.sendButton.disabled = false;
    elements.naturalMode.disabled = false;
    elements.questionInput.focus();
  }
}

function resizeTextarea() {
  elements.questionInput.style.height = "auto";
  elements.questionInput.style.height = `${Math.min(elements.questionInput.scrollHeight, 150)}px`;
}

function bindSuggestions() {
  elements.chatArea.querySelectorAll(".suggestion").forEach((button) => {
    button.addEventListener("click", () => sendQuestion(button.textContent));
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
    : "문서에 대해 질문하세요.";
}

elements.chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
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
elements.saveApiKey.addEventListener("click", () => {
  state.apiKey = elements.apiKey.value.trim();
  if (state.apiKey) sessionStorage.setItem("finance-doc-api-key", state.apiKey);
  else sessionStorage.removeItem("finance-doc-api-key");
  showToast(state.apiKey ? "API Key를 적용했습니다." : "API Key를 비웠습니다.");
  loadDocuments();
});
elements.clearChat.addEventListener("click", () => {
  elements.chatArea.innerHTML = initialChat;
  bindSuggestions();
});
elements.closeDetail.addEventListener("click", () => elements.detailDialog.close());
elements.detailDialog.addEventListener("click", (event) => { if (event.target === elements.detailDialog) elements.detailDialog.close(); });
elements.openSidebar.addEventListener("click", () => elements.sidebar.classList.add("open"));
elements.closeSidebar.addEventListener("click", () => elements.sidebar.classList.remove("open"));

bindSuggestions();
checkServer();
loadDocuments();
updateNaturalMode();
elements.questionInput.focus();
