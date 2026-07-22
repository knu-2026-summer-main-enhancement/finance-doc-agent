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
    if (node.parentElement?.closest(".contact-name")) continue;
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
      badge.textContent = route === "natural" ? "자연어 검색" : route.toUpperCase();
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
  let streamedMessage;
  let streamedBody;
  const startStreamedMessage = () => {
    if (streamedMessage) return;
    loading.remove();
    streamedMessage = appendMessage(
      "assistant",
      "",
      request.mode === "natural" ? "natural" : "",
      [],
      request,
      true,
    );
    streamedBody = streamedMessage.querySelector(".message-body");
  };

  try {
    const response = await fetch("/chat/stream", {
      method: "POST",
      headers: apiHeaders(true),
      body: JSON.stringify(request),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(await errorMessage(response));
    if (!response.body) throw new Error("응답 스트림을 시작하지 못했습니다.");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let pendingText = "";
    while (true) {
      const { done, value: chunk } = await reader.read();
      if (done) break;
      pendingText += decoder.decode(chunk, { stream: true });
      if (pendingText.trim()) {
        startStreamedMessage();
        streamedBody.textContent += pendingText;
        pendingText = "";
        streamedMessage.querySelector(".message-actions").hidden = false;
      }
      elements.chatArea.scrollTop = elements.chatArea.scrollHeight;
    }
    pendingText += decoder.decode();
    if (!streamedMessage) {
      startStreamedMessage();
      streamedBody.textContent = pendingText.trim() || "답변이 비어 있습니다.";
    } else {
      streamedBody.textContent += pendingText;
    }
    if (!streamedBody.textContent.trim()) {
      streamedBody.textContent = "답변이 비어 있습니다.";
    }
    collapseCalculationEvidence(streamedBody);
    linkContactNames(streamedBody);
    streamedMessage.querySelector(".message-actions").hidden = false;
  } catch (error) {
    loading.remove();
    if (error.name === "AbortError") {
      if (streamedBody?.textContent.trim()) {
        streamedBody.textContent += "\n\n[답변 생성을 중단했습니다.]";
      } else {
        streamedMessage?.remove();
        appendMessage("assistant", "답변 생성을 중단했습니다.", "error", [], request);
      }
    } else {
      streamedMessage?.remove();
      appendMessage("assistant", error.message || "답변 처리 중 오류가 발생했습니다.", "error", [], request);
    }
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

function collapseCalculationEvidence(body) {
  const text = body.textContent;
  const marker = "계산 근거:";
  const markerIndex = text.indexOf(marker);
  if (markerIndex <= 0) return;

  const answer = text.slice(0, markerIndex).trimEnd();
  const evidence = text.slice(markerIndex + marker.length).trim();
  if (!evidence) return;

  body.replaceChildren();
  const answerText = document.createElement("div");
  answerText.textContent = answer;
  body.append(answerText);

  const details = document.createElement("details");
  details.className = "calculation-evidence";
  const summary = document.createElement("summary");
  summary.textContent = "계산 근거";
  const evidenceText = document.createElement("div");
  evidenceText.className = "calculation-evidence-body";
  evidenceText.textContent = evidence;
  details.append(summary, evidenceText);
  body.append(details);
}

function setChatBusy(busy) {
  state.busy = busy;
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
    : "문서에 대해 질문하세요.";
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
  elements.chatArea.innerHTML = initialChat;
  bindSuggestions();
});
elements.chatArea.addEventListener("click", (event) => {
  const contactName = event.target.closest(".contact-name");
  if (contactName) toggleContactCard(contactName);
});
elements.openSidebar.addEventListener("click", () => elements.sidebar.classList.add("open"));
elements.closeSidebar.addEventListener("click", () => elements.sidebar.classList.remove("open"));

bindSuggestions();
loadDocuments();
updateNaturalMode();
elements.questionInput.focus();
