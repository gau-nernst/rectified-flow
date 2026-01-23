const ui = {
  modelSelect: document.getElementById("model"),
  loadModelBtn: document.getElementById("load_model"),
  modelStatus: document.getElementById("model_status"),
  vramStatus: document.getElementById("vram_status"),
  numStepsInput: document.getElementById("num_steps"),
  cfgScaleInput: document.getElementById("cfg_scale"),
  promptInput: document.getElementById("prompt"),
  negPromptInput: document.getElementById("neg_prompt"),
  widthInput: document.getElementById("width"),
  heightInput: document.getElementById("height"),
  seedInput: document.getElementById("seed"),
  generateBtn: document.getElementById("generate"),
  outputImage: document.getElementById("output_image"),
  outputMeta: document.getElementById("output_meta"),
  progressWrap: document.getElementById("progress_wrap"),
  progressFill: document.getElementById("progress_fill"),
  progressText: document.getElementById("progress_text"),
  copyOutputBtn: document.getElementById("copy_output"),
  saveOutputBtn: document.getElementById("save_output"),
  shelf: document.getElementById("image_shelf"),
  inputStackEl: document.getElementById("input_stack"),
  importBtn: document.getElementById("import_btn"),
  fileInput: document.getElementById("file_input"),
  status: document.getElementById("status"),
  urlInput: document.getElementById("image_url"),
  loadUrlBtn: document.getElementById("load_url"),
  importServerBtn: document.getElementById("import_server"),
  preview: document.getElementById("image_preview"),
  previewImage: document.getElementById("preview_image"),
  previewPrev: document.getElementById("preview_prev"),
  previewNext: document.getElementById("preview_next"),
  previewLabel: document.getElementById("preview_label"),
};

const state = {
  modelDefaults: {},
  shelfImages: [],
  inputStack: [],
  latestOutputBlob: null,
  shelfIdCounter: 0,
  previewItems: [],
  previewIndex: -1,
  activeRenameId: null,
  nameCounters: {},
};

function base64ToBlob(base64, mimeType) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Blob([bytes], { type: mimeType });
}

function showProgress(label) {
  ui.progressWrap.classList.remove("hidden");
  ui.progressFill.style.width = "0%";
  ui.progressText.textContent = label || "Preparing...";
}

function updateProgress(step, total) {
  const safeTotal = total || 1;
  const percent = Math.max(0, Math.min(100, Math.round((step / safeTotal) * 100)));
  ui.progressWrap.classList.remove("hidden");
  ui.progressFill.style.width = `${percent}%`;
  ui.progressText.textContent = `Step ${step} of ${total} (${percent}%)`;
}

function hideProgress() {
  ui.progressWrap.classList.add("hidden");
  ui.progressFill.style.width = "0%";
  ui.progressText.textContent = "";
}

function setStatus(message, isError = false) {
  ui.status.textContent = message;
  ui.status.classList.toggle("error", isError);
  ui.status.classList.remove("visible", "pop");
  void ui.status.offsetWidth;
  ui.status.classList.toggle("visible", Boolean(message));
  ui.status.classList.add("pop");
  if (!message) return;
  clearTimeout(setStatus._hideTimer);
  setStatus._hideTimer = setTimeout(() => {
    ui.status.classList.remove("visible");
  }, 3200);
}

function getUniqueFilename(prefix, ext) {
  const existing = new Set(state.shelfImages.map((item) => item.filename));
  let counter = state.nameCounters[prefix] || 0;
  let candidate = "";
  do {
    counter += 1;
    candidate = `${prefix}${String(counter).padStart(4, "0")}.${ext}`;
  } while (existing.has(candidate));
  state.nameCounters[prefix] = counter;
  return candidate;
}

async function copyBlobToClipboard(blob) {
  const type = blob.type || "image/png";
  try {
    const item = new ClipboardItem({ [type]: blob });
    await navigator.clipboard.write([item]);
    return;
  } catch (err) {
    const pngBlob = await convertBlobToPng(blob);
    const item = new ClipboardItem({ "image/png": pngBlob });
    await navigator.clipboard.write([item]);
  }
}

async function convertBlobToPng(blob) {
  if (blob.type === "image/png") return blob;
  const bitmap = await createImageBitmap(blob);
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, 0, 0);
  return canvas.convertToBlob({ type: "image/png" });
}

// Populate model selector and apply defaults once.
function applyDefaults(modelName) {
  const defaults = state.modelDefaults[modelName] || {};
  if (defaults.num_steps !== undefined) {
    ui.numStepsInput.value = defaults.num_steps;
  }
  if (defaults.cfg_scale !== undefined) {
    ui.cfgScaleInput.value = defaults.cfg_scale;
  }
}

async function addShelfItem(blob, filename, saved = false) {
  const url = URL.createObjectURL(blob);
  let width = 0;
  let height = 0;

  await new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      width = img.naturalWidth;
      height = img.naturalHeight;
      resolve();
    };
    img.onerror = () => resolve();
    img.src = url;
  });

  state.shelfImages.unshift({
    id: `item_${++state.shelfIdCounter}`,
    filename,
    blob,
    url,
    width,
    height,
    saved,
    error: "",
  });

  const row = buildShelfRow(state.shelfImages[0]);
  ui.shelf.prepend(row);
}

// Pull shelf entries from server and add missing files.
async function importFromServer() {
  let data;
  try {
    const res = await fetch("/image");
    if (!res.ok) {
      setStatus("Import from server failed", true);
      return;
    }
    data = await res.json();
  } catch (err) {
    setStatus(err?.message || "Import from server failed", true);
    return;
  }
  const existing = new Set(state.shelfImages.map((item) => item.filename));
  const serverSet = new Set((data || []).map((item) => item.filename || item));
  state.shelfImages.forEach((item) => {
    if (item.saved && !serverSet.has(item.filename)) {
      item.saved = false;
    }
  });
  for (const item of data || []) {
    const filename = item.filename || item;
    if (!filename || existing.has(filename)) continue;
    const imgRes = await fetch(`/image/${encodeURIComponent(filename)}`);
    if (!imgRes.ok) continue;
    const blob = await imgRes.blob();
    await addShelfItem(blob, filename, true);
  }
}

function buildShelfRow(item) {
  const row = document.createElement("div");
  row.className = "shelf-item";
  row.dataset.id = item.id;

  const img = document.createElement("img");
  img.src = item.url;
  img.alt = item.filename;
  img.dataset.id = item.id;

  const meta = document.createElement("div");
  meta.className = "shelf-meta";

  const name = document.createElement("span");
  name.className = "name";
  name.textContent = item.filename;
  name.dataset.action = "rename";

  const size = document.createElement("span");
  size.className = "size";
  size.textContent = `${item.width}x${item.height}`;

  meta.appendChild(name);
  meta.appendChild(size);

  const actions = document.createElement("div");
  actions.className = "shelf-actions";

  const addBtn = document.createElement("button");
  addBtn.type = "button";
  addBtn.dataset.action = "add";
  addBtn.textContent = "Add";

  const saveBtn = document.createElement("button");
  saveBtn.type = "button";
  saveBtn.dataset.action = "save";
  saveBtn.textContent = item.saved ? "Saved" : "Save";
  saveBtn.disabled = item.saved;

  const deleteBtn = document.createElement("button");
  deleteBtn.type = "button";
  deleteBtn.dataset.action = "delete";
  deleteBtn.textContent = "Remove";

  const copyBtn = document.createElement("button");
  copyBtn.type = "button";
  copyBtn.dataset.action = "copy";
  copyBtn.textContent = "Copy";

  actions.appendChild(addBtn);
  actions.appendChild(saveBtn);
  actions.appendChild(copyBtn);
  actions.appendChild(deleteBtn);

  row.appendChild(img);
  row.appendChild(meta);
  if (item.error) {
    const error = document.createElement("span");
    error.className = "shelf-error";
    error.textContent = item.error;
    meta.appendChild(error);
  }
  row.appendChild(actions);
  return row;
}

// Render shelf as a list with actions.
function renderShelf() {
  ui.shelf.innerHTML = "";
  state.shelfImages.forEach((item) => {
    ui.shelf.appendChild(buildShelfRow(item));
  });
}

function setShelfRowError(row, message) {
  const meta = row.querySelector(".shelf-meta");
  if (!meta) return;
  let errorEl = meta.querySelector(".shelf-error");
  if (!message) {
    if (errorEl) {
      errorEl.remove();
    }
    return;
  }
  if (!errorEl) {
    errorEl = document.createElement("span");
    errorEl.className = "shelf-error";
    meta.appendChild(errorEl);
  }
  errorEl.textContent = message;
}

function startRename(nameEl) {
  const row = nameEl.closest(".shelf-item");
  if (!row) return;
  const itemId = row.dataset.id;
  if (!itemId) return;
  const item = state.shelfImages.find((entry) => entry.id === itemId);
  if (!item) return;
  if (state.activeRenameId && state.activeRenameId !== itemId) {
    const existing = ui.shelf.querySelector(
      `.shelf-item[data-id="${state.activeRenameId}"] [data-action="rename"]`,
    );
    if (existing) {
      existing.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    }
  }
  state.activeRenameId = itemId;
  nameEl.contentEditable = "true";
  nameEl.spellcheck = false;
  nameEl.dataset.originalName = item.filename;
  nameEl.focus();
  const selection = window.getSelection();
  if (selection) {
    const range = document.createRange();
    range.selectNodeContents(nameEl);
    selection.removeAllRanges();
    selection.addRange(range);
  }

  const finish = (commit) => {
    nameEl.removeEventListener("blur", onBlur);
    nameEl.removeEventListener("keydown", onKeydown);
    nameEl.contentEditable = "false";
    nameEl.spellcheck = true;
    state.activeRenameId = null;

    const originalName = nameEl.dataset.originalName || item.filename;
    delete nameEl.dataset.originalName;
    let nextName = commit ? nameEl.textContent.trim() : originalName;
    if (!nextName) {
      nextName = originalName;
    }
    if (nextName !== item.filename) {
      const duplicate = state.shelfImages.some(
        (entry) => entry.filename === nextName && entry.id !== item.id,
      );
      if (duplicate) {
        item.error = "Name already exists";
        nextName = item.filename;
      } else {
        item.filename = nextName;
        item.error = "";
      }
    }
    nameEl.textContent = nextName;
    setShelfRowError(row, item.error);
    renderInputStack();
  };

  const onBlur = () => finish(true);
  const onKeydown = (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      finish(true);
    } else if (event.key === "Escape") {
      event.preventDefault();
      finish(false);
    }
  };

  nameEl.addEventListener("blur", onBlur);
  nameEl.addEventListener("keydown", onKeydown);
}

function buildInputRow(item, index) {
  const row = document.createElement("div");
  row.className = "stack-item";
  row.dataset.index = index.toString();

  const img = document.createElement("img");
  img.src = item.url;
  img.alt = item.filename;
  img.dataset.id = item.id;

  const label = document.createElement("span");
  label.textContent = item.filename;

  const removeBtn = document.createElement("button");
  removeBtn.type = "button";
  removeBtn.dataset.action = "remove";
  removeBtn.textContent = "Remove";

  row.appendChild(img);
  row.appendChild(label);
  row.appendChild(removeBtn);
  return row;
}

// Render selected input images as a list.
function renderInputStack() {
  ui.inputStackEl.innerHTML = "";
  state.inputStack.forEach((itemId, index) => {
    const item = state.shelfImages.find((entry) => entry.id === itemId);
    if (!item) return;
    ui.inputStackEl.appendChild(buildInputRow(item, index));
  });
}

ui.modelSelect.addEventListener("change", () => applyDefaults(ui.modelSelect.value));

// Preview large image on click.
function openPreview(itemId) {
  state.previewItems = getPreviewItems();
  state.previewIndex = state.previewItems.findIndex((entry) => entry.id === itemId);
  if (state.previewIndex === -1) return;
  renderPreview();
}

function closePreview() {
  ui.previewImage.src = "";
  ui.previewLabel.textContent = "";
  ui.preview.classList.add("hidden");
  state.previewItems = [];
  state.previewIndex = -1;
}
ui.preview.addEventListener("click", (event) => {
  if (event.target === ui.preview) {
    closePreview();
  }
});

function getPreviewItems() {
  const items = [];
  state.inputStack.forEach((itemId) => {
    const item = state.shelfImages.find((entry) => entry.id === itemId);
    if (!item) return;
    items.push({ id: item.id, src: item.url, label: item.filename });
  });
  if (ui.outputImage.src) {
    items.push({ id: "output", src: ui.outputImage.src, label: "Output" });
  }
  state.shelfImages.forEach((item) => {
    items.push({ id: item.id, src: item.url, label: item.filename });
  });
  return items;
}

function renderPreview() {
  if (state.previewIndex < 0 || state.previewIndex >= state.previewItems.length) return;
  const current = state.previewItems[state.previewIndex];
  ui.previewImage.src = current.src;
  ui.previewLabel.textContent = `${current.label || "Image"} (${state.previewIndex + 1}/${state.previewItems.length})`;
  ui.preview.classList.remove("hidden");
}

function stepPreview(delta) {
  if (state.previewItems.length === 0) return;
  state.previewIndex = (state.previewIndex + delta + state.previewItems.length) % state.previewItems.length;
  renderPreview();
}

ui.previewPrev.addEventListener("click", (event) => {
  event.stopPropagation();
  stepPreview(-1);
});

ui.previewNext.addEventListener("click", (event) => {
  event.stopPropagation();
  stepPreview(1);
});

ui.outputImage.addEventListener("click", () => {
  if (!ui.outputImage.src) return;
  openPreview("output");
});

document.addEventListener("keydown", (event) => {
  if (ui.preview.classList.contains("hidden")) return;
  if (event.key === "ArrowLeft") {
    stepPreview(-1);
  } else if (event.key === "ArrowRight") {
    stepPreview(1);
  } else if (event.key === "Escape") {
    closePreview();
  }
});

// Shelf actions (add/delete) via event delegation.
ui.shelf.addEventListener("click", async (event) => {
  const img = event.target.closest("img");
  if (img?.dataset.id) {
    openPreview(img.dataset.id);
    return;
  }
  const addBtn = event.target.closest('button[data-action="add"]');
  if (addBtn) {
    const row = event.target.closest(".shelf-item");
    if (!row) return;
    const itemId = row.dataset.id;
    if (!itemId) return;
    state.inputStack.push(itemId);
    const item = state.shelfImages.find((entry) => entry.id === itemId);
    if (!item) return;
    ui.inputStackEl.appendChild(buildInputRow(item, state.inputStack.length - 1));
    return;
  }
  const saveBtn = event.target.closest('button[data-action="save"]');
  if (saveBtn) {
    const row = event.target.closest(".shelf-item");
    if (!row) return;
    const itemId = row.dataset.id;
    const item = state.shelfImages.find((entry) => entry.id === itemId);
    if (!item || item.saved) return;
    const form = new FormData();
    form.append("file", item.blob, item.filename);
    const res = await fetch("/image", { method: "POST", body: form });
    if (res.ok) {
      item.saved = true;
      item.error = "";
      setStatus("");
      saveBtn.textContent = "Saved";
      saveBtn.disabled = true;
      setShelfRowError(row, "");
      return;
    }
    let errorText = "Save failed";
    try {
      const data = await res.json();
      if (data.detail) {
        errorText = data.detail;
      }
    } catch {
      const text = await res.text();
      if (text) {
        errorText = text;
      }
    }
    item.error = errorText;
    setStatus(`Save failed: ${errorText}`, true);
    setShelfRowError(row, errorText);
    return;
  }
  const renameTarget = event.target.closest('[data-action="rename"]');
  if (renameTarget) {
    startRename(renameTarget);
    return;
  }
  const copyBtn = event.target.closest('button[data-action="copy"]');
  if (copyBtn) {
    const row = event.target.closest(".shelf-item");
    if (!row) return;
    const itemId = row.dataset.id;
    const item = state.shelfImages.find((entry) => entry.id === itemId);
    if (!item) return;
    try {
      await copyBlobToClipboard(item.blob);
      setStatus("Copied to clipboard");
      if (item.error) {
        item.error = "";
        setShelfRowError(row, "");
      }
    } catch (err) {
      item.error = err?.message || "Copy failed";
      setStatus(item.error, true);
      setShelfRowError(row, item.error);
    }
    return;
  }
  const deleteBtn = event.target.closest('button[data-action="delete"]');
  if (!deleteBtn) return;
  const row = event.target.closest(".shelf-item");
  if (!row) return;
  const itemId = row.dataset.id;
  const item = state.shelfImages.find((entry) => entry.id === itemId);
  if (item) {
    URL.revokeObjectURL(item.url);
  }
  state.shelfImages = state.shelfImages.filter((entry) => entry.id !== itemId);
  state.inputStack = state.inputStack.filter((entry) => entry !== itemId);
  row.remove();
  renderInputStack();
});

// Input list actions via event delegation.
ui.inputStackEl.addEventListener("click", (event) => {
  const img = event.target.closest("img");
  if (img?.dataset.id) {
    openPreview(img.dataset.id);
    return;
  }
  const removeBtn = event.target.closest('button[data-action="remove"]');
  if (!removeBtn) return;
  const row = event.target.closest(".stack-item");
  if (!row) return;
  const index = Number(row.dataset.index);
  if (Number.isNaN(index)) return;
  state.inputStack = state.inputStack.filter((_, idx) => idx !== index);
  row.remove();
  Array.from(ui.inputStackEl.querySelectorAll(".stack-item")).forEach((el, idx) => {
    el.dataset.index = idx.toString();
  });
});
ui.loadUrlBtn.addEventListener("click", async () => {
  if (!ui.urlInput.value) return;
  const res = await fetch(`/proxy?url=${encodeURIComponent(ui.urlInput.value)}`);
  if (!res.ok) {
    setStatus("Load URL failed", true);
    return;
  }
  const blob = await res.blob();
  const type = blob.type.split("/")[1] || "webp";
  const ext = type === "jpeg" ? "jpg" : type;
  let name = "";
  try {
    const parsed = new URL(ui.urlInput.value);
    name = parsed.pathname.split("/").filter(Boolean).pop() || "";
  } catch {
    name = "";
  }
  if (!name) {
    name = getUniqueFilename("url_", ext);
  }
  if (!name.includes(".")) {
    name = `${name}.${ext}`;
  }
  await addShelfItem(blob, name);
  ui.urlInput.value = "";
});
ui.importServerBtn.addEventListener("click", async () => {
  await importFromServer();
});
ui.loadModelBtn.addEventListener("click", async () => {
  ui.loadModelBtn.disabled = true;
  ui.modelStatus.textContent = "Loading model...";
  const res = await fetch(`/model/${encodeURIComponent(ui.modelSelect.value)}`, {
    method: "POST",
  });
  if (res.ok) {
    ui.modelStatus.textContent = `Loaded: ${ui.modelSelect.value}`;
    setStatus("");
  } else {
    const text = await res.text();
    ui.modelStatus.textContent = `Load failed: ${text}`;
    setStatus(`Model load failed: ${text}`, true);
  }
  ui.loadModelBtn.disabled = false;
});
ui.importBtn.addEventListener("click", () => ui.fileInput.click());
ui.fileInput.addEventListener("change", async () => {
  const files = Array.from(ui.fileInput.files || []);
  if (!files.length) return;
  for (const file of files) {
    await addShelfItem(file, file.name || "upload");
  }
  ui.fileInput.value = "";
});

document.addEventListener("paste", async (event) => {
  const target = event.target;
  if (
    target &&
    (target.tagName === "INPUT" ||
      target.tagName === "TEXTAREA" ||
      target.isContentEditable)
  ) {
    return;
  }
  const items = event.clipboardData.items || [];
  for (const item of items) {
    if (!item.type.startsWith("image/")) continue;
    const file = item.getAsFile();
    if (!file) continue;
    const ext = file.type.split("/")[1] || "png";
    const filename = getUniqueFilename("paste_", ext);
    await addShelfItem(file, filename);
    event.preventDefault();
    return;
  }
});

// Save output back into the shelf.
ui.saveOutputBtn.addEventListener("click", async () => {
  if (!state.latestOutputBlob) return;
  const filename = getUniqueFilename("flux_", "webp");
  await addShelfItem(state.latestOutputBlob, filename);
});

// Run generation using current settings and selected inputs.
ui.generateBtn.addEventListener("click", async () => {
  ui.generateBtn.disabled = true;
  ui.saveOutputBtn.disabled = true;
  setStatus("");
  ui.outputMeta.textContent = "Generating...";
  showProgress("Starting...");
  const prompt = ui.promptInput.value || "";
  const negPrompt = ui.negPromptInput.value || "";
  const width = Number(ui.widthInput.value) || 512;
  const height = Number(ui.heightInput.value) || 512;
  const numSteps = Number(ui.numStepsInput.value) || 4;
  const cfgScale = Number(ui.cfgScaleInput.value) || 1.0;
  const seed = ui.seedInput.value ? Number(ui.seedInput.value) : null;

  const form = new FormData();
  form.append("prompt", prompt);
  form.append("neg_prompt", negPrompt);
  form.append("width", String(width));
  form.append("height", String(height));
  form.append("num_steps", String(numSteps));
  form.append("cfg_scale", String(cfgScale));
  if (seed !== null) {
    form.append("seed", String(seed));
  }
  state.inputStack.forEach((itemId) => {
    const item = state.shelfImages.find((entry) => entry.id === itemId);
    if (item) {
      form.append("images", item.blob, item.filename);
    }
  });

  const res = await fetch("/generate", { method: "POST", body: form });
  if (!res.ok) {
    const text = await res.text();
    ui.outputMeta.textContent = `Error: ${text}`;
    setStatus(`Generation failed: ${text}`, true);
    hideProgress();
    ui.generateBtn.disabled = false;
    return;
  }

  if (!res.body) {
    ui.outputMeta.textContent = "Error: empty response";
    setStatus("Generation failed: empty response", true);
    hideProgress();
    ui.generateBtn.disabled = false;
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finished = false;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    while (true) {
      const newlineIndex = buffer.indexOf("\n");
      if (newlineIndex === -1) break;
      const line = buffer.slice(0, newlineIndex).trim();
      buffer = buffer.slice(newlineIndex + 1);
      if (!line) continue;
      let msg;
      try {
        msg = JSON.parse(line);
      } catch {
        continue;
      }
      if (msg.type === "progress") {
        updateProgress(msg.step, msg.total);
      } else if (msg.type === "done") {
        const blob = base64ToBlob(msg.image_base64, "image/webp");
        state.latestOutputBlob = blob;
        if (ui.outputImage.src.startsWith("blob:")) {
          URL.revokeObjectURL(ui.outputImage.src);
        }
        ui.outputImage.src = URL.createObjectURL(blob);
        ui.outputMeta.textContent = `Generated ${msg.width}x${msg.height}`;
        setStatus("");
        hideProgress();
        ui.saveOutputBtn.disabled = false;
        ui.generateBtn.disabled = false;
        finished = true;
        break;
      } else if (msg.type === "error") {
        const message = msg.message || "generation failed";
        ui.outputMeta.textContent = `Error: ${message}`;
        setStatus(`Generation failed: ${message}`, true);
        hideProgress();
        ui.generateBtn.disabled = false;
        finished = true;
        break;
      }
    }
    if (finished) break;
  }

  if (!finished) {
    ui.outputMeta.textContent = "Error: stream ended unexpectedly";
    setStatus("Generation failed: stream ended unexpectedly", true);
    hideProgress();
    ui.generateBtn.disabled = false;
  }
});

ui.copyOutputBtn.addEventListener("click", async () => {
  if (!state.latestOutputBlob) {
    setStatus("No output image to copy", true);
    return;
  }
  try {
    await copyBlobToClipboard(state.latestOutputBlob);
  } catch (err) {
    setStatus(err?.message || "Copy failed", true);
  }
});

// Initial load (models + shelf).
(async () => {
  const res = await fetch("/models");
  const data = await res.json();
  state.modelDefaults = data.defaults || {};
  ui.modelSelect.innerHTML = "";
  data.models.forEach((name) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    ui.modelSelect.appendChild(opt);
  });
  if (data.active_model) {
    ui.modelSelect.value = data.active_model;
    ui.modelStatus.textContent = `Loaded: ${data.active_model}`;
  } else {
    ui.modelStatus.textContent = "No model loaded";
  }
  applyDefaults(ui.modelSelect.value);
  await importFromServer();
})();

// VRAM status polling.
async function refreshVram() {
  try {
    const res = await fetch("/vram");
    if (!res.ok) {
      ui.vramStatus.textContent = "VRAM: unavailable";
      return;
    }
    const data = await res.json();
    const toGiB = (bytes) => (bytes / (1024 ** 3)).toFixed(2);
    ui.vramStatus.textContent = `VRAM used: ${toGiB(data.used_bytes)} / ${toGiB(data.total_bytes)} GiB`;
  } catch {
    ui.vramStatus.textContent = "VRAM: unavailable";
  }
}

refreshVram();
setInterval(refreshVram, 4000);
