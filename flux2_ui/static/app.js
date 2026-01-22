const modelSelect = document.getElementById("model");
const loadModelBtn = document.getElementById("load_model");
const modelStatus = document.getElementById("model_status");
const vramStatus = document.getElementById("vram_status");
const numStepsInput = document.getElementById("num_steps");
const cfgScaleInput = document.getElementById("cfg_scale");
const promptInput = document.getElementById("prompt");
const negPromptInput = document.getElementById("neg_prompt");
const widthInput = document.getElementById("width");
const heightInput = document.getElementById("height");
const seedInput = document.getElementById("seed");
const generateBtn = document.getElementById("generate");
const outputImage = document.getElementById("output_image");
const outputMeta = document.getElementById("output_meta");
const progressWrap = document.getElementById("progress_wrap");
const progressFill = document.getElementById("progress_fill");
const progressText = document.getElementById("progress_text");
const saveOutputBtn = document.getElementById("save_output");
const shelf = document.getElementById("image_shelf");
const inputStackEl = document.getElementById("input_stack");
const importBtn = document.getElementById("import_btn");
const fileInput = document.getElementById("file_input");
const urlInput = document.getElementById("image_url");
const loadUrlBtn = document.getElementById("load_url");
const importServerBtn = document.getElementById("import_server");
const preview = document.getElementById("image_preview");
const previewImage = document.getElementById("preview_image");
const previewPrev = document.getElementById("preview_prev");
const previewNext = document.getElementById("preview_next");
const previewLabel = document.getElementById("preview_label");

let modelDefaults = {};
let shelfImages = [];
let inputStack = [];
let latestOutputBlob = null;
let shelfIdCounter = 0;
let fluxCounter = 0;
let urlCounter = 0;
let previewItems = [];
let previewIndex = -1;
let activeRenameId = null;

function base64ToBlob(base64, mimeType) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Blob([bytes], { type: mimeType });
}

function showProgress(label) {
  progressWrap.classList.remove("hidden");
  progressFill.style.width = "0%";
  progressText.textContent = label || "Preparing...";
}

function updateProgress(step, total) {
  const safeTotal = total || 1;
  const percent = Math.max(0, Math.min(100, Math.round((step / safeTotal) * 100)));
  progressWrap.classList.remove("hidden");
  progressFill.style.width = `${percent}%`;
  progressText.textContent = `Step ${step} of ${total} (${percent}%)`;
}

function hideProgress() {
  progressWrap.classList.add("hidden");
  progressFill.style.width = "0%";
  progressText.textContent = "";
}

// Populate model selector and apply defaults once.
function applyDefaults(modelName) {
  const defaults = modelDefaults[modelName] || {};
  if (defaults.num_steps !== undefined) {
    numStepsInput.value = defaults.num_steps;
  }
  if (defaults.cfg_scale !== undefined) {
    cfgScaleInput.value = defaults.cfg_scale;
  }
}

async function addShelfItem(blob, filename, isFlux, saved = false) {
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

  shelfImages.unshift({
    id: `item_${++shelfIdCounter}`,
    filename,
    blob,
    url,
    width,
    height,
    isFlux,
    saved,
    error: "",
  });

  renderShelf();
}

function getUniqueFilename(prefix, ext) {
  const existing = new Set(shelfImages.map((item) => item.filename));
  let counter = fluxCounter;
  let candidate = "";
  do {
    counter += 1;
    candidate = `${prefix}${String(counter).padStart(4, "0")}.${ext}`;
  } while (existing.has(candidate));
  fluxCounter = counter;
  return candidate;
}

// Pull shelf entries from server and add missing files.
async function importFromServer() {
  const res = await fetch("/image");
  if (!res.ok) return;
  const data = await res.json();
  const existing = new Set(shelfImages.map((item) => item.filename));
  const serverSet = new Set((data || []).map((item) => item.filename || item));
  shelfImages.forEach((item) => {
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
    await addShelfItem(blob, filename, filename.startsWith("flux_"), true);
  }
  renderShelf();
}

// Render shelf as a list with actions.
function renderShelf() {
  shelf.innerHTML = "";
  shelfImages.forEach((item) => {
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

    actions.appendChild(addBtn);
    actions.appendChild(saveBtn);
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
    shelf.appendChild(row);
  });
}

function startRename(nameEl) {
  const row = nameEl.closest(".shelf-item");
  if (!row) return;
  const itemId = row.dataset.id;
  if (!itemId) return;
  const item = shelfImages.find((entry) => entry.id === itemId);
  if (!item) return;
  if (activeRenameId && activeRenameId !== itemId) {
    renderShelf();
  }
  activeRenameId = itemId;
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
    activeRenameId = null;

    const originalName = nameEl.dataset.originalName || item.filename;
    delete nameEl.dataset.originalName;
    let nextName = commit ? nameEl.textContent.trim() : originalName;
    if (!nextName) {
      nextName = originalName;
    }
    if (nextName !== item.filename) {
      const duplicate = shelfImages.some(
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
    renderShelf();
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

// Render selected input images as a list.
function renderInputStack() {
  inputStackEl.innerHTML = "";
  inputStack.forEach((itemId, index) => {
    const item = shelfImages.find((entry) => entry.id === itemId);
    if (!item) return;
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
    inputStackEl.appendChild(row);
  });
}

modelSelect.addEventListener("change", () => applyDefaults(modelSelect.value));

// Preview large image on click.
function openPreview(itemId) {
  previewItems = getPreviewItems();
  previewIndex = previewItems.findIndex((entry) => entry.id === itemId);
  if (previewIndex === -1) return;
  renderPreview();
}

function closePreview() {
  previewImage.src = "";
  previewLabel.textContent = "";
  preview.classList.add("hidden");
  previewItems = [];
  previewIndex = -1;
}
preview.addEventListener("click", (event) => {
  if (event.target === preview) {
    closePreview();
  }
});

function getPreviewItems() {
  const items = [];
  inputStack.forEach((itemId) => {
    const item = shelfImages.find((entry) => entry.id === itemId);
    if (!item) return;
    items.push({ id: item.id, src: item.url, label: item.filename });
  });
  if (outputImage.src) {
    items.push({ id: "output", src: outputImage.src, label: "Output" });
  }
  shelfImages.forEach((item) => {
    items.push({ id: item.id, src: item.url, label: item.filename });
  });
  return items;
}

function renderPreview() {
  if (previewIndex < 0 || previewIndex >= previewItems.length) return;
  const current = previewItems[previewIndex];
  previewImage.src = current.src;
  previewLabel.textContent = `${current.label || "Image"} (${previewIndex + 1}/${previewItems.length})`;
  preview.classList.remove("hidden");
}

function stepPreview(delta) {
  if (previewItems.length === 0) return;
  previewIndex = (previewIndex + delta + previewItems.length) % previewItems.length;
  renderPreview();
}

previewPrev.addEventListener("click", (event) => {
  event.stopPropagation();
  stepPreview(-1);
});

previewNext.addEventListener("click", (event) => {
  event.stopPropagation();
  stepPreview(1);
});

outputImage.addEventListener("click", () => {
  if (!outputImage.src) return;
  openPreview("output");
});

document.addEventListener("keydown", (event) => {
  if (preview.classList.contains("hidden")) return;
  if (event.key === "ArrowLeft") {
    stepPreview(-1);
  } else if (event.key === "ArrowRight") {
    stepPreview(1);
  } else if (event.key === "Escape") {
    closePreview();
  }
});

// Shelf actions (add/delete) via event delegation.
shelf.addEventListener("click", async (event) => {
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
    inputStack.push(itemId);
    renderInputStack();
    return;
  }
  const saveBtn = event.target.closest('button[data-action="save"]');
  if (saveBtn) {
    const row = event.target.closest(".shelf-item");
    if (!row) return;
    const itemId = row.dataset.id;
    const item = shelfImages.find((entry) => entry.id === itemId);
    if (!item || item.saved) return;
    const form = new FormData();
    form.append("file", item.blob, item.filename);
    const res = await fetch("/image", { method: "POST", body: form });
    if (res.ok) {
      item.saved = true;
      item.error = "";
      renderShelf();
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
    renderShelf();
    return;
  }
  const renameTarget = event.target.closest('[data-action="rename"]');
  if (renameTarget) {
    startRename(renameTarget);
    return;
  }
  const deleteBtn = event.target.closest('button[data-action="delete"]');
  if (!deleteBtn) return;
  const row = event.target.closest(".shelf-item");
  if (!row) return;
  const itemId = row.dataset.id;
  const item = shelfImages.find((entry) => entry.id === itemId);
  if (item) {
    URL.revokeObjectURL(item.url);
  }
  shelfImages = shelfImages.filter((entry) => entry.id !== itemId);
  inputStack = inputStack.filter((entry) => entry !== itemId);
  renderShelf();
  renderInputStack();
});

// Input list actions via event delegation.
inputStackEl.addEventListener("click", (event) => {
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
  inputStack = inputStack.filter((_, idx) => idx !== index);
  renderInputStack();
});
loadUrlBtn.addEventListener("click", async () => {
  if (!urlInput.value) return;
  const res = await fetch(`/proxy?url=${encodeURIComponent(urlInput.value)}`);
  if (!res.ok) return;
  const blob = await res.blob();
  const type = blob.type.split("/")[1] || "webp";
  const ext = type === "jpeg" ? "jpg" : type;
  let name = "";
  try {
    const parsed = new URL(urlInput.value);
    name = parsed.pathname.split("/").filter(Boolean).pop() || "";
  } catch {
    name = "";
  }
  if (!name) {
    urlCounter += 1;
    name = `url_${String(urlCounter).padStart(4, "0")}.${ext}`;
  }
  if (!name.includes(".")) {
    name = `${name}.${ext}`;
  }
  await addShelfItem(blob, name, false);
  urlInput.value = "";
});
importServerBtn.addEventListener("click", async () => {
  await importFromServer();
});
loadModelBtn.addEventListener("click", async () => {
  loadModelBtn.disabled = true;
  modelStatus.textContent = "Loading model...";
  const res = await fetch(`/model/${encodeURIComponent(modelSelect.value)}`, {
    method: "POST",
  });
  if (res.ok) {
    modelStatus.textContent = `Loaded: ${modelSelect.value}`;
  } else {
    const text = await res.text();
    modelStatus.textContent = `Load failed: ${text}`;
  }
  loadModelBtn.disabled = false;
});
importBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", async () => {
  const files = Array.from(fileInput.files || []);
  if (!files.length) return;
  for (const file of files) {
    await addShelfItem(file, file.name || "upload", false);
  }
  fileInput.value = "";
});

// Save output back into the shelf.
saveOutputBtn.addEventListener("click", async () => {
  if (!latestOutputBlob) return;
  const filename = getUniqueFilename("flux_", "webp");
  await addShelfItem(latestOutputBlob, filename, true);
});

// Run generation using current settings and selected inputs.
generateBtn.addEventListener("click", async () => {
  generateBtn.disabled = true;
  saveOutputBtn.disabled = true;
  outputMeta.textContent = "Generating...";
  showProgress("Starting...");
  const payload = {
    prompt: promptInput.value || "",
    neg_prompt: negPromptInput.value || "",
    width: Number(widthInput.value) || 512,
    height: Number(heightInput.value) || 512,
    num_steps: Number(numStepsInput.value) || 4,
    cfg_scale: Number(cfgScaleInput.value) || 1.0,
    seed: seedInput.value ? Number(seedInput.value) : null,
  };

  const form = new FormData();
  form.append("prompt", payload.prompt);
  form.append("neg_prompt", payload.neg_prompt);
  form.append("width", String(payload.width));
  form.append("height", String(payload.height));
  form.append("num_steps", String(payload.num_steps));
  form.append("cfg_scale", String(payload.cfg_scale));
  if (payload.seed !== null) {
    form.append("seed", String(payload.seed));
  }
  inputStack.forEach((itemId) => {
    const item = shelfImages.find((entry) => entry.id === itemId);
    if (item) {
      form.append("images", item.blob, item.filename);
    }
  });

  const res = await fetch("/generate", { method: "POST", body: form });
  if (!res.ok) {
    const text = await res.text();
    outputMeta.textContent = `Error: ${text}`;
    hideProgress();
    generateBtn.disabled = false;
    return;
  }

  if (!res.body) {
    outputMeta.textContent = "Error: empty response";
    hideProgress();
    generateBtn.disabled = false;
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
        latestOutputBlob = blob;
        if (outputImage.src.startsWith("blob:")) {
          URL.revokeObjectURL(outputImage.src);
        }
        outputImage.src = URL.createObjectURL(blob);
        outputMeta.textContent = `Generated ${msg.width}x${msg.height}`;
        hideProgress();
        saveOutputBtn.disabled = false;
        generateBtn.disabled = false;
        finished = true;
        break;
      } else if (msg.type === "error") {
        outputMeta.textContent = `Error: ${msg.message || "generation failed"}`;
        hideProgress();
        generateBtn.disabled = false;
        finished = true;
        break;
      }
    }
    if (finished) break;
  }

  if (!finished) {
    outputMeta.textContent = "Error: stream ended unexpectedly";
    hideProgress();
    generateBtn.disabled = false;
  }
});

// Initial load (models + shelf).
(async () => {
  const res = await fetch("/models");
  const data = await res.json();
  modelDefaults = data.defaults || {};
  modelSelect.innerHTML = "";
  data.models.forEach((name) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    modelSelect.appendChild(opt);
  });
  if (data.active_model) {
    modelSelect.value = data.active_model;
    modelStatus.textContent = `Loaded: ${data.active_model}`;
  } else {
    modelStatus.textContent = "No model loaded";
  }
  applyDefaults(modelSelect.value);
  await importFromServer();
})();

// VRAM status polling.
async function refreshVram() {
  try {
    const res = await fetch("/vram");
    if (!res.ok) {
      vramStatus.textContent = "VRAM: unavailable";
      return;
    }
    const data = await res.json();
    const toGiB = (bytes) => (bytes / (1024 ** 3)).toFixed(2);
    vramStatus.textContent = `VRAM used: ${toGiB(data.used_bytes)} / ${toGiB(data.total_bytes)} GiB`;
  } catch {
    vramStatus.textContent = "VRAM: unavailable";
  }
}

refreshVram();
setInterval(refreshVram, 4000);
