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
const saveOutputBtn = document.getElementById("save_output");
const shelf = document.getElementById("image_shelf");
const inputStackEl = document.getElementById("input_stack");
const importBtn = document.getElementById("import_btn");
const fileInput = document.getElementById("file_input");
const urlInput = document.getElementById("image_url");
const loadUrlBtn = document.getElementById("load_url");
const importServerBtn = document.getElementById("import_server");

let modelDefaults = {};
let shelfImages = [];
let inputStack = [];
let latestOutputBlob = null;
let shelfIdCounter = 0;
let fluxCounter = 0;
let urlCounter = 0;

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

// Pull shelf entries from server and add missing files.
async function importFromServer() {
  const res = await fetch("/image");
  if (!res.ok) return;
  const data = await res.json();
  const existing = new Set(shelfImages.map((item) => item.filename));
  for (const item of data || []) {
    const filename = item.filename || item;
    if (!filename || existing.has(filename)) continue;
    const imgRes = await fetch(`/image/${encodeURIComponent(filename)}`);
    if (!imgRes.ok) continue;
    const blob = await imgRes.blob();
    await addShelfItem(blob, filename, filename.startsWith("flux_"), true);
  }
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

    const meta = document.createElement("div");
    meta.className = "shelf-meta";

    const name = document.createElement("span");
    name.className = "name";
    name.textContent = item.filename;

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

// Shelf actions (add/delete) via event delegation.
shelf.addEventListener("click", async (event) => {
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
  const res = await fetch(urlInput.value);
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
  fluxCounter += 1;
  const filename = `flux_${String(fluxCounter).padStart(4, "0")}.webp`;
  await addShelfItem(latestOutputBlob, filename, true);
});

// Run generation using current settings and selected inputs.
generateBtn.addEventListener("click", async () => {
  generateBtn.disabled = true;
  outputMeta.textContent = "Generating...";
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
    generateBtn.disabled = false;
    return;
  }

  const blob = await res.blob();
  latestOutputBlob = blob;
  outputImage.src = URL.createObjectURL(blob);
  outputMeta.textContent = `Generated ${payload.width}x${payload.height}`;
  saveOutputBtn.disabled = false;

  generateBtn.disabled = false;
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
