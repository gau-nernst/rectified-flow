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

let modelDefaults = {};
let shelfImages = [];
let inputStack = [];
let latestOutputBlob = null;

async function fetchModels() {
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
}

function applyDefaults(modelName) {
  const defaults = modelDefaults[modelName] || {};
  if (defaults.num_steps !== undefined) {
    numStepsInput.value = defaults.num_steps;
  }
  if (defaults.cfg_scale !== undefined) {
    cfgScaleInput.value = defaults.cfg_scale;
  }
}

async function fetchShelf() {
  const res = await fetch("/image/");
  const data = await res.json();
  shelfImages = data.items || [];
  renderShelf();
  inputStack = inputStack.filter((filename) => shelfImages.some((item) => item.filename === filename));
  renderInputStack();
}

function renderShelf() {
  shelf.innerHTML = "";
  shelfImages.forEach((item) => {
    const card = document.createElement("div");
    card.className = "shelf-card";
    card.addEventListener("dblclick", () => {
      toggleInputItem(item.filename);
    });

    const img = document.createElement("img");
    img.src = `/image/${item.filename}`;
    img.alt = item.filename;

    const actions = document.createElement("div");
    actions.className = "shelf-actions";

    const label = document.createElement("span");
    label.textContent = item.filename;

    const del = document.createElement("button");
    del.textContent = "Delete";
    del.addEventListener("click", async (event) => {
      event.stopPropagation();
      await fetch(`/image/${item.filename}`, { method: "DELETE" });
      inputStack = inputStack.filter((filename) => filename !== item.filename);
      await fetchShelf();
    });

    actions.appendChild(label);
    actions.appendChild(del);

    card.appendChild(img);
    card.appendChild(actions);
    shelf.appendChild(card);
  });
}

function renderInputStack() {
  inputStackEl.innerHTML = "";
  inputStack.forEach((filename, index) => {
    const item = shelfImages.find((entry) => entry.filename === filename);
    if (!item) return;

    const row = document.createElement("div");
    row.className = "stack-item";

    const img = document.createElement("img");
    img.src = `/image/${item.filename}`;
    img.alt = item.filename;

    const label = document.createElement("span");
    label.textContent = item.filename;

    const removeBtn = document.createElement("button");
    removeBtn.textContent = "Remove";
    removeBtn.addEventListener("click", () => {
      inputStack = inputStack.filter((_, idx) => idx !== index);
      renderInputStack();
    });

    row.appendChild(img);
    row.appendChild(label);
    row.appendChild(removeBtn);
    inputStackEl.appendChild(row);
  });
}

function toggleInputItem(filename) {
  if (!filename) return;
  if (inputStack.includes(filename)) {
    inputStack = inputStack.filter((entry) => entry !== filename);
  } else {
    inputStack.push(filename);
  }
  renderInputStack();
}

async function handleGenerate() {
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
    image_input_filenames: inputStack,
  };

  const res = await fetch("/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
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
}

async function saveOutputToShelf() {
  if (!latestOutputBlob) return;

  const form = new FormData();
  const suffix = globalThis.crypto?.randomUUID
    ? globalThis.crypto.randomUUID()
    : `${Date.now()}_${Math.random().toString(16).slice(2)}`;
  form.append("file", latestOutputBlob, `flux_${suffix}.webp`);
  const res = await fetch("/image", { method: "POST", body: form });
  if (res.ok) {
    await fetchShelf();
  }
}

async function loadFromUrl() {
  if (!urlInput.value) return;
  const form = new FormData();
  form.append("url", urlInput.value);
  const res = await fetch("/image", { method: "POST", body: form });
  if (res.ok) {
    urlInput.value = "";
    await fetchShelf();
  }
}

async function loadModel() {
  loadModelBtn.disabled = true;
  modelStatus.textContent = "Loading model...";
  const res = await fetch("/model/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: modelSelect.value }),
  });
  if (res.ok) {
    modelStatus.textContent = `Loaded: ${modelSelect.value}`;
  } else {
    const text = await res.text();
    modelStatus.textContent = `Load failed: ${text}`;
  }
  loadModelBtn.disabled = false;
}

modelSelect.addEventListener("change", () => applyDefaults(modelSelect.value));
loadUrlBtn.addEventListener("click", loadFromUrl);
loadModelBtn.addEventListener("click", loadModel);
importBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", async () => {
  const files = Array.from(fileInput.files || []);
  if (!files.length) return;
  await Promise.all(
    files.map(async (file) => {
      const form = new FormData();
      form.append("file", file);
      await fetch("/image", { method: "POST", body: form });
    })
  );
  fileInput.value = "";
  await fetchShelf();
});

saveOutputBtn.addEventListener("click", saveOutputToShelf);
generateBtn.addEventListener("click", handleGenerate);

fetchModels().then(fetchShelf);

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
