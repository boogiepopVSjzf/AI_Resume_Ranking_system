const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const statusEl = document.getElementById("status");
const outputEl = document.getElementById("output");
const structuredEl = document.getElementById("structured");

let lastResumeId = null;
let lastText = "";

const setStatus = (text, isError = false) => {
  statusEl.textContent = text;
  statusEl.className = isError ? "status error" : "status";
};

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = fileInput.files[0];
  if (!file) {
    setStatus("请先选择 PDF 文件", true);
    return;
  }

  setStatus("上传中...");
  outputEl.textContent = "";
  structuredEl.textContent = "";
  lastResumeId = null;
  lastText = "";

  const data = new FormData();
  data.append("file", file);

  try {
    const response = await fetch("/api/upload", {
      method: "POST",
      body: data,
    });

    const result = await response.json();

    if (!response.ok) {
      setStatus(result.detail || "解析失败", true);
      return;
    }

    setStatus(`解析成功，ID: ${result.resume_id}`);
    lastResumeId = result.resume_id;
    lastText = result.text || "";
    outputEl.textContent = lastText || "未提取到文本";

    setStatus("抽取结构化 JSON 中...");
    const extractResp = await fetch("/api/extract", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ resume_id: lastResumeId, text: lastText }),
    });
    const extractResult = await extractResp.json();
    if (!extractResp.ok) {
      setStatus(extractResult.detail || "结构化抽取失败", true);
      return;
    }
    structuredEl.textContent = JSON.stringify(extractResult, null, 2);
    setStatus("解析与结构化完成");
  } catch (error) {
    setStatus("网络错误或服务未启动", true);
  }
});
