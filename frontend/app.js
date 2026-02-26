const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const statusEl = document.getElementById("status");
const outputEl = document.getElementById("output");

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
    outputEl.textContent = result.text || "未提取到文本";
  } catch (error) {
    setStatus("网络错误或服务未启动", true);
  }
});
