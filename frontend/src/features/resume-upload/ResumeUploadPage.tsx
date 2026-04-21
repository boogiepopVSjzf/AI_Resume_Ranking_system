import { ChangeEvent, DragEvent, useMemo, useRef, useState } from "react";
import { uploadResumeBatch } from "./resumeUploadApi";
import type { BatchParseResponse } from "./resumeUploadTypes";

const ACCEPTED_EXTENSIONS = [".pdf", ".docx"];

function fileKey(file: File) {
  return `${file.name}-${file.size}-${file.lastModified}`;
}

function isSupportedResume(file: File) {
  const lowerName = file.name.toLowerCase();
  return ACCEPTED_EXTENSIONS.some((extension) => lowerName.endsWith(extension));
}

function formatFileSize(size: number) {
  if (size < 1024 * 1024) return `${Math.max(1, Math.round(size / 1024))} KB`;
  return `${(size / 1024 / 1024).toFixed(1)} MB`;
}

export function ResumeUploadPage() {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [result, setResult] = useState<BatchParseResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const totalSize = useMemo(() => {
    return files.reduce((sum, file) => sum + file.size, 0);
  }, [files]);

  const addFiles = (incoming: FileList | File[]) => {
    const supported = Array.from(incoming).filter(isSupportedResume);
    setResult(null);
    setError(null);
    setFiles((current) => {
      const existing = new Set(current.map(fileKey));
      const next = [...current];
      for (const file of supported) {
        if (!existing.has(fileKey(file))) next.push(file);
      }
      return next;
    });
    if (supported.length === 0 && Array.from(incoming).length > 0) {
      setError("Only PDF and DOCX resumes are supported.");
    }
  };

  const onFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) addFiles(event.target.files);
    event.target.value = "";
  };

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    addFiles(event.dataTransfer.files);
  };

  const removeFile = (target: File) => {
    setFiles((current) => current.filter((file) => fileKey(file) !== fileKey(target)));
  };

  const clearAll = () => {
    setFiles([]);
    setResult(null);
    setError(null);
  };

  const submit = async () => {
    if (files.length === 0) {
      setError("Please add at least one resume before uploading.");
      return;
    }
    setIsUploading(true);
    setError(null);
    setResult(null);
    try {
      const response = await uploadResumeBatch(files);
      setResult(response);
      if (response.failed_count === 0) setFiles([]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Resume upload failed.");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <section className="resume-upload page-grid">
      <div className="upload-hero">
        <p className="eyebrow">Resume Upload</p>
        <h2>Upload resumes.</h2>
      </div>

      <div className="upload-layout">
        <div className="upload-panel">
          <input
            ref={inputRef}
            className="hidden-file-input"
            type="file"
            multiple
            accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            onChange={onFileChange}
          />

          <div
            className={`drop-zone ${isDragging ? "dragging" : ""}`}
            onDragOver={(event) => {
              event.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={onDrop}
            role="button"
            tabIndex={0}
            onClick={() => inputRef.current?.click()}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") inputRef.current?.click();
            }}
          >
            <div className="drop-orbit">
              <span />
              <span />
              <strong>+</strong>
            </div>
            <div>
              <h3>Batch upload</h3>
              <p>PDF or DOCX</p>
            </div>
          </div>

          <div className="upload-actions">
            <button type="button" className="secondary-action" onClick={clearAll}>
              Clear
            </button>
            <button
              type="button"
              className="primary-action"
              onClick={submit}
              disabled={isUploading || files.length === 0}
            >
              {isUploading ? "Processing..." : `Upload ${files.length || ""} resumes`}
            </button>
          </div>

          {error && <div className="error-panel">{error}</div>}
        </div>

        <aside className="upload-summary-panel">
          <div>
            <p className="eyebrow">Current Batch</p>
            <h3>{files.length} files ready</h3>
            <p className="muted">{files.length > 0 ? formatFileSize(totalSize) : "No files yet."}</p>
          </div>

          {files.length > 0 ? (
            <div className="file-stack">
              {files.map((file) => (
                <article className="file-row" key={fileKey(file)}>
                  <div>
                    <strong>{file.name}</strong>
                    <span>{formatFileSize(file.size)}</span>
                  </div>
                  <button type="button" onClick={() => removeFile(file)}>
                    Remove
                  </button>
                </article>
              ))}
            </div>
          ) : (
            <div className="empty-result">
              <span />
              <p>Selected files appear here.</p>
            </div>
          )}
        </aside>
      </div>

      {result && (
        <div className="upload-result-panel">
          <div className="result-stat success">
            <span>Stored</span>
            <strong>{result.succeeded_count}</strong>
          </div>
          <div className="result-stat">
            <span>Processed</span>
            <strong>{result.total}</strong>
          </div>
          <div className={`result-stat ${result.failed_count > 0 ? "danger" : ""}`}>
            <span>Need attention</span>
            <strong>{result.failed_count}</strong>
          </div>

          <div className="result-list">
            {result.succeeded.map((item) => (
              <div className="result-item stored" key={item.resume_id ?? item.filename}>
                <span>{item.filename}</span>
                <strong>Stored</strong>
              </div>
            ))}
            {result.failed.map((item) => (
              <div className="result-item failed" key={item.filename}>
                <span>{item.filename}</span>
                <strong>{item.reason}</strong>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
