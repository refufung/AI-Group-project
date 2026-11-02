import { useEffect, useMemo, useState, type ChangeEvent } from "react";
import "./App.css";
import {
  API_BASE,
  fetchHealth,
  predictBatch,
  predictOne,
  type BatchResponse,
  type Health,
  type PredictResult,
} from "./api";

type Mode = "single" | "batch";

const IMAGE_EXT = /\.(jpe?g|png|webp|bmp)$/i;

function prettyBreed(name: string) {
  return name.replaceAll("_", " ");
}

function ProbBars({ result }: { result: PredictResult }) {
  return (
    <div className="bars">
      {result.probabilities.map((p) => (
        <div className="bar-row" key={p.class_name}>
          <span>{prettyBreed(p.class_name)}</span>
          <div className="bar-track">
            <div
              className="bar-fill"
              style={{ width: `${Math.max(2, p.probability * 100)}%` }}
            />
          </div>
          <span>{(p.probability * 100).toFixed(0)}%</span>
        </div>
      ))}
    </div>
  );
}

export default function App() {
  const [mode, setMode] = useState<Mode>("single");
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);

  const [singleFile, setSingleFile] = useState<File | null>(null);
  const [singlePreview, setSinglePreview] = useState<string | null>(null);
  const [singleResult, setSingleResult] = useState<PredictResult | null>(null);

  const [batchFiles, setBatchFiles] = useState<File[]>([]);
  const [batchPaths, setBatchPaths] = useState<string[]>([]);
  const [batchResult, setBatchResult] = useState<BatchResponse | null>(null);
  const [thumbUrls, setThumbUrls] = useState<Record<string, string>>({});

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchHealth()
      .then((h) => {
        setHealth(h);
        setHealthError(null);
      })
      .catch((e: Error) => {
        setHealth(null);
        setHealthError(e.message);
      });
  }, []);

  useEffect(() => {
    if (!singleFile) {
      setSinglePreview(null);
      return;
    }
    const url = URL.createObjectURL(singleFile);
    setSinglePreview(url);
    return () => URL.revokeObjectURL(url);
  }, [singleFile]);

  useEffect(() => {
    const urls: Record<string, string> = {};
    for (const f of batchFiles.slice(0, 80)) {
      urls[f.name + (f as File & { webkitRelativePath?: string }).webkitRelativePath] =
        URL.createObjectURL(f);
    }
    setThumbUrls(urls);
    return () => {
      Object.values(urls).forEach((u) => URL.revokeObjectURL(u));
    };
  }, [batchFiles]);

  const batchCountLabel = useMemo(() => {
    if (!batchFiles.length) return "No images selected";
    return `${batchFiles.length} image${batchFiles.length === 1 ? "" : "s"} selected`;
  }, [batchFiles]);

  function onSingleChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0] ?? null;
    setSingleFile(file);
    setSingleResult(null);
    setError(null);
  }

  function collectBatch(fileList: FileList | null) {
    if (!fileList) return;
    const files: File[] = [];
    const paths: string[] = [];
    Array.from(fileList).forEach((file) => {
      if (!IMAGE_EXT.test(file.name)) return;
      const rel =
        (file as File & { webkitRelativePath?: string }).webkitRelativePath ||
        file.name;
      files.push(file);
      paths.push(rel);
    });
    setBatchFiles(files);
    setBatchPaths(paths);
    setBatchResult(null);
    setError(null);
  }

  async function runSingle() {
    if (!singleFile) return;
    setLoading(true);
    setError(null);
    try {
      const result = await predictOne(singleFile);
      setSingleResult(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  async function runBatch() {
    if (!batchFiles.length) return;
    setLoading(true);
    setError(null);
    try {
      const result = await predictBatch(batchFiles, batchPaths);
      setBatchResult(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <header className="hero">
        <h1 className="brand">Dog Breed EfficientNet</h1>
        <p className="lede">
          Upload one photo or a whole folder. The trained EfficientNet-B0
          classifies French Bulldog, Cardigan, Labrador, and Golden Retriever.
        </p>
        {healthError ? (
          <p className="status err">
            API offline at {API_BASE}. Start the backend first. ({healthError})
          </p>
        ) : health ? (
          <p className="status ok">
            Model ready on {health.device} · {health.class_names.join(" · ")}
          </p>
        ) : (
          <p className="status">Checking API…</p>
        )}
      </header>

      <div className="tabs">
        <button
          type="button"
          className={`tab ${mode === "single" ? "active" : ""}`}
          onClick={() => setMode("single")}
        >
          Single image
        </button>
        <button
          type="button"
          className={`tab ${mode === "batch" ? "active" : ""}`}
          onClick={() => setMode("batch")}
        >
          Batch / folder
        </button>
      </div>

      <section className="panel">
        {mode === "single" ? (
          <div className="uploader">
            <div className="drop">
              <label htmlFor="single-input">Choose a .jpg / .png</label>
              <p className="hint">One dog photo → breed + confidence bars</p>
              <input
                id="single-input"
                type="file"
                accept=".jpg,.jpeg,.png,.webp,.bmp,image/*"
                onChange={onSingleChange}
              />
            </div>
            <div className="actions">
              <button
                type="button"
                className="btn"
                disabled={!singleFile || loading || !health}
                onClick={runSingle}
              >
                {loading ? "Predicting…" : "Predict"}
              </button>
              {singleFile && (
                <button
                  type="button"
                  className="btn ghost"
                  onClick={() => {
                    setSingleFile(null);
                    setSingleResult(null);
                  }}
                >
                  Clear
                </button>
              )}
            </div>

            {(singlePreview || singleResult) && (
              <div className="preview-row">
                <div>
                  {singlePreview && (
                    <img
                      className="preview-img"
                      src={singlePreview}
                      alt="Selected dog"
                    />
                  )}
                </div>
                <div>
                  {singleResult ? (
                    <>
                      <h2 className="result-head">
                        {prettyBreed(singleResult.predicted_class)}
                      </h2>
                      <p className="confidence">
                        Confidence {(singleResult.confidence * 100).toFixed(1)}%
                      </p>
                      <ProbBars result={singleResult} />
                    </>
                  ) : (
                    <p className="hint">Run Predict to see the breed scores.</p>
                  )}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="uploader">
            <div className="drop">
              <label htmlFor="folder-input">Choose a folder</label>
              <p className="hint">
                Pick a folder of dog photos. Subfolder names like{" "}
                <code>Cardigan/</code> are used as true labels when present.
              </p>
              <input
                id="folder-input"
                type="file"
                multiple
                onChange={(e) => collectBatch(e.target.files)}
                {...({
                  webkitdirectory: "",
                  directory: "",
                } as Record<string, string>)}
              />
              <p className="hint" style={{ marginTop: "1rem" }}>
                Or multi-select files:
              </p>
              <input
                type="file"
                accept=".jpg,.jpeg,.png,.webp,.bmp,image/*"
                multiple
                onChange={(e) => collectBatch(e.target.files)}
              />
              <p className="hint">{batchCountLabel}</p>
            </div>
            <div className="actions">
              <button
                type="button"
                className="btn"
                disabled={!batchFiles.length || loading || !health}
                onClick={runBatch}
              >
                {loading ? "Running…" : "Run batch"}
              </button>
              {batchFiles.length > 0 && (
                <button
                  type="button"
                  className="btn ghost"
                  onClick={() => {
                    setBatchFiles([]);
                    setBatchPaths([]);
                    setBatchResult(null);
                  }}
                >
                  Clear
                </button>
              )}
            </div>

            {batchResult && (
              <>
                <div className="summary">
                  <span>
                    Predicted <strong>{batchResult.summary.n_ok}</strong> /{" "}
                    {batchResult.summary.n_total}
                  </span>
                  {batchResult.summary.accuracy != null && (
                    <span>
                      Accuracy{" "}
                      <strong>
                        {(batchResult.summary.accuracy * 100).toFixed(1)}%
                      </strong>{" "}
                      ({batchResult.summary.n_correct}/
                      {batchResult.summary.n_labeled} labeled)
                    </span>
                  )}
                </div>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th></th>
                        <th>File</th>
                        <th>Prediction</th>
                        <th>Conf.</th>
                        <th>True</th>
                        <th></th>
                      </tr>
                    </thead>
                    <tbody>
                      {batchResult.results.map((r, i) => {
                        const key =
                          (r.relative_path || r.filename || String(i)) + i;
                        const thumbKey =
                          (batchFiles[i]
                            ? batchFiles[i].name +
                              ((
                                batchFiles[i] as File & {
                                  webkitRelativePath?: string;
                                }
                              ).webkitRelativePath || "")
                            : "") || "";
                        return (
                          <tr key={key}>
                            <td>
                              {thumbUrls[thumbKey] ? (
                                <img
                                  className="thumb"
                                  src={thumbUrls[thumbKey]}
                                  alt=""
                                />
                              ) : null}
                            </td>
                            <td>
                              {r.relative_path || r.filename || "—"}
                              {r.error && (
                                <div className="status err">{r.error}</div>
                              )}
                            </td>
                            <td>
                              {r.predicted_class
                                ? prettyBreed(r.predicted_class)
                                : "—"}
                            </td>
                            <td>
                              {r.confidence != null
                                ? `${(r.confidence * 100).toFixed(0)}%`
                                : "—"}
                            </td>
                            <td>
                              {r.true_label
                                ? prettyBreed(r.true_label)
                                : "—"}
                            </td>
                            <td>
                              {r.correct === true && (
                                <span className="badge ok">OK</span>
                              )}
                              {r.correct === false && (
                                <span className="badge bad">Wrong</span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </div>
        )}

        {error && <div className="error-box">{error}</div>}
      </section>
    </div>
  );
}
