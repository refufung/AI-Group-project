// Dev: talk to local API. Production (same-origin Docker/tunnel): relative URLs.
const API_BASE =
  import.meta.env.VITE_API_URL !== undefined
    ? import.meta.env.VITE_API_URL
    : import.meta.env.DEV
      ? "http://127.0.0.1:8000"
      : "";

export type Prob = { class_name: string; probability: number };

export type PredictResult = {
  filename?: string | null;
  predicted_class: string;
  confidence: number;
  probabilities: Prob[];
  true_label?: string | null;
  correct?: boolean;
  relative_path?: string | null;
  error?: string;
};

export type BatchResponse = {
  summary: {
    n_total: number;
    n_ok: number;
    n_labeled: number;
    n_correct: number;
    accuracy: number | null;
  };
  results: PredictResult[];
  class_names: string[];
};

export type Health = {
  status: string;
  device: string;
  class_names: string[];
  checkpoint: string;
};

export async function fetchHealth(): Promise<Health> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`API health failed (${res.status})`);
  return res.json();
}

export async function predictOne(file: File): Promise<PredictResult> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/predict`, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || `Predict failed (${res.status})`);
  }
  return res.json();
}

export async function predictBatch(
  files: File[],
  relativePaths: string[],
): Promise<BatchResponse> {
  const form = new FormData();
  files.forEach((file) => form.append("files", file));
  relativePaths.forEach((p) => form.append("relative_paths", p));
  const res = await fetch(`${API_BASE}/predict/batch`, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || `Batch predict failed (${res.status})`);
  }
  return res.json();
}

export { API_BASE };
