import { useEffect, useRef, useState } from "react";
import {
  type BrowserModelMetadata,
  type BrowserTrainingMode,
  isBrowserModelActive,
  readBrowserModelMetadata,
} from "./browserModelStore";
import type { BrowserBackendInfo, BrowserTrainingProgress } from "./browserTraining";

interface TrainingStudioProps {
  open: boolean;
  onClose: () => void;
}

const MODE_COPY: Record<BrowserTrainingMode, { title: string; detail: string; facts: string }> = {
  quick: {
    title: "Quick experiment",
    detail: "Train one model while keeping the Lab responsive. Good for learning and parameter experiments.",
    facts: "10,000 transitions · 5 epochs · 1 seed",
  },
  full: {
    title: "Full reproduction",
    detail: "Train three independent models and select the median validation run, as in the reference pipeline.",
    facts: "50,000 transitions · 5 epochs · 3 seeds",
  },
};

function formatLoss(value: number | null) {
  return value === null || !Number.isFinite(value) ? "—" : value.toFixed(4);
}

function formatMetric(value: number) {
  return Number.isFinite(value) ? value.toFixed(3) : "—";
}

export default function TrainingStudio({ open, onClose }: TrainingStudioProps) {
  const [mode, setMode] = useState<BrowserTrainingMode>("quick");
  const [backend, setBackend] = useState<BrowserBackendInfo | null>(null);
  const [metadata, setMetadata] = useState<BrowserModelMetadata | null>(() => readBrowserModelMetadata());
  const [active, setActive] = useState(() => isBrowserModelActive());
  const [progress, setProgress] = useState<BrowserTrainingProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortController = useRef<AbortController | null>(null);
  const preparing = useRef(false);

  useEffect(() => {
    if (!open || backend || preparing.current) return;
    preparing.current = true;
    void import("./browserTraining")
      .then((module) => module.prepareTrainingBackend())
      .then(setBackend)
      .catch((reason) => setError(`Could not initialize browser training: ${String(reason)}`))
      .finally(() => { preparing.current = false; });
  }, [backend, open]);

  const training = progress !== null && !["complete", "cancelled", "failed"].includes(progress.stage);

  async function startTraining() {
    setError(null);
    const controller = new AbortController();
    abortController.current = controller;
    try {
      const module = await import("./browserTraining");
      const preparedBackend = backend ?? await module.prepareTrainingBackend();
      setBackend(preparedBackend);
      const result = await module.trainBrowserModel(
        mode,
        (nextProgress) => setProgress({ ...nextProgress }),
        controller.signal,
      );
      setMetadata(result);
      setActive(false);
    } catch (reason) {
      if (controller.signal.aborted) {
        setProgress((current) => current ? { ...current, stage: "cancelled", message: "Training cancelled. The default model is unchanged." } : null);
      } else {
        setError(String(reason));
        setProgress((current) => current ? { ...current, stage: "failed", message: "Training failed. The default model is unchanged." } : null);
      }
    } finally {
      abortController.current = null;
    }
  }

  async function activateModel() {
    if (!metadata?.qualityPassed) return;
    const module = await import("./browserTraining");
    await module.activateBrowserModel();
    setActive(true);
  }

  async function restoreDefault() {
    const module = await import("./browserTraining");
    module.deactivateBrowserModel();
    setActive(false);
  }

  return (
    <div className={`training-layer ${open ? "open" : ""}`} aria-hidden={!open}>
      <button className="training-backdrop" type="button" aria-label="Close training" onClick={onClose} />
      <aside className="training-studio" role="dialog" aria-modal="true" aria-labelledby="training-title">
        <header className="training-heading">
          <div>
            <p className="eyebrow">OPTIONAL · RUNS ON THIS DEVICE</p>
            <h2 id="training-title">Train the world model</h2>
            <p>The pretrained model always remains available. A new model must pass the same quality gates before it can shape the Lab's imagined futures.</p>
          </div>
          <button type="button" aria-label="Close training" onClick={onClose}>×</button>
        </header>

        <section className="training-runtime" aria-label="Training runtime">
          <div>
            <span className={`runtime-light ${backend?.accelerated ? "accelerated" : ""}`} />
            <span><strong>{backend?.label ?? "Detecting hardware…"}</strong><small>{backend?.detail ?? "Selecting the fastest compatible TensorFlow.js backend."}</small></span>
          </div>
          <span className="model-safety"><strong>Default safe</strong><small>Pretrained model is never overwritten</small></span>
        </section>

        <section className="training-modes" aria-label="Training depth">
          {(Object.keys(MODE_COPY) as BrowserTrainingMode[]).map((key) => (
            <label className={mode === key ? "selected" : ""} key={key}>
              <input type="radio" name="training-mode" value={key} checked={mode === key} disabled={training} onChange={() => setMode(key)} />
              <span><strong>{MODE_COPY[key].title}</strong><small>{MODE_COPY[key].detail}</small><em>{MODE_COPY[key].facts}</em></span>
            </label>
          ))}
        </section>

        {progress && (
          <section className="training-progress" aria-live="polite">
            <div className="training-progress-heading">
              <span><strong>{progress.message}</strong><small>{progress.seed ? `Seed ${progress.seed} · epoch ${progress.epoch}/${progress.epochs}` : "Preparing the training graph"}</small></span>
              <span><strong>{Math.round(progress.fraction * 100)}%</strong><small>loss {formatLoss(progress.loss)}</small></span>
            </div>
            <div className="training-progress-track"><i style={{ width: `${progress.fraction * 100}%` }} /></div>
          </section>
        )}

        {error && <p className="training-error">{error}</p>}

        {metadata && !training && (
          <section className={`training-result ${metadata.qualityPassed ? "passed" : "failed"}`}>
            <header>
              <span><strong>{metadata.qualityPassed ? "Model passed validation" : "Model needs more training"}</strong><small>{metadata.mode === "full" ? "Full reproduction" : "Quick experiment"} · seed {metadata.selectedSeed} · validation loss {metadata.validationLoss.toFixed(4)}</small></span>
              <b>{metadata.qualityPassed ? "READY" : "NOT ACTIVE"}</b>
            </header>
            <div className="quality-grid">
              {metadata.checks.map((check) => (
                <div className={check.passed ? "passed" : "failed"} key={check.key}>
                  <span>{check.passed ? "✓" : "×"}</span>
                  <p><strong>{check.label}</strong><small>{formatMetric(check.value)} · target {check.threshold}</small></p>
                </div>
              ))}
            </div>
            <p>{metadata.qualityPassed
              ? "Activation changes the neural futures shown during Imagine. Belief updates and policy scoring remain exact and analytic."
              : "The candidate is stored locally for inspection, but cannot replace the validated model. Try Full reproduction or a faster GPU backend."}</p>
          </section>
        )}

        <footer className="training-actions">
          {training ? (
            <button type="button" className="secondary-training-button" onClick={() => abortController.current?.abort()}>Cancel training</button>
          ) : (
            <button type="button" className="primary-training-button" disabled={!backend} onClick={() => void startTraining()}>Train {mode === "quick" ? "quick model" : "full model"}</button>
          )}
          {metadata?.qualityPassed && !active && !training && <button type="button" onClick={() => void activateModel()}>Use in Lab</button>}
          {active && <button type="button" onClick={() => void restoreDefault()}>Restore pretrained model</button>}
          <span>{active ? "Browser-trained model active" : "Pretrained model active"}</span>
        </footer>
      </aside>
    </div>
  );
}
