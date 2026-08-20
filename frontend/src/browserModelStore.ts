export type BrowserTrainingMode = "quick" | "full";

export interface BrowserQualityCheck {
  key: "actionConditioning" | "latentVariation" | "positionAccuracy" | "beliefAccuracy";
  label: string;
  value: number;
  threshold: string;
  passed: boolean;
}

export interface BrowserModelMetadata {
  version: 1;
  modelId: string;
  createdAt: string;
  backend: string;
  mode: BrowserTrainingMode;
  transitions: number;
  epochs: number;
  selectedSeed: number;
  validationLoss: number;
  qualityPassed: boolean;
  checks: BrowserQualityCheck[];
  projection: {
    mean: number[];
    standardDeviation: number[];
    dimensions: [number, number];
  };
}

const METADATA_KEY = "two-gates-browser-model-metadata-v1";
const ACTIVE_KEY = "two-gates-browser-model-active-v1";
export const BROWSER_MODEL_CHANGE_EVENT = "two-gates-browser-model-change";

export function readBrowserModelMetadata(): BrowserModelMetadata | null {
  try {
    const value = window.localStorage.getItem(METADATA_KEY);
    if (!value) return null;
    const parsed = JSON.parse(value) as BrowserModelMetadata;
    return {
      ...parsed,
      validationLoss: typeof parsed.validationLoss === "number" ? parsed.validationLoss : Number.NaN,
      checks: parsed.checks.map((check) => ({
        ...check,
        value: typeof check.value === "number" ? check.value : Number.NaN,
      })),
      projection: {
        ...parsed.projection,
        mean: parsed.projection.mean.map((item) => typeof item === "number" ? item : Number.NaN),
        standardDeviation: parsed.projection.standardDeviation.map((item) => typeof item === "number" ? item : Number.NaN),
      },
    };
  } catch {
    return null;
  }
}

export function writeBrowserModelMetadata(metadata: BrowserModelMetadata): void {
  window.localStorage.setItem(METADATA_KEY, JSON.stringify(metadata));
  notifyBrowserModelChange();
}

export function isBrowserModelActive(): boolean {
  try {
    const metadata = readBrowserModelMetadata();
    return Boolean(metadata?.qualityPassed && window.localStorage.getItem(ACTIVE_KEY) === "true");
  } catch {
    return false;
  }
}

export function setBrowserModelActiveFlag(active: boolean): void {
  if (active) {
    window.localStorage.setItem(ACTIVE_KEY, "true");
  } else {
    window.localStorage.removeItem(ACTIVE_KEY);
  }
  notifyBrowserModelChange();
}

function notifyBrowserModelChange(): void {
  window.dispatchEvent(new CustomEvent(BROWSER_MODEL_CHANGE_EVENT));
}
