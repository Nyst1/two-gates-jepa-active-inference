import { afterEach, describe, expect, it } from "vitest";
import { readBrowserModelMetadata } from "./browserModelStore";

const METADATA_KEY = "two-gates-browser-model-metadata-v1";

describe("browser model metadata", () => {
  afterEach(() => window.localStorage.clear());

  it("turns JSON null metrics from interrupted numerical runs into safe non-finite values", () => {
    window.localStorage.setItem(METADATA_KEY, JSON.stringify({
      version: 1,
      modelId: "diagnostic-model",
      createdAt: "2026-08-20T00:00:00.000Z",
      backend: "webgpu",
      mode: "quick",
      transitions: 100,
      epochs: 1,
      selectedSeed: 11,
      validationLoss: null,
      qualityPassed: false,
      checks: [{
        key: "actionConditioning",
        label: "Action conditioning",
        value: null,
        threshold: ">= 1.20",
        passed: false,
      }],
      projection: {
        mean: [null, 0],
        standardDeviation: [null, 1],
        dimensions: [0, 1],
      },
    }));

    const metadata = readBrowserModelMetadata();
    expect(metadata).not.toBeNull();
    expect(Number.isNaN(metadata?.validationLoss)).toBe(true);
    expect(Number.isNaN(metadata?.checks[0]?.value)).toBe(true);
    expect(Number.isNaN(metadata?.projection.mean[0])).toBe(true);
    expect(Number.isNaN(metadata?.projection.standardDeviation[0])).toBe(true);
  });
});
