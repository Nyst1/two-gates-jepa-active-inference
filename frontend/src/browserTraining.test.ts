import { describe, expect, it } from "vitest";
import { runBrowserTrainingSmokeTest } from "./browserTraining";

describe("browser world-model training", () => {
  it("runs a finite forward and backward update through the JEPA objective", async () => {
    const result = await runBrowserTrainingSmokeTest();
    expect(Number.isFinite(result.loss)).toBe(true);
    expect(result.loss).toBeGreaterThan(1.5);
    expect(Number.isFinite(result.gradientNorm)).toBe(true);
    expect(result.gradientNorm).toBeGreaterThan(0);
    expect(result.parameterCount).toBe(182_392);
  }, 30_000);
});
