import { describe, expect, it } from "vitest";
import { LocalEpisode, PythonRandom, expectedPosteriorEntropy, stateInformationGain } from "./localEpisode";
import type { EpisodeConfig } from "./types";

function config(overrides: Partial<EpisodeConfig> = {}): EpisodeConfig {
  return {
    agentType: "balanced",
    seed: 0,
    beta: 1,
    prior: 0.5,
    cueReliability: 0.9,
    gateTesting: "allowed",
    source: "local",
    ...overrides,
  };
}

describe("in-browser analytic episode", () => {
  it("keeps Python seed semantics for layouts, hidden states, paths, and cues", () => {
    const random = new PythonRandom(3);
    expect(Array.from({ length: 4 }, () => random.random())).toEqual([
      0.23796462709189137,
      0.5442292252959519,
      0.36995516654807925,
      0.6039200385961945,
    ]);
    const gaussian = new PythonRandom(0);
    expect(Array.from({ length: 4 }, () => gaussian.gauss())).toEqual([
      0.9417154046806644,
      -1.3965781047011498,
      -0.6797144480784211,
      0.3705035674606598,
    ]);
  });

  it("selects the earlier sensor cue when path cost is tied", () => {
    const episode = new LocalEpisode(config({ seed: 3, cueReliability: 0.6 }));
    episode.initialFrame();
    let frame = episode.step();
    for (let index = 1; index < 4; index += 1) frame = episode.step();
    const up = frame.candidates.find((candidate) => candidate.action === "up")!;
    const down = frame.candidates.find((candidate) => candidate.action === "down")!;
    const cueGain = stateInformationGain(0.5, 0.6);
    const entropyAfterCue = expectedPosteriorEntropy(0.5, 0.6);

    expect(up.preferenceCost).toBe(down.preferenceCost);
    expect(up.informationSequence).toEqual(["gate"]);
    expect(down.informationSequence).toEqual(["sensor", "gate"]);
    expect(down.scoredSensorInformationGain).toBeCloseTo(cueGain * 4 / 6, 5);
    expect(down.scoredGateInformationGain).toBeCloseTo(entropyAfterCue * 3 / 6, 5);
    expect(down.informationGain).toBeCloseTo(down.scoredSensorInformationGain + down.scoredGateInformationGain, 5);
    expect(down.efeScore).toBeLessThan(up.efeScore);
    expect(frame.selectedAction).toBe("down");
  });

  it("lets changed Lab controls run a complete local episode", () => {
    const episode = new LocalEpisode(config({ seed: 12, prior: 0.7, cueReliability: 0.75, gateTesting: "prohibited" }));
    let frame = episode.initialFrame();
    for (let index = 0; index < 48 && !frame.done; index += 1) frame = episode.step();
    expect(frame.done).toBe(true);
    expect(frame.metrics.success).toBe(true);
    expect(frame.world.gateTesting).toBe("prohibited");
    expect(frame.step).toBeLessThanOrEqual(48);
  });
});
