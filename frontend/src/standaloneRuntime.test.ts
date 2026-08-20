import { describe, expect, it } from "vitest";
import { createStandaloneEpisode } from "./standaloneRuntime";
import type { EpisodeConfig } from "./types";

const defaultConfig: EpisodeConfig = {
  agentType: "balanced",
  seed: 0,
  beta: 1,
  prior: 0.5,
  cueReliability: 0.9,
  gateTesting: "allowed",
  source: "local",
};

describe("standalone runtime routing", () => {
  it("keeps the accepted checkpoint replay as the default", () => {
    const episode = createStandaloneEpisode(defaultConfig);
    const frame = episode.initialFrame();
    expect(episode.source).toBe("replay");
    expect(frame.checkpointHash).toBe("60e9141873e99464188fb358ae4e43ffa383c3395d97f755f0c599881a9cc4f1");
    expect(frame.modelInspection).not.toBeNull();
  });

  it("routes changed controls through the in-browser planner", () => {
    const episode = createStandaloneEpisode({ ...defaultConfig, cueReliability: 0.6, seed: 3 });
    const frame = episode.initialFrame();
    expect(episode.source).toBe("local");
    expect(frame.checkpointHash).toBe("browser-analytic-v1");
    expect(frame.world.cueReliability).toBe(0.6);
  });
});
