import balancedReplay from "../public/replays/two-gates-balanced-seed-0.json";
import informationReplay from "../public/replays/two-gates-information-seed-0.json";
import pragmaticReplay from "../public/replays/two-gates-pragmatic-seed-0.json";
import { LocalEpisode } from "./localEpisode";
import type { AgentType, EpisodeConfig, MetaResponse, ReplayTrace, SourceType, StepFrame } from "./types";

export interface StandaloneEpisode {
  readonly source: SourceType;
  initialFrame(): StepFrame;
  step(): StepFrame;
}

const replays: Record<AgentType, ReplayTrace> = {
  balanced: balancedReplay as unknown as ReplayTrace,
  pragmatic: pragmaticReplay as unknown as ReplayTrace,
  information: informationReplay as unknown as ReplayTrace,
};

export const STANDALONE_META: MetaResponse = {
  schemaVersion: "1.4",
  appName: "Two Gates",
  liveModel: {
    available: false,
    reason: "The standalone build uses an embedded accepted checkpoint replay or the in-browser analytic planner.",
    device: "browser",
    checkpointHash: replays.balanced.checkpointHash,
    torchAvailable: false,
  },
  replayAvailable: true,
  futureSamples: 16,
  planningHorizon: 5,
  worldVariation: {
    seedControls: ["hidden gate", "upper gate row", "lower gate row", "sensor position", "cue draw"],
    layoutRange: {
      upperGateRows: [1, 2, 3],
      lowerGateRows: [5, 6, 7],
      sensorColumns: [2, 3, 4, 5],
      sensorRows: [1, 2, 3, 4, 5, 6, 7],
    },
  },
  modelArchitecture: {
    observation: [3, 32, 32],
    encoderChannels: [16, 32, 48],
    denseUnits: 64,
    latentUnits: 8,
    denoiserUnits: [96, 96, 8],
    diffusionSteps: 8,
  },
  gateTestingModes: {
    allowed: "Diagnostic gate contact gives exact evidence, but a closed gate still blocks movement and counts as a collision.",
    prohibited: "The agent must visit the sensor before contacting an unresolved gate, unless prior confidence is at least 98%.",
  },
  scientificLabels: [
    "minimal JEPA-style latent world model",
    "conditional latent diffusion",
    "exact Bayesian information gain in a toy POMDP",
    "operational EFE approximation",
  ],
};

function isBundledCheckpointConfig(config: EpisodeConfig) {
  return config.seed === 0
    && config.beta === 1
    && config.prior === 0.5
    && config.cueReliability === 0.9
    && config.gateTesting === "allowed";
}

class ReplayEpisode implements StandaloneEpisode {
  readonly source = "replay" as const;
  private index = 0;

  constructor(private readonly trace: ReplayTrace) {}

  initialFrame() {
    this.index = 0;
    return this.trace.frames[0];
  }

  step() {
    this.index = Math.min(this.trace.frames.length - 1, this.index + 1);
    return this.trace.frames[this.index];
  }
}

export function createStandaloneEpisode(config: EpisodeConfig): StandaloneEpisode {
  if (isBundledCheckpointConfig(config)) return new ReplayEpisode(replays[config.agentType]);
  return new LocalEpisode(config);
}
