export type AgentType = "balanced" | "pragmatic" | "information";
export type SourceType = "live" | "replay" | "local";
export type Position = [number, number];

export interface EpisodeConfig {
  agentType: AgentType;
  seed: number;
  beta: number;
  prior: number;
  cueReliability: number;
  gateTesting: "allowed" | "prohibited";
  source: SourceType;
}

export interface Candidate {
  action: "right" | "up" | "down" | "left";
  preferenceCost: number;
  informationGain: number;
  sensorInformationGain: number;
  sensorRawInformationGain: number;
  sensorReachFactor: number;
  gateInformationGain: number;
  gateRawInformationGain: number;
  gateReachFactor: number;
  scoredSensorInformationGain: number;
  scoredSensorRawInformationGain: number;
  scoredSensorReachFactor: number;
  scoredGateInformationGain: number;
  scoredGateRawInformationGain: number;
  scoredGateReachFactor: number;
  informationSource: "sensor" | "gate" | "sensor_then_gate" | "none";
  informationSequence: Array<"sensor" | "gate">;
  planObjective: "sensor" | "gate" | "goal";
  informationTarget: Position | null;
  efeScore: number;
  collisionProbability: number;
  sampledPaths: Array<{
    hypothesis: "upper" | "lower";
    points: Position[];
    objective: "sensor" | "gate" | "goal";
  }>;
  latentSamples: Array<[number, number]>;
  selected: boolean;
  admissible: boolean;
  constraintReason: string | null;
}

export interface WorldState {
  width: number;
  height: number;
  agent: Position;
  previousAgent: Position;
  start: Position;
  goal: Position;
  sensor: Position;
  layoutId: string;
  upperGate: { position: Position; status: "unknown" | "open" | "closed" };
  lowerGate: { position: Position; status: "unknown" | "open" | "closed" };
  walls: Position[];
  sensorVisited: boolean;
  cue: "upper" | "lower" | null;
  cueReliability: number;
  gateTesting: "allowed" | "prohibited";
  gateTestingRestricted: boolean;
  reachedGoal: boolean;
}

export interface NeuralLayer {
  id: string;
  label: string;
  kind: string;
  values: number[];
  totalUnits: number;
}

export interface NeuralEdge {
  sourceLayer: string;
  source: number;
  targetLayer: string;
  target: number;
  weight: number;
  contribution: number;
}

export interface ModelInspection {
  mode: "live checkpoint inference" | "recorded checkpoint inference";
  selectedAction: string;
  observation: {
    shape: [number, number, number];
    thumbnail: Array<Array<[number, number, number]>>;
  };
  conditioning: {
    beliefUpper: number;
    actionOneHot: number[];
    diffusionStepShown: number;
  };
  encoder: { layers: NeuralLayer[]; edges: NeuralEdge[] };
  denoiser: { layers: NeuralLayer[]; edges: NeuralEdge[] };
  diffusionTrajectory: Array<{
    step: number;
    latentNorm: number;
    predictedNoiseNorm: number;
  }>;
  outputs: {
    current: { position: number[]; upperGateProbe: number; goalProbability: number };
    meanFuture: { position: number[]; upperGateProbe: number; goalProbability: number };
    futureLatentMean: number[];
  };
  explanation: string;
}

export interface StepFrame {
  schemaVersion: string;
  checkpointHash: string;
  source: SourceType;
  step: number;
  phase: "observe" | "imagine" | "evaluate" | "act" | "update";
  world: WorldState;
  belief: {
    upper: number;
    lower: number;
    entropyBefore: number;
    entropyAfter: number;
    updateReason: string;
  };
  candidates: Candidate[];
  selectedAction: string;
  modelInspection: ModelInspection | null;
  transition: {
    action: string;
    moved: boolean;
    collision: boolean;
    cue: "upper" | "lower" | null;
    revealedGate: "upper" | "lower" | null;
    reachedGoal: boolean;
  } | null;
  metrics: {
    steps: number;
    collisions: number;
    wrongGateCommitments: number;
    gateTests: number;
    pathLength: number;
    success: boolean;
  };
  done: boolean;
}

export interface ReplayTrace {
  id: string;
  schemaVersion: string;
  checkpointHash: string;
  config: EpisodeConfig;
  frames: StepFrame[];
  summary: StepFrame["metrics"];
  traceHash: string;
  modelMode: string;
}

export interface ReplayIndexItem {
  id: string;
  agentType: AgentType;
  seed: number;
  schemaVersion: string;
  checkpointHash: string;
  modelMode: string;
  summary: StepFrame["metrics"];
  file: string;
}

export interface MetaResponse {
  schemaVersion: string;
  appName: string;
  liveModel: {
    available: boolean;
    reason: string;
    device: string;
    checkpointHash: string;
    torchAvailable: boolean;
  };
  replayAvailable: boolean;
  futureSamples: number;
  planningHorizon: number;
  worldVariation: {
    seedControls: string[];
    layoutRange: Record<string, number[]>;
  };
  modelArchitecture: {
    observation: number[];
    encoderChannels: number[];
    denseUnits: number;
    latentUnits: number;
    denoiserUnits: number[];
    diffusionSteps: number;
  };
  gateTestingModes: Record<"allowed" | "prohibited", string>;
  scientificLabels: string[];
}
