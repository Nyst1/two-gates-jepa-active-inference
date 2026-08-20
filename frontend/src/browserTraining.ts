import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgpu";
import {
  type BrowserModelMetadata,
  type BrowserQualityCheck,
  type BrowserTrainingMode,
  isBrowserModelActive,
  readBrowserModelMetadata,
  setBrowserModelActiveFlag,
  writeBrowserModelMetadata,
} from "./browserModelStore";
import type { StepFrame, WorldState } from "./types";

export interface BrowserBackendInfo {
  id: "webgpu" | "webgl" | "cpu";
  label: string;
  detail: string;
  accelerated: boolean;
}

export interface BrowserTrainingProgress {
  stage: "preparing" | "training" | "validating" | "saving" | "complete" | "cancelled" | "failed";
  message: string;
  fraction: number;
  seed: number | null;
  epoch: number;
  epochs: number;
  loss: number | null;
}

interface TrainingDefinition {
  transitions: number;
  epochs: number;
  batchSize: number;
  seeds: number[];
}

interface TransitionBatch {
  observation: tf.Tensor4D;
  nextObservation: tf.Tensor4D;
  actionIndices: Int32Array;
  actionOneHot: tf.Tensor2D;
  belief: tf.Tensor2D;
  position: tf.Tensor2D;
  nextPosition: tf.Tensor2D;
  goal: tf.Tensor2D;
}

interface ModelBundle {
  encoder: tf.LayersModel;
  targetEncoder: tf.LayersModel;
  denoiser: tf.LayersModel;
  deterministicPredictor: tf.LayersModel;
  inverseHead: tf.LayersModel;
  positionHead: tf.LayersModel;
  gateHead: tf.LayersModel;
  goalHead: tf.LayersModel;
}

interface QualityResult {
  passed: boolean;
  checks: BrowserQualityCheck[];
  projection: BrowserModelMetadata["projection"];
}

const DEFINITIONS: Record<BrowserTrainingMode, TrainingDefinition> = {
  quick: { transitions: 10_000, epochs: 5, batchSize: 128, seeds: [11] },
  full: { transitions: 50_000, epochs: 5, batchSize: 256, seeds: [11, 29, 47] },
};

const WIDTH = 13;
const HEIGHT = 9;
const WALL_X = 6;
const START: Position = [1, 4];
const GOAL: Position = [11, 4];
const ACTIONS: Position[] = [[1, 0], [0, -1], [0, 1], [-1, 0]];
const MODEL_PREFIX = "two-gates-browser-user-v1";
const COMPONENT_NAMES = [
  "encoder",
  "targetEncoder",
  "denoiser",
  "deterministicPredictor",
  "inverseHead",
  "positionHead",
  "gateHead",
  "goalHead",
] as const;

type Position = [number, number];

let backendPromise: Promise<BrowserBackendInfo> | null = null;
let inferenceCache: { modelId: string; encoder: tf.LayersModel; denoiser: tf.LayersModel } | null = null;

function componentUrl(modelId: string, name: typeof COMPONENT_NAMES[number]) {
  return `indexeddb://${MODEL_PREFIX}-${modelId}-${name}`;
}

async function hasWebGpu() {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) return false;
  try {
    const gpu = (navigator as Navigator & { gpu?: { requestAdapter: () => Promise<unknown | null> } }).gpu;
    return Boolean(await gpu?.requestAdapter());
  } catch {
    return false;
  }
}

function hasWebGl() {
  if (typeof document === "undefined") return false;
  try {
    const canvas = document.createElement("canvas");
    return Boolean(canvas.getContext("webgl2") || canvas.getContext("webgl"));
  } catch {
    return false;
  }
}

async function backendSmokeTest() {
  const value = tf.tidy(() => tf.matMul(tf.ones([2, 2]), tf.ones([2, 2])).sum());
  await value.data();
  value.dispose();
}

export async function prepareTrainingBackend(): Promise<BrowserBackendInfo> {
  if (backendPromise) return backendPromise;
  backendPromise = (async () => {
    if (await hasWebGpu()) {
      try {
        if (await tf.setBackend("webgpu")) {
          await tf.ready();
          await backendSmokeTest();
          return {
            id: "webgpu",
            label: "WebGPU acceleration",
            detail: "The browser can run tensor operations on the available GPU.",
            accelerated: true,
          };
        }
      } catch {
        // Continue to the broadly supported WebGL backend.
      }
    }
    if (hasWebGl()) {
      try {
        if (await tf.setBackend("webgl")) {
          await tf.ready();
          await backendSmokeTest();
          return {
            id: "webgl",
            label: "WebGL acceleration",
            detail: "GPU acceleration is available through the browser's compatibility backend.",
            accelerated: true,
          };
        }
      } catch {
        // CPU is a slower but complete training fallback.
      }
    }
    await tf.setBackend("cpu");
    await tf.ready();
    return {
      id: "cpu",
      label: "CPU training",
      detail: "No compatible browser GPU was found. Quick training is recommended.",
      accelerated: false,
    };
  })();
  return backendPromise;
}

class RandomStream {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  next() {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let value = this.state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  }

  integer(maxExclusive: number) {
    return Math.floor(this.next() * maxExclusive);
  }

  choice<T>(items: readonly T[]): T {
    return items[this.integer(items.length)];
  }
}

interface SimulatorState {
  position: Position;
  upperGate: Position;
  lowerGate: Position;
  sensor: Position;
  hiddenUpper: boolean;
  sensorVisited: boolean;
  cue: "upper" | "lower" | null;
  belief: number;
  steps: number;
}

interface GeneratedTransition {
  observation: Float32Array;
  nextObservation: Float32Array;
  action: number;
  belief: number;
  position: Position;
  nextPosition: Position;
  goal: number;
}

const observationBases = new Map<string, Float32Array>();

class TransitionStream {
  private state: SimulatorState;
  private readonly random: RandomStream;

  constructor(seed: number) {
    this.random = new RandomStream(seed);
    this.state = this.newEpisode();
  }

  private newEpisode(): SimulatorState {
    const upperGate: Position = [WALL_X, 1 + this.random.integer(3)];
    const lowerGate: Position = [WALL_X, 5 + this.random.integer(3)];
    const sensorCandidates: Position[] = [];
    for (let x = 2; x < WALL_X; x += 1) {
      for (let y = 1; y < HEIGHT - 1; y += 1) {
        if (x !== START[0] || y !== START[1]) sensorCandidates.push([x, y]);
      }
    }
    return {
      position: [...START],
      upperGate,
      lowerGate,
      sensor: this.random.choice(sensorCandidates),
      hiddenUpper: this.random.next() < 0.5,
      sensorVisited: false,
      cue: null,
      belief: 0.5,
      steps: 0,
    };
  }

  private isBoundary([x, y]: Position) {
    return x === 0 || x === WIDTH - 1 || y === 0 || y === HEIGHT - 1;
  }

  private isBlocked(position: Position) {
    const [x, y] = position;
    if (this.isBoundary(position)) return true;
    const isGate = samePosition(position, this.state.upperGate) || samePosition(position, this.state.lowerGate);
    if (x === WALL_X && !isGate) return true;
    const closedGate = this.state.hiddenUpper ? this.state.lowerGate : this.state.upperGate;
    return x === closedGate[0] && y === closedGate[1];
  }

  private chooseAction() {
    if (this.random.next() < 0.55) return this.random.integer(ACTIONS.length);
    const target = this.state.sensorVisited || this.random.next() < 0.55 ? GOAL : this.state.sensor;
    const dx = target[0] - this.state.position[0];
    const dy = target[1] - this.state.position[1];
    const candidates: number[] = [];
    if (dx > 0) candidates.push(0);
    else if (dx < 0) candidates.push(3);
    if (dy > 0) candidates.push(2);
    else if (dy < 0) candidates.push(1);
    return this.random.choice(candidates.length ? candidates : [0, 1, 2, 3]);
  }

  next(): GeneratedTransition {
    if (samePosition(this.state.position, GOAL) || this.state.steps >= 36) this.state = this.newEpisode();
    const observation = renderObservation(this.state);
    const position: Position = [...this.state.position];
    const belief = this.state.belief;
    const action = this.chooseAction();
    const [dx, dy] = ACTIONS[action];
    const proposed: Position = [position[0] + dx, position[1] + dy];
    const collision = this.isBlocked(proposed);
    if (!collision) this.state.position = proposed;

    if (samePosition(proposed, this.state.upperGate) || samePosition(proposed, this.state.lowerGate)) {
      const observedUpper = samePosition(proposed, this.state.upperGate);
      this.state.belief = collision ? (observedUpper ? 0 : 1) : (observedUpper ? 1 : 0);
    }

    if (samePosition(this.state.position, this.state.sensor) && !this.state.sensorVisited) {
      this.state.sensorVisited = true;
      const truthful = this.state.hiddenUpper ? "upper" : "lower";
      this.state.cue = this.random.next() <= 0.9 ? truthful : (truthful === "upper" ? "lower" : "upper");
      const cueUpper = this.state.cue === "upper";
      const numerator = (cueUpper ? 0.9 : 0.1) * this.state.belief;
      const denominator = numerator + (cueUpper ? 0.1 : 0.9) * (1 - this.state.belief);
      this.state.belief = denominator > 0 ? numerator / denominator : this.state.belief;
    }

    this.state.steps += 1;
    return {
      observation,
      nextObservation: renderObservation(this.state),
      action,
      belief,
      position: [position[0] / 12, position[1] / 8],
      nextPosition: [this.state.position[0] / 12, this.state.position[1] / 8],
      goal: samePosition(position, GOAL) ? 1 : 0,
    };
  }

  batch(size: number): TransitionBatch {
    const imageSize = 32 * 32 * 3;
    const observations = new Float32Array(size * imageSize);
    const nextObservations = new Float32Array(size * imageSize);
    const actions = new Int32Array(size);
    const actionOneHot = new Float32Array(size * 4);
    const beliefs = new Float32Array(size);
    const positions = new Float32Array(size * 2);
    const nextPositions = new Float32Array(size * 2);
    const goals = new Float32Array(size);
    for (let index = 0; index < size; index += 1) {
      const transition = this.next();
      observations.set(transition.observation, index * imageSize);
      nextObservations.set(transition.nextObservation, index * imageSize);
      actions[index] = transition.action;
      actionOneHot[index * 4 + transition.action] = 1;
      beliefs[index] = transition.belief;
      positions.set(transition.position, index * 2);
      nextPositions.set(transition.nextPosition, index * 2);
      goals[index] = transition.goal;
    }
    return {
      observation: tf.tensor4d(observations, [size, 32, 32, 3]),
      nextObservation: tf.tensor4d(nextObservations, [size, 32, 32, 3]),
      actionIndices: actions,
      actionOneHot: tf.tensor2d(actionOneHot, [size, 4]),
      belief: tf.tensor2d(beliefs, [size, 1]),
      position: tf.tensor2d(positions, [size, 2]),
      nextPosition: tf.tensor2d(nextPositions, [size, 2]),
      goal: tf.tensor2d(goals, [size, 1]),
    };
  }
}

function samePosition(left: Position, right: Position) {
  return left[0] === right[0] && left[1] === right[1];
}

function renderObservation(state: SimulatorState): Float32Array {
  const cacheKey = `${state.upperGate[1]}-${state.lowerGate[1]}`;
  let base = observationBases.get(cacheKey);
  if (!base) {
    base = new Float32Array(32 * 32 * 3);
    for (let pixel = 0; pixel < 32 * 32; pixel += 1) {
      base[pixel * 3] = 0.035;
      base[pixel * 3 + 1] = 0.045;
      base[pixel * 3 + 2] = 0.075;
    }
    for (let x = 0; x < WIDTH; x += 1) {
      paint(base, [x, 0], [0.22, 0.24, 0.34]);
      paint(base, [x, HEIGHT - 1], [0.22, 0.24, 0.34]);
    }
    for (let y = 0; y < HEIGHT; y += 1) {
      paint(base, [0, y], [0.22, 0.24, 0.34]);
      paint(base, [WIDTH - 1, y], [0.22, 0.24, 0.34]);
    }
    for (let y = 1; y < HEIGHT - 1; y += 1) {
      if (y !== state.upperGate[1] && y !== state.lowerGate[1]) paint(base, [WALL_X, y], [0.22, 0.24, 0.34]);
    }
    paint(base, state.upperGate, [0.15, 0.19, 0.28]);
    paint(base, state.lowerGate, [0.15, 0.19, 0.28]);
    paint(base, GOAL, [0.23, 0.82, 0.48], 1);
    observationBases.set(cacheKey, base);
  }
  const image = base.slice();
  const sensorColor: [number, number, number] = state.cue === "upper"
    ? [0.21, 0.78, 0.94]
    : state.cue === "lower"
      ? [0.78, 0.48, 0.94]
      : state.sensorVisited
        ? [0.95, 0.62, 0.21]
        : [0.42, 0.31, 0.18];
  paint(image, state.sensor, sensorColor, 1);
  paint(image, state.position, [0.96, 0.93, 0.79], 1);
  for (let y = 0; y < 2; y += 1) {
    for (let x = 0; x < 32; x += 1) {
      const offset = (y * 32 + x) * 3;
      image[offset] = state.belief;
      image[offset + 2] = 1 - state.belief;
    }
  }
  for (let index = 0; index < image.length; index += 1) {
    image[index] = Math.round(image[index] * 255) / 255;
  }
  return image;
}

function paint(image: Float32Array, [cellX, cellY]: Position, color: [number, number, number], inset = 0) {
  const x0 = Math.round(cellX * 32 / WIDTH) + inset;
  const x1 = Math.round((cellX + 1) * 32 / WIDTH) - inset;
  const y0 = Math.round(cellY * 32 / HEIGHT) + inset;
  const y1 = Math.round((cellY + 1) * 32 / HEIGHT) - inset;
  for (let y = Math.max(0, y0); y < Math.min(32, y1); y += 1) {
    for (let x = Math.max(0, x0); x < Math.min(32, x1); x += 1) {
      const offset = (y * 32 + x) * 3;
      image[offset] = color[0];
      image[offset + 1] = color[1];
      image[offset + 2] = color[2];
    }
  }
}

function makeInitializer(seed: number) {
  let index = 0;
  return () => tf.initializers.glorotUniform({ seed: seed * 101 + index++ });
}

function createEncoder(seed: number, name: string) {
  const initializer = makeInitializer(seed);
  const input = tf.input({ shape: [32, 32, 3], name: `${name}-input` });
  let value = tf.layers.conv2d({ filters: 16, kernelSize: 4, strides: 2, padding: "same", kernelInitializer: initializer(), name: `${name}-conv1` }).apply(input) as tf.SymbolicTensor;
  value = tf.layers.activation({ activation: "swish", name: `${name}-silu1` }).apply(value) as tf.SymbolicTensor;
  value = tf.layers.conv2d({ filters: 32, kernelSize: 4, strides: 2, padding: "same", kernelInitializer: initializer(), name: `${name}-conv2` }).apply(value) as tf.SymbolicTensor;
  value = tf.layers.activation({ activation: "swish", name: `${name}-silu2` }).apply(value) as tf.SymbolicTensor;
  value = tf.layers.conv2d({ filters: 48, kernelSize: 4, strides: 2, padding: "same", kernelInitializer: initializer(), name: `${name}-conv3` }).apply(value) as tf.SymbolicTensor;
  value = tf.layers.activation({ activation: "swish", name: `${name}-silu3` }).apply(value) as tf.SymbolicTensor;
  value = tf.layers.flatten({ name: `${name}-flatten` }).apply(value) as tf.SymbolicTensor;
  value = tf.layers.dense({ units: 64, kernelInitializer: initializer(), name: `${name}-dense64` }).apply(value) as tf.SymbolicTensor;
  value = tf.layers.layerNormalization({ axis: -1, epsilon: 1e-5, name: `${name}-norm` }).apply(value) as tf.SymbolicTensor;
  value = tf.layers.activation({ activation: "swish", name: `${name}-silu4` }).apply(value) as tf.SymbolicTensor;
  const output = tf.layers.dense({ units: 8, kernelInitializer: initializer(), name: `${name}-latent` }).apply(value) as tf.SymbolicTensor;
  return tf.model({ inputs: input, outputs: output, name });
}

function createDenseModel(seed: number, inputSize: number, units: number[], name: string, outputActivation?: "sigmoid") {
  const initializer = makeInitializer(seed);
  const model = tf.sequential({ name });
  units.forEach((width, index) => {
    const last = index === units.length - 1;
    model.add(tf.layers.dense({
      units: width,
      inputShape: index === 0 ? [inputSize] : undefined,
      activation: last ? outputActivation : "swish",
      kernelInitializer: initializer(),
      name: `${name}-dense${index + 1}`,
    }));
  });
  return model;
}

function createBundle(seed: number): ModelBundle {
  const encoder = createEncoder(seed, `encoder-${seed}`);
  const targetEncoder = createEncoder(seed + 10_000, `target-encoder-${seed}`);
  const onlineWeights = encoder.getWeights();
  targetEncoder.setWeights(onlineWeights);
  targetEncoder.trainable = false;
  return {
    encoder,
    targetEncoder,
    denoiser: createDenseModel(seed + 1, 22, [96, 96, 8], `denoiser-${seed}`),
    deterministicPredictor: createDenseModel(seed + 2, 13, [64, 8], `deterministic-${seed}`),
    inverseHead: createDenseModel(seed + 3, 16, [48, 4], `inverse-${seed}`),
    positionHead: createDenseModel(seed + 4, 8, [32, 2], `position-${seed}`, "sigmoid"),
    gateHead: createDenseModel(seed + 5, 8, [16, 1], `gate-${seed}`),
    goalHead: createDenseModel(seed + 6, 8, [16, 1], `goal-${seed}`),
  };
}

function predict(model: tf.LayersModel, inputs: tf.Tensor | tf.Tensor[]) {
  return model.apply(inputs, { training: true }) as tf.Tensor;
}

function sigmoidCrossEntropy(labels: tf.Tensor, logits: tf.Tensor) {
  return tf.mean(tf.add(tf.sub(tf.relu(logits), tf.mul(logits, labels)), tf.softplus(tf.neg(tf.abs(logits)))));
}

function sumTensors(values: tf.Tensor[]) {
  if (values.length === 0) throw new Error("Cannot sum an empty tensor list.");
  return values.slice(1).reduce((total, value) => tf.add(total, value), values[0]);
}

function batchLoss(bundle: ModelBundle, batch: TransitionBatch, randomSeed: number): tf.Scalar {
  return tf.tidy(() => {
    const current = predict(bundle.encoder, batch.observation);
    const nextOnline = predict(bundle.encoder, batch.nextObservation);
    const target = predict(bundle.targetEncoder, batch.nextObservation);
    const deterministicDelta = predict(bundle.deterministicPredictor, tf.concat([current, batch.actionOneHot, batch.belief], 1));
    const deterministic = tf.add(current, deterministicDelta);
    const deterministicLoss = tf.mean(tf.square(tf.sub(deterministic, target)));

    const batchSize = batch.observation.shape[0];
    const diffusionIndices = tf.randomUniform([batchSize, 1], 0, 8, "int32", randomSeed);
    const betas = tf.linspace(0.025, 0.18, 8);
    const alphaBars = tf.cumprod(tf.sub(1, betas));
    const selectedAlphaBars = tf.gather(alphaBars, diffusionIndices.flatten()).reshape([batchSize, 1]);
    const noise = tf.randomNormal([batchSize, 8], 0, 1, "float32", randomSeed + 1);
    const noisyTarget = tf.add(tf.mul(tf.sqrt(selectedAlphaBars), target), tf.mul(tf.sqrt(tf.sub(1, selectedAlphaBars)), noise));
    const normalizedTime = tf.div(diffusionIndices.toFloat(), 7);
    const predictedNoise = predict(bundle.denoiser, tf.concat([noisyTarget, current, batch.actionOneHot, batch.belief, normalizedTime], 1));
    const diffusionLoss = tf.mean(tf.square(tf.sub(predictedNoise, noise)));

    const centered = tf.sub(current, tf.mean(current, 0, true));
    const moments = tf.moments(centered, 0);
    const standardDeviation = tf.sqrt(tf.add(moments.variance, 1e-4));
    const varianceLoss = tf.mean(tf.relu(tf.sub(1, standardDeviation)));
    const covariance = tf.div(tf.matMul(centered, centered, true, false), Math.max(1, batchSize - 1));
    const offDiagonal = tf.mul(covariance, tf.sub(1, tf.eye(8)));
    const covarianceLoss = tf.div(tf.sum(tf.square(offDiagonal)), 8);

    const inverseLogits = predict(bundle.inverseHead, tf.concat([current, nextOnline], 1));
    const inverseLoss = tf.mean(tf.losses.softmaxCrossEntropy(batch.actionOneHot, inverseLogits));
    const positionLoss = tf.mean(tf.square(tf.sub(predict(bundle.positionHead, current), batch.position)));
    const actionPositionLoss = tf.mean(tf.square(tf.sub(predict(bundle.positionHead, deterministic), batch.nextPosition)));
    const gateLoss = sigmoidCrossEntropy(batch.belief, predict(bundle.gateHead, current));
    const goalLoss = sigmoidCrossEntropy(batch.goal, predict(bundle.goalHead, current));

    return sumTensors([
      diffusionLoss,
      deterministicLoss,
      tf.mul(varianceLoss, 0.2),
      tf.mul(covarianceLoss, 0.05),
      tf.mul(inverseLoss, 0.5),
      tf.mul(positionLoss, 5),
      tf.mul(actionPositionLoss, 5),
      tf.mul(gateLoss, 0.35),
      tf.mul(goalLoss, 0.1),
    ]) as tf.Scalar;
  });
}

function disposeBatch(batch: TransitionBatch) {
  batch.observation.dispose();
  batch.nextObservation.dispose();
  batch.actionOneHot.dispose();
  batch.belief.dispose();
  batch.position.dispose();
  batch.nextPosition.dispose();
  batch.goal.dispose();
}

function trainableVariables(bundle: ModelBundle) {
  return [
    bundle.encoder,
    bundle.denoiser,
    bundle.deterministicPredictor,
    bundle.inverseHead,
    bundle.positionHead,
    bundle.gateHead,
    bundle.goalHead,
  ].flatMap((model) => model.trainableWeights.map((weight) => (weight as unknown as { val: tf.Variable }).val));
}

function updateTargetEncoder(bundle: ModelBundle, momentum = 0.99) {
  const online = bundle.encoder.getWeights();
  const target = bundle.targetEncoder.getWeights();
  const updated = online.map((weight, index) => tf.tidy(() => tf.add(tf.mul(target[index], momentum), tf.mul(weight, 1 - momentum))));
  bundle.targetEncoder.setWeights(updated);
  updated.forEach((weight) => weight.dispose());
}

function disposeBundle(bundle: ModelBundle) {
  COMPONENT_NAMES.forEach((name) => bundle[name].dispose());
}

function throwIfAborted(signal: AbortSignal) {
  if (signal.aborted) throw new DOMException("Training cancelled", "AbortError");
}

async function trainOneSeed(
  seed: number,
  definition: TrainingDefinition,
  datasetSeed: number,
  startStep: number,
  totalSteps: number,
  onProgress: (progress: BrowserTrainingProgress) => void,
  signal: AbortSignal,
) {
  const bundle = createBundle(seed);
  const variables = trainableVariables(bundle);
  const optimizer = tf.train.adam(3e-4);
  const trainingTransitions = Math.floor(definition.transitions * 0.8);
  const batchesPerEpoch = Math.ceil(trainingTransitions / definition.batchSize);
  let completedSteps = startStep;
  let lastLoss = Number.NaN;
  try {
    for (let epoch = 1; epoch <= definition.epochs; epoch += 1) {
      // Recreate the same finite dataset each epoch instead of silently increasing
      // the advertised transition count. The generator is streamed to cap memory.
      const stream = new TransitionStream(datasetSeed);
      for (let batchIndex = 0; batchIndex < batchesPerEpoch; batchIndex += 1) {
        throwIfAborted(signal);
        const size = Math.min(definition.batchSize, trainingTransitions - batchIndex * definition.batchSize || definition.batchSize);
        const batch = stream.batch(size);
        const gradientResult = tf.variableGrads(() => batchLoss(bundle, batch, seed * 1_000_000 + epoch * 10_000 + batchIndex), variables);
        lastLoss = (await gradientResult.value.data())[0];
        if (!Number.isFinite(lastLoss)) {
          gradientResult.value.dispose();
          Object.values(gradientResult.grads).forEach((gradient) => gradient.dispose());
          disposeBatch(batch);
          throw new Error(`Training became numerically unstable at epoch ${epoch}, batch ${batchIndex + 1}.`);
        }
        const gradients = variables.map((variable) => gradientResult.grads[variable.name]);
        const clipped = tf.tidy(() => {
          const norm = tf.sqrt(sumTensors(gradients.map((gradient) => tf.sum(tf.square(gradient)))));
          const scale = tf.minimum(1, tf.div(2, tf.add(norm, 1e-6)));
          return gradients.map((gradient) => tf.mul(gradient, scale));
        });
        optimizer.applyGradients(variables.map((variable, index) => ({ name: variable.name, tensor: clipped[index] })));
        tf.tidy(() => variables.forEach((variable) => variable.assign(tf.mul(variable, 1 - 3e-4 * 1e-4))));
        updateTargetEncoder(bundle);
        gradientResult.value.dispose();
        Object.values(gradientResult.grads).forEach((gradient) => gradient.dispose());
        clipped.forEach((gradient) => gradient.dispose());
        disposeBatch(batch);
        completedSteps += 1;
        if (batchIndex % 4 === 0 || batchIndex === batchesPerEpoch - 1) {
          onProgress({
            stage: "training",
            message: `Learning representations · run ${seed}`,
            fraction: Math.min(0.92, completedSteps / totalSteps * 0.92),
            seed,
            epoch,
            epochs: definition.epochs,
            loss: lastLoss,
          });
          await tf.nextFrame();
        }
      }
    }
    const validationLoss = await evaluateValidation(bundle, definition, datasetSeed + 90_000, seed, signal);
    return { bundle, validationLoss, completedSteps };
  } catch (error) {
    disposeBundle(bundle);
    throw error;
  } finally {
    optimizer.dispose();
  }
}

async function evaluateValidation(bundle: ModelBundle, definition: TrainingDefinition, seed: number, modelSeed: number, signal: AbortSignal) {
  const validationTransitions = Math.max(definition.batchSize, Math.floor(definition.transitions * 0.2));
  const batches = Math.min(20, Math.ceil(validationTransitions / definition.batchSize));
  const stream = new TransitionStream(seed);
  let total = 0;
  for (let index = 0; index < batches; index += 1) {
    throwIfAborted(signal);
    const batch = stream.batch(definition.batchSize);
    const loss = batchLoss(bundle, batch, modelSeed * 10_000 + index);
    total += (await loss.data())[0];
    loss.dispose();
    disposeBatch(batch);
    if (index % 4 === 0) await tf.nextFrame();
  }
  return total / batches;
}

async function assessQuality(bundle: ModelBundle, seed: number): Promise<QualityResult> {
  const batch = new TransitionStream(seed).batch(512);
  const tensors = tf.tidy(() => {
    const current = predict(bundle.encoder, batch.observation);
    const target = predict(bundle.targetEncoder, batch.nextObservation);
    const correct = tf.add(current, predict(bundle.deterministicPredictor, tf.concat([current, batch.actionOneHot, batch.belief], 1)));
    const shiftedIndices = Int32Array.from(batch.actionIndices, (value) => (value + 1) % 4);
    const shiftedActions = tf.oneHot(tf.tensor1d(shiftedIndices, "int32"), 4);
    const shuffled = tf.add(current, predict(bundle.deterministicPredictor, tf.concat([current, shiftedActions, batch.belief], 1)));
    const correctMse = tf.mean(tf.square(tf.sub(correct, target)));
    const shuffledMse = tf.mean(tf.square(tf.sub(shuffled, target)));
    const moments = tf.moments(current, 0);
    const standardDeviation = tf.sqrt(tf.add(moments.variance, 1e-4));
    const positionMse = tf.mean(tf.square(tf.sub(predict(bundle.positionHead, current), batch.position)));
    const gatePrediction = tf.sigmoid(predict(bundle.gateHead, current));
    const gateBrier = tf.mean(tf.square(tf.sub(gatePrediction, batch.belief)));
    return { correctMse, shuffledMse, mean: moments.mean, standardDeviation, positionMse, gateBrier };
  });
  const [correctMse, shuffledMse, mean, standardDeviation, positionMse, gateBrier] = await Promise.all([
    tensors.correctMse.data(),
    tensors.shuffledMse.data(),
    tensors.mean.data(),
    tensors.standardDeviation.data(),
    tensors.positionMse.data(),
    tensors.gateBrier.data(),
  ]);
  Object.values(tensors).forEach((tensor) => tensor.dispose());
  disposeBatch(batch);
  const actionRatio = shuffledMse[0] / Math.max(1e-8, correctMse[0]);
  const minStd = Math.min(...standardDeviation);
  const checks: BrowserQualityCheck[] = [
    { key: "actionConditioning", label: "Action changes prediction", value: actionRatio, threshold: "≥ 1.20×", passed: actionRatio >= 1.2 },
    { key: "latentVariation", label: "Latent avoids collapse", value: minStd, threshold: "≥ 0.25", passed: minStd >= 0.25 },
    { key: "positionAccuracy", label: "Position remains readable", value: positionMse[0], threshold: "≤ 0.03 MSE", passed: positionMse[0] <= 0.03 },
    { key: "beliefAccuracy", label: "Gate belief remains readable", value: gateBrier[0], threshold: "≤ 0.08 Brier", passed: gateBrier[0] <= 0.08 },
  ];
  const dimensions = Array.from(standardDeviation, (value, index) => ({ value, index }))
    .sort((left, right) => right.value - left.value)
    .slice(0, 2)
    .map((entry) => entry.index) as [number, number];
  return {
    passed: checks.every((check) => check.passed),
    checks,
    projection: {
      mean: Array.from(mean),
      standardDeviation: Array.from(standardDeviation),
      dimensions,
    },
  };
}

async function saveBundle(bundle: ModelBundle, modelId: string) {
  for (const name of COMPONENT_NAMES) {
    await bundle[name].save(componentUrl(modelId, name));
  }
}

async function removeStoredBundle(modelId: string) {
  for (const name of COMPONENT_NAMES) {
    try {
      await tf.io.removeModel(componentUrl(modelId, name));
    } catch {
      // Best-effort cleanup after an atomically saved replacement.
    }
  }
}

export async function trainBrowserModel(
  mode: BrowserTrainingMode,
  onProgress: (progress: BrowserTrainingProgress) => void,
  signal: AbortSignal,
): Promise<BrowserModelMetadata> {
  const definition = DEFINITIONS[mode];
  const backend = await prepareTrainingBackend();
  onProgress({ stage: "preparing", message: "Building the JEPA training graph", fraction: 0.01, seed: null, epoch: 0, epochs: definition.epochs, loss: null });
  await tf.nextFrame();
  const trainingSteps = definition.seeds.length * definition.epochs * Math.ceil(Math.floor(definition.transitions * 0.8) / definition.batchSize);
  const trained: Array<{ seed: number; validationLoss: number; bundle: ModelBundle }> = [];
  let completedSteps = 0;
  try {
    for (const seed of definition.seeds) {
      throwIfAborted(signal);
      const result = await trainOneSeed(seed, definition, 20260818, completedSteps, trainingSteps, onProgress, signal);
      completedSteps = result.completedSteps;
      trained.push({ seed, validationLoss: result.validationLoss, bundle: result.bundle });
      onProgress({ stage: "validating", message: `Validation complete · run ${seed}`, fraction: Math.min(0.94, completedSteps / trainingSteps * 0.92 + 0.02), seed, epoch: definition.epochs, epochs: definition.epochs, loss: result.validationLoss });
    }
    const ordered = [...trained].sort((left, right) => left.validationLoss - right.validationLoss);
    const selected = ordered[Math.floor(ordered.length / 2)];
    onProgress({ stage: "validating", message: "Running scientific quality gates", fraction: 0.95, seed: selected.seed, epoch: definition.epochs, epochs: definition.epochs, loss: selected.validationLoss });
    const quality = await assessQuality(selected.bundle, 20260818 + 180_000);
    throwIfAborted(signal);
    onProgress({ stage: "saving", message: "Saving the candidate in this browser", fraction: 0.98, seed: selected.seed, epoch: definition.epochs, epochs: definition.epochs, loss: selected.validationLoss });
    const previousMetadata = readBrowserModelMetadata();
    const modelId = `${Date.now().toString(36)}-${selected.seed}`;
    await saveBundle(selected.bundle, modelId);
    const metadata: BrowserModelMetadata = {
      version: 1,
      modelId,
      createdAt: new Date().toISOString(),
      backend: backend.id,
      mode,
      transitions: definition.transitions,
      epochs: definition.epochs,
      selectedSeed: selected.seed,
      validationLoss: selected.validationLoss,
      qualityPassed: quality.passed,
      checks: quality.checks,
      projection: quality.projection,
    };
    deactivateBrowserModel();
    writeBrowserModelMetadata(metadata);
    if (previousMetadata?.modelId && previousMetadata.modelId !== modelId) {
      void removeStoredBundle(previousMetadata.modelId);
    }
    onProgress({ stage: "complete", message: quality.passed ? "Training and validation complete" : "Training complete · quality gates need attention", fraction: 1, seed: selected.seed, epoch: definition.epochs, epochs: definition.epochs, loss: selected.validationLoss });
    return metadata;
  } finally {
    trained.forEach((run) => disposeBundle(run.bundle));
  }
}

/** Runs one real forward/backward update without persisting a model. Used by automated verification. */
export async function runBrowserTrainingSmokeTest() {
  await tf.setBackend("cpu");
  await tf.ready();
  const bundle = createBundle(71);
  const variables = trainableVariables(bundle);
  const batch = new TransitionStream(20260818).batch(16);
  const optimizer = tf.train.adam(3e-4);
  try {
    const result = tf.variableGrads(() => batchLoss(bundle, batch, 710001), variables);
    const loss = (await result.value.data())[0];
    const gradients = variables.map((variable) => result.grads[variable.name]);
    const gradientNormTensor = tf.tidy(() => tf.sqrt(sumTensors(gradients.map((gradient) => tf.sum(tf.square(gradient))))));
    const gradientNorm = (await gradientNormTensor.data())[0];
    optimizer.applyGradients(variables.map((variable, index) => ({ name: variable.name, tensor: gradients[index] })));
    updateTargetEncoder(bundle);
    result.value.dispose();
    gradientNormTensor.dispose();
    Object.values(result.grads).forEach((gradient) => gradient.dispose());
    return {
      loss,
      gradientNorm,
      parameterCount: COMPONENT_NAMES.reduce((total, name) => total + bundle[name].countParams(), 0),
    };
  } finally {
    optimizer.dispose();
    disposeBatch(batch);
    disposeBundle(bundle);
  }
}

async function loadInferenceModels(modelId: string) {
  if (inferenceCache?.modelId === modelId) return inferenceCache;
  if (inferenceCache) {
    inferenceCache.encoder.dispose();
    inferenceCache.denoiser.dispose();
    inferenceCache = null;
  }
  await prepareTrainingBackend();
  const [encoder, denoiser] = await Promise.all([
    tf.loadLayersModel(componentUrl(modelId, "encoder")),
    tf.loadLayersModel(componentUrl(modelId, "denoiser")),
  ]);
  inferenceCache = { modelId, encoder, denoiser };
  return inferenceCache;
}

export async function activateBrowserModel() {
  const metadata = readBrowserModelMetadata();
  if (!metadata?.qualityPassed) throw new Error("The browser model has not passed validation.");
  await loadInferenceModels(metadata.modelId);
  setBrowserModelActiveFlag(true);
}

export function deactivateBrowserModel() {
  setBrowserModelActiveFlag(false);
  if (inferenceCache) {
    inferenceCache.encoder.dispose();
    inferenceCache.denoiser.dispose();
    inferenceCache = null;
  }
}

function renderFrameObservation(world: WorldState, belief: number) {
  const state: SimulatorState = {
    position: [...world.agent],
    upperGate: [...world.upperGate.position],
    lowerGate: [...world.lowerGate.position],
    sensor: [...world.sensor],
    hiddenUpper: true,
    sensorVisited: world.sensorVisited,
    cue: world.cue,
    belief,
    steps: 0,
  };
  return renderObservation(state);
}

export async function applyActiveBrowserModel(frame: StepFrame): Promise<StepFrame> {
  if (!isBrowserModelActive()) return frame;
  const metadata = readBrowserModelMetadata();
  if (!metadata) return frame;
  const models = await loadInferenceModels(metadata.modelId);
  const actionIndex: Record<string, number> = { right: 0, up: 1, down: 2, left: 3 };
  const samplesPerAction = 16;
  const actionCount = frame.candidates.length;
  const samples = tf.tidy(() => {
    const observation = tf.tensor4d(renderFrameObservation(frame.world, frame.belief.upper), [1, 32, 32, 3]);
    const currentSingle = models.encoder.predict(observation) as tf.Tensor2D;
    const current = tf.tile(currentSingle, [actionCount * samplesPerAction, 1]);
    const actionValues = frame.candidates.flatMap((candidate) => Array(samplesPerAction).fill(actionIndex[candidate.action]));
    const actions = tf.oneHot(tf.tensor1d(actionValues, "int32"), 4);
    const beliefs = tf.fill([actionCount * samplesPerAction, 1], frame.belief.upper);
    let latent = tf.randomNormal([actionCount * samplesPerAction, 8], 0, 1, "float32", frame.step * 97 + 17);
    const betas = tf.linspace(0.025, 0.18, 8);
    const alphas = tf.sub(1, betas);
    const alphaBars = tf.cumprod(alphas);
    for (let step = 7; step >= 0; step -= 1) {
      const time = tf.fill([actionCount * samplesPerAction, 1], step / 7);
      const predictedNoise = models.denoiser.predict(tf.concat([latent, current, actions, beliefs, time], 1)) as tf.Tensor2D;
      const alpha = alphas.slice([step], [1]);
      const alphaBar = alphaBars.slice([step], [1]);
      const beta = betas.slice([step], [1]);
      const next = tf.div(tf.sub(latent, tf.mul(tf.div(tf.sub(1, alpha), tf.sqrt(tf.sub(1, alphaBar))), predictedNoise)), tf.sqrt(alpha));
      const withNoise = step > 0 ? tf.add(next, tf.mul(tf.sqrt(beta), tf.randomNormal(latent.shape, 0, 1, "float32", frame.step * 1_000 + step))) : next;
      latent.dispose();
      latent = withNoise;
    }
    return latent;
  });
  const values = await samples.array() as number[][];
  samples.dispose();
  const [firstDimension, secondDimension] = metadata.projection.dimensions;
  const project = (row: number): [number, number] => {
    const vector = values[row];
    const first = (vector[firstDimension] - metadata.projection.mean[firstDimension]) / Math.max(0.1, metadata.projection.standardDeviation[firstDimension] * 3);
    const second = (vector[secondDimension] - metadata.projection.mean[secondDimension]) / Math.max(0.1, metadata.projection.standardDeviation[secondDimension] * 3);
    return [Math.max(-1, Math.min(1, first)), Math.max(-1, Math.min(1, second))];
  };
  return {
    ...frame,
    modelInspection: null,
    candidates: frame.candidates.map((candidate, candidateIndex) => ({
      ...candidate,
      latentSamples: Array.from({ length: samplesPerAction }, (_, sampleIndex) => project(candidateIndex * samplesPerAction + sampleIndex)),
    })),
  };
}
