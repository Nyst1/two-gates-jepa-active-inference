import type { Candidate, EpisodeConfig, Position, StepFrame } from "./types";

type Action = Candidate["action"];
type InformationSource = Candidate["informationSource"];
type PlanObjective = Candidate["planObjective"];

const SCHEMA_VERSION = "1.4";
const CHECKPOINT_HASH = "browser-analytic-v1";
const WIDTH = 13;
const HEIGHT = 9;
const WALL_X = 6;
const START: Position = [1, 4];
const GOAL: Position = [11, 4];
const PLANNING_HORIZON = 5;
const FUTURE_SAMPLES = 16;
const ACTIONS: Array<{ name: Action; delta: Position }> = [
  { name: "right", delta: [1, 0] },
  { name: "up", delta: [0, -1] },
  { name: "down", delta: [0, 1] },
  { name: "left", delta: [-1, 0] },
];

interface Layout {
  upperGate: Position;
  lowerGate: Position;
  sensor: Position;
  layoutId: string;
}

interface Transition {
  action: Action;
  positionBefore: Position;
  positionAfter: Position;
  moved: boolean;
  collision: boolean;
  cue: "upper" | "lower" | null;
  revealedGate: "upper" | "lower" | null;
  reachedGoal: boolean;
}

interface Evidence {
  raw: number;
  reach: number;
  value: number;
  target: Position | null;
  stepsAfterAction: number | null;
}

interface PolicyEvidence {
  source: InformationSource;
  sequence: Array<"sensor" | "gate">;
  target: Position | null;
  sensorRaw: number;
  sensorReach: number;
  sensorValue: number;
  gateRaw: number;
  gateReach: number;
  gateValue: number;
  value: number;
}

/** Python's MT19937 behavior, used so browser seeds retain the backend's meaning. */
export class PythonRandom {
  private readonly state = new Uint32Array(624);
  private index = 624;
  private nextGaussian: number | null = null;

  constructor(seed: number) {
    const absolute = BigInt(Math.abs(Math.trunc(seed)));
    const words: number[] = [];
    let remaining = absolute;
    do {
      words.push(Number(remaining & 0xffff_ffffn));
      remaining >>= 32n;
    } while (remaining > 0n);
    this.seedWithArray(words);
  }

  private seedInt(seed: number) {
    this.state[0] = seed >>> 0;
    for (let index = 1; index < 624; index += 1) {
      const previous = this.state[index - 1];
      this.state[index] = (Math.imul(previous ^ (previous >>> 30), 1812433253) + index) >>> 0;
    }
    this.index = 624;
  }

  private seedWithArray(words: number[]) {
    this.seedInt(19650218);
    let stateIndex = 1;
    let wordIndex = 0;
    let remaining = Math.max(624, words.length);
    while (remaining > 0) {
      const previous = this.state[stateIndex - 1];
      this.state[stateIndex] = (
        (this.state[stateIndex] ^ Math.imul(previous ^ (previous >>> 30), 1664525))
        + words[wordIndex]
        + wordIndex
      ) >>> 0;
      stateIndex += 1;
      wordIndex += 1;
      if (stateIndex >= 624) {
        this.state[0] = this.state[623];
        stateIndex = 1;
      }
      if (wordIndex >= words.length) wordIndex = 0;
      remaining -= 1;
    }
    remaining = 623;
    while (remaining > 0) {
      const previous = this.state[stateIndex - 1];
      this.state[stateIndex] = (
        (this.state[stateIndex] ^ Math.imul(previous ^ (previous >>> 30), 1566083941))
        - stateIndex
      ) >>> 0;
      stateIndex += 1;
      if (stateIndex >= 624) {
        this.state[0] = this.state[623];
        stateIndex = 1;
      }
      remaining -= 1;
    }
    this.state[0] = 0x8000_0000;
  }

  private uint32() {
    if (this.index >= 624) {
      for (let index = 0; index < 624; index += 1) {
        const value = (this.state[index] & 0x8000_0000) | (this.state[(index + 1) % 624] & 0x7fff_ffff);
        let twisted = this.state[(index + 397) % 624] ^ (value >>> 1);
        if (value & 1) twisted ^= 0x9908_b0df;
        this.state[index] = twisted >>> 0;
      }
      this.index = 0;
    }
    let value = this.state[this.index++];
    value ^= value >>> 11;
    value ^= (value << 7) & 0x9d2c_5680;
    value ^= (value << 15) & 0xefc6_0000;
    value ^= value >>> 18;
    return value >>> 0;
  }

  random() {
    const high = this.uint32() >>> 5;
    const low = this.uint32() >>> 6;
    return (high * 67_108_864 + low) / 9_007_199_254_740_992;
  }

  private getBits(bits: number) {
    return this.uint32() >>> (32 - bits);
  }

  integer(maxExclusive: number) {
    if (maxExclusive <= 0) throw new Error("Random choice requires a non-empty range.");
    const bits = Math.floor(Math.log2(maxExclusive)) + 1;
    let value = this.getBits(bits);
    while (value >= maxExclusive) value = this.getBits(bits);
    return value;
  }

  choice<T>(values: readonly T[]): T {
    return values[this.integer(values.length)];
  }

  gauss(mean = 0, standardDeviation = 1) {
    let value: number;
    if (this.nextGaussian !== null) {
      value = this.nextGaussian;
      this.nextGaussian = null;
    } else {
      const angle = 2 * Math.PI * this.random();
      const radius = Math.sqrt(-2 * Math.log(1 - this.random()));
      value = Math.cos(angle) * radius;
      this.nextGaussian = Math.sin(angle) * radius;
    }
    return mean + value * standardDeviation;
  }
}

function samePosition(left: Position, right: Position) {
  return left[0] === right[0] && left[1] === right[1];
}

function positionKey([x, y]: Position) {
  return `${x},${y}`;
}

function round(value: number) {
  return Number(value.toFixed(5));
}

function clampProbability(value: number) {
  return Math.min(1, Math.max(0, value));
}

export function binaryEntropy(probability: number) {
  const value = clampProbability(probability);
  if (value <= 0 || value >= 1) return 0;
  return -(value * Math.log(value) + (1 - value) * Math.log(1 - value));
}

function posteriorAfterCue(priorUpper: number, cue: "upper" | "lower", reliability: number) {
  const prior = clampProbability(priorUpper);
  const cueIfUpper = cue === "upper" ? reliability : 1 - reliability;
  const cueIfLower = cue === "upper" ? 1 - reliability : reliability;
  const evidence = prior * cueIfUpper + (1 - prior) * cueIfLower;
  return evidence <= 0 ? prior : clampProbability(prior * cueIfUpper / evidence);
}

export function expectedPosteriorEntropy(priorUpper: number, reliability: number) {
  const cueUpperProbability = priorUpper * reliability + (1 - priorUpper) * (1 - reliability);
  return cueUpperProbability * binaryEntropy(posteriorAfterCue(priorUpper, "upper", reliability))
    + (1 - cueUpperProbability) * binaryEntropy(posteriorAfterCue(priorUpper, "lower", reliability));
}

export function stateInformationGain(priorUpper: number, reliability: number) {
  return Math.max(0, binaryEntropy(priorUpper) - expectedPosteriorEntropy(priorUpper, reliability));
}

function layoutFromSeed(seed: number): Layout {
  if (seed === 0) {
    return { upperGate: [6, 2], lowerGate: [6, 6], sensor: [3, 3], layoutId: "g2-6-s3-3" };
  }
  const random = new PythonRandom(seed ^ 0x5eed);
  const upperGate: Position = [WALL_X, random.choice([1, 2, 3])];
  const lowerGate: Position = [WALL_X, random.choice([5, 6, 7])];
  const sensors: Position[] = [];
  for (let x = 2; x < WALL_X; x += 1) {
    for (let y = 1; y < HEIGHT - 1; y += 1) {
      if (x !== START[0] || y !== START[1]) sensors.push([x, y]);
    }
  }
  const sensor = random.choice(sensors);
  return { upperGate, lowerGate, sensor, layoutId: `g${upperGate[1]}-${lowerGate[1]}-s${sensor[0]}-${sensor[1]}` };
}

class LocalEnvironment {
  readonly layout: Layout;
  readonly hiddenUpper: boolean;
  private readonly cueRandom: PythonRandom;
  position: Position = [...START];
  stepCount = 0;
  sensorVisited = false;
  lastCue: "upper" | "lower" | null = null;
  revealedUpper: boolean | null = null;
  collisions = 0;
  path: Position[] = [[...START]];

  constructor(readonly seed: number, readonly priorUpper: number, readonly cueReliability: number) {
    this.layout = layoutFromSeed(seed);
    this.hiddenUpper = new PythonRandom(seed ^ 0xa11ce).random() < priorUpper;
    this.cueRandom = new PythonRandom(seed ^ 0xc0de);
  }

  get done() {
    return samePosition(this.position, GOAL);
  }

  boundaryWalls() {
    const result = new Map<string, Position>();
    for (let x = 0; x < WIDTH; x += 1) {
      result.set(positionKey([x, 0]), [x, 0]);
      result.set(positionKey([x, HEIGHT - 1]), [x, HEIGHT - 1]);
    }
    for (let y = 0; y < HEIGHT; y += 1) {
      result.set(positionKey([0, y]), [0, y]);
      result.set(positionKey([WIDTH - 1, y]), [WIDTH - 1, y]);
    }
    return result;
  }

  dividerWalls() {
    const result = new Map<string, Position>();
    for (let y = 1; y < HEIGHT - 1; y += 1) {
      const position: Position = [WALL_X, y];
      if (!samePosition(position, this.layout.upperGate) && !samePosition(position, this.layout.lowerGate)) {
        result.set(positionKey(position), position);
      }
    }
    return result;
  }

  walls(hiddenUpper = this.hiddenUpper) {
    const result = new Map([...this.boundaryWalls(), ...this.dividerWalls()]);
    const closed = hiddenUpper ? this.layout.lowerGate : this.layout.upperGate;
    result.set(positionKey(closed), closed);
    return result;
  }

  isBlocked(position: Position, hiddenUpper = this.hiddenUpper) {
    return this.walls(hiddenUpper).has(positionKey(position));
  }

  shortestPath(start: Position, goal: Position = GOAL, hiddenUpper = this.hiddenUpper, firstAction?: Action) {
    const walls = this.walls(hiddenUpper);
    let initial: Position = [...start];
    const prefix: Position[] = [[...start]];
    if (firstAction) {
      const delta = ACTIONS.find((item) => item.name === firstAction)!.delta;
      const proposed: Position = [start[0] + delta[0], start[1] + delta[1]];
      initial = walls.has(positionKey(proposed)) ? [...start] : proposed;
      prefix.push(initial);
    }
    if (samePosition(initial, goal)) return prefix;
    const queue: Position[] = [initial];
    const parents = new Map<string, Position | null>([[positionKey(initial), null]]);
    let cursor = 0;
    while (cursor < queue.length) {
      const current = queue[cursor++];
      if (samePosition(current, goal)) break;
      for (const { delta } of ACTIONS) {
        const neighbor: Position = [current[0] + delta[0], current[1] + delta[1]];
        const key = positionKey(neighbor);
        if (walls.has(key) || parents.has(key)) continue;
        parents.set(key, current);
        queue.push(neighbor);
      }
    }
    if (!parents.has(positionKey(goal))) return prefix;
    const suffix: Position[] = [];
    let current: Position | null = goal;
    while (current) {
      suffix.push(current);
      current = parents.get(positionKey(current)) ?? null;
    }
    suffix.reverse();
    return prefix.slice(0, -1).concat(suffix);
  }

  publicWalls(): Position[] {
    return Array.from(new Map([...this.boundaryWalls(), ...this.dividerWalls()]).values())
      .sort((left, right) => left[0] - right[0] || left[1] - right[1]);
  }

  step(action: Action): Transition {
    const before: Position = [...this.position];
    const delta = ACTIONS.find((item) => item.name === action)!.delta;
    const proposed: Position = [before[0] + delta[0], before[1] + delta[1]];
    const collision = this.isBlocked(proposed);
    if (collision) this.collisions += 1;
    else this.position = proposed;
    let revealedGate: "upper" | "lower" | null = null;
    if (samePosition(proposed, this.layout.upperGate) || samePosition(proposed, this.layout.lowerGate)) {
      const observedUpper = samePosition(proposed, this.layout.upperGate);
      this.revealedUpper = collision ? !observedUpper : observedUpper;
      revealedGate = this.revealedUpper ? "upper" : "lower";
    }
    let cue: "upper" | "lower" | null = null;
    if (samePosition(this.position, this.layout.sensor) && !this.sensorVisited) {
      this.sensorVisited = true;
      const truthful = this.hiddenUpper ? "upper" : "lower";
      cue = this.cueRandom.random() <= this.cueReliability ? truthful : truthful === "upper" ? "lower" : "upper";
      this.lastCue = cue;
    }
    this.stepCount += 1;
    this.path.push([...this.position]);
    return {
      action,
      positionBefore: before,
      positionAfter: [...this.position],
      moved: !samePosition(before, this.position),
      collision,
      cue,
      revealedGate,
      reachedGoal: this.done,
    };
  }
}

function nextPosition(environment: LocalEnvironment, action: Action, hiddenUpper: boolean): [Position, boolean] {
  const delta = ACTIONS.find((item) => item.name === action)!.delta;
  const proposed: Position = [environment.position[0] + delta[0], environment.position[1] + delta[1]];
  const blocked = environment.isBlocked(proposed, hiddenUpper);
  return [blocked ? [...environment.position] : proposed, blocked];
}

function pathViaSensor(environment: LocalEnvironment, action: Action, hiddenUpper: boolean) {
  const first = environment.shortestPath(environment.position, environment.layout.sensor, hiddenUpper, action);
  if (!samePosition(first[first.length - 1], environment.layout.sensor)) {
    return environment.shortestPath(environment.position, GOAL, hiddenUpper, action);
  }
  const second = environment.shortestPath(environment.layout.sensor, GOAL, hiddenUpper);
  return first.concat(second.slice(1));
}

function pathViaGate(environment: LocalEnvironment, action: Action, hiddenUpper: boolean, targetGate: Position) {
  const delta = ACTIONS.find((item) => item.name === action)!.delta;
  const proposed: Position = [environment.position[0] + delta[0], environment.position[1] + delta[1]];
  const [next] = nextPosition(environment, action, hiddenUpper);
  let prefix: Position[] = [[...environment.position], next];
  if (samePosition(proposed, targetGate)) {
    return prefix.concat(environment.shortestPath(next, GOAL, hiddenUpper).slice(1));
  }
  const approach: Position = [targetGate[0] - 1, targetGate[1]];
  const approachPath = environment.shortestPath(next, approach, hiddenUpper);
  if (!samePosition(approachPath[approachPath.length - 1], approach)) {
    return environment.shortestPath(environment.position, GOAL, hiddenUpper, action);
  }
  prefix = prefix.slice(0, -1).concat(approachPath);
  const testingUpper = samePosition(targetGate, environment.layout.upperGate);
  const gateOpen = testingUpper ? hiddenUpper : !hiddenUpper;
  if (gateOpen) return prefix.concat(environment.shortestPath(targetGate, GOAL, hiddenUpper));
  return prefix.concat([approach], environment.shortestPath(approach, GOAL, hiddenUpper).slice(1));
}

function gateStatus(environment: LocalEnvironment, beliefUpper: number, upper: boolean): "unknown" | "open" | "closed" {
  if (environment.revealedUpper !== null) {
    const open = upper ? environment.revealedUpper : !environment.revealedUpper;
    return open ? "open" : "closed";
  }
  const confidence = upper ? beliefUpper : 1 - beliefUpper;
  if (confidence >= 0.995) return "open";
  if (confidence <= 0.005) return "closed";
  return "unknown";
}

export class LocalEpisode {
  readonly config: EpisodeConfig;
  readonly environment: LocalEnvironment;
  private beliefUpper: number;
  private wrongGateCommitments = 0;
  private gateTests = 0;
  private lastFrame: StepFrame | null = null;

  constructor(config: EpisodeConfig) {
    this.config = {
      ...config,
      seed: Math.trunc(config.seed),
      beta: Math.min(8, Math.max(0, config.beta)),
      prior: Math.min(0.99, Math.max(0.01, config.prior)),
      cueReliability: Math.min(1, Math.max(0.5, config.cueReliability)),
      source: "local",
    };
    this.beliefUpper = this.config.prior;
    this.environment = new LocalEnvironment(this.config.seed, this.config.prior, this.config.cueReliability);
  }

  get source() {
    return "local" as const;
  }

  private get done() {
    return this.environment.done || this.environment.stepCount >= 48;
  }

  private gateTestingRestricted() {
    return this.config.gateTesting === "prohibited"
      && !this.environment.sensorVisited
      && this.environment.revealedUpper === null
      && Math.max(this.beliefUpper, 1 - this.beliefUpper) < 0.98;
  }

  private prohibitedGateContact(action: Action) {
    if (!this.gateTestingRestricted()) return false;
    const delta = ACTIONS.find((item) => item.name === action)!.delta;
    const proposed: Position = [this.environment.position[0] + delta[0], this.environment.position[1] + delta[1]];
    return samePosition(proposed, this.environment.layout.upperGate) || samePosition(proposed, this.environment.layout.lowerGate);
  }

  private expectedPreferenceCost(action: Action): [number, number] {
    let weightedDistance = 0;
    let collisionProbability = 0;
    for (const [hiddenUpper, probability] of [[true, this.beliefUpper], [false, 1 - this.beliefUpper]] as const) {
      const [next, collision] = nextPosition(this.environment, action, hiddenUpper);
      const path = this.gateTestingRestricted()
        ? pathViaSensor(this.environment, action, hiddenUpper)
        : this.environment.shortestPath(next, GOAL, hiddenUpper);
      weightedDistance += probability * Math.max(0, path.length - 1);
      collisionProbability += probability * Number(collision);
    }
    const delta = ACTIONS.find((item) => item.name === action)!.delta;
    const proposed: Position = [this.environment.position[0] + delta[0], this.environment.position[1] + delta[1]];
    const knownWall = this.environment.boundaryWalls().has(positionKey(proposed)) || this.environment.dividerWalls().has(positionKey(proposed));
    return [weightedDistance / (WIDTH + HEIGHT) + (knownWall ? 0.8 : 0) + (this.prohibitedGateContact(action) ? 4 : 0), collisionProbability];
  }

  private sensorEvidence(action: Action): Evidence {
    const raw = this.environment.sensorVisited || binaryEntropy(this.beliefUpper) <= 1e-9
      ? 0
      : stateInformationGain(this.beliefUpper, this.config.cueReliability);
    const [upper] = nextPosition(this.environment, action, true);
    const [lower] = nextPosition(this.environment, action, false);
    const upperPath = this.environment.shortestPath(upper, this.environment.layout.sensor, true);
    const lowerPath = this.environment.shortestPath(lower, this.environment.layout.sensor, false);
    const steps = this.beliefUpper * Math.max(0, upperPath.length - 1)
      + (1 - this.beliefUpper) * Math.max(0, lowerPath.length - 1);
    const reach = Math.max(0, 1 - steps / (PLANNING_HORIZON + 1));
    return { raw, reach, value: raw * reach, target: this.environment.layout.sensor, stepsAfterAction: steps };
  }

  private gateTarget(action: Action): [Position | null, number] {
    const delta = ACTIONS.find((item) => item.name === action)!.delta;
    const proposed: Position = [this.environment.position[0] + delta[0], this.environment.position[1] + delta[1]];
    const gates = [this.environment.layout.upperGate, this.environment.layout.lowerGate];
    const direct = gates.find((gate) => samePosition(gate, proposed));
    if (direct) return [direct, 0];
    const [next] = nextPosition(this.environment, action, true);
    const options: Array<[number, Position]> = [];
    for (const gate of gates) {
      const approach: Position = [gate[0] - 1, gate[1]];
      const path = this.environment.shortestPath(next, approach, true);
      if (samePosition(path[path.length - 1], approach)) options.push([Math.max(0, path.length - 1) + 1, gate]);
    }
    options.sort((left, right) => left[0] - right[0] || left[1][1] - right[1][1]);
    return options.length > 0 ? [options[0][1], options[0][0]] : [null, PLANNING_HORIZON + 1];
  }

  private gateEvidence(action: Action): Evidence {
    if (this.gateTestingRestricted()
      || this.environment.revealedUpper !== null
      || this.environment.position[0] >= this.environment.layout.upperGate[0]
      || binaryEntropy(this.beliefUpper) <= 1e-9) {
      return { raw: 0, reach: 0, value: 0, target: null, stepsAfterAction: null };
    }
    const [target, steps] = this.gateTarget(action);
    const raw = binaryEntropy(this.beliefUpper);
    const reach = Math.max(0, 1 - steps / (PLANNING_HORIZON + 1));
    return { raw, reach, value: raw * reach, target, stepsAfterAction: steps };
  }

  private sensorStepsBeforeGate(action: Action, targetGate: Position) {
    if (this.environment.sensorVisited) return null;
    const delta = ACTIONS.find((item) => item.name === action)!.delta;
    const proposed: Position = [this.environment.position[0] + delta[0], this.environment.position[1] + delta[1]];
    if (samePosition(proposed, targetGate)) return null;
    const [next] = nextPosition(this.environment, action, true);
    const approach: Position = [targetGate[0] - 1, targetGate[1]];
    const path = this.environment.shortestPath(next, approach, true);
    if (!samePosition(path[path.length - 1], approach)) return null;
    const index = path.findIndex((position) => samePosition(position, this.environment.layout.sensor));
    return index < 0 ? null : index;
  }

  private policyEvidence(action: Action, sensor: Evidence, gate: Evidence): PolicyEvidence {
    const sensorOnly: PolicyEvidence = {
      source: "sensor", sequence: ["sensor"], target: sensor.target,
      sensorRaw: sensor.raw, sensorReach: sensor.reach, sensorValue: sensor.value,
      gateRaw: 0, gateReach: 0, gateValue: 0, value: sensor.value,
    };
    let gatePlan: PolicyEvidence = {
      source: "gate", sequence: ["gate"], target: gate.target,
      sensorRaw: 0, sensorReach: 0, sensorValue: 0,
      gateRaw: gate.raw, gateReach: gate.reach, gateValue: gate.value, value: gate.value,
    };
    if (gate.target && gate.value > 1e-9 && sensor.raw > 1e-9) {
      const sensorSteps = this.sensorStepsBeforeGate(action, gate.target);
      if (sensorSteps !== null) {
        const sensorReach = Math.max(0, 1 - sensorSteps / (PLANNING_HORIZON + 1));
        const sensorValue = sensor.raw * sensorReach;
        const remainingEntropy = expectedPosteriorEntropy(this.beliefUpper, this.config.cueReliability);
        const gateValue = remainingEntropy * gate.reach;
        gatePlan = {
          source: "sensor_then_gate", sequence: ["sensor", "gate"], target: gate.target,
          sensorRaw: sensor.raw, sensorReach, sensorValue,
          gateRaw: remainingEntropy, gateReach: gate.reach, gateValue, value: sensorValue + gateValue,
        };
      }
    }
    const best = gatePlan.value >= sensorOnly.value ? gatePlan : sensorOnly;
    if (best.value > 1e-9) return best;
    return {
      source: "none", sequence: [], target: null,
      sensorRaw: 0, sensorReach: 0, sensorValue: 0,
      gateRaw: 0, gateReach: 0, gateValue: 0, value: 0,
    };
  }

  private planObjective(source: InformationSource): PlanObjective {
    if (this.gateTestingRestricted()) return "sensor";
    const epistemic = this.config.agentType === "information" || (this.config.agentType === "balanced" && this.config.beta > 0);
    if (epistemic && source === "sensor") return "sensor";
    if (epistemic && (source === "gate" || source === "sensor_then_gate")) return "gate";
    return "goal";
  }

  private sampledPaths(action: Action, objective: PlanObjective, target: Position | null): Candidate["sampledPaths"] {
    const actionIndex = ACTIONS.findIndex((item) => item.name === action);
    const random = new PythonRandom(this.config.seed * 10_000 + this.environment.stepCount * 101 + actionIndex);
    return Array.from({ length: FUTURE_SAMPLES }, () => {
      const hiddenUpper = random.random() < this.beliefUpper;
      const fullPath = objective === "sensor"
        ? pathViaSensor(this.environment, action, hiddenUpper)
        : objective === "gate" && target
          ? pathViaGate(this.environment, action, hiddenUpper, target)
          : this.environment.shortestPath(this.environment.position, GOAL, hiddenUpper, action);
      return {
        hypothesis: hiddenUpper ? "upper" as const : "lower" as const,
        points: fullPath.slice(0, PLANNING_HORIZON + 1),
        objective,
      };
    });
  }

  private latentSamples(action: Action): Array<[number, number]> {
    const actionIndex = ACTIONS.findIndex((item) => item.name === action);
    const random = new PythonRandom(this.config.seed * 1_000 + this.environment.stepCount * 17 + actionIndex);
    return Array.from({ length: FUTURE_SAMPLES }, () => {
      const hiddenUpper = random.random() < this.beliefUpper;
      const [next, collision] = nextPosition(this.environment, action, hiddenUpper);
      const branch = hiddenUpper ? 0.28 : -0.28;
      const uncertainty = binaryEntropy(this.beliefUpper) / Math.log(2);
      return [
        round((next[0] / (WIDTH - 1) - 0.5) * 2 + branch * uncertainty + random.gauss(0, 0.025)),
        round((next[1] / (HEIGHT - 1) - 0.5) * 2 + (collision ? 0.08 : 0) + random.gauss(0, 0.025)),
      ];
    });
  }

  candidates(): Candidate[] {
    const candidates: Candidate[] = ACTIONS.map(({ name: action }) => {
      const [preferenceCost, collisionProbability] = this.expectedPreferenceCost(action);
      const sensor = this.sensorEvidence(action);
      const gate = this.gateEvidence(action);
      const policy = this.policyEvidence(action, sensor, gate);
      const objective = this.planObjective(policy.source);
      const target = objective === "gate" ? policy.target : objective === "sensor" ? this.environment.layout.sensor : null;
      const efeScore = this.config.agentType === "pragmatic"
        ? preferenceCost
        : this.config.agentType === "information"
          ? -policy.value + 0.03 * preferenceCost
          : preferenceCost - this.config.beta * policy.value;
      const prohibited = this.prohibitedGateContact(action);
      return {
        action,
        preferenceCost: round(preferenceCost),
        informationGain: round(policy.value),
        sensorInformationGain: round(sensor.value),
        sensorRawInformationGain: round(sensor.raw),
        sensorReachFactor: round(sensor.reach),
        gateInformationGain: round(gate.value),
        gateRawInformationGain: round(gate.raw),
        gateReachFactor: round(gate.reach),
        scoredSensorInformationGain: round(policy.sensorValue),
        scoredSensorRawInformationGain: round(policy.sensorRaw),
        scoredSensorReachFactor: round(policy.sensorReach),
        scoredGateInformationGain: round(policy.gateValue),
        scoredGateRawInformationGain: round(policy.gateRaw),
        scoredGateReachFactor: round(policy.gateReach),
        informationSource: policy.source,
        informationSequence: policy.sequence,
        planObjective: objective,
        informationTarget: target ? [...target] as Position : null,
        efeScore: round(efeScore),
        collisionProbability: round(collisionProbability),
        sampledPaths: this.sampledPaths(action, objective, target),
        latentSamples: this.latentSamples(action),
        selected: false,
        admissible: !prohibited,
        constraintReason: prohibited ? "sensor evidence required before gate contact" : null,
      };
    });
    candidates.reduce((best, candidate) => candidate.efeScore < best.efeScore ? candidate : best).selected = true;
    return candidates;
  }

  private world(positionBefore?: Position): StepFrame["world"] {
    return {
      width: WIDTH,
      height: HEIGHT,
      agent: [...this.environment.position],
      previousAgent: [...(positionBefore ?? this.environment.position)],
      start: [...START],
      goal: [...GOAL],
      sensor: [...this.environment.layout.sensor],
      layoutId: this.environment.layout.layoutId,
      upperGate: { position: [...this.environment.layout.upperGate], status: gateStatus(this.environment, this.beliefUpper, true) },
      lowerGate: { position: [...this.environment.layout.lowerGate], status: gateStatus(this.environment, this.beliefUpper, false) },
      walls: this.environment.publicWalls(),
      sensorVisited: this.environment.sensorVisited,
      cue: this.environment.lastCue,
      cueReliability: this.config.cueReliability,
      gateTesting: this.config.gateTesting,
      gateTestingRestricted: this.gateTestingRestricted(),
      reachedGoal: this.environment.done,
    };
  }

  private metrics(): StepFrame["metrics"] {
    return {
      steps: this.environment.stepCount,
      collisions: this.environment.collisions,
      wrongGateCommitments: this.wrongGateCommitments,
      gateTests: this.gateTests,
      pathLength: Math.max(0, this.environment.path.length - 1),
      success: this.environment.done,
    };
  }

  initialFrame(): StepFrame {
    const candidates = this.candidates();
    this.lastFrame = {
      schemaVersion: SCHEMA_VERSION,
      checkpointHash: CHECKPOINT_HASH,
      source: "local",
      step: 0,
      phase: "observe",
      world: this.world(),
      belief: {
        upper: this.beliefUpper,
        lower: 1 - this.beliefUpper,
        entropyBefore: binaryEntropy(this.beliefUpper),
        entropyAfter: binaryEntropy(this.beliefUpper),
        updateReason: "prior",
      },
      candidates,
      selectedAction: candidates.find((candidate) => candidate.selected)!.action,
      modelInspection: null,
      transition: null,
      metrics: this.metrics(),
      done: this.done,
    };
    return this.lastFrame;
  }

  step(): StepFrame {
    if (this.done) return this.lastFrame ?? this.initialFrame();
    const candidates = this.candidates();
    const selected = candidates.find((candidate) => candidate.selected)!;
    const beliefBefore = this.beliefUpper;
    const deliberateGateTest = this.config.gateTesting === "allowed"
      && selected.planObjective === "gate"
      && (this.config.agentType === "information" || (this.config.agentType === "balanced" && this.config.beta > 0));
    const transition = this.environment.step(selected.action);
    let updateReason = "prediction only";
    if (transition.cue) {
      this.beliefUpper = posteriorAfterCue(beliefBefore, transition.cue, this.config.cueReliability);
      updateReason = `${transition.cue} sensor cue`;
    }
    if (transition.revealedGate) {
      if (deliberateGateTest) this.gateTests += 1;
      if (transition.collision && !deliberateGateTest) this.wrongGateCommitments += 1;
      this.beliefUpper = transition.revealedGate === "upper" ? 1 : 0;
      updateReason = deliberateGateTest
        ? `${transition.revealedGate} gate learned by direct test`
        : `${transition.revealedGate} gate revealed`;
    }
    const displayedCandidates = this.done ? candidates : this.candidates();
    const displayedAction = this.done ? selected.action : displayedCandidates.find((candidate) => candidate.selected)!.action;
    this.lastFrame = {
      schemaVersion: SCHEMA_VERSION,
      checkpointHash: CHECKPOINT_HASH,
      source: "local",
      step: this.environment.stepCount,
      phase: transition.cue || transition.revealedGate ? "update" : transition.reachedGoal ? "act" : "evaluate",
      world: this.world(transition.positionBefore),
      belief: {
        upper: this.beliefUpper,
        lower: 1 - this.beliefUpper,
        entropyBefore: binaryEntropy(beliefBefore),
        entropyAfter: binaryEntropy(this.beliefUpper),
        updateReason,
      },
      candidates: displayedCandidates,
      selectedAction: displayedAction,
      modelInspection: null,
      transition: {
        action: transition.action,
        moved: transition.moved,
        collision: transition.collision,
        cue: transition.cue,
        revealedGate: transition.revealedGate,
        reachedGoal: transition.reachedGoal,
      },
      metrics: this.metrics(),
      done: this.done,
    };
    return this.lastFrame;
  }
}
