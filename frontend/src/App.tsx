import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createEpisode, fetchMeta, fetchReplay, stepEpisode } from "./api";
import { BROWSER_MODEL_CHANGE_EVENT, isBrowserModelActive } from "./browserModelStore";
import ModelInspector from "./ModelInspector";
import ProjectGuide from "./ProjectGuide";
import TrainingStudio from "./TrainingStudio";
import type {
  AgentType,
  Candidate,
  EpisodeConfig,
  MetaResponse,
  ReplayTrace,
  SourceType,
  StepFrame,
} from "./types";
import type { StandaloneEpisode } from "./standaloneRuntime";

const ACTION_SYMBOL: Record<string, string> = { right: "→", up: "↑", down: "↓", left: "←" };
const AGENT_LABEL: Record<AgentType, string> = {
  balanced: "Balanced",
  pragmatic: "Pragmatic only",
  information: "Information only",
};

export type PresentationPhase = "observe" | "imagine" | "evaluate" | "act" | "update";
export const PRESENTATION_PHASES: PresentationPhase[] = ["observe", "imagine", "evaluate", "act", "update"];

export function nextPresentationPhase(phase: PresentationPhase): PresentationPhase {
  const index = PRESENTATION_PHASES.indexOf(phase);
  return PRESENTATION_PHASES[(index + 1) % PRESENTATION_PHASES.length];
}

function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`;
}

function formatNats(value: number) {
  return `${value.toFixed(3)} nats`;
}

function useEpisode(config: EpisodeConfig) {
  const [meta, setMeta] = useState<MetaResponse | null>(null);
  const [frame, setFrame] = useState<StepFrame | null>(null);
  const [episodeId, setEpisodeId] = useState<string | null>(null);
  const [trace, setTrace] = useState<ReplayTrace | null>(null);
  const [traceIndex, setTraceIndex] = useState(0);
  const [runtimeSource, setRuntimeSource] = useState<SourceType | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const requestSerial = useRef(0);
  const standaloneEpisode = useRef<StandaloneEpisode | null>(null);

  const restart = useCallback(async () => {
    const serial = ++requestSerial.current;
    setLoading(true);
    setError(null);
    setFrame(null);
    setTrace(null);
    setTraceIndex(0);
    setEpisodeId(null);
    setRuntimeSource(null);
    standaloneEpisode.current = null;
    if (import.meta.env.MODE === "standalone") {
      const runtimeModule = await import("./standaloneRuntime");
      if (serial !== requestSerial.current) return;
      const runtime = runtimeModule.createStandaloneEpisode(config);
      standaloneEpisode.current = runtime;
      setMeta(runtimeModule.STANDALONE_META);
      setRuntimeSource(runtime.source);
      setFrame(runtime.initialFrame());
      setLoading(false);
      return;
    }
    const discoveredMeta = await fetchMeta();
    if (serial !== requestSerial.current) return;
    setMeta(discoveredMeta);
    if (discoveredMeta?.liveModel.available) {
      try {
        const response = await createEpisode({ ...config, source: "live" });
        if (serial !== requestSerial.current) return;
        setEpisodeId(response.episodeId);
        setFrame(response.frame);
        setLoading(false);
        return;
      } catch (liveError) {
        setError(`Live model unavailable; showing verified replay. ${String(liveError)}`);
      }
    }
    try {
      const replay = await fetchReplay(config.agentType);
      if (serial !== requestSerial.current) return;
      setTrace(replay);
      setFrame(replay.frames[0]);
      setLoading(false);
    } catch (replayError) {
      if (serial !== requestSerial.current) return;
      setError(`Could not load live model or replay: ${String(replayError)}`);
      setLoading(false);
    }
  }, [config.agentType, config.beta, config.cueReliability, config.gateTesting, config.prior, config.seed]);

  useEffect(() => {
    void restart();
  }, [restart]);

  const advance = useCallback(async () => {
    if (!frame || frame.done || loading) return;
    if (standaloneEpisode.current) {
      setFrame(standaloneEpisode.current.step());
      return;
    }
    if (episodeId) {
      setLoading(true);
      try {
        setFrame(await stepEpisode(episodeId));
      } catch (stepError) {
        setError(`Live step failed: ${String(stepError)}`);
      } finally {
        setLoading(false);
      }
      return;
    }
    if (trace) {
      const nextIndex = Math.min(trace.frames.length - 1, traceIndex + 1);
      setTraceIndex(nextIndex);
      setFrame(trace.frames[nextIndex]);
    }
  }, [episodeId, frame, loading, trace, traceIndex]);

  return {
    frame,
    meta,
    loading,
    error,
    restart,
    advance,
    source: runtimeSource ?? (episodeId ? "live" : "replay"),
    replayConfigMismatch:
      runtimeSource === null &&
      !episodeId &&
      trace !== null &&
      (trace.config.seed !== config.seed ||
        trace.config.beta !== config.beta ||
        trace.config.prior !== config.prior ||
        trace.config.cueReliability !== config.cueReliability ||
        trace.config.gateTesting !== config.gateTesting),
  };
}

function SourceBadge({ source, meta, browserModelActive = false }: { source: SourceType; meta: MetaResponse | null; browserModelActive?: boolean }) {
  const live = source === "live";
  if (browserModelActive) {
    return (
      <div className="source-badge is-browser-trained">
        <span className="source-dot" aria-hidden="true" />
        <span>BROWSER-TRAINED WORLD MODEL + ANALYTIC PLANNER</span>
        <small>LOCAL LATENTS · EXACT BELIEF AND POLICY</small>
      </div>
    );
  }
  if (source === "local") {
    return (
      <div className="source-badge is-local">
        <span className="source-dot" aria-hidden="true" />
        <span>IN-BROWSER ANALYTIC PLANNER</span>
        <small>ANALYTIC LATENTS · EXACT BELIEF AND POLICY</small>
      </div>
    );
  }
  return (
    <div className={`source-badge ${live ? "is-live" : "is-replay"}`}>
      <span className="source-dot" aria-hidden="true" />
      <span>{live ? "LIVE MODEL + ANALYTIC PLANNER" : "VERIFIED REPLAY"}</span>
      <small>{live ? `${meta?.liveModel.device.toUpperCase() ?? "CPU"} LATENTS · EXACT BELIEF` : "CHECKPOINT LATENTS · ANALYTIC POLICY"}</small>
    </div>
  );
}

function CycleRail({ phase }: { phase: PresentationPhase }) {
  const stages = ["Observe", "Imagine", "Evaluate", "Act", "Update belief"];
  const activeMap: Record<PresentationPhase, number> = {
    observe: 0,
    imagine: 1,
    evaluate: 2,
    act: 3,
    update: 4,
  };
  const active = activeMap[phase];
  return (
    <ol className="cycle-rail" aria-label="Agent cycle">
      {stages.map((stage, index) => (
        <li className={index === active ? "active" : index < active ? "complete" : ""} aria-current={index === active ? "step" : undefined} key={stage}>
          <span>{String(index + 1).padStart(2, "0")}</span>
          {stage}
        </li>
      ))}
    </ol>
  );
}

function WorldSvg({
  frame,
  compact = false,
  presentationPhase = "evaluate",
}: {
  frame: StepFrame;
  compact?: boolean;
  presentationPhase?: PresentationPhase;
}) {
  const { world } = frame;
  const cell = 1;
  const selected = frame.candidates.find((candidate) => candidate.selected);
  const allPaths = frame.candidates.flatMap((candidate) =>
    candidate.sampledPaths.map((path, index) => ({ candidate, path, index })),
  );
  return (
    <svg
      className={`world-svg ${compact ? "is-compact" : ""}`}
      viewBox={`0 0 ${world.width} ${world.height}`}
      role="img"
      aria-labelledby={`world-title-${frame.step}-${compact}`}
    >
      <title id={`world-title-${frame.step}-${compact}`}>
        Two Gates world at step {frame.step}. The agent is at {world.agent.join(", ")}.
      </title>
      <desc>
        A vertical barrier has two uncertain gates. Translucent lines are analytic policy futures. The emphasized family follows the same objective used in policy scoring.
      </desc>
      <defs>
        <pattern id="fog" width="0.3" height="0.3" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="0.3" className="fog-line" />
        </pattern>
        <filter id="agent-glow" x="-100%" y="-100%" width="300%" height="300%">
          <feGaussianBlur stdDeviation="0.16" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      <rect width={world.width} height={world.height} className="world-field" rx="0.28" />
      {Array.from({ length: world.width + 1 }, (_, x) => (
        <line key={`vx-${x}`} x1={x} y1={0} x2={x} y2={world.height} className="grid-line" />
      ))}
      {Array.from({ length: world.height + 1 }, (_, y) => (
        <line key={`hy-${y}`} x1={0} y1={y} x2={world.width} y2={y} className="grid-line" />
      ))}

      {!compact && ["imagine", "evaluate"].includes(presentationPhase) &&
        allPaths.map(({ candidate, path, index }) => {
          const points = path.points.map(([x, y]) => `${x + 0.5},${y + 0.5}`).join(" ");
          return (
            <polyline
              key={`${candidate.action}-${index}`}
              points={points}
              className={`future-path ${presentationPhase === "evaluate" && candidate.selected ? "is-selected" : ""} ${presentationPhase === "imagine" ? "is-imagining" : ""} ${path.hypothesis}`}
            />
          );
        })}

      {world.walls.map(([x, y]) => (
        <rect key={`wall-${x}-${y}`} x={x + 0.08} y={y + 0.08} width={0.84} height={0.84} rx={0.08} className="wall" />
      ))}

      {[world.upperGate, world.lowerGate].map((gate, index) => {
        const [x, y] = gate.position;
        return (
          <g key={`gate-${index}`} className={`gate ${gate.status}`}>
            <rect x={x + 0.06} y={y + 0.06} width={0.88} height={0.88} rx={0.12} />
            {gate.status === "unknown" && <rect x={x + 0.06} y={y + 0.06} width={0.88} height={0.88} rx={0.12} fill="url(#fog)" />}
            <text x={x + 0.5} y={y + 0.63} textAnchor="middle">
              {gate.status === "unknown" ? "?" : gate.status === "open" ? "○" : "×"}
            </text>
          </g>
        );
      })}

      {selected?.informationTarget && selected.planObjective !== "goal" && !compact && presentationPhase === "evaluate" && (
        <g
          className={`evidence-target ${selected.planObjective}`}
          transform={`translate(${selected.informationTarget[0] + 0.5} ${selected.informationTarget[1] + 0.5})`}
        >
          <circle r={0.53} />
          <text y={-0.68} textAnchor="middle">{selected.planObjective === "gate" ? "TEST TARGET" : "CUE TARGET"}</text>
        </g>
      )}

      <g className={`sensor ${world.sensorVisited ? "visited" : ""}`}>
        <rect x={world.sensor[0] + 0.15} y={world.sensor[1] + 0.15} width={0.7} height={0.7} rx={0.16} />
        <path
          d={`M${world.sensor[0] + 0.32} ${world.sensor[1] + 0.56} Q${world.sensor[0] + 0.5} ${world.sensor[1] + 0.34} ${world.sensor[0] + 0.68} ${world.sensor[1] + 0.56}`}
        />
        <circle cx={world.sensor[0] + 0.5} cy={world.sensor[1] + 0.62} r={0.06} />
      </g>
      {!compact && <text x={world.sensor[0] + 0.5} y={world.sensor[1] - 0.14} textAnchor="middle" className="world-label">SENSOR</text>}

      <g className="goal">
        <rect x={world.goal[0] + 0.14} y={world.goal[1] + 0.14} width={0.72} height={0.72} rx={0.18} />
        <path d={`M${world.goal[0] + 0.32} ${world.goal[1] + 0.52} l0.12 0.13 0.25 -0.3`} />
      </g>
      {!compact && <text x={world.goal[0] + 0.5} y={world.goal[1] - 0.14} textAnchor="middle" className="world-label">PREFERRED STATE</text>}

      {presentationPhase === "act" && frame.transition && !compact && (
        <g className={`action-transition ${frame.transition.collision ? "blocked" : "moved"}`}>
          <line
            x1={world.previousAgent[0] + 0.5}
            y1={world.previousAgent[1] + 0.5}
            x2={world.agent[0] + 0.5}
            y2={world.agent[1] + 0.5}
          />
          <circle cx={world.agent[0] + 0.5} cy={world.agent[1] + 0.5} r={0.48} />
          <text x={world.agent[0] + 0.5} y={world.agent[1] - 0.18} textAnchor="middle">
            {frame.transition.collision ? "BLOCKED" : `${frame.transition.action.toUpperCase()} EXECUTED`}
          </text>
        </g>
      )}
      {selected && !compact && presentationPhase === "evaluate" && (
        <g className="selected-action-marker" transform={`translate(${world.agent[0] + 0.5} ${world.agent[1] + 0.5})`}>
          <circle r={0.6} />
          <text y={-0.82} textAnchor="middle">
            {ACTION_SYMBOL[selected.action]} selected
          </text>
        </g>
      )}
      <g className="agent" filter="url(#agent-glow)" transform={`translate(${world.agent[0] + 0.5} ${world.agent[1] + 0.5})`}>
        <circle r={0.28} />
        <path d="M-0.12 0.02 L0 0.13 L0.16 -0.12" />
      </g>
    </svg>
  );
}

function BeliefPanel({ frame, active = true }: { frame: StepFrame; active?: boolean }) {
  const entropyMax = Math.log(2);
  const entropyRatio = frame.belief.entropyAfter / entropyMax;
  return (
    <section className={`data-section belief-section ${active ? "is-phase-active" : "is-phase-muted"}`} aria-labelledby="belief-title">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Hidden state belief</p>
          <h2 id="belief-title">Which gate is open?</h2>
        </div>
        <span className="entropy-value">H = {formatNats(frame.belief.entropyAfter)}</span>
      </div>
      <div className="belief-track" aria-label={`Upper gate probability ${formatPercent(frame.belief.upper)}`}>
        <div className="belief-upper" style={{ width: `${frame.belief.upper * 100}%` }} />
        <div className="belief-divider" style={{ left: `${frame.belief.upper * 100}%` }} />
      </div>
      <div className="belief-labels">
        <span><i className="upper-swatch" />Upper gate <strong>{formatPercent(frame.belief.upper)}</strong></span>
        <span><i className="lower-swatch" />Lower gate <strong>{formatPercent(frame.belief.lower)}</strong></span>
      </div>
      <div className="entropy-row">
        <span>Uncertainty</span>
        <div className="mini-track"><div style={{ width: `${entropyRatio * 100}%` }} /></div>
        <span>{entropyRatio > 0.75 ? "high" : entropyRatio > 0.25 ? "partial" : "low"}</span>
      </div>
      <p className="update-note">{frame.belief.updateReason}</p>
    </section>
  );
}

function CandidatePanel({ frame, active = true }: { frame: StepFrame; active?: boolean }) {
  const values = frame.candidates.flatMap((candidate) => [candidate.preferenceCost, candidate.informationGain]);
  const scale = Math.max(0.001, ...values);
  const scoreMin = Math.min(...frame.candidates.map((candidate) => candidate.efeScore));
  const scoreMax = Math.max(...frame.candidates.map((candidate) => candidate.efeScore));
  const selected = frame.candidates.find((candidate) => candidate.selected) ?? frame.candidates[0];
  return (
    <section className={`data-section candidate-section ${active ? "is-phase-active" : "is-phase-muted"}`} aria-labelledby="candidate-title">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Analytic policy evaluation</p>
          <h2 id="candidate-title">Why this action?</h2>
        </div>
        <span className="lower-note">lower G is preferred</span>
      </div>
      <div className="candidate-legend" aria-hidden="true">
        <span><i className="preference-key" />Preference cost</span>
        <span><i className="information-key" />Prospective epistemic value</span>
      </div>
      <div className={`gate-testing-note ${frame.world.gateTesting}`}>
        <strong>{frame.world.gateTesting === "allowed" ? "Diagnostic gate contact allowed" : "Sensor-first constraint"}</strong>
        <span>{frame.world.gateTesting === "allowed" ? "Contact reveals the state exactly; a closed gate still blocks movement and counts as a collision." : "An unresolved gate requires sensor evidence or at least 98% prior confidence."}</span>
      </div>
      <div className="candidate-list">
        {frame.candidates.map((candidate) => {
          const scoreRatio = scoreMax === scoreMin ? 0.5 : (candidate.efeScore - scoreMin) / (scoreMax - scoreMin);
          return (
            <div className={`candidate-row ${active && candidate.selected ? "selected" : ""} ${candidate.admissible ? "" : "inadmissible"}`} key={candidate.action}>
              <div className="candidate-action">
                <span>{ACTION_SYMBOL[candidate.action]}</span>
                <strong>{candidate.action}</strong>
              </div>
              <div className="candidate-bars">
                <div className="metric-bar preference"><i style={{ width: `${(candidate.preferenceCost / scale) * 100}%` }} /></div>
                <div className="metric-bar information"><i style={{ width: `${(candidate.informationGain / scale) * 100}%` }} /></div>
              </div>
              <div className="candidate-score">
                <span>G</span>
                <strong>{candidate.efeScore.toFixed(3)}</strong>
                <i style={{ opacity: 0.25 + (1 - scoreRatio) * 0.75 }} />
              </div>
            </div>
          );
        })}
      </div>
      {selected && active && (
        <div className="information-breakdown">
          <div className="information-heading">
            <span>Selected action's epistemic calculation</span>
            <strong>
              scored sequence: {selected.informationSequence.length > 0 ? selected.informationSequence.join(" → ") : "none"}
              {selected.informationSequence.length > 0 ? ` · ${selected.informationGain.toFixed(3)} nats` : ""}
            </strong>
          </div>
          <div className="information-equation">
            <span>Sensor cue <small>{formatPercent(frame.world.cueReliability)} reliable</small></span>
            <code>{selected.scoredSensorRawInformationGain.toFixed(3)} × {selected.scoredSensorReachFactor.toFixed(3)} = {selected.scoredSensorInformationGain.toFixed(3)}</code>
          </div>
          <div className="information-equation">
            <span>Gate contact <small>{selected.informationSource === "sensor_then_gate" ? "remaining uncertainty" : "diagnostic"}</small></span>
            <code>{selected.scoredGateRawInformationGain.toFixed(3)} × {selected.scoredGateReachFactor.toFixed(3)} = {selected.scoredGateInformationGain.toFixed(3)}</code>
          </div>
          <small>Incremental information gains are time-weighted and added in order. Later evidence can only remove the uncertainty that remains.</small>
        </div>
      )}
    </section>
  );
}

function LatentPlot({
  frame,
  active = true,
  browserModelApplied = false,
}: {
  frame: StepFrame;
  active?: boolean;
  browserModelApplied?: boolean;
}) {
  const selected = frame.candidates.find((candidate) => candidate.selected) ?? frame.candidates[0];
  const plottedCandidates = active ? frame.candidates : selected ? [selected] : [];
  const points = plottedCandidates.flatMap((candidate) =>
    candidate.latentSamples.map(([x, y], index) => ({ x, y, action: candidate.action, index })),
  );
  return (
    <section className={`latent-section ${active ? "is-phase-active" : "is-phase-muted"}`} aria-labelledby="latent-title">
      <div className="section-heading compact-heading">
        <div>
          <p className="eyebrow">{browserModelApplied ? "Browser-trained imagination" : "Checkpoint imagination"}</p>
          <h2 id="latent-title">{points.length} latent futures · PCA view</h2>
        </div>
        <span>{active ? "4 candidate actions" : `${ACTION_SYMBOL[selected?.action ?? "right"]} ${selected?.action}`}</span>
      </div>
      <svg className="latent-plot" viewBox="0 0 320 170" role="img" aria-label="Projected latent future samples">
        <line x1="160" y1="12" x2="160" y2="158" />
        <line x1="14" y1="85" x2="306" y2="85" />
        <text x="299" y="79">PC1</text>
        <text x="166" y="20">PC2</text>
        {points.map(({ x, y, action, index }) => (
          <circle className={`latent-${action}`} key={`${action}-${x}-${y}-${index}`} cx={160 + x * 112} cy={85 - y * 58} r={index % 3 === 0 ? 4.5 : 3.2} />
        ))}
      </svg>
      <p>{active
        ? browserModelApplied
          ? "Each candidate action contributes 16 samples from the locally trained browser model before evaluation."
          : "Each candidate action contributes 16 checkpoint samples before evaluation."
        : browserModelApplied
          ? "These latent samples come from the locally trained action-conditioned model."
          : "These latent samples come from the selected checkpoint-conditioned action."} The grid paths and policy score are analytic.</p>
    </section>
  );
}

function phaseInsight(frame: StepFrame, phase: PresentationPhase, browserModelApplied = false) {
  const selected = frame.candidates.find((candidate) => candidate.selected);
  if (phase === "observe") {
    return {
      number: "01",
      title: "Observe the public state.",
      body: `The agent sees its position, the barrier, sensor and goal. The gate state remains hidden; current belief is ${formatPercent(frame.belief.upper)} upper and ${formatPercent(frame.belief.lower)} lower.`,
    };
  }
  if (phase === "imagine") {
    return {
      number: "02",
      title: "Imagine futures before choosing.",
      body: `Each of four actions contributes 16 ${browserModelApplied ? "locally trained latent" : "checkpoint"} samples, while the translucent grid traces show analytic five-step futures under both gate hypotheses. No winner is highlighted yet.`,
    };
  }
  if (phase === "evaluate") {
    if (selected?.informationSource === "sensor_then_gate" && selected.planObjective === "gate") {
      return {
        number: "03",
        title: "Evaluate: earlier evidence improves the policy.",
        body: `The ${selected.action} route reaches the sensor before the diagnostic gate. The cue contributes ${selected.scoredSensorInformationGain.toFixed(3)} nats now, and later gate contact contributes ${selected.scoredGateInformationGain.toFixed(3)} from the remaining uncertainty. Their time-weighted total is ${selected.informationGain.toFixed(3)}, giving ${selected.action} the lowest G.`,
      };
    }
    if (selected?.informationSource === "gate" && selected.planObjective === "gate") {
      return {
        number: "03",
        title: "Evaluate: the nearer gate yields evidence sooner.",
        body: `Gate contact offers ${selected.scoredGateRawInformationGain.toFixed(3)} nats of raw information. Reach discount gives ${selected.scoredGateInformationGain.toFixed(3)}, so ${selected.action} has the lowest G. The highlighted paths and TEST TARGET show that same plan.`,
      };
    }
    if (selected?.informationSource === "sensor" && selected.planObjective === "sensor") {
      return {
        number: "03",
        title: "Evaluate: the sensor's information changes the choice.",
        body: `The selected move is ${selected.action}. Its sensor value is ${selected.scoredSensorRawInformationGain.toFixed(3)} × ${selected.scoredSensorReachFactor.toFixed(3)} = ${selected.scoredSensorInformationGain.toFixed(3)}, and the highlighted paths lead to the CUE TARGET.`,
      };
    }
    return {
      number: "03",
      title: "Evaluate: preference cost determines the move.",
      body: `The selected move is ${selected?.action ?? frame.selectedAction}. It has the lowest G; the highlighted analytic paths show the goal-directed plan being scored.`,
    };
  }
  if (phase === "act") {
    const transition = frame.transition;
    if (!transition) {
      return { number: "04", title: "Act on the selected policy.", body: "The chosen move is now executed in the world." };
    }
    return {
      number: "04",
      title: `Act: ${ACTION_SYMBOL[transition.action] ?? ""} ${transition.action}.`,
      body: transition.collision
        ? "The attempted move was blocked. Position stays fixed; the observation produced by that contact is revealed in the next phase."
        : "The selected action moved the agent one cell. Any resulting cue or gate observation is applied in the next phase.",
    };
  }
  if (frame.world.reachedGoal) {
    return {
      number: "05",
      title: "Update: the preferred state is reached.",
      body: `The episode ends after ${frame.metrics.steps} actions and ${frame.metrics.collisions} collisions.`,
    };
  }
  if (frame.transition?.cue) {
    return {
      number: "05",
      title: `Update: a ${frame.transition.cue} cue changes belief.`,
      body: `Bayesian updating changes entropy from ${formatNats(frame.belief.entropyBefore)} to ${formatNats(frame.belief.entropyAfter)}.`,
    };
  }
  if (frame.belief.updateReason.includes("direct test")) {
    return {
      number: "05",
      title: "Update: gate contact resolves the hidden state.",
      body: `The diagnostic observation changes entropy from ${formatNats(frame.belief.entropyBefore)} to ${formatNats(frame.belief.entropyAfter)}. A closed gate still counted as a blocked action.`,
    };
  }
  return {
    number: "05",
    title: "Update: no new evidence, belief is unchanged.",
    body: `This move produced neither a cue nor gate contact. Entropy remains ${formatNats(frame.belief.entropyAfter)} before the next Observe phase.`,
  };
}

function PlaybackControls({
  playing,
  setPlaying,
  advance,
  restart,
  loading,
  done,
  stepLabel = "Step",
}: {
  playing: boolean;
  setPlaying: (playing: boolean) => void;
  advance: () => void;
  restart: () => void;
  loading: boolean;
  done: boolean;
  stepLabel?: string;
}) {
  return (
    <div className="playback-controls">
      <button className="primary-button" type="button" onClick={() => setPlaying(!playing)} disabled={loading || done}>
        <span aria-hidden="true">{playing ? "Ⅱ" : "▶"}</span>{playing ? "Pause" : "Run agent"}
      </button>
      <button type="button" onClick={advance} disabled={loading || done}>{stepLabel}</button>
      <button type="button" onClick={restart}>Reset</button>
    </div>
  );
}

function EpisodeExperience({ config, lab = false }: { config: EpisodeConfig; lab?: boolean }) {
  const episode = useEpisode(config);
  const [playing, setPlaying] = useState(false);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [presentationPhase, setPresentationPhase] = useState<PresentationPhase>("observe");
  const [decisionFrame, setDecisionFrame] = useState<StepFrame | null>(null);
  const [browserModelActive, setBrowserModelActive] = useState(() => isBrowserModelActive());
  const [browserFrame, setBrowserFrame] = useState<StepFrame | null>(null);
  const pausedForDecision = useRef(false);
  const { frame } = episode;

  useEffect(() => {
    const updateStatus = () => setBrowserModelActive(isBrowserModelActive());
    window.addEventListener(BROWSER_MODEL_CHANGE_EVENT, updateStatus);
    return () => window.removeEventListener(BROWSER_MODEL_CHANGE_EVENT, updateStatus);
  }, []);

  useEffect(() => {
    let cancelled = false;
    if (!frame || !browserModelActive) {
      setBrowserFrame(null);
      return () => { cancelled = true; };
    }
    setBrowserFrame(null);
    void import("./browserTraining")
      .then((module) => module.applyActiveBrowserModel(frame))
      .then((nextFrame) => { if (!cancelled) setBrowserFrame(nextFrame); })
      .catch(() => { if (!cancelled) setBrowserFrame(null); });
    return () => { cancelled = true; };
  }, [browserModelActive, frame]);

  useEffect(() => {
    if (frame?.step === 0 && frame.transition === null) {
      setPresentationPhase("observe");
      setDecisionFrame(null);
      pausedForDecision.current = false;
    }
  }, [frame]);

  const advancePresentation = useCallback(async () => {
    if (!frame || episode.loading) return;
    if (presentationPhase === "evaluate") {
      setDecisionFrame(frame);
      await episode.advance();
      setPresentationPhase("act");
      return;
    }
    if (presentationPhase === "update") {
      if (frame.done) return;
      setDecisionFrame(null);
      setPresentationPhase("observe");
      return;
    }
    setPresentationPhase(nextPresentationPhase(presentationPhase));
  }, [episode.advance, episode.loading, frame, presentationPhase]);

  const restartPresentation = useCallback(() => {
    setPlaying(false);
    setPresentationPhase("observe");
    setDecisionFrame(null);
    pausedForDecision.current = false;
    void episode.restart();
  }, [episode.restart]);

  useEffect(() => {
    if (presentationPhase !== "imagine" && presentationPhase !== "evaluate") {
      setInspectorOpen(false);
    }
  }, [presentationPhase]);

  useEffect(() => {
    if (!playing || !frame) return;
    if (frame.done && presentationPhase === "update") {
      setPlaying(false);
      return;
    }
    const selected = frame.candidates.find((candidate) => candidate.selected);
    const epistemicDecision =
      presentationPhase === "evaluate" &&
      selected?.planObjective !== "goal" &&
      !frame.world.sensorVisited;
    if (!lab && epistemicDecision && !pausedForDecision.current) {
      pausedForDecision.current = true;
      setPlaying(false);
      return;
    }
    const delay: Record<PresentationPhase, number> = {
      observe: 700,
      imagine: 1100,
      evaluate: 1350,
      act: 750,
      update: 900,
    };
    const timer = window.setTimeout(() => void advancePresentation(), delay[presentationPhase]);
    return () => window.clearTimeout(timer);
  }, [advancePresentation, frame, lab, playing, presentationPhase]);

  if (episode.loading && !frame) return <div className="loading-state"><i />Preparing the agent's world model…</div>;
  if (!frame) return <div className="error-state">{episode.error ?? "No episode available."}</div>;
  const browserModelApplied = browserModelActive && browserFrame?.step === frame.step;
  const displayFrame = browserModelApplied ? browserFrame : frame;
  const presentationFrame: StepFrame = presentationPhase === "act" && decisionFrame
    ? {
        ...displayFrame,
        belief: decisionFrame.belief,
        world: {
          ...displayFrame.world,
          upperGate: decisionFrame.world.upperGate,
          lowerGate: decisionFrame.world.lowerGate,
          sensorVisited: decisionFrame.world.sensorVisited,
          cue: decisionFrame.world.cue,
        },
      }
    : displayFrame;
  const insight = phaseInsight(displayFrame, presentationPhase, browserModelApplied);
  const nextPhase = nextPresentationPhase(presentationPhase);
  const nextPhaseLabel = nextPhase === "update" ? "Update belief" : nextPhase[0].toUpperCase() + nextPhase.slice(1);
  return (
    <>
      <div className="experience-toolbar">
        <SourceBadge source={episode.source} meta={episode.meta} browserModelActive={browserModelApplied} />
        <div className="experience-actions">
          <button
            className={`inspector-button ${inspectorOpen ? "active" : ""}`}
            type="button"
            onClick={() => setInspectorOpen((value) => !value)}
            disabled={browserModelApplied || !displayFrame.modelInspection || (presentationPhase !== "imagine" && presentationPhase !== "evaluate")}
            title={browserModelApplied ? "Detailed ANN inspection is available for the pretrained checkpoint" : presentationPhase === "imagine" || presentationPhase === "evaluate" ? "Inspect checkpoint inference" : "ANN activity is presented during Imagine and Evaluate"}
          >
            <span aria-hidden="true">⌁</span> ANN activity
          </button>
          <PlaybackControls
            playing={playing}
            setPlaying={setPlaying}
            advance={() => void advancePresentation()}
            restart={restartPresentation}
            loading={episode.loading}
            done={frame.done && presentationPhase === "update"}
            stepLabel={`Next: ${nextPhaseLabel}`}
          />
        </div>
      </div>
      {episode.error && <p className="runtime-note">{episode.error}</p>}
      {episode.replayConfigMismatch && lab && (
        <p className="runtime-note">Replay mode uses the locked default configuration. Start the local model to apply Lab controls.</p>
      )}
      <CycleRail phase={presentationPhase} />
      <div className="experience-grid">
        <div className="world-column">
          <div className="world-frame">
            <div className="world-topline">
              <span>PUBLIC OBSERVATION</span>
              <span>LAYOUT {frame.world.layoutId} · ACTION {String(frame.step).padStart(2, "0")}</span>
            </div>
            <WorldSvg frame={presentationFrame} presentationPhase={presentationPhase} />
            <div className="world-legend">
              <span><i className="legend-agent" />Agent</span>
              <span><i className="legend-future" />Analytic policy future</span>
              <span><i className="legend-sensor" />Information source</span>
              <span><i className="legend-goal" />Preferred state</span>
            </div>
          </div>
          <aside className="insight-strip">
            <span>{insight.number}</span>
            <div><h2>{insight.title}</h2><p>{insight.body}</p></div>
          </aside>
        </div>
        <div className="data-column">
          <BeliefPanel frame={presentationFrame} active={presentationPhase === "observe" || presentationPhase === "act" || presentationPhase === "update"} />
          <CandidatePanel frame={displayFrame} active={presentationPhase === "evaluate"} />
          <LatentPlot frame={displayFrame} active={presentationPhase === "imagine"} browserModelApplied={browserModelApplied} />
        </div>
      </div>
      {inspectorOpen && <ModelInspector frame={displayFrame} onClose={() => setInspectorOpen(false)} />}
    </>
  );
}

function LabView() {
  const [beta, setBeta] = useState(1);
  const [prior, setPrior] = useState(0.5);
  const [reliability, setReliability] = useState(0.9);
  const [seed, setSeed] = useState(0);
  const [agentType, setAgentType] = useState<AgentType>("balanced");
  const [gateTesting, setGateTesting] = useState<"allowed" | "prohibited">("allowed");
  const config = useMemo<EpisodeConfig>(() => ({
    agentType,
    seed,
    beta,
    prior,
    cueReliability: reliability,
    gateTesting,
    source: "live",
  }), [agentType, beta, gateTesting, prior, reliability, seed]);
  return (
    <main className="lab-main">
      <section className="hero-copy lab-hero">
        <div>
          <p className="eyebrow">EXPERIMENT WITH THE AGENT'S DECISION MODEL</p>
          <h1>Test when information is worth the detour.</h1>
        </div>
        <p>Change the agent's prior, evidence, and objective, then watch how uncertainty changes its policy.</p>
      </section>
      <section className="lab-controls" aria-label="Lab controls">
        <label>
          <span>Agent objective<strong>{AGENT_LABEL[agentType]}</strong></span>
          <select value={agentType} onChange={(event) => setAgentType(event.target.value as AgentType)}>
            <option value="balanced">Balanced</option>
            <option value="pragmatic">Pragmatic only</option>
            <option value="information">Information only</option>
          </select>
        </label>
        <label>
          <span>Epistemic weight β<strong>{beta.toFixed(1)}</strong></span>
          <input type="range" min="0" max="3" step="0.1" value={beta} onChange={(event) => setBeta(Number(event.target.value))} />
        </label>
        <label>
          <span>Prior P(upper)<strong>{formatPercent(prior)}</strong></span>
          <input type="range" min="0.05" max="0.95" step="0.05" value={prior} onChange={(event) => setPrior(Number(event.target.value))} />
        </label>
        <label>
          <span>Cue reliability<strong>{formatPercent(reliability)}</strong></span>
          <input type="range" min="0.5" max="1" step="0.05" value={reliability} onChange={(event) => setReliability(Number(event.target.value))} />
        </label>
        <label>
          <span>Pre-cue gate contact<strong>{gateTesting === "allowed" ? "diagnostic" : "prohibited"}</strong></span>
          <select
            value={gateTesting}
            title="A closed gate reveals the state but blocks the move. Prohibited sends the agent to the sensor first."
            onChange={(event) => setGateTesting(event.target.value as "allowed" | "prohibited")}
          >
            <option value="allowed">Allowed: diagnostic test</option>
            <option value="prohibited">Prohibited: sensor first</option>
          </select>
        </label>
        <label>
          <span>World seed<strong>{seed}</strong></span>
          <input
            type="number"
            min="0"
            max="9999"
            value={seed}
            title="Changes the layout, hidden gate, and cue."
            onChange={(event) => setSeed(Number(event.target.value))}
          />
        </label>
      </section>
      <EpisodeExperience config={config} lab />
    </main>
  );
}

export default function App() {
  const [methodsOpen, setMethodsOpen] = useState(false);
  const [trainingOpen, setTrainingOpen] = useState(false);
  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark"><i /><i /></span>
          <span><strong>TWO GATES</strong><small>JEPA × ACTIVE INFERENCE</small></span>
        </div>
        <nav aria-label="Main views">
          <button type="button" className="active" aria-current="page">Lab</button>
        </nav>
        <div className="header-actions">
          <button className="training-button" type="button" onClick={() => setTrainingOpen(true)}>
            <span aria-hidden="true">↗</span> Train model
          </button>
          <button className="method-button" type="button" onClick={() => setMethodsOpen(true)}>
            <span aria-hidden="true">i</span> Project info
          </button>
        </div>
      </header>
      <LabView />
      <footer>
        <span>Built as an inspectable teaching model.</span>
        <span>Observed state ≠ hidden state · uncertainty is part of the decision</span>
      </footer>
      <ProjectGuide open={methodsOpen} onClose={() => setMethodsOpen(false)} />
      <TrainingStudio open={trainingOpen} onClose={() => setTrainingOpen(false)} />
    </div>
  );
}
