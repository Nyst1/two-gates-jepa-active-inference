import { useMemo, useState } from "react";
import type { ModelInspection, NeuralEdge, NeuralLayer, StepFrame } from "./types";

type InspectorTab = "encoder" | "denoiser" | "readouts";

function topUnitIndices(layer: NeuralLayer, edges: NeuralEdge[], limit = 10) {
  const edgeScores = new Map<number, number>();
  edges.forEach((edge) => {
    if (edge.sourceLayer === layer.id) {
      edgeScores.set(edge.source, Math.max(edgeScores.get(edge.source) ?? 0, Math.abs(edge.contribution)));
    }
    if (edge.targetLayer === layer.id) {
      edgeScores.set(edge.target, Math.max(edgeScores.get(edge.target) ?? 0, Math.abs(edge.contribution)));
    }
  });
  return layer.values
    .map((value, index) => ({ index, score: Math.max(Math.abs(value), edgeScores.get(index) ?? 0) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, Math.min(limit, layer.values.length))
    .map((entry) => entry.index)
    .sort((a, b) => a - b);
}

function NeuralGraph({ layers, edges, title }: { layers: NeuralLayer[]; edges: NeuralEdge[]; title: string }) {
  const visible = useMemo(
    () => layers.map((layer) => ({ layer, units: topUnitIndices(layer, edges) })),
    [edges, layers],
  );
  const positions = new Map<string, { x: number; y: number; value: number }>();
  const width = 920;
  const height = 330;
  visible.forEach(({ layer, units }, layerIndex) => {
    const x = 70 + (layerIndex * (width - 140)) / Math.max(1, visible.length - 1);
    units.forEach((unit, unitIndex) => {
      const y = 70 + (unitIndex * 205) / Math.max(1, units.length - 1);
      positions.set(`${layer.id}:${unit}`, { x, y, value: layer.values[unit] });
    });
  });
  const maxActivation = Math.max(0.001, ...visible.flatMap(({ layer, units }) => units.map((index) => Math.abs(layer.values[index]))));
  const visibleEdges = edges.filter(
    (edge) => positions.has(`${edge.sourceLayer}:${edge.source}`) && positions.has(`${edge.targetLayer}:${edge.target}`),
  );
  const maxContribution = Math.max(0.001, ...visibleEdges.map((edge) => Math.abs(edge.contribution)));

  return (
    <svg className="neural-graph" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
      <title>{title}</title>
      <desc>Nodes show the strongest current activations. Solid connections contribute positively and dashed connections negatively.</desc>
      {visible.slice(0, -1).map(({ layer }, index) => {
        const from = positions.get(`${layer.id}:${visible[index].units[0]}`)?.x ?? 0;
        const nextLayer = visible[index + 1];
        const to = positions.get(`${nextLayer.layer.id}:${nextLayer.units[0]}`)?.x ?? 0;
        return <line key={`flow-${layer.id}`} x1={from + 15} y1={294} x2={to - 15} y2={294} className="network-flow-line" />;
      })}
      {visibleEdges.map((edge, index) => {
        const source = positions.get(`${edge.sourceLayer}:${edge.source}`)!;
        const target = positions.get(`${edge.targetLayer}:${edge.target}`)!;
        const ratio = Math.abs(edge.contribution) / maxContribution;
        return (
          <line
            key={`${edge.sourceLayer}-${edge.source}-${edge.targetLayer}-${edge.target}-${index}`}
            x1={source.x}
            y1={source.y}
            x2={target.x}
            y2={target.y}
            className={`active-connection ${edge.contribution < 0 ? "negative" : "positive"}`}
            style={{ strokeOpacity: 0.16 + ratio * 0.72, strokeWidth: 0.7 + ratio * 2.4 }}
          >
            <title>weight {edge.weight.toFixed(3)}, contribution {edge.contribution.toFixed(3)}</title>
          </line>
        );
      })}
      {visible.map(({ layer, units }, layerIndex) => {
        const x = 70 + (layerIndex * (width - 140)) / Math.max(1, visible.length - 1);
        return (
          <g key={layer.id}>
            <text x={x} y={24} textAnchor="middle" className="network-layer-label">{layer.label}</text>
            <text x={x} y={42} textAnchor="middle" className="network-layer-count">showing {units.length}/{layer.totalUnits}</text>
            {units.map((unit, unitIndex) => {
              const point = positions.get(`${layer.id}:${unit}`)!;
              const value = layer.values[unit];
              const ratio = Math.abs(value) / maxActivation;
              return (
                <g key={unit} className={`neuron ${value < 0 ? "negative" : "positive"}`}>
                  <circle cx={point.x} cy={point.y} r={4.5 + ratio * 5.5} style={{ fillOpacity: 0.18 + ratio * 0.76 }}>
                    <title>unit {unit}: activation {value.toFixed(4)}</title>
                  </circle>
                  <text x={point.x + 13} y={point.y + 3}>n{unit}</text>
                </g>
              );
            })}
            <text x={x} y={319} textAnchor="middle" className="network-layer-kind">{layer.kind}</text>
          </g>
        );
      })}
    </svg>
  );
}

function ObservationThumbnail({ inspection }: { inspection: ModelInspection }) {
  const pixels = inspection.observation.thumbnail;
  return (
    <svg className="observation-thumbnail" viewBox="0 0 8 8" role="img" aria-label="Actual 32 by 32 model input, pooled to 8 by 8 for display">
      <title>Actual model input, spatially pooled for display</title>
      {pixels.flatMap((row, y) => row.map((pixel, x) => {
        const color = pixel.map((channel) => Math.max(0, Math.min(255, Math.round(channel * 255))));
        return <rect key={`${x}-${y}`} x={x} y={y} width={1.02} height={1.02} fill={`rgb(${color.join(",")})`} />;
      }))}
    </svg>
  );
}

function ReadoutView({ inspection }: { inspection: ModelInspection }) {
  const current = inspection.outputs.current;
  const future = inspection.outputs.meanFuture;
  return (
    <div className="readout-view">
      <div className="readout-latent" aria-label="Mean future latent values">
        {inspection.outputs.futureLatentMean.map((value, index) => {
          const magnitude = Math.min(1, Math.abs(value) / 3);
          return (
            <div key={index}>
              <span>L{index + 1}</span>
              <i className={value < 0 ? "negative" : "positive"} style={{ height: `${16 + magnitude * 82}%` }} />
              <strong>{value.toFixed(2)}</strong>
            </div>
          );
        })}
      </div>
      <div className="readout-comparison">
        <div><span>Probe</span><strong>Current latent</strong><strong>Mean imagined latent</strong></div>
        <div><span>Position x, y</span><strong>{current.position.map((value) => value.toFixed(2)).join(", ")}</strong><strong>{future.position.map((value) => value.toFixed(2)).join(", ")}</strong></div>
        <div><span>P(upper), learned probe</span><strong>{Math.round(current.upperGateProbe * 100)}%</strong><strong>{Math.round(future.upperGateProbe * 100)}%</strong></div>
        <div><span>P(goal), learned probe</span><strong>{Math.round(current.goalProbability * 100)}%</strong><strong>{Math.round(future.goalProbability * 100)}%</strong></div>
      </div>
      <p className="inspector-caveat">These auxiliary heads probe what the latent contains. They do not replace the exact explicit belief used by policy evaluation.</p>
    </div>
  );
}

export default function ModelInspector({ frame, onClose }: { frame: StepFrame; onClose: () => void }) {
  const [tab, setTab] = useState<InspectorTab>("encoder");
  const inspection = frame.modelInspection;
  return (
    <section className="model-inspector" aria-labelledby="model-inspector-title">
      <div className="inspector-heading">
        <div>
          <p className="eyebrow">ACTUAL CHECKPOINT ACTIVITY</p>
          <h2 id="model-inspector-title">Inside the inference pass</h2>
          <p>{inspection?.mode ?? "No checkpoint activity in this frame"} · selected action {frame.selectedAction}</p>
        </div>
        <button type="button" onClick={onClose}>Close ANN view</button>
      </div>
      {!inspection ? (
        <div className="inspector-empty">This frame has no recorded checkpoint activations. Start the live model or load a newly exported verified replay.</div>
      ) : (
        <>
          <div className="inspector-input-row">
            <ObservationThumbnail inspection={inspection} />
            <div>
              <span>Model input</span>
              <strong>3 × 32 × 32 RGB observation</strong>
              <small>The 8 × 8 preview is pooled from the exact tensor sent to the encoder.</small>
            </div>
            <div>
              <span>Condition</span>
              <strong>action = {inspection.selectedAction}</strong>
              <small>one-hot [{inspection.conditioning.actionOneHot.join(", ")}] · belief {Math.round(inspection.conditioning.beliefUpper * 100)}% upper · shown at t{inspection.conditioning.diffusionStepShown}</small>
            </div>
            <div>
              <span>Model output</span>
              <strong>16 possible next latents</strong>
              <small>Eight denoising iterations per sample.</small>
            </div>
          </div>
          <div className="inspector-tabs" role="tablist" aria-label="Model inspection stage">
            {(["encoder", "denoiser", "readouts"] as const).map((item) => (
              <button key={item} type="button" role="tab" aria-selected={tab === item} onClick={() => setTab(item)}>
                {item === "readouts" ? "Latent & readouts" : item[0].toUpperCase() + item.slice(1)}
              </button>
            ))}
          </div>
          {tab === "encoder" && (
            <>
              <div className="neural-graph-scroll"><NeuralGraph layers={inspection.encoder.layers} edges={inspection.encoder.edges} title="Encoder channel and neuron activity" /></div>
              <p className="inspector-explanation">Convolutional circles summarize each feature map over space. Dense circles are individual neurons. Detailed edges are shown where a dense weight matrix makes a neuron-to-neuron contribution meaningful.</p>
            </>
          )}
          {tab === "denoiser" && (
            <>
              <div className="neural-graph-scroll"><NeuralGraph layers={inspection.denoiser.layers} edges={inspection.denoiser.edges} title="Conditional denoiser neuron and connection activity" /></div>
              <div className="diffusion-strip" aria-label="Eight denoising iterations from noise to imagined latent">
                {inspection.diffusionTrajectory.map((point) => (
                  <div key={point.step}>
                    <span>t{point.step}</span>
                    <i style={{ height: `${Math.min(100, 12 + point.predictedNoiseNorm * 13)}%` }} />
                    <small>‖ε̂‖ {point.predictedNoiseNorm.toFixed(2)}</small>
                  </div>
                ))}
              </div>
              <p className="inspector-explanation">The graph shows the final denoising iteration. Solid cyan edges add to the target unit; dashed amber edges subtract. Line strength is |activation × weight|.</p>
            </>
          )}
          {tab === "readouts" && <ReadoutView inspection={inspection} />}
        </>
      )}
    </section>
  );
}
