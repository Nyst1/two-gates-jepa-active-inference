import { useEffect, useState } from "react";

type GuideSection = "why" | "architecture" | "decision" | "limits" | "references";
type ArchitecturePhaseId = "observe" | "imagine" | "evaluate" | "act" | "update";
type ArchitectureTone = "learned" | "analytic" | "environment" | "state";

type ArchitecturePhase = {
  id: ArchitecturePhaseId;
  number: string;
  label: string;
  plainTitle: string;
  summary: string;
  why: string;
  labShows: string[];
  nodes: Array<{
    label: string;
    title: string;
    detail: string;
    tone: ArchitectureTone;
    connector?: "+" | "→";
  }>;
  details: string[];
};

const ARCHITECTURE_PHASES: ArchitecturePhase[] = [
  {
    id: "observe",
    number: "01",
    label: "Observe",
    plainTitle: "Take in what can be seen",
    summary: "The agent receives the public scene and keeps a separate, explicit belief about the gate it cannot see.",
    why: "A useful decision starts by separating observation from assumption. The hidden gate is never handed to the agent as a fact.",
    labShows: ["The grid world and visible objects", "The upper/lower gate belief and uncertainty"],
    nodes: [
      { label: "Environment", title: "Public world", detail: "Walls, agent, sensor and goal", tone: "environment", connector: "→" },
      { label: "Observation", title: "32 × 32 RGB", detail: "Visible state plus a narrow belief band", tone: "state", connector: "→" },
      { label: "Learned model", title: "Online encoder", detail: "CNN compresses the image", tone: "learned", connector: "→" },
      { label: "Representation", title: "8D latent zₜ", detail: "Compact current-state features", tone: "learned" },
    ],
    details: [
      "The encoder uses three convolutional stages (16 → 32 → 48 channels), a 64-unit dense layer and an eight-dimensional latent.",
      "The scalar belief bₜ = P(upper gate open | history) remains explicit and inspectable. A learned probe can read belief-related information, but it is not the source of the Bayesian belief.",
    ],
  },
  {
    id: "imagine",
    number: "02",
    label: "Imagine",
    plainTitle: "Try several futures before committing",
    summary: "For every possible move, the model creates several plausible next latent states instead of one supposedly certain answer.",
    why: "Several futures make uncertainty visible. The agent can inspect what might follow from each action before it acts.",
    labShows: ["64 latent dots: 16 futures for each of four actions", "Transparent five-step route families in the grid"],
    nodes: [
      { label: "Current state", title: "Latent zₜ", detail: "Encoded observation", tone: "learned", connector: "+" },
      { label: "Candidate", title: "Action a", detail: "Right, up, down or left", tone: "state", connector: "+" },
      { label: "Uncertainty", title: "Belief bₜ", detail: "Current gate probability", tone: "analytic", connector: "→" },
      { label: "Learned model", title: "8-step denoiser", detail: "Conditional latent diffusion", tone: "learned", connector: "→" },
      { label: "Output", title: "16 futures/action", detail: "Possible next latents", tone: "learned" },
    ],
    details: [
      "The diffusion model is conditioned on current latent, candidate action, explicit belief and diffusion time.",
      "The default latent dots come from the accepted checkpoint replay. A validated browser-trained model can replace them; changed offline controls otherwise use clearly labelled analytic reference samples.",
      "The paths drawn through grid cells are transparent analytic planning traces, not decoded diffusion samples.",
      "During training only, an EMA target encoder maps the real next observation xₜ₊₁ to a stable target latent. The model learns both deterministic next-latent prediction and diffusion noise prediction.",
    ],
  },
  {
    id: "evaluate",
    number: "03",
    label: "Evaluate",
    plainTitle: "Compare progress with what can be learned",
    summary: "Each action is scored for progress toward the goal and for useful information it could reveal.",
    why: "A detour can be rational when the evidence it provides prevents a more costly wrong commitment later.",
    labShows: ["Preference-cost and information-value bars", "The selected action, target and lowest G score"],
    nodes: [
      { label: "Analytic", title: "Path cost", detail: "Expected remaining distance", tone: "analytic", connector: "+" },
      { label: "Exact belief math", title: "Information value", detail: "Incremental gain × temporal reach", tone: "analytic", connector: "→" },
      { label: "Policy score", title: "G(a)", detail: "Cost − β × information value", tone: "analytic", connector: "→" },
      { label: "Decision", title: "Lowest G", detail: "One action is selected", tone: "analytic" },
    ],
    details: [
      "Evidence is scored in the order a policy would reveal it. If a cue comes before a gate test, its early gain is added to the later gain from only the uncertainty that remains.",
      "This incremental decomposition avoids counting the same hidden state twice while still valuing earlier uncertainty reduction more highly.",
      "The neural latent futures are displayed for inspection, but the current G score and action choice are calculated analytically. This keeps the teaching model auditable.",
      "Pragmatic ignores the information term, Information emphasizes it, and Balanced uses the selected epistemic weight β.",
    ],
  },
  {
    id: "act",
    number: "04",
    label: "Act",
    plainTitle: "Make one move in the world",
    summary: "The chosen action is executed once. The world, not the model, determines whether the agent moves or is blocked.",
    why: "Prediction becomes testable only when an action meets the real environment and produces a consequence.",
    labShows: ["The executed direction and new position", "A visible block when a wall or closed gate stops the move"],
    nodes: [
      { label: "Policy", title: "Selected action", detail: "The lowest-scoring move", tone: "analytic", connector: "→" },
      { label: "Environment", title: "Grid transition", detail: "One attempted cell movement", tone: "environment", connector: "→" },
      { label: "Consequence", title: "Move or collision", detail: "Position and counters change", tone: "environment", connector: "→" },
      { label: "Possible evidence", title: "Cue or gate contact", detail: "An observation may be produced", tone: "environment" },
    ],
    details: [
      "The interface advances through explanatory phases, but the environment changes only when the cycle enters Act.",
      "Diagnostic does not mean free: contacting a closed gate reveals its state, blocks the move and counts as a collision.",
    ],
  },
  {
    id: "update",
    number: "05",
    label: "Update",
    plainTitle: "Use the result to revise belief",
    summary: "New evidence changes the probability assigned to each gate, and the next decision begins from that revised belief.",
    why: "Learning is visible as a specific before-and-after change in uncertainty, not as a vague claim that the agent became smarter.",
    labShows: ["Belief and entropy before and after evidence", "The exact reason for the update"],
    nodes: [
      { label: "Evidence", title: "Cue or gate result", detail: "What the last action revealed", tone: "environment", connector: "→" },
      { label: "Exact update", title: "Bayes' rule", detail: "Uses configured cue reliability", tone: "analytic", connector: "→" },
      { label: "New state", title: "Belief bₜ₊₁", detail: "Updated gate probability", tone: "state", connector: "→" },
      { label: "Next cycle", title: "Observe again", detail: "Plan from the new belief", tone: "state" },
    ],
    details: [
      "A noisy sensor cue produces an exact binary Bayesian posterior. Direct gate evidence resolves the hidden state, so belief becomes 0 or 1.",
      "If the action produces no evidence, belief and entropy remain unchanged. The interface says this explicitly.",
    ],
  },
];

const REFERENCES = [
  {
    id: "jedi",
    title: "JEDI: Joint Embedding Diffusion World Model for Online Model-Based Reinforcement Learning",
    detail: "Lim et al. (2026) · the closest architectural motivation for end-to-end latent diffusion world models.",
    arxiv: "https://arxiv.org/abs/2605.13013",
  },
  {
    id: "eb-jepa",
    title: "A Lightweight Library for Energy-Based Joint-Embedding Predictive Architectures",
    detail: "Terver et al. (2026) · action-conditioned JEPA examples and collapse-prevention ablations.",
    arxiv: "https://arxiv.org/abs/2602.03604",
  },
  {
    id: "fep-deep-learning",
    title: "The Free Energy Principle for Perception and Action: A Deep Learning Perspective",
    detail: "Mazzaglia et al. (2022) · connects preferences, epistemics, inference and planning to deep-learning implementations.",
    arxiv: "https://arxiv.org/abs/2207.06415",
  },
  {
    id: "pymdp",
    title: "pymdp: A Python library for active inference in discrete state spaces",
    detail: "Heins et al. (2022) · reference for discrete POMDP beliefs, utility and state-information gain.",
    arxiv: "https://arxiv.org/abs/2201.03904",
  },
  {
    id: "value-guided-jepa",
    title: "Value-Guided Action Planning with JEPA World Models",
    detail: "Destrade et al. (2026) · motivates shaping predictive representations for planning-relevant distance.",
    arxiv: "https://arxiv.org/abs/2601.00844",
  },
  {
    id: "i-jepa",
    title: "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture",
    detail: "Assran et al. (2023) · foundational representation-space prediction reference.",
    arxiv: "https://arxiv.org/abs/2301.08243",
  },
  {
    id: "d-jepa",
    title: "Denoising with a Joint-Embedding Predictive Architecture",
    detail: "Chen et al. (2024/2025) · reference for combining JEPA representations and diffusion objectives.",
    arxiv: "https://arxiv.org/abs/2410.03755",
  },
];

function WhySection() {
  return (
    <div className="guide-content">
      <p className="guide-lead">Most AI demos hide uncertainty behind one confident output. Two Gates makes a different capability visible: an agent can represent several futures, value information, and change its policy after evidence.</p>
      <div className="guide-contrast">
        <div><span>Reactive pattern</span><strong>Observe → choose an action</strong><p>Useful when the current observation contains enough information and the objective rewards immediate task progress.</p></div>
        <div><span>This experiment</span><strong>Observe → imagine → evaluate → act → update</strong><p>Useful when the world has hidden causes and an action can be valuable because of what it helps the agent learn.</p></div>
      </div>
      <h3>Why the combination is interesting</h3>
      <ul className="guide-points">
        <li><strong>JEPA-style prediction</strong><span>asks the model to predict compact representations rather than reconstruct every pixel.</span></li>
        <li><strong>Latent diffusion</strong><span>allows several plausible next representations instead of forcing one averaged prediction.</span></li>
        <li><strong>Active Inference</strong><span>adds epistemic value: uncertainty reduction can influence action selection.</span></li>
        <li><strong>Explicit Bayesian belief</strong><span>keeps the hidden binary gate state exact and inspectable instead of pretending the neural network is calibrated by default.</span></li>
      </ul>
    </div>
  );
}

function ArchitectureSection() {
  const [phaseId, setPhaseId] = useState<ArchitecturePhaseId>("observe");
  const phase = ARCHITECTURE_PHASES.find((item) => item.id === phaseId) ?? ARCHITECTURE_PHASES[0];
  return (
    <div className="guide-content architecture-guide">
      <div className="architecture-intro">
        <p className="eyebrow">ONE DECISION · FIVE VISIBLE PHASES</p>
        <h3>The Lab is a window into this cycle</h3>
        <p>Select a phase to connect what appears on screen with what the system is doing underneath. Start with the plain-language explanation; open the detail only when you want it.</p>
      </div>

      <div className="architecture-cycle" role="tablist" aria-label="Lab cycle phases">
        {ARCHITECTURE_PHASES.map((item) => (
          <button
            key={item.id}
            type="button"
            role="tab"
            aria-label={`${item.number} ${item.label}`}
            aria-selected={phase.id === item.id}
            aria-controls="architecture-phase-panel"
            className={phase.id === item.id ? "active" : ""}
            data-phase={item.id}
            onClick={() => setPhaseId(item.id)}
          >
            <span>{item.number}</span>
            <strong>{item.label}</strong>
          </button>
        ))}
      </div>

      <section
        id="architecture-phase-panel"
        className="architecture-phase-card"
        data-phase={phase.id}
        role="tabpanel"
        aria-label={`${phase.label}: ${phase.plainTitle}`}
      >
        <header className="architecture-phase-heading">
          <span>{phase.number}</span>
          <div>
            <p className="eyebrow">{phase.label}</p>
            <h3>{phase.plainTitle}</h3>
            <p>{phase.summary}</p>
          </div>
        </header>

        <div className="architecture-bridge">
          <div>
            <span>In the Lab, you see</span>
            <ul>{phase.labShows.map((item) => <li key={item}>{item}</li>)}</ul>
          </div>
          <div>
            <span>Why this phase exists</span>
            <p>{phase.why}</p>
          </div>
        </div>

        <div className="architecture-flow" aria-label={`${phase.label} system flow`}>
          {phase.nodes.map((node, index) => (
            <div className="architecture-flow-group" key={`${phase.id}-${node.title}`}>
              <article className={`architecture-flow-node ${node.tone}`}>
                <span>{node.label}</span>
                <strong>{node.title}</strong>
                <small>{node.detail}</small>
              </article>
              {index < phase.nodes.length - 1 && <i aria-hidden="true">{node.connector ?? "→"}</i>}
            </div>
          ))}
        </div>

        <details className="architecture-details">
          <summary>More about this step <span aria-hidden="true">+</span></summary>
          <ul>{phase.details.map((item) => <li key={item}>{item}</li>)}</ul>
        </details>
      </section>

      <div className="architecture-legend" aria-label="System responsibility legend">
        <div><i className="learned" /><span><strong>Learned model</strong>Encoder, latent diffusion, target encoder during training and probe heads.</span></div>
        <div><i className="analytic" /><span><strong>Exact / analytic</strong>Belief update, information value, path cost, G score and action selection.</span></div>
        <div><i className="environment" /><span><strong>Environment</strong>The grid transition and the evidence produced after an action.</span></div>
      </div>

      <details className="architecture-training">
        <summary>How the learned world model is trained <span aria-hidden="true">+</span></summary>
        <div className="training-map">
          <div><span>Current observation xₜ</span><strong>Online encoder → zₜ</strong></div>
          <i>+</i>
          <div><span>Condition</span><strong>Action aₜ + belief bₜ</strong></div>
          <i>→</i>
          <div><span>Prediction</span><strong>Next latent / diffusion noise</strong></div>
          <i>↔</i>
          <div><span>Training target only</span><strong>xₜ₊₁ → EMA target encoder</strong></div>
        </div>
        <p>Prediction and denoising losses train the learned transition model. Variance/covariance regularization discourages collapse, while inverse-dynamics, position, belief and goal probes test whether useful state remains readable.</p>
      </details>
    </div>
  );
}

function DecisionSection() {
  return (
    <div className="guide-content">
      <h3>1. Belief</h3>
      <p><code>b<sub>t</sub> = P(upper gate open | history)</code> is an explicit scalar probability, initialized by the prior. It is not read from a hidden simulator variable.</p>
      <h3>2. Bayesian cue update</h3>
      <div className="guide-formula">P(upper | “upper”) = r b / [r b + (1 − r)(1 − b)]</div>
      <p><code>r</code> is cue reliability. A gate collision or successful crossing reveals the state exactly, so the posterior becomes 0 or 1.</p>
      <h3>3. Uncertainty and information</h3>
      <div className="guide-formula">H(b) = −b ln b − (1 − b) ln(1 − b)</div>
      <div className="guide-formula">Information gain = H(b<sub>t</sub>) − E[H(b<sub>t+1</sub>)]</div>
      <p>The binary cue's raw information gain is computed exactly. Each incremental gain is multiplied by a transparent reach factor based on when it arrives within the five-step planning horizon. The interface shows both factors and calls their product <strong>prospective epistemic value</strong>, rather than presenting the time-weighted product as raw information gain.</p>
      <h3>4. Two ways to obtain evidence</h3>
      <p>A physical gate test is perfectly diagnostic in this toy world: both a successful crossing and a blocked attempt identify the open gate, so its raw information value is the current entropy <code>H(b)</code>. “Diagnostic” does not mean cost-free: a closed gate blocks movement and counts as a collision. The Lab control makes the task assumption explicit:</p>
      <ul className="guide-points">
        <li><strong>Allowed: diagnostic test</strong><span>The planner scores the evidence sequence along each route. If the sensor comes first, its gain is followed by a gate gain based only on expected remaining entropy. This values early evidence without double-counting.</span></li>
        <li><strong>Prohibited: sensor first</strong><span>Contact with an unresolved gate is inadmissible until the sensor has been visited, unless prior confidence is already at least 98%. Preference paths therefore route via the sensor.</span></li>
      </ul>
      <p>This switch represents a task constraint, not a change in intelligence. In a real system, a test might be unsafe, irreversible, expensive, ethically barred, or simply unavailable.</p>
      <h3>5. Preference and policy</h3>
      <div className="guide-formula">G(π) = expected normalized path cost − β × Σ time-weighted incremental information gain</div>
      <p>Expected path cost averages shortest remaining grid distance under both gate hypotheses, so every extra move carries cost. The temporal reach factor makes earlier evidence more valuable than the same uncertainty reduction later. Walking into a known wall adds a fixed penalty. Lower <code>G</code> wins. Pragmatic-only sets the epistemic term to zero; information-only emphasizes information with a small path tie-breaker; balanced uses the selected β.</p>
      <h3>6. Learned and analytic responsibilities</h3>
      <p>The neural checkpoint produces the latent imagination shown in the interface. The current navigation score and grid paths use analytic grid distance and the exact binary belief model. The selected grid-path family now follows the same sensor, gate-test or goal objective used by the score. This separation makes the teaching loop auditable, but it also means the learned world model does not yet control the full policy end to end.</p>
    </div>
  );
}

function LimitsSection() {
  return (
    <div className="guide-content">
      <p className="guide-lead">This is a small operational bridge between research ideas, not evidence that the combined architecture solves general intelligence or reproduces a full Active Inference agent.</p>
      <ul className="guide-points limits-list">
        <li><strong>Toy POMDP</strong><span>Only one binary hidden variable is updated exactly. Real environments need richer state and observation models.</span></li>
        <li><strong>Operational EFE approximation</strong><span>The score borrows the utility/information decomposition but is not the full expected-free-energy machinery in pymdp.</span></li>
        <li><strong>Minimal JEPA-style model</strong><span>The CNN and EMA target borrow the predictive-representation pattern; this is not a reproduction of I-JEPA, EB-JEPA or JEDI.</span></li>
        <li><strong>Conditional latent diffusion</strong><span>Eight denoising steps model next-latent uncertainty. The five-step paths in the grid remain analytic planning traces.</span></li>
        <li><strong>Activation is not explanation</strong><span>Seeing a neuron fire shows computation, not a causal semantic interpretation. Probe heads and intervention tests provide stronger evidence.</span></li>
      </ul>
      <h3>Useful next scientific tests</h3>
      <p>Hold out entire layouts during training, intervene on the explicit belief while keeping pixels fixed, ablate action conditioning, compare latent samples with empirical transition outcomes, and replace analytic path cost with a learned calibrated outcome model only after it passes out-of-distribution tests.</p>
    </div>
  );
}

function ReferencesSection() {
  const standalone = import.meta.env.MODE === "standalone";
  return (
    <div className="guide-content reference-list">
      <p className="guide-lead">These papers motivate different pieces. None of them describes this exact demo architecture.</p>
      {REFERENCES.map((reference) => (
        <article key={reference.id}>
          <h3>{reference.title}</h3>
          <p>{reference.detail}</p>
          <div>
            {!standalone && <a href={`/api/papers/${reference.id}`} target="_blank" rel="noreferrer">Open bundled paper</a>}
            <a href={reference.arxiv} target="_blank" rel="noreferrer">arXiv record</a>
          </div>
        </article>
      ))}
      {standalone && <p className="standalone-reference-note">The standalone Lab includes the complete reference list and summaries. Full paper PDFs remain in the repository edition to keep this shareable file compact.</p>}
    </div>
  );
}

export default function ProjectGuide({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [section, setSection] = useState<GuideSection>("why");
  useEffect(() => {
    if (!open) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKey);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKey);
    };
  }, [onClose, open]);
  return (
    <div className={`drawer-layer guide-layer ${open ? "open" : ""}`} aria-hidden={!open}>
      <button className="drawer-scrim" aria-label="Close project guide" onClick={onClose} tabIndex={open ? 0 : -1} />
      <aside className="technical-drawer project-guide" aria-label="Project explanation, architecture and references">
        <div className="drawer-heading">
          <div><p className="eyebrow">PROJECT GUIDE</p><h2>Why this agent imagines before it acts</h2></div>
          <button type="button" onClick={onClose} aria-label="Close project guide">×</button>
        </div>
        <nav className="guide-nav" aria-label="Guide sections">
          {([
            ["why", "Why it matters"],
            ["architecture", "Lab & architecture"],
            ["decision", "Beliefs & policy"],
            ["limits", "Limits"],
            ["references", "Papers"],
          ] as Array<[GuideSection, string]>).map(([id, label]) => (
            <button key={id} type="button" className={section === id ? "active" : ""} onClick={() => setSection(id)}>{label}</button>
          ))}
        </nav>
        {section === "why" && <WhySection />}
        {section === "architecture" && <ArchitectureSection />}
        {section === "decision" && <DecisionSection />}
        {section === "limits" && <LimitsSection />}
        {section === "references" && <ReferencesSection />}
      </aside>
    </div>
  );
}
