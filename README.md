# Two Gates

An interactive, science-museum-style teaching demo of a small JEPA-style latent world model, conditional latent diffusion, and epistemic policy selection.

The central question is simple: **when the shortest route may be blocked, is it worth taking a small detour to learn which route is open?**

The demo makes five operations visible:

1. **Observe** a partially observable world.
2. **Imagine** 16 possible latent futures per action.
3. **Evaluate** preference cost and prospective epistemic value.
4. **Act** on the lowest operational expected-free-energy score.
5. **Update belief** with an exact Bayesian update after evidence.

## Run the demo

Double-click **`Start Two Gates.cmd`** in the project folder. The launcher checks the local setup, installs or rebuilds missing parts when needed, starts the API, waits until it is ready, and opens the demo in the default browser.

In the Lab, each click on **Next** advances exactly one teaching phase: Observe → Imagine → Evaluate → Act → Update belief. The environment moves only when entering Act. A new action cycle starts after Update belief.

Keep the launcher window open while using the demo. Close it or press `Ctrl+C` to stop the server. Double-clicking the launcher again while the server is running simply opens the demo.

The checked-in replay mode works without a GPU. Live mode uses the included trained PyTorch checkpoint. A first launch on a new machine requires Python 3.12, Node.js, and an internet connection for dependency installation; later launches work from the local installation.

For terminal use or a custom port, run:

```powershell
.\start-demo.ps1 -Port 8000
```

The launcher uses `ExecutionPolicy Bypass` only for its own local PowerShell process. It does not change the machine's PowerShell policy.

## What is implemented

- A 13×9 **Two Gates** POMDP with one hidden binary gate state, a configurable cue, and a preferred goal state.
- Stable seed-generated layouts that vary both gate rows and the sensor position while keeping every world navigable.
- An explicit Lab assumption for gate evidence: diagnostic contact can be allowed, or unresolved gate contact can require sensor evidence first. A closed gate still blocks movement and counts as a collision.
- An explicit belief `P(upper gate open | history)` and exact binary state-information gain.
- A 182k-parameter PyTorch model with:
  - 32×32 RGB observations;
  - an 8-dimensional JEPA-style latent;
  - an EMA target encoder;
  - an 8-step action- and belief-conditioned latent denoiser;
  - variance/covariance and inverse-dynamics regularization;
  - position, belief, and goal probes.
- A React/TypeScript interface focused on a single interactive **Lab** view.
- Optional in-browser training with a permanently available pretrained default, Quick and Full modes, WebGPU/WebGL/CPU selection, streamed transition generation, local model storage, and scientific quality gates before activation.
- An optional ANN activity view with the actual checkpoint input, feature-channel activity, dense-unit activations, signed activation-times-weight contributions, diffusion iterations, latent output, and learned probe readouts.
- A project guide covering motivation, architecture, Bayesian belief updates, policy scoring, scientific limits, and links to every bundled source paper.
- Schema- and checkpoint-locked replay traces that use the same `StepFrame` shape as live inference.
- FastAPI endpoints for metadata, episode reset, episode stepping, replay access, and the bundled papers.

## Scientific scope

The operational policy score is:

`G(π) = expected preference cost − β Στ wτ I(hidden state; observationτ | earlier observations, π)`

The epistemic term uses exact Bayesian information gain for the toy binary hidden state. A noisy sensor cue uses its expected posterior entropy; a diagnostic gate test can remove all uncertainty that remains. Each incremental gain is multiplied by a transparent five-step temporal reach factor. Sequential gains are added in observation order, with later evidence scored only against remaining entropy. This makes early evidence valuable without double-counting the same hidden state. The learned model supplies compact representation learning and conditional latent samples. The environment and belief layer keep the demonstration reliable and inspectable.

Use these labels when describing the project:

- **minimal JEPA-style latent world model**
- **conditional latent diffusion**
- **exact Bayesian information gain in a toy POMDP**
- **operational EFE approximation**

This is not a reproduction of JEDI and not a full pymdp agent.

## Reproduce the model and evidence

### Optional browser training

Choose **Train model** in the Lab header to train without installing Python. **Quick experiment** trains one seed on 10,000 deterministic transitions for five epochs. **Full reproduction** uses 50,000 transitions, three seeds, and median validation-run selection. Both modes regenerate the same finite transition stream per epoch so the advertised dataset size remains truthful without retaining hundreds of megabytes of images in browser memory.

The browser attempts WebGPU first, then WebGL, and finally CPU. GPU access uses the browser abstraction rather than CUDA, so availability depends on the browser and graphics configuration. CPU users should start with Quick experiment.

Every candidate is checked for action conditioning, non-collapsed latent variation, position accuracy, and belief accuracy. Only a passing model can be activated. Activation replaces the latent futures displayed during **Imagine**; the explicit Bayesian belief update, information value, path cost, and policy choice remain analytic. **Restore pretrained model** returns to the bundled accepted checkpoint at any time.

### Standalone HTML

Run `pnpm build:standalone` in `frontend/` to create `Two-Gates-Standalone.html` in the repository root. The generated file embeds the Lab UI, styles, three accepted checkpoint replays, the in-browser analytic episode engine, TensorFlow.js training, and the favicon. It can be moved and opened directly without the Python server or companion assets.

The accepted checkpoint replay is used for the locked default configuration. Changing a Lab control switches to the browser's analytic episode engine so the displayed paths, policy scores, belief updates, and explanations continue to share one calculation. A browser-trained model can still replace the latent samples after it passes the quality gates.

The reference list and paper summaries are included, while the seven full PDF files remain in the repository edition. Excluding those PDFs keeps the standalone file small enough to share easily; the arXiv links remain optional and require internet access.

The browser implementation preserves the model architecture and loss terms but is not bit-identical to PyTorch because random-number generation, floating-point order, optimizer kernels, and device backends differ. The PyTorch workflow below remains the reference reproduction and evaluation path.

### Reference PyTorch training

Train 50,000 transitions over three seeds and select the median validation checkpoint:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\train.ps1
```

Run the checkpoint quality gates:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\validate.ps1
```

Run the locked 200-world behavioral evaluation:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\evaluate.ps1
```

Regenerate backend and static replay files from the accepted checkpoint:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\export-replays.ps1
```


Run all automated tests and rebuild the interface:

```powershell
.\.venv\Scripts\python.exe -m pytest
cd frontend
pnpm run test
pnpm run build
```

## Current accepted checkpoint

- Dataset: 50,000 transitions
- Training seeds: 11, 29, 47
- Dataset version: `two-gates-varied-layout-v2`
- Selection: median validation loss, seed 11
- Checkpoint hash: `60e9141873e99464188fb358ae4e43ffa383c3395d97f755f0c599881a9cc4f1`
- Held-out action-shuffle ratio: 1.248× worse than correct actions
- Minimum latent-dimension standard deviation: 0.559
- Position-head MSE: 0.0129
- Belief-head Brier score: 0.00739

Across 200 locked, varied-layout worlds, balanced and pragmatic agents both reached 100% success with a median path-length ratio of 1.000. With diagnostic gate contact enabled, balanced made no wrong-gate commitments, but averaged 0.455 physical collisions versus pragmatic's 0.395 because testing a closed gate blocks movement. The demo reports diagnostic tests and wrong commitments separately.

## Source material

The papers in `jepa_active_inference_starter/jepa_active_inference_starter/papers/` remain unchanged. The original starter brief is preserved beside them. Key implementation references include JEDI, EB-JEPA, the deep-learning perspective on Active Inference, pymdp, value-guided JEPA planning, I-JEPA, and D-JEPA.
