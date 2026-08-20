# Minimal JEPA + Latent Diffusion + Active Inference Demo

## Goal
Build the smallest convincing demo of an agent that:
1. encodes observations into a compact latent state (JEPA-style),
2. predicts a distribution over future latent states conditioned on actions (latent diffusion),
3. evaluates candidate actions using both pragmatic/extrinsic value and epistemic/information value,
4. acts, observes the result, and updates its world model.

The intended first version should run locally on ordinary consumer hardware and be visually inspectable.

## Recommended minimal environment
A tiny 2D grid world rendered as 16x16 or 32x32 images.
- Agent position is visible.
- One goal/food location gives preferred outcomes.
- One or more hidden/uncertain properties (e.g. which corridor is open, or which colored tile is rewarding).
- Actions: up/down/left/right.
- The environment should include a situation where the shortest-looking route is not always optimal because an information-gathering move can reduce uncertainty.

This makes the epistemic/pragmatic trade-off visible rather than merely mathematical.

## Minimal architecture (v0)

Observation x_t (small image)
  -> Encoder E_phi
  -> latent z_t (start with 4-16 dimensions)

Action a_t + z_t + noisy candidate future latent z_{t+1}^tau
  -> tiny conditional denoiser D_theta
  -> predicted clean future latent z_{t+1}

Sample several possible futures for each candidate action.

Action score:
  score(a) = pragmatic_value(a) + beta * epistemic_value(a)

Pragmatic value:
- proximity/probability of preferred outcome (goal state)

Epistemic value, v0 approximation:
- expected reduction in uncertainty after taking the action
- easiest first implementation: ensemble disagreement / posterior variance / entropy reduction over sampled latent futures

Then choose action, observe x_{t+1}, train/update, repeat.

## Important implementation choice
Do NOT start by reproducing full JEDI, V-JEPA 2, Atari, or full pymdp.
The demo should preserve the computational ideas while reducing scale aggressively.

Suggested first-pass components:
- PyTorch
- custom tiny grid-world environment (no Gym dependency necessary)
- MLP or tiny CNN encoder
- latent dimension: 8 (adjustable)
- tiny MLP denoiser
- diffusion steps: 4-16, not hundreds
- planning horizon: 1 initially, then 3-5
- discrete candidate actions: 4
- 8-32 future samples/action

## What each paper contributes

### 01 JEDI (2026)
Core technical bridge. Shows an end-to-end latent diffusion world model built in a JEPA framework. The encoder learns from conditional denoising and future latents are predicted from current latent state/action context. This is the closest direct precedent for our world-model core.
Source: https://arxiv.org/abs/2605.13013

### 02 EB-JEPA (2026)
Most useful engineering reference for a small implementation. Provides modular, educational JEPA examples including action-conditioned planning in a procedurally generated Two Rooms environment, explicitly designed for single-GPU experimentation.
Source: https://arxiv.org/abs/2602.03604
Code: https://github.com/facebookresearch/eb_jepa

### 03 Free Energy Principle for Perception and Action - Deep Learning Perspective (2022)
Best bridge paper for translating Active Inference into deep-learning engineering terms. Especially relevant sections: expected free energy, epistemic value, extrinsic value, variational world models.
Source: https://arxiv.org/abs/2207.06415

### 04 pymdp (2022)
Practical reference implementation of discrete Active Inference/POMDPs. Useful as a correctness reference for action selection and expected free energy, even if our first demo uses continuous learned latents rather than pymdp directly.
Source: https://arxiv.org/abs/2201.03904
Code: https://github.com/infer-actively/pymdp

### 05 Value-guided action planning with JEPA world models (2026)
Useful bridge between JEPA latent geometry and action selection/planning. It shows that shaping latent representations around value can materially improve planning.
Source: https://arxiv.org/abs/2601.00844

### 06 I-JEPA (2023)
Foundational JEPA reference: predict representations rather than pixels. Useful for understanding target/context encoders and collapse avoidance.
Source: https://arxiv.org/abs/2301.08243
Code: https://github.com/facebookresearch/ijepa

### 07 D-JEPA (2024)
Shows a different JEPA+diffusion combination, especially useful for understanding how diffusion losses can be coupled to JEPA-style prediction in continuous spaces. Less directly world-model/action-oriented than JEDI.
Source: https://arxiv.org/abs/2410.03755

## Proposed build sequence

### Stage 0 - deterministic sanity check
Learn z_t and deterministic z_{t+1}=f(z_t,a_t). Plan to goal using latent distance. Verify that latent world-model planning works.

### Stage 1 - diffusion future model
Replace deterministic predictor with a tiny conditional diffusion model over z_{t+1}. Visualize multiple sampled futures per action.

### Stage 2 - pragmatic action selection
Define preferences over outcomes and select actions from predicted futures.

### Stage 3 - epistemic action selection
Add uncertainty/information-gain term. Construct a fork where the agent should sometimes move toward an informative observation before committing to the goal route.

### Stage 4 - online adaptation
Let the agent update the world model from its own trajectories and show uncertainty falling as it learns.

### Stage 5 - compare agents
Run three agents in the same uncertain environment:
A. reward/pragmatic only
B. uncertainty-seeking only
C. pragmatic + epistemic (Active-Inference-like)
Plot success, path length, uncertainty, and cumulative preferred-outcome score.

## What would count as a convincing demo
The key visual should not be a high-quality generated image. It should be behavior:
- the model imagines several possible latent futures,
- uncertainty differs by action,
- the agent sometimes chooses an action that is not immediately goal-directed because it is informative,
- after gathering information, it exploits what it learned and reaches the preferred state efficiently.

That is the point where JEPA + diffusion becomes meaningfully Active-Inference-like rather than merely a stochastic world model.
