import type { AgentType, EpisodeConfig, MetaResponse, ReplayIndexItem, ReplayTrace, StepFrame } from "./types";

const SCHEMA_VERSION = "1.4";

async function checkedJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

export async function fetchMeta(): Promise<MetaResponse | null> {
  try {
    return await checkedJson<MetaResponse>(await fetch("/api/meta", { signal: AbortSignal.timeout(1500) }));
  } catch {
    return null;
  }
}

export async function createEpisode(config: EpisodeConfig): Promise<{ episodeId: string; frame: StepFrame }> {
  return checkedJson(
    await fetch("/api/episodes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    }),
  );
}

export async function stepEpisode(episodeId: string): Promise<StepFrame> {
  return checkedJson(await fetch(`/api/episodes/${episodeId}/step`, { method: "POST" }));
}

export async function fetchReplayIndex(): Promise<ReplayIndexItem[]> {
  try {
    return await checkedJson<ReplayIndexItem[]>(await fetch("/api/replays"));
  } catch {
    return checkedJson<ReplayIndexItem[]>(await fetch("/replays/index.json"));
  }
}

export async function fetchReplay(agentType: AgentType): Promise<ReplayTrace> {
  const index = await fetchReplayIndex();
  const item = index.find((entry) => entry.agentType === agentType);
  if (!item) throw new Error(`No replay for ${agentType}`);
  let trace: ReplayTrace;
  try {
    trace = await checkedJson<ReplayTrace>(await fetch(`/api/replays/${item.id}`));
  } catch {
    trace = await checkedJson<ReplayTrace>(await fetch(`/replays/${item.file}`));
  }
  if (trace.schemaVersion !== SCHEMA_VERSION || trace.checkpointHash !== item.checkpointHash) {
    throw new Error("Replay schema or checkpoint hash mismatch");
  }
  return trace;
}
