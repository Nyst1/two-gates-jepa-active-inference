import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App, { nextPresentationPhase, type PresentationPhase } from "./App";
import replayJson from "../public/replays/two-gates-balanced-seed-0.json";
import type { ReplayTrace } from "./types";

const replay = replayJson as unknown as ReplayTrace;

describe("Two Gates app shell", () => {
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("opens directly into the single Lab view", () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline test")));
    render(<App />);
    expect(screen.getByRole("button", { name: "Lab" }).getAttribute("aria-current")).toBe("page");
    expect(screen.queryByRole("button", { name: "Story" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Compare" })).toBeNull();
    expect(screen.getByRole("heading", { name: "Test when information is worth the detour." })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Train model" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Project info" })).toBeTruthy();
  });

  it("visits every teaching phase in order before the next decision cycle", () => {
    const phases: PresentationPhase[] = ["observe"];
    for (let index = 0; index < 5; index += 1) {
      phases.push(nextPresentationPhase(phases[phases.length - 1]));
    }
    expect(phases).toEqual(["observe", "imagine", "evaluate", "act", "update", "observe"]);
  });

  it("maps every Lab phase to an understandable architecture explanation", () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline test")));
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "Project info" }));
    fireEvent.click(screen.getByRole("button", { name: "Lab & architecture" }));

    expect(screen.getByRole("tab", { name: "01 Observe" }).getAttribute("aria-selected")).toBe("true");
    expect(screen.getByRole("heading", { name: "Take in what can be seen" })).toBeTruthy();

    fireEvent.click(screen.getByRole("tab", { name: "03 Evaluate" }));
    expect(screen.getByRole("heading", { name: "Compare progress with what can be learned" })).toBeTruthy();
    expect(screen.getByText("Cost − β × information value")).toBeTruthy();
  });

  it("keeps the active rail, explanation number, and Step control synchronized", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "/api/meta") {
        return new Response(JSON.stringify({ liveModel: { available: true, device: "cpu" } }), { status: 200 });
      }
      if (url === "/api/episodes" && init?.method === "POST") {
        return new Response(JSON.stringify({ episodeId: "phase-test", frame: replay.frames[0] }), { status: 200 });
      }
      if (url === "/api/episodes/phase-test/step") {
        return new Response(JSON.stringify(replay.frames[1]), { status: 200 });
      }
      return new Response("not found", { status: 404 });
    });
    vi.stubGlobal("fetch", fetchMock);
    const { container } = render(<App />);

    const activePhase = () => container.querySelector(".cycle-rail li.active")?.textContent ?? "";
    const insightNumber = () => container.querySelector(".insight-strip > span")?.textContent ?? "";

    await screen.findByRole("button", { name: "Next: Imagine" });
    expect(activePhase()).toContain("Observe");
    expect(insightNumber()).toBe("01");

    fireEvent.click(screen.getByRole("button", { name: "Next: Imagine" }));
    await waitFor(() => expect(activePhase()).toContain("Imagine"));
    await waitFor(() => expect(insightNumber()).toBe("02"));

    fireEvent.click(screen.getByRole("button", { name: "Next: Evaluate" }));
    await waitFor(() => expect(activePhase()).toContain("Evaluate"));
    await waitFor(() => expect(insightNumber()).toBe("03"));

    fireEvent.click(screen.getByRole("button", { name: "Next: Act" }));
    await waitFor(() => expect(activePhase()).toContain("Act"));
    expect(insightNumber()).toBe("04");

    fireEvent.click(screen.getByRole("button", { name: "Next: Update belief" }));
    await waitFor(() => expect(activePhase()).toContain("Update belief"));
    await waitFor(() => expect(insightNumber()).toBe("05"));
  });
});
