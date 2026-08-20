import { readFile, readdir, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const frontendRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const buildRoot = join(frontendRoot, "standalone-dist");
const outputPath = resolve(frontendRoot, "..", "Two-Gates-Standalone.html");
const assetNames = await readdir(join(buildRoot, "assets"));
const scriptName = assetNames.find((name) => name.endsWith(".js"));
const styleName = assetNames.find((name) => name.endsWith(".css"));

if (!scriptName || !styleName) {
  throw new Error("Standalone build did not produce one JavaScript and one CSS asset.");
}

const [document, script, style, favicon] = await Promise.all([
  readFile(join(buildRoot, "index.html"), "utf8"),
  readFile(join(buildRoot, "assets", scriptName), "utf8"),
  readFile(join(buildRoot, "assets", styleName), "utf8"),
  readFile(join(frontendRoot, "public", "favicon.svg")),
]);

const faviconUrl = `data:image/svg+xml;base64,${favicon.toString("base64")}`;
const safeScript = script.replaceAll("</script", "<\\/script");
let standalone = document
  .replace(/<link rel="stylesheet"[^>]*>/, () => `<style>${style}</style>`)
  .replace(/<script[^>]*\bsrc="[^"]+"[^>]*><\/script>/, () => `<script type="module">${safeScript}</script>`)
  .replace(/href="\.\/favicon\.svg"|href="\/favicon\.svg"/, `href="${faviconUrl}"`)
  .replace(/^/, "<!-- Generated standalone build: move or share this file without companion assets. -->\n");

const documentShell = standalone
  .replace(/<script\b[^>]*>[\s\S]*?<\/script>/g, "<script></script>")
  .replace(/<style\b[^>]*>[\s\S]*?<\/style>/g, "<style></style>");
const externalReferences = documentShell.match(/(?:src|href)="(?!data:|#)[^"]+"/g) ?? [];
if (externalReferences.length > 0) {
  throw new Error(`Standalone HTML still contains external references: ${externalReferences.join(", ")}`);
}
if ((standalone.match(/<\/script>/g) ?? []).length !== 1 || (standalone.match(/<\/style>/g) ?? []).length !== 1) {
  throw new Error("Standalone HTML contains a prematurely closed inline asset.");
}
if (standalone.includes("./assets/") || standalone.includes("/api/meta") || standalone.includes("/replays/")) {
  throw new Error("Standalone HTML still contains a build, API, or replay-file dependency.");
}
if (!standalone.includes("IN-BROWSER ANALYTIC PLANNER") || !standalone.includes("Train the world model")) {
  throw new Error("Standalone HTML is missing the Lab runtime or browser training UI.");
}

await writeFile(outputPath, standalone, "utf8");
console.log(`${outputPath}\n${Buffer.byteLength(standalone).toLocaleString("en-US")} bytes`);
