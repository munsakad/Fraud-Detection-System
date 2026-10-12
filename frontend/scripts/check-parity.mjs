// Verifies that the browser port (src/demo/inference.js) reproduces the
// Python models' scores on held-out rows. Runs in CI before the Pages build,
// so the demo can never silently drift from the trained models.
import { readFileSync } from "node:fs";
import { scoreRow } from "../src/demo/inference.js";

const read = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url), "utf8"));
const model = read("../public/demo/model.json");
const parity = read("../public/demo/parity.json");

const KEYS = ["random_forest", "xgboost", "deep_learning", "ensemble"];
const diffs = Object.fromEntries(KEYS.map((k) => [k, []]));
let riskMismatches = 0;

for (const { x, s } of parity) {
  const js = scoreRow(model, x);
  for (const k of KEYS) diffs[k].push(Math.abs(js[k] - s[k]));
  if (js.risk_level !== s.risk_level) riskMismatches += 1;
}

let ok = true;
for (const k of KEYS) {
  const d = diffs[k].sort((a, b) => a - b);
  const mean = d.reduce((a, b) => a + b, 0) / d.length;
  const p99 = d[Math.floor(d.length * 0.99)];
  console.log(
    k.padEnd(14) + " mean|diff|=" + mean.toExponential(2) + "  p99=" + p99.toExponential(2) + "  max=" + d.at(-1).toExponential(2)
  );
  if (mean > 1e-3 || p99 > 1e-2) ok = false;
}
console.log("risk level mismatches: " + riskMismatches + "/" + parity.length);

if (!ok || riskMismatches > parity.length * 0.02) {
  console.error("JS inference diverges from the Python models");
  process.exit(1);
}
console.log("JS inference matches the Python models");
