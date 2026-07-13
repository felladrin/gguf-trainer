// Standalone assert check for the ETA duration formatter (src/eta.ts).
// Run:  deno run tests/eta_fmt.ts
import { fmtEta } from "../src/eta.ts";

function eq(got: string, want: string, msg: string): void {
  if (got !== want) throw new Error(`${msg}: got ${got}, want ${want}`);
}

eq(fmtEta(0), "0:00:00:00", "zero");
eq(fmtEta(59), "0:00:00:59", "seconds only");
eq(fmtEta(60), "0:00:01:00", "one minute");
eq(fmtEta(3661), "0:01:01:01", "hour/min/sec carry");
eq(fmtEta(86400), "1:00:00:00", "one day");
eq(fmtEta(90061), "1:01:01:01", "day plus h:m:s");
eq(fmtEta(15 * 86400 + 14 * 3600 + 49 * 60 + 48), "15:14:49:48", "realistic eta");
eq(fmtEta(89.6), "0:00:01:30", "rounds to nearest second");
eq(fmtEta(Infinity), "--:--:--:--", "infinite (rate still 0)");
eq(fmtEta(-5), "--:--:--:--", "negative");
eq(fmtEta(NaN), "--:--:--:--", "nan");

console.log("eta_fmt: all assertions passed");
