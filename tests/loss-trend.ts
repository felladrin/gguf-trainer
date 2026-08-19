// Standalone assert check for the endpoint-mean loss summary (src/loss-trend.ts).
// Run:  deno run tests/loss-trend.ts
import { lossTrend } from "../src/loss-trend.ts";

function ok(cond: boolean, msg: string): void {
  if (!cond) throw new Error(msg);
}

function near(got: number, want: number, msg: string): void {
  if (Math.abs(got - want) > 1e-9) throw new Error(`${msg}: got ${got}, want ${want}`);
}

ok(lossTrend([]) === null, "no logged losses reports nothing rather than 0 -> 0");

const one = lossTrend([2.5])!;
near(one.first, 2.5, "a single log is both ends");
near(one.last, 2.5, "a single log is both ends");
ok(one.window === 1, "the window cannot be smaller than one sample");

// The windows must not overlap, or a short run reports the same batches as its
// own before and after and every run looks flat.
const three = lossTrend([3, 2, 1])!;
ok(three.window === 1, "three logs give a window of one, not two");
near(three.first, 3, "first window is the head");
near(three.last, 1, "last window is the tail");

const four = lossTrend([4, 3, 2, 1])!;
ok(four.window === 2, "four logs give a window of two");
near(four.first, 3.5, "mean of the first two");
near(four.last, 1.5, "mean of the last two");

// Past 2*window the window stops growing, so long runs compare fixed-size ends.
const many = lossTrend(Array.from({ length: 100 }, (_, i) => i))!;
ok(many.window === 10, "the window saturates at its cap");
near(many.first, 4.5, "mean of losses 0..9");
near(many.last, 94.5, "mean of losses 90..99");

// The regression this exists for: endpoint batches that disagree with the trend.
// Head mean 3.0 and tail mean 2.0, but the first batch is low and the last high.
const noisy = [2.0, 3.5, 3.5, 3.0, 1.0, 1.5, 2.0, 3.5];
const t = lossTrend(noisy, 4)!;
near(t.first, 3.0, "head mean ignores the lucky first batch");
near(t.last, 2.0, "tail mean ignores the unlucky last batch");
ok(noisy[0] < noisy[noisy.length - 1], "the raw endpoints do report an increase");
ok(t.first > t.last, "the means report the improvement that actually happened");

console.log("loss-trend: all checks passed");
