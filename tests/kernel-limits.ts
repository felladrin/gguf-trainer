// Standalone assert check: no kernel `src/backend/wgsl.ts` emits may declare
// more workgroup storage than WebGPU's portable floor. Every builder the module
// exports is swept, not only the ones that stage a tile today, so a kernel that
// starts using workgroup memory is covered the moment it does.
//
// Why this is a source check and not a GPU one. `maxComputeWorkgroupStorageSize`
// defaults to 16384, and an implementation that validates it must reject a
// pipeline over that. Deno's wgpu does not validate it at all: probed on an
// M1 Max, `createComputePipeline` accepted a 16640-byte shader on a device
// reporting a granted limit of 16384, with an empty validation scope. So a GPU
// test cannot gate this on the runtime the trainer uses, and an over-budget
// kernel would only fail for someone else, on a stack that does validate.
//
// The property this actually proves is independence: `attnBwdTile` sizes the
// query tile from the formula 8*BT*(hd+1), while this file sums the array
// declarations in the emitted WGSL. The two never consult each other, so this
// catches the formula drifting away from the source it is supposed to describe,
// not merely the source exceeding the floor.
//
// srcAttnBwdDkv is the only kernel that has ever been near the limit: its query
// tile scales with the head dim, and a fixed 32-row tile overflowed from
// head-dim 64 up, the size every published checkpoint trains at. srcGemm is the
// only other real consumer, at a fixed 8 KiB of vec4 tiles: half the floor, which
// is why doubling GEMM_BK again would spend all of it. The rest declare a fixed
// reduction array or nothing, and are here so that stops being true silently. The GPU optimizers (adamw-gpu, muon-gpu) are NOT covered: their
// kernel builders are module-private, and their only workgroup array is a fixed
// 256-lane reduction, 1 KiB, that no parameter grows.
//
// Run:  deno run tests/kernel-limits.ts

import type { AttnDims } from "../src/backend/wgsl.ts";
import {
  bindF32,
  srcAttnBwdD,
  srcAttnBwdDkv,
  srcAttnBwdDq,
  srcAttnDkv,
  srcAttnDq,
  srcAttnDScore,
  srcAttnFwd,
  srcAttnOut,
  srcAttnProbs,
  srcCeBwd,
  srcCeFwd,
  srcCeReduce,
  srcElementwise,
  srcEmbeddingBwd,
  srcEmbeddingFwd,
  srcGemm,
  srcRmsNormBwdW,
  srcRmsNormBwdX,
  srcRmsNormFwd,
  srcRope,
  srcSoftCeBwdP,
  srcSoftCeBwdQ,
  srcSoftCeFwd,
} from "../src/backend/wgsl.ts";

/** WebGPU's spec-default maxComputeWorkgroupStorageSize, in bytes. */
const FLOOR = 16384;

const SIZEOF: Record<string, number> = { f32: 4, u32: 4, i32: 4, "vec4<f32>": 16, "vec2<f32>": 8 };

/**
 * Bytes of `var<workgroup>` a WGSL source declares. Throws rather than
 * under-reporting: a declaration this cannot size would otherwise be skipped
 * while the others still summed, and the total would look fine. Both an unknown
 * element type (a returning f16 variant) and an unknown count form (a WGSL const
 * as the array length, which the private arrays in these same kernels already
 * use) have to fail closed, or the gate can be weakened by an edit elsewhere.
 */
function workgroupBytes(label: string, src: string): number {
  const declared = src.match(/var<workgroup>/g)?.length ?? 0;
  let total = 0;
  let sized = 0;
  const re = /var<workgroup>\s+\w+\s*:\s*array<\s*(.+?)\s*,\s*(\d+)\s*>/g;
  for (const m of src.matchAll(re)) {
    const [, ty, count] = m;
    const size = SIZEOF[ty];
    if (size === undefined) throw new Error(`${label}: unknown workgroup element type: ${ty}`);
    total += size * Number(count);
    sized++;
  }
  if (sized !== declared) {
    throw new Error(
      `${label}: ${declared - sized} var<workgroup> declaration(s) this parser cannot size`,
    );
  }
  return total;
}

function check(label: string, src: string) {
  const bytes = workgroupBytes(label, src);
  if (bytes > FLOOR) {
    throw new Error(
      `${label}: ${bytes} bytes of workgroup storage exceeds the ${FLOOR}-byte portable floor`,
    );
  }
}

// Head dims the trainer can be asked for: 6 and 12 exercise the scalar and
// odd-lane paths, 64 is the published shape, 128 and 256 are the qwen3-style
// sizes that are free of the model width. Only srcAttnBwdDkv stages a tile
// today; the others are swept to catch the day one of them starts to.
for (const hd of [6, 12, 24, 32, 64, 128, 256]) {
  for (const window of [0, 1024]) {
    const a: AttnDims = { T: 2048, Hq: 4, Hkv: 2, hd, window };
    const at = `hd=${hd}, W=${window}`;
    check(`srcAttnFwd(${at})`, srcAttnFwd(a));
    check(`srcAttnBwdD(${at})`, srcAttnBwdD(a));
    check(`srcAttnBwdDq(${at})`, srcAttnBwdDq(a));
    check(`srcAttnBwdDkv(${at})`, srcAttnBwdDkv(a));
    // The materialized small-T path over the same shapes.
    check(`srcAttnProbs(${at})`, srcAttnProbs(a));
    check(`srcAttnOut(${at})`, srcAttnOut(a));
    check(`srcAttnDScore(${at})`, srcAttnDScore(a));
    check(`srcAttnDq(${at})`, srcAttnDq(a));
    check(`srcAttnDkv(${at}, probs)`, srcAttnDkv(a, true));
    check(`srcAttnDkv(${at}, dScore)`, srcAttnDkv(a, false));
  }
  // RoPE has no window; both directions emit different bodies.
  check(`srcRope(hd=${hd}, fwd)`, srcRope(2048, 4, hd, 1e6, 0, false));
  check(`srcRope(hd=${hd}, bwd)`, srcRope(2048, 4, hd, 1e6, 0, true));
}

// The GEMM tiles are fixed-size, so one shape per transpose flavor covers them.
for (const kind of ["NT", "NN", "TN"] as const) {
  check(`srcGemm(${kind})`, srcGemm(kind, false, 70, 75, 83));
}

// The rest, at the largest vocab and width the trainer is used at.
check("srcRmsNormFwd", srcRmsNormFwd(2048, 640, 1e-6));
check("srcRmsNormBwdX", srcRmsNormBwdX(2048, 640));
check("srcRmsNormBwdW", srcRmsNormBwdW(2048, 640));
check("srcEmbeddingFwd", srcEmbeddingFwd(2048, 640));
check("srcEmbeddingBwd", srcEmbeddingBwd(2048, 640, 32768));
check(
  "srcElementwise",
  srcElementwise(
    [bindF32(0, "A", "read"), bindF32(1, "B", "read_write")],
    2048 * 640,
    "B[i] = A[i];",
  ),
);
check("srcCeFwd", srcCeFwd(2048, 32768));
check("srcCeBwd", srcCeBwd(2048, 32768));
check("srcCeReduce", srcCeReduce(2048));
check("srcSoftCeFwd", srcSoftCeFwd(2048, 32768, 16));
check("srcSoftCeBwdP", srcSoftCeBwdP(2048, 32768, 16));
check("srcSoftCeBwdQ", srcSoftCeBwdQ(2048, 32768, 16));

// Prove the attention tile sizing is load-bearing rather than incidentally
// satisfied: it must fall below 32 rows by the head dim that would overflow.
const tileRows = (hd: number) => {
  const m = srcAttnBwdDkv({ T: 2048, Hq: 4, Hkv: 2, hd }).match(/const BT: u32 = (\d+)u;/);
  if (!m) throw new Error(`srcAttnBwdDkv(hd=${hd}): no BT constant found`);
  return Number(m[1]);
};
if (tileRows(32) !== 32) throw new Error(`hd=32 should still stage 32 rows, got ${tileRows(32)}`);
if (tileRows(64) >= 32) throw new Error(`hd=64 must stage fewer than 32 rows, got ${tileRows(64)}`);

console.log("kernel_limits: all assertions passed");
