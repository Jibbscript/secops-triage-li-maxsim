#![cfg(target_arch = "x86_64")]

use core::arch::x86_64::*;

/// Opportunistic AVX-512 VPOPCNTDQ path.
/// Runtime dispatch is mandatory because many consumer Intel mobile parts fuse AVX-512 off.
#[target_feature(enable = "avx512f,avx512vpopcntdq,avx512bw,avx512vl")]
pub unsafe fn hamming_min_avx512(q: u64, docs: &[u64]) -> u32 {
    let qv = _mm512_set1_epi64(q as i64);
    let mut best = _mm512_set1_epi64(48);
    let chunks = docs.chunks_exact(8);
    let rem = chunks.remainder();

    for c in chunks {
        let d = _mm512_loadu_si512(c.as_ptr() as *const __m512i);
        let x = _mm512_xor_si512(qv, d);
        let p = _mm512_popcnt_epi64(x);
        best = _mm512_min_epu64(best, p);
    }

    let lo = _mm512_castsi512_si256(best);
    let hi = _mm512_extracti64x4_epi64(best, 1);
    let m = _mm256_min_epu64(lo, hi);

    let mut buf = [0u64; 4];
    _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, m);

    let mut out = buf.iter().copied().min().unwrap() as u32;
    for &d in rem {
        out = out.min((q ^ d).count_ones());
    }
    out
}
