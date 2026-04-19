#![cfg(target_arch = "x86_64")]

use core::arch::x86_64::*;

/// Practical AVX2 path for 8-byte operands:
/// vector xor, then scalar popcount extracted per lane.
#[target_feature(enable = "avx2,popcnt")]
pub unsafe fn hamming_min_avx2(q: u64, docs: &[u64]) -> u32 {
    let qv = _mm256_set1_epi64x(q as i64);
    let mut best = u32::MAX;
    let chunks = docs.chunks_exact(4);
    let rem = chunks.remainder();

    for c in chunks {
        let d = _mm256_loadu_si256(c.as_ptr() as *const __m256i);
        let x = _mm256_xor_si256(qv, d);
        let p0 = (_mm256_extract_epi64::<0>(x) as u64).count_ones();
        let p1 = (_mm256_extract_epi64::<1>(x) as u64).count_ones();
        let p2 = (_mm256_extract_epi64::<2>(x) as u64).count_ones();
        let p3 = (_mm256_extract_epi64::<3>(x) as u64).count_ones();
        best = best.min(p0).min(p1).min(p2).min(p3);
    }

    for &d in rem {
        best = best.min((q ^ d).count_ones());
    }
    best
}
