/// Placeholder for an explicit NEON implementation.
/// For 8-byte operands on Apple silicon, an aggressively unrolled scalar popcount path
/// often ties or wins. Keep this file so the crate layout is stable when benchmarking says
/// a NEON path is worthwhile.
#[inline]
/// # Safety
///
/// `docs` must contain packed 64-bit token rows that use only the low 48 bits.
/// This placeholder path performs scalar operations only and does not assume alignment.
pub unsafe fn hamming_min_neon(q: u64, docs: &[u64]) -> u32 {
    let mut best = u32::MAX;
    for chunk in docs.chunks_exact(8) {
        let p0 = (q ^ chunk[0]).count_ones();
        let p1 = (q ^ chunk[1]).count_ones();
        let p2 = (q ^ chunk[2]).count_ones();
        let p3 = (q ^ chunk[3]).count_ones();
        let p4 = (q ^ chunk[4]).count_ones();
        let p5 = (q ^ chunk[5]).count_ones();
        let p6 = (q ^ chunk[6]).count_ones();
        let p7 = (q ^ chunk[7]).count_ones();
        best = best
            .min(p0)
            .min(p1)
            .min(p2)
            .min(p3)
            .min(p4)
            .min(p5)
            .min(p6)
            .min(p7);
    }
    for &d in docs.chunks_exact(8).remainder() {
        best = best.min((q ^ d).count_ones());
    }
    best
}
