#[inline]
pub unsafe fn hamming_min_scalar(q: u64, docs: &[u64]) -> u32 {
    let mut best = u32::MAX;
    for &d in docs {
        best = best.min((q ^ d).count_ones());
    }
    best
}
