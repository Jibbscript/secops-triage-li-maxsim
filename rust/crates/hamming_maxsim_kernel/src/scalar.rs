#[inline]
/// # Safety
///
/// `docs` must contain packed 64-bit token rows that use only the low 48 bits.
/// The function performs no pointer arithmetic and does not retain references.
pub unsafe fn hamming_min_scalar(q: u64, docs: &[u64]) -> u32 {
    let mut best = u32::MAX;
    for &d in docs {
        best = best.min((q ^ d).count_ones());
    }
    best
}
