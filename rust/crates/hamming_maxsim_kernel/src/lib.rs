pub mod dispatch;
pub mod scalar;
#[cfg(target_arch = "x86_64")]
pub mod avx2;
#[cfg(target_arch = "x86_64")]
pub mod avx512;
#[cfg(target_arch = "aarch64")]
pub mod neon;
pub mod tiled;

pub const BITS: u32 = 48;
pub const STRIDE: usize = 8;

pub type InnerKernel = unsafe fn(u64, &[u64]) -> u32;

/// similarity(query, doc) = sum_q (48 - min_hamming(q, doc_tokens))
#[inline]
pub fn maxsim_hamming(query: &[u64], docs: &[u64], kernel: InnerKernel) -> u32 {
    let mut sim = 0u32;
    for &q in query {
        let min_h = unsafe { kernel(q, docs) };
        sim += BITS.saturating_sub(min_h);
    }
    sim
}
