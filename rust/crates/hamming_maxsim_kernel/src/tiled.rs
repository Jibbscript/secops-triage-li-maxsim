use crate::InnerKernel;

/// Process large doc token arrays in tiles to improve cache locality across query tokens.
///
/// This is intentionally conservative in v0 and simply falls back to the untiled path when the
/// caller provides a tiny tile size.
pub fn maxsim_hamming_tiled(query: &[u64], docs: &[u64], kernel: InnerKernel, tile_tokens: usize) -> u32 {
    if tile_tokens == 0 || docs.len() <= tile_tokens {
        return crate::maxsim_hamming(query, docs, kernel);
    }

    let mut total = 0u32;
    for &q in query {
        let mut best = crate::BITS;
        for tile in docs.chunks(tile_tokens) {
            let local = unsafe { kernel(q, tile) };
            if local < best {
                best = local;
            }
            if best == 0 {
                break;
            }
        }
        total += crate::BITS.saturating_sub(best);
    }
    total
}
