use bytemuck::cast_slice;
use datafusion::arrow::array::{FixedSizeListArray, UInt8Array};
use datafusion::common::Result;

use crate::errors::invalid_arg;

pub const LOW48_MASK: u64 = (1u64 << 48) - 1;

pub fn fixed_size_list_to_u64_vec(arr: &FixedSizeListArray) -> Result<Vec<u64>> {
    let values = arr
        .values()
        .as_any()
        .downcast_ref::<UInt8Array>()
        .ok_or_else(|| invalid_arg("expected inner UInt8Array"))?;

    let raw = values.values();
    if raw.len() % 8 != 0 {
        return Err(invalid_arg(
            "binary token buffer length must be a multiple of 8",
        ));
    }
    Ok(cast_slice::<u8, u64>(raw)
        .iter()
        .map(|value| value & LOW48_MASK)
        .collect())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::array::{FixedSizeListArray, UInt8Array};
    use datafusion::arrow::datatypes::{DataType, Field};

    use super::*;

    #[test]
    fn fixed_size_list_masks_upper_16_bits() {
        let raw = Arc::new(UInt8Array::from(vec![
            0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA, 0x99, 0x88,
        ]));
        let arr = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::UInt8, false)),
            8,
            raw,
            None,
        );

        let out = fixed_size_list_to_u64_vec(&arr).unwrap();

        assert_eq!(out, vec![0x0000_AABB_CCDD_EEFF]);
    }
}
