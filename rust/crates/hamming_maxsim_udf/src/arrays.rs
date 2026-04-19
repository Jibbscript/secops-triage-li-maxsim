use datafusion::arrow::array::{FixedSizeListArray, UInt8Array};
use bytemuck::cast_slice;
use datafusion::common::Result;

use crate::errors::invalid_arg;

pub fn fixed_size_list_to_u64_slice(arr: &FixedSizeListArray) -> Result<&[u64]> {
    let values = arr
        .values()
        .as_any()
        .downcast_ref::<UInt8Array>()
        .ok_or_else(|| invalid_arg("expected inner UInt8Array"))?;

    let raw = values.values();
    if raw.len() % 8 != 0 {
        return Err(invalid_arg("binary token buffer length must be a multiple of 8"));
    }
    Ok(cast_slice(raw))
}
