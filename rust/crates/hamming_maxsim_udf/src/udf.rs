use std::any::Any;
use std::sync::Arc;

use datafusion::arrow::array::{
    Array, ArrayRef, FixedSizeListArray, Float32Builder, GenericListArray,
};
use datafusion::arrow::datatypes::{DataType, Field};
use datafusion::common::Result;
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};

use hamming_maxsim_kernel::{dispatch, maxsim_hamming, STRIDE};

use crate::arrays::fixed_size_list_to_u64_slice;
use crate::errors::invalid_arg;

#[derive(Debug)]
pub struct HammingMaxSim {
    sig: Signature,
}

impl HammingMaxSim {
    pub fn new() -> Self {
        let token = DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::UInt8, false)),
            STRIDE as i32,
        );
        let mv = DataType::List(Arc::new(Field::new("item", token, false)));
        Self {
            sig: Signature::exact(vec![mv.clone(), mv], Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for HammingMaxSim {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "hamming_maxsim"
    }

    fn signature(&self) -> &Signature {
        &self.sig
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float32)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let arrs = ColumnarValue::values_to_arrays(&args.args)?;
        let q_list = arrs[0]
            .as_any()
            .downcast_ref::<GenericListArray<i32>>()
            .ok_or_else(|| invalid_arg("arg0 must be List<FixedSizeList<UInt8,8>>"))?;
        let d_list = arrs[1]
            .as_any()
            .downcast_ref::<GenericListArray<i32>>()
            .ok_or_else(|| invalid_arg("arg1 must be List<FixedSizeList<UInt8,8>>"))?;

        if q_list.len() != d_list.len() {
            return Err(invalid_arg("argument row counts must match"));
        }

        let kernel = dispatch::pick();
        let mut out = Float32Builder::with_capacity(q_list.len());

        for i in 0..q_list.len() {
            if q_list.is_null(i) || d_list.is_null(i) {
                out.append_null();
                continue;
            }

            let q_row = q_list.value(i);
            let d_row = d_list.value(i);

            let q_fsl = q_row
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| invalid_arg("arg0 row must be FixedSizeList"))?;
            let d_fsl = d_row
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| invalid_arg("arg1 row must be FixedSizeList"))?;

            let q_u64 = fixed_size_list_to_u64_slice(q_fsl)?;
            let d_u64 = fixed_size_list_to_u64_slice(d_fsl)?;

            let score = maxsim_hamming(q_u64, d_u64, kernel);
            out.append_value(score as f32);
        }

        Ok(ColumnarValue::Array(Arc::new(out.finish()) as ArrayRef))
    }
}

pub fn build_udf() -> ScalarUDF {
    ScalarUDF::from(HammingMaxSim::new())
}
