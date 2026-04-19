use datafusion::common::DataFusionError;

pub fn invalid_arg(msg: impl Into<String>) -> DataFusionError {
    DataFusionError::Execution(msg.into())
}
