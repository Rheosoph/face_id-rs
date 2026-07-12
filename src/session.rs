use crate::error::FaceIdError;
use ort::{ep::ExecutionProviderDispatch, session::builder::SessionBuilder};

pub(crate) fn configured_session_builder(
    execution_providers: &[ExecutionProviderDispatch],
) -> Result<SessionBuilder, FaceIdError> {
    let has_directml = execution_providers
        .iter()
        .any(|provider| provider.downcast_ref::<ort::ep::DirectML>().is_some());
    let mut builder =
        ort::session::Session::builder()?.with_execution_providers(execution_providers)?;

    // DirectML requires sequential execution and disabled memory patterns for every session.
    if has_directml {
        builder = builder
            .with_memory_pattern(false)?
            .with_parallel_execution(false)?;
    }

    Ok(builder)
}
