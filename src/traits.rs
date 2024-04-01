use crate::{error::Error, ModuleParam};

pub trait Adapter: Send + Sync {
    fn adapt(&self, input: ModuleParam) -> ModuleParam;
}

impl<F: Fn(ModuleParam) -> ModuleParam + Send + Sync + 'static> Adapter for F {
    fn adapt(&self, input: ModuleParam) -> ModuleParam {
        self(input)
    }
}

#[async_trait::async_trait]
pub trait UnitProcess: Send + Sync {
    fn get_name(&self) -> &str;
    async fn process(&self, input: ModuleParam) -> Result<ModuleParam, Error>;
}
