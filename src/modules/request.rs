use crate::models::{AgentModuleTrait, Error, ModuleParam};
use sllm::{
    message::{PromptMessageBuilder, PromptMessageGroup},
    Model,
};

#[derive(Debug)]
pub struct RequestModule {
    name: String,
}

impl RequestModule {
    pub fn new() -> Self {
        Self {
            name: "RequestModule".into(),
        }
    }
}

#[async_trait::async_trait]
impl AgentModuleTrait for RequestModule {
    fn get_name(&self) -> String {
        self.name.clone()
    }

    async fn execute(&mut self, model: &Model, input: ModuleParam) -> Result<ModuleParam, Error> {
        log::debug!("[{}] intput - {:?}", self.name, input);
        // ignore the input
        let groups = match input {
            ModuleParam::Str(req) => {
                let mut group = PromptMessageGroup::new("");
                group.insert("Request", req.as_str());
                vec![group]
            }
            ModuleParam::MessageBuilders(builder) => builder,
            ModuleParam::None => {
                return Err(Error::InputRequiredError);
            }
        };

        // generate the response
        model
            .generate_response(&PromptMessageBuilder::new(groups))
            .await
            .map(|result| result.into())
            .map_err(|e| e.into())
    }
}
