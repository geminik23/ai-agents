use sllm::message::PromptMessageGroup;

use crate::{Error, Model, ModuleParam, UnitProcess};

#[derive(Debug, Clone)]
pub struct ModelUnit {
    name: String,
    model: Model,
}

impl ModelUnit {
    pub fn new(name: &str, model: Model) -> Self {
        Self {
            name: name.into(),
            model,
        }
    }
}

#[async_trait::async_trait]
impl UnitProcess for ModelUnit {
    fn get_name(&self) -> &str {
        self.name.as_str()
    }

    async fn process(&self, input: ModuleParam) -> Result<ModuleParam, Error> {
        log::debug!("[{}] intput - {:?}", self.name, input);
        // ignore the input
        let groups = match input {
            ModuleParam::Str(req) => {
                let mut group = PromptMessageGroup::new("");
                group.insert("", req.as_str());
                vec![group]
            }
            ModuleParam::MessageBuilders(builder) => builder,
            ModuleParam::None => {
                vec![]
                // return Err(Error::InputRequiredError);
            }
        };

        // generate the response
        self.model
            .generate_response(groups)
            .await
            .map(|result| result.into())
            .map_err(|e| e.into())
    }
}
