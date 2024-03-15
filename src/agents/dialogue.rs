use sllm::{
    message::{PromptMessageBuilder, PromptMessageGroup},
    Model,
};

use crate::{
    models::{Error, ModuleCascade, ModuleParam},
    modules::RequestModule,
    prelude::*,
};

#[derive(Debug, Default)]
pub struct DialogueEntry {
    name: String,
    message: String,
}

#[derive(Debug, Default)]
pub struct DialogueAgent {
    background: Vec<PromptMessageGroup>,

    dialogues: Vec<DialogueEntry>,
    responder_name: Option<String>,

    output: ModuleParam,
    agent: ModuleCascade,
}

impl DialogueAgent {
    pub fn new() -> Self {
        let mut agent = ModuleCascade::new();
        agent.add_module(RequestModule::new());

        Self {
            agent,
            ..Default::default()
        }
    }

    pub fn clear_dialogue(&mut self) {
        self.dialogues.clear();
    }

    pub fn add_instruction(&mut self, instruction: &str) {
        self.add_dialogue("", instruction)
    }

    pub fn add_dialogue(&mut self, name: &str, message: &str) {
        let entry = DialogueEntry {
            name: name.into(),
            message: message.into(),
        };
        self.dialogues.push(entry);
    }

    pub fn set_responder_name(&mut self, name: &str) {
        self.responder_name = if name.is_empty() {
            None
        } else {
            Some(name.into())
        };
    }

    pub fn set_background(&mut self, background: Vec<PromptMessageGroup>) {
        self.background = background;
    }

    pub fn update_last_reponse(&mut self) {
        let responder_name = self.responder_name.take();
        match self.output.clone() {
            ModuleParam::Str(s) => {
                self.add_dialogue(responder_name.as_deref().map_or("", |v| v), s.as_str());
            }
            _ => {}
        }
        self.output = ModuleParam::None;
    }
}

#[async_trait::async_trait]
impl AgentTrait for DialogueAgent {
    fn construct_param(&mut self) -> ModuleParam {
        // FIXME changed to ref later.
        let mut group = PromptMessageGroup::new("Dialogue");
        self.dialogues.iter().for_each(|entry| {
            group.insert(&entry.name, &entry.message);
        });
        if let Some(responder_name) = &self.responder_name {
            group.insert(responder_name, "");
        }

        let mut args = self.background.clone();
        args.push(group);
        ModuleParam::MessageBuilders(args)
    }

    async fn execute(&mut self, model: &Model) -> Result<(), Error> {
        let args = self.construct_param();

        // match &args {
        //     ModuleParam::Str(s) => {
        //         log::debug!("Context Message: {}", s);
        //     }
        //     ModuleParam::MessageBuilders(msgs) => {
        //         log::debug!(
        //             "Context Message: \n{}",
        //             PromptMessageBuilder::new(msgs.clone()).build()
        //         );
        //     }
        //     ModuleParam::None => {
        //         log::debug!("Context Message: None");
        //     }
        // }

        let result = self.agent.execute(model, args).await?;
        self.output = result;
        Ok(())
    }

    fn get_result(&self) -> &ModuleParam {
        &self.output
    }
}

#[cfg(test)]
mod tests {
    use crate::{models::ModuleParam, prelude::*, sync::block_on};
    use sllm::message::{MessageBuilder, PromptMessageBuilder};

    use crate::tests::get_model;

    use super::DialogueAgent;

    /// # Result
    ///
    /// ```text
    /// Simply respond what the participant says
    /// Tom: Hello
    /// Jack: Hi Tom, how are you doing?
    /// Tom: I'm doing well, thanks for asking. How about you?
    /// Jack: I'm doing great, thanks for asking.
    /// Tom: That's good to hear. Anything exciting happening with you lately?
    /// Jack: Not much really, just the usual. How about you, anything new going on in your life?
    /// Tom: Well, I recently started a new job, so that's been keeping me pretty busy.
    /// ```
    #[ignore]
    #[test]
    fn test_memory_dialogue() {
        let mut model = get_model();

        model.set_temperature(0.8);

        let participants = ["Tom", "Jack"];

        let mut agent = DialogueAgent::new();
        agent.add_instruction("Simply respond what the participant says");
        agent.add_dialogue(participants[0], "Hello");
        agent.set_responder_name(participants[1]);

        block_on(async move {
            for i in 0..6 {
                if let Ok(_) = agent.execute(&model).await {
                    agent.update_last_reponse();
                    agent.set_responder_name(participants[i % 2]);
                }
            }

            match agent.construct_param() {
                ModuleParam::MessageBuilders(arg) => {
                    println!("::Prompt::\n{}", PromptMessageBuilder::new(arg).build());
                }
                _ => {}
            }
        });
    }

    /// # Result
    ///
    /// ```text
    /// ::Prompt::
    /// Simply respond what the player says
    /// Player: Hello
    /// Jack:
    /// Str("Hey there! How can I help you today?")
    ///
    /// ```
    #[ignore]
    #[test]
    fn test_onetime_dialogue() {
        let mut model = get_model();

        model.set_temperature(0.8);

        let mut agent = DialogueAgent::new();

        agent.add_instruction("Simply respond what the player says");
        agent.add_dialogue("Player", "Hello");
        agent.set_responder_name("Jack");

        block_on(async move {
            match agent.construct_param() {
                ModuleParam::MessageBuilders(arg) => {
                    println!("::Prompt::\n{}", PromptMessageBuilder::new(arg).build());
                }
                _ => {}
            }
            if let Ok(_) = agent.execute(&model).await {
                println!("{:?}", agent.get_result());
            }
        });
    }
}
