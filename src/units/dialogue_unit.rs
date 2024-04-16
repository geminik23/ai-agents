use std::slice::Iter;

use sllm::message::PromptMessageGroup;

use crate::{Error, ModuleParam, UnitProcess};

//
// 1. make the PipelineNet
// :
//
// 2. Unit itself?
// : problem is unit only generate the
// : issue is that...

#[derive(Debug, Default)]
pub struct DialogueEntry {
    name: String,
    message: String,
}

#[derive(Debug, Default)]
pub struct DialogueUnit {
    name: String,

    dialogues: Vec<DialogueEntry>,
    responder_name: Option<String>,
}

impl DialogueUnit {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    pub fn clear_dialogue(&mut self) {
        self.dialogues.clear();
    }

    pub fn iter_dialogue(&self) -> Iter<'_, DialogueEntry> {
        self.dialogues.iter()
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

    pub fn remove_last_dialogue(&mut self) -> Option<DialogueEntry> {
        self.dialogues.pop()
    }

    pub fn set_responder_name(&mut self, name: &str) {
        self.responder_name = if name.is_empty() {
            None
        } else {
            Some(name.into())
        };
    }

    fn construct_param(&self) -> PromptMessageGroup {
        let mut group = PromptMessageGroup::new_key_value("Dialogue");
        self.dialogues.iter().for_each(|entry| {
            group.add_message(&entry.name, &entry.message);
        });
        if let Some(responder_name) = &self.responder_name {
            group.add_message(responder_name, "");
        }
        group
    }
}

#[async_trait::async_trait]
impl UnitProcess for DialogueUnit {
    fn get_name(&self) -> &str {
        self.name.as_str()
    }

    async fn process(&self, input: ModuleParam) -> Result<ModuleParam, Error> {
        // input +
        log::debug!("[{}] intput - {:?}", self.name, input);

        // ignore the input
        let mut groups = match input {
            ModuleParam::Str(req) => {
                let mut group = PromptMessageGroup::new_key_value("");
                group.add_message("", req.as_str());
                vec![group]
            }
            ModuleParam::MessageBuilders(builder) => builder,
            ModuleParam::None => {
                vec![]
                // return Err(Error::InputRequiredError);
            }
        };

        groups.push(self.construct_param());

        Ok(ModuleParam::MessageBuilders(groups))
    }
}

#[cfg(test)]
mod tests {
    use sllm::message::PromptMessageBuilder;

    use crate::{prelude::*, sync::block_on, ModuleParam, UnitProcess};

    use super::DialogueUnit;

    /// # Result
    ///
    /// ```text
    /// [Dialogue]
    /// Simply respond what the participant says
    /// Tom: Hello
    /// Jack:
    /// ```
    #[ignore]
    #[test]
    fn test_dialogue_unit() {
        let participants = ["Tom", "Jack"];

        let mut agent = DialogueUnit::new("dialogue");
        agent.add_instruction("Simply respond what the participant says");
        agent.add_dialogue(participants[0], "Hello");
        agent.set_responder_name(participants[1]);

        let Ok(ModuleParam::MessageBuilders(groups)) =
            block_on(async move { agent.process(ModuleParam::None).await })
        else {
            assert!(false);
            return;
        };
        println!("{}", PromptMessageBuilder::new(groups).build().as_str());
    }
}
