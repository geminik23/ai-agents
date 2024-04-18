use sllm::message::TemplatedMessage;

use crate::{prelude::*, Error, ModuleParam, UnitProcess};

#[derive(Debug, Default)]
pub struct JsonGeneratorUnit {
    name: String,
    response_template: String,
}

impl JsonGeneratorUnit {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    pub fn update_response_template<T: ToKeywordString>(&mut self) {
        self.response_template = T::to_keyword_string();
    }

    fn construct_param(&self) -> PromptMessage {
        let mut templated = TemplatedMessage::new(
            "{{ output_template }}\n\nYou are a json generator. Generate in json template above.",
        );
        templated.insert("output_template", &self.response_template);
        templated.into()
    }
}

#[async_trait::async_trait]
impl UnitProcess for JsonGeneratorUnit {
    fn get_name(&self) -> &str {
        self.name.as_str()
    }

    async fn process(&self, input: ModuleParam) -> Result<ModuleParam, Error> {
        // input +
        log::debug!("[{}] intput - {:?}", self.name, input);

        // ignore the input
        let mut groups = match input {
            ModuleParam::Str(req) => {
                vec![req.into()]
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
    use serde::Deserialize;
    use sllm::message::{PromptMessageBuilder, TemplatedMessage};

    use crate::{prelude::*, sync::block_on};
    use crate::{ModuleParam, UnitProcess};

    use super::JsonGeneratorUnit;

    const REQUEST_FIND_TREASURE_STR: &'static str = r#"This is the one episode in RPG game.

The goal is to find the treasure in the town.

The way of finding treasure is to talk to NPCs in town in specific order and goes to some place.
Generate background of town include facilities and NPCs with background, and the orders player talk to who to finish the game.

There are {{ num_npcs }} NPCs and {{ num_facilities }} facilities in Town, And only {{ num_clues }} NPCs to visit have clue among them.
Treasure location is in facility. Do not duplicate treasure location and the locations of characters who have clue."#;

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct EntityDescription {
        name: String,
        description: String,
    }

    impl ToKeywordString for EntityDescription {
        fn to_keyword_string() -> String {
            "{name, description}".into()
        }
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct ScenarioResponse {
        town: EntityDescription,
        facilities: Vec<EntityDescription>,
        characters: Vec<EntityDescription>,
        visit_order: Vec<String>,
        treasure_place: String,
    }

    impl ToKeywordString for ScenarioResponse {
        fn to_keyword_string() -> String {
            "{town{name, description}, facilities[{name, description}], characters[{name, description}], visit_order, treasure_place}".into()
        }
    }

    #[ignore]
    #[test]
    fn test_find_treasure() {
        let mut unit = JsonGeneratorUnit::new("ScenarioUnit");
        unit.update_response_template::<ScenarioResponse>();

        let mut templated = TemplatedMessage::new(REQUEST_FIND_TREASURE_STR);
        templated.insert("num_npcs", &5);
        templated.insert("num_facilities", &8);
        templated.insert("num_clues", &3);

        let Ok(ModuleParam::MessageBuilders(groups)) = block_on(async move {
            unit.process(ModuleParam::MessageBuilders(vec![templated.into()]))
                .await
        }) else {
            assert!(false);
            return;
        };
        println!("{}", PromptMessageBuilder::new(groups).build().as_str());
    }
}
