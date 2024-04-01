use std::collections::HashMap;

use tera::{Context, Tera};

use serde::Serialize;

use crate::{prelude::*, Error, ModuleParam, UnitProcess};

#[derive(Debug, Default)]
pub struct ScenarioUnit {
    name: String,
    scenario_template: String,
    response_template: String,
    context: HashMap<String, serde_json::Value>,
}

impl ScenarioUnit {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    pub fn update_response_template<T: ToKeywordString>(&mut self) {
        self.response_template = T::to_keyword_string();
    }

    pub fn set_scenario_template(&mut self, scenario: &str) {
        self.scenario_template = scenario.to_string();
    }

    pub fn insert_context<T: Serialize>(&mut self, key: &str, value: T) {
        self.context
            .insert(key.into(), serde_json::to_value(value).unwrap());
    }

    fn construct_param(&self) -> PromptMessageGroup {
        let mut tera = Tera::default();
        tera.add_raw_template("req", &format!("{}\n\n{{{{ output_template }}}}\n\nYou are a json generator. Generate in json template above.", self.scenario_template))
            .unwrap();
        let mut context = Context::new();
        self.context.iter().for_each(|(k, v)| {
            context.insert(k, v);
        });

        context.insert("output_template", &self.response_template);

        let mut group = PromptMessageGroup::new("");
        group.insert("", tera.render("req", &context).unwrap().as_str());
        group
    }
}

#[async_trait::async_trait]
impl UnitProcess for ScenarioUnit {
    fn get_name(&self) -> &str {
        self.name.as_str()
    }

    async fn process(&self, input: ModuleParam) -> Result<ModuleParam, Error> {
        // input +
        log::debug!("[{}] intput - {:?}", self.name, input);

        // ignore the input
        let mut groups = match input {
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

        groups.push(self.construct_param());

        Ok(ModuleParam::MessageBuilders(groups))
    }
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;
    use sllm::message::PromptMessageBuilder;

    use crate::{prelude::*, sync::block_on};
    use crate::{ModuleParam, UnitProcess};

    use super::ScenarioUnit;

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
            "{town{name, descriptoin}, facilities[{name, description}], characters[{name, description}], visit_order, treasure_place}".into()
        }
    }

    #[ignore]
    #[test]
    fn test_find_treasure() {
        let mut unit = ScenarioUnit::new("ScenarioUnit");
        unit.set_scenario_template(REQUEST_FIND_TREASURE_STR);
        unit.insert_context("num_npcs", 5);
        unit.insert_context("num_facilities", 8);
        unit.insert_context("num_clues", 3);
        unit.update_response_template::<ScenarioResponse>();

        let Ok(ModuleParam::MessageBuilders(groups)) =
            block_on(async move { unit.process(ModuleParam::None).await })
        else {
            assert!(false);
            return;
        };
        println!("{}", PromptMessageBuilder::new(groups).build().as_str());
    }
}
