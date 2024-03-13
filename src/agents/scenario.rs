use std::collections::HashMap;

use tera::{Context, Tera};

use serde::Serialize;
use sllm::Model;

use crate::{
    models::{Error, ModuleCascade, ModuleParam},
    modules::RequestModule,
    prelude::*,
};

#[derive(Debug, Default)]
pub struct ScenarioAgent {
    scenario_template: String,
    response_template: String,
    context: HashMap<String, serde_json::Value>,

    output: ModuleParam,
    agent: ModuleCascade,
}

impl ScenarioAgent {
    pub fn new() -> Self {
        let mut agent = ModuleCascade::new();
        agent.add_module(RequestModule::new());

        Self {
            output: ModuleParam::default(),
            agent,
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
}

#[async_trait::async_trait]
impl AgentTrait for ScenarioAgent {
    fn construct_param(&mut self) -> ModuleParam {
        let mut tera = Tera::default();
        tera.add_raw_template("req", &format!("{}\n\n{{{{ output_template }}}}\n\nYou are a json generator. Generate in json template above.", self.scenario_template))
            .unwrap();
        let mut context = Context::new();
        self.context.iter().for_each(|(k, v)| {
            context.insert(k, v);
        });

        context.insert("output_template", &self.response_template);

        tera.render("req", &context).unwrap().into()
    }

    async fn execute(&mut self, model: &Model) -> Result<(), Error> {
        let args = self.construct_param();
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
    use serde::Deserialize;

    use crate::agents::scenario::ScenarioAgent;

    use crate::models::ModuleParam;
    use crate::{prelude::*, sync::block_on};

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
        let model = crate::tests::get_model();
        env_logger::init();

        let mut agent = ScenarioAgent::new();
        agent.set_scenario_template(REQUEST_FIND_TREASURE_STR);
        agent.insert_context("num_npcs", 5);
        agent.insert_context("num_facilities", 8);
        agent.insert_context("num_clues", 3);
        agent.update_response_template::<ScenarioResponse>();

        match agent.construct_param() {
            ModuleParam::Str(arg) => {
                println!("::Prompt::\n{}", arg);
            }
            _ => {}
        }
        block_on(async move {
            if let Ok(_) = agent.execute(&model).await {
                let result = agent.get_typed_result::<ScenarioResponse>().unwrap();
                println!("{:?}", result);
            }
        });
    }
}
