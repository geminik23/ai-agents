use std::collections::HashMap;

use serde::Deserialize;
use std::sync::Arc;

use ai_agents::{
    prelude::*,
    sync::RwLock,
    units::{DialogueUnit, JsonGeneratorUnit, ModelUnit},
    Error, Model, ModuleParam, PipelineNet,
};

#[derive(Clone, Debug, Deserialize)]
pub struct EntityDescription {
    pub name: String,
    pub description: String,
}

impl ToString for EntityDescription {
    fn to_string(&self) -> String {
        format!("{} - {}", self.name, self.description)
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct CharacterDescription {
    pub name: String,
    pub job: String,
    pub location: String,
    pub description: String,
}

impl ToString for CharacterDescription {
    fn to_string(&self) -> String {
        format!(
            "{}({} in {}) - {}",
            self.name, self.job, self.location, self.description
        )
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug, Deserialize)]
pub struct Scenario {
    pub town: EntityDescription,
    pub facilities: Vec<EntityDescription>,
    pub characters: Vec<CharacterDescription>,
    pub visit_order: Vec<String>,
    pub treasure_place: String,
}

impl ToString for Scenario {
    fn to_string(&self) -> String {
        format!("\nTown {}\n\nFacilities({}): {}\n\nCharacters({}): {}\n\nVisit Order: {}\n\nTreasure Place: {}\n\n", 
            self.town.to_string(),
            self.facilities.len(), self.facilities.iter().map(|v|v.to_string()).collect::<Vec<_>>().join("\n"),
            self.characters.len(), self.characters.iter().map(|v|v.to_string()).collect::<Vec<_>>().join("\n"),
            self.visit_order.join(", "),
            self.treasure_place,
            )
    }
}

impl Scenario {
    pub fn construct_background_message(&self) -> PromptMessage {
        let mut group = PromptMessage::new_key_value("Background");
        group.add_message("Town", self.town.to_string().as_str());
        group.add_message(
            "Facilities",
            self.facilities
                .iter()
                .map(|e| e.to_string())
                .collect::<Vec<String>>()
                .join(";")
                .as_str(),
        );
        group.add_message(
            "Characters",
            self.characters
                .iter()
                .map(|e| e.to_string())
                .collect::<Vec<String>>()
                .join(";")
                .as_str(),
        );
        group.add_message("Treasure Place", &self.treasure_place);
        group.add_message("Goal", "Find the treasure.");
        group.add_message("Play method", "Get the clue by visiting NPCs in order.");
        group.add_message("Visit order", &self.visit_order.join(","));
        group
    }
}

impl ToKeywordString for Scenario {
    fn to_keyword_string() -> String {
        "{town{name, description}, facilities[{name, description}], characters[{name, job, location, description}], visit_order, treasure_place}".into()
    }
}

// [Rules]
// 1. If the player talks to the NPC listed under 'Next', then the NPC must immediately mention the next NPC in the visit order.
// 2. If it's not the NPC in that order, the NPC should engage in small talk.
// 3. The NPC in the last order must reveal the location of the treasure.
const RULES: [&'static str; 3] = [
    "If the player talks to the NPC listed under 'Next', then the NPC must immediately mention the next NPC in the visit order.", 
    "If it's not the NPC in that order, the NPC should engage in small talk.", 
    "The NPC in the last order must reveal the location of the treasure."
];

#[derive(Debug)]
pub struct GameState {
    description: String,
    treasure_place: String,

    facilities: HashMap<String, Vec<String>>,
    npcs: HashMap<String, CharacterDescription>,
    play_order: Vec<String>,
    visited_count: usize, // index for visit_order

    current_location: Option<String>,
    current_npc: Option<String>,
    dialogue: String,
}

impl GameState {
    pub fn new(scenario: &Scenario) -> Self {
        let npcs = scenario
            .characters
            .iter()
            .map(|v| (v.name.clone(), v.clone()))
            .collect::<HashMap<_, _>>();
        let mut facilities = scenario
            .facilities
            .iter()
            .map(|v| (v.name.clone(), Vec::new()))
            .collect::<HashMap<_, _>>();

        npcs.iter().for_each(|(v, desc)| {
            facilities
                .get_mut(&desc.location)
                .map(|l| l.push(v.clone()));
        });

        Self {
            description: String::new(),
            npcs,
            play_order: scenario.visit_order.clone(),
            visited_count: 0,
            facilities,
            current_location: None,
            current_npc: None,
            dialogue: String::new(),
            treasure_place: scenario.treasure_place.clone(),
        }
    }

    pub fn visited_count(&self) -> usize {
        self.visited_count
    }

    pub fn set_game_introduction(&mut self, desc: String) {
        self.description = desc;
    }

    pub fn game_introduction(&self) -> &str {
        &self.description
    }

    pub fn visited_all(&self) -> bool {
        self.play_order.len() == self.visited_count
    }

    pub fn has_npc(&self, name: &str) -> bool {
        self.npcs.contains_key(name)
    }

    pub fn visit(&mut self, name: &str) {
        self.current_npc = Some(name.into());
        if self.play_order.len() == self.visited_count {
            return;
        }

        if let Some(n) = self.play_order.get(self.visited_count) {
            if n == name {
                self.visited_count += 1;
            }
        }
    }

    pub fn is_in_treasure_place(&self) -> bool {
        self.current_location
            .as_deref()
            .map(|v| v == self.treasure_place)
            .unwrap_or(false)
    }

    pub fn construct_game_state(
        &self,
        scene_response: &Scenario,
        completed_order_count: usize,
        cur_npc: &str,
    ) -> PromptMessage {
        let mut game_state = PromptMessage::new_key_value("Game State");
        game_state.add_message("Goal", "Find the treasure location.");
        game_state.add_message("Player", "Treasure Hunter.");
        game_state.add_message("Visit Order", scene_response.visit_order.join(",").as_str());
        game_state.add_message(
            "Visited",
            scene_response
                .visit_order
                .iter()
                .take(completed_order_count)
                .enumerate()
                .map(|(_, v)| format!("{}", v))
                .collect::<Vec<_>>()
                .join(", ")
                .as_str(),
        );

        if let Some(next) = scene_response.visit_order.get(completed_order_count) {
            let is_last_npc = completed_order_count >= scene_response.visit_order.len() - 1;
            game_state.add_message("Next", next.as_str());
            game_state.add_message("", "");
            game_state.add_message(
                "",
                RULES[if next == cur_npc {
                    if is_last_npc {
                        2
                    } else {
                        0
                    }
                } else {
                    1
                }],
            );
        }

        game_state
    }

    pub fn list_of_facilities(&self) -> Vec<String> {
        self.facilities.keys().cloned().collect()
    }

    pub fn has_found_treasure(&self) -> bool {
        self.visited_all() && self.is_in_treasure_place()
    }

    pub fn go_out(&mut self) {
        self.current_location = None;
        self.current_npc = None;
    }

    pub fn visit_npc(&mut self, npc: &str) {
        self.current_npc = Some(npc.to_string());
    }

    pub fn move_to_facility(&mut self, facility_name: &str) {
        if self.facilities.contains_key(facility_name) {
            self.current_location = Some(facility_name.into());
            self.current_npc = None;
        }
    }

    // Queries
    pub fn get_npc_info(&self, npc_name: &str) -> Option<CharacterDescription> {
        self.npcs.get(npc_name).cloned()
    }

    pub fn list_npcs_in_facility(&self, facility_name: &str) -> Vec<String> {
        self.facilities
            .get(facility_name)
            .cloned()
            .unwrap_or_default()
    }

    pub fn current_location(&self) -> Option<String> {
        self.current_location.clone()
    }

    pub fn current_npc(&self) -> Option<String> {
        self.current_npc.clone()
    }

    pub fn set_dialogue(&mut self, dialogue: String) {
        self.dialogue = dialogue;
    }

    pub fn dialogue(&self) -> String {
        self.dialogue.clone()
    }

    pub fn treasure_place(&self) -> String {
        self.treasure_place.clone()
    }
}

const REQUEST_FIND_TREASURE_STR: &'static str = r#"This is the one episode in RPG game.

The goal is to find the treasure in the town.

Generate background of town include facilities and NPCs with background, and the orders player talk to who to finish the game.

The way of finding treasure is to talk to NPCs in town in specific order and goes to some place.

There are {{ num_npcs }} NPCs and {{ num_facilities }} facilities in Town, And only {{ num_clues }} NPCs have clue. 

Treasure location is facility. 

Do not duplicate treasure location with locations of characters who have clue. 

The number of NPCs who have clues should be equal with the number of visit order."#;
// #[derive(Debug)]
pub struct FindTreasureAgent {
    // scenario_unit: Arc<RwLock<JsonGeneratorUnit>>,
    dialogue_unit: Arc<RwLock<DialogueUnit>>,

    pipeline_net: PipelineNet,

    param: FindTreasureParam,
}

impl FindTreasureAgent {
    pub fn new(model: Model, param: FindTreasureParam) -> Self {
        let mut scenario_unit = JsonGeneratorUnit::new("scenario");
        scenario_unit.update_response_template::<Scenario>();
        let scenario_unit = Arc::new(RwLock::new(scenario_unit));

        let mut dialogue_unit = DialogueUnit::new("dialogue");
        dialogue_unit.add_dialogue("", "Player meets NPC.");
        let dialogue_unit = Arc::new(RwLock::new(dialogue_unit));

        let model_unit = Arc::new(RwLock::new(ModelUnit::new("chatgpt", model)));

        let mut pipeline_net = PipelineNet::new();
        pipeline_net.add_node("scene_in", scenario_unit.clone());
        pipeline_net.add_node("dialogue_in", dialogue_unit.clone());
        pipeline_net.add_node("out", model_unit);

        pipeline_net.add_edge("scene_in", "out");
        pipeline_net.add_edge("dialogue_in", "out");

        pipeline_net.set_group_input("scenario", "scene_in");
        pipeline_net.set_group_input("dialogue", "dialogue_in");

        Self {
            // scenario_unit,
            dialogue_unit,
            pipeline_net,
            param,
        }
    }

    pub fn update_param(&mut self, param: FindTreasureParam) {
        self.param = param;
    }

    // generate the Scenario
    pub async fn generate_scenario(&mut self) -> Result<(GameState, Scenario), Error> {
        // update into the scenario_agent
        let mut in_param = TemplatedMessage::new(REQUEST_FIND_TREASURE_STR);
        in_param.insert("num_npcs", &self.param.num_npcs);
        in_param.insert("num_facilities", &self.param.num_facilities);
        in_param.insert("num_clues", &self.param.num_clues);

        let responses = self
            .pipeline_net
            .process_group("scenario", ModuleParam::None)
            .await?;

        let scenario: Scenario =
            serde_json::from_str(responses.get("out").unwrap().as_string().unwrap())?;

        // set the game state and background
        let game_state = GameState::new(&scenario);

        Ok((game_state, scenario))
    }

    //
    //  ACTIONS
    pub async fn talk_to(
        &mut self,
        scenario_prompt: PromptMessage,
        game_state_prompt: PromptMessage,
        npc_name: &str,
    ) -> Result<String, Error> {
        // let Some(scenario) = &self.scenario else {
        //     return Err(Error::NotFound("Scenario Response".into()));
        // };

        {
            let mut unit = self.dialogue_unit.write().await;
            unit.set_responder_name(npc_name);
        }

        // let state = self.construct_game_state(&scenario, game_state.visited_count, npc_name);
        let param = ModuleParam::MessageBuilders(vec![scenario_prompt, game_state_prompt]);
        let mut responses = self.pipeline_net.process_group("dialogue", param).await?;
        let dialogue = responses
            .remove("out")
            .unwrap()
            .into_string()
            .ok_or(Error::WrongOutputType)?;

        // udpate the visit
        // if let Some(game_state) = &mut self.game_state {
        //     game_state.visit(npc_name);
        // }

        Ok(dialogue)
    }
}

#[derive(Debug, Default)]
pub struct FindTreasureParam {
    pub num_npcs: u32,
    pub num_facilities: u32,
    pub num_clues: u32,
}

impl FindTreasureParam {
    pub fn new(num_npcs: u32, num_facilities: u32, num_clues: u32) -> Self {
        Self {
            num_npcs,
            num_facilities,
            num_clues,
        }
    }
}
