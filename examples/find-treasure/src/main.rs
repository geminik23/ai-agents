mod utils;

use ai_agents::{models::Error, sllm::Model};
use find_treasure::{FindTreasure, FindTreasureParam};

struct Simulator {
    agent: FindTreasure,
}

impl Simulator {
    fn new(api_key: &str) -> Self {
        let mut llmodel = Model::new(ai_agents::sllm::Backend::ChatGPT {
            api_key: api_key.to_string(),
            model: "gpt-3.5-turbo".into(),
        })
        .unwrap();
        llmodel.set_temperature(0.1);

        let agent = FindTreasure::new(llmodel, FindTreasureParam::new(4, 6, 3));

        Self { agent }
    }

    async fn generate_scenario(&mut self) -> Result<(), Error> {
        // Generate the scenario
        loop {
            println!("");
            utils::printout_progress("Generating scenario...");

            // Generate new scenario
            self.agent.generate_scenario().await?;

            // get information of game.
            let scenario = self.agent.scenario();

            let Some(scenario) = scenario else {
                return Ok(());
            };

            // PRINT GENERATED BACKGROUND
            println!("::Generated scenario::\n{:?}", scenario);

            println!("");
            if utils::get_yes_or_no("Play this scenario? (y/n) ") {
                break;
            }
        }
        Ok(())
    }

    async fn explore_location(&mut self, location: &str) -> Result<(), Error> {
        let npcs = self.agent.list_npcs_in_facility(location);
        println!("");
        println!("Select the action.");
        npcs.iter().enumerate().for_each(|(i, v)| {
            println!("{}. {}", i + 1, v);
        });

        println!("{}. go out", npcs.len() + 1);
        let Ok(selection) = utils::get_user_response("Type the number : ").parse::<usize>() else {
            return Ok(());
        };

        if selection > 0 && selection <= npcs.len() {
            println!("");
            utils::printout_progress("Generating dialogue...");
            let npc_name = npcs[selection - 1].as_str();
            match self.agent.talk_to(npc_name).await {
                Ok(txt) => {
                    println!("{} : {}", npc_name, txt);
                    println!("");
                }
                Err(err) => {
                    eprintln!("Err - {:?}", err);
                    return Ok(());
                }
            }
        } else {
            self.agent.go_out();
            return Ok(());
        }
        Ok(())
    }

    fn choose_new_location(&mut self) {
        println!("");
        println!("Where will you visit?");
        let selections = self.agent.list_of_facilities();
        selections.iter().enumerate().for_each(|(i, v)| {
            println!("{}. {}", i + 1, v);
        });

        let Ok(selection) = utils::get_user_response("Type the number : ").parse::<usize>() else {
            return;
        };
        if selection > 0 && selection <= selections.len() {
            self.agent
                .move_to_facility(selections.get(selection - 1).unwrap().as_str());
        }
    }

    pub async fn play_game(&mut self) -> Result<(), Error> {
        self.generate_scenario().await?;

        let Some(scenario) = self.agent.scenario() else {
            return Ok(());
        };

        // All description is pre-generated via ChatGPT.
        println!("");
        println!(
            "In the quaint town of {}, legends speak of hidden treasure buried beneath its grounds. {} This place holds secrets waiting to be unearthed by a daring treasure hunter. Your quest begins amidst its lively streets and shadowy alleys, where every encounter could lead you closer to fortune—or into the depths of mystery.",
            scenario.town.name,
            scenario.town.description);
        println!("");
        println!("Welcome to the mystical town of {}. Rumors of hidden treasure have drawn many to its streets.", scenario.town.name);

        let treasure_location = scenario.treasure_place.clone();

        // Main game loop
        while !self.agent.has_found_treasure() {
            if let Some(location) = self.agent.current_location() {
                self.explore_location(location.as_str()).await?;
            } else {
                self.choose_new_location();
            }
        }
        utils::printout_progress(&format!("After a long journey filled with challenges and revelations, you finally uncover the hidden treasure beneath the {}.", treasure_location));

        Ok(())
    }
}

fn main() {
    dotenv::dotenv().ok();

    let api_key = std::env::var("OPEN_API_KEY").expect("Failed to find OPEN_API_KEY");
    let mut simulator = Simulator::new(&api_key);

    loop {
        println!("");
        if utils::get_yes_or_no("Start new simulation of 'Find Treasure'? (y/n) : ") {
            let _ = ai_agents::sync::block_on(simulator.play_game());
        } else {
            break;
        }
    }
}
