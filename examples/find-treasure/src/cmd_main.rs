use ai_agents::{Backend, Error, Model};
use find_treasure::{FindTreasureAgent, FindTreasureParam, GameState, Scenario};

use crossterm::{style, ExecutableCommand};

pub fn printout_text(text: &str, color: style::Color) {
    let mut stdout: std::io::Stdout = std::io::stdout();
    stdout.execute(style::SetForegroundColor(color)).unwrap();
    println!("{}", text);
    stdout.execute(style::ResetColor).unwrap();
}

pub fn printout_progress(text: &str) {
    printout_text(text, style::Color::DarkRed)
}

pub fn get_yes_or_no(prompt: &str) -> bool {
    match get_user_response(prompt).to_lowercase().as_str() {
        "y" => true,
        _ => false,
    }
}

pub fn get_user_response(prompt: &str) -> String {
    let mut stdout: std::io::Stdout = std::io::stdout();

    stdout
        .execute(style::SetForegroundColor(style::Color::DarkBlue))
        .unwrap();
    print!("{}", prompt);

    stdout.execute(style::ResetColor).unwrap();

    let mut response: String = String::new();
    std::io::stdin()
        .read_line(&mut response)
        .expect("Failed to read response");

    response.trim().to_string()
}

struct Simulator {
    agent: FindTreasureAgent,
}

impl Simulator {
    fn new(api_key: &str) -> Self {
        let llmodel = Model::new(Backend::ChatGPT {
            api_key: api_key.to_string(),
            model: "gpt-3.5-turbo".into(),
        })
        .unwrap();

        let agent = FindTreasureAgent::new(llmodel, FindTreasureParam::new(4, 6, 3));

        Self { agent }
    }

    async fn generate_scenario(&mut self) -> Result<(GameState, Scenario), Error> {
        // Generate the scenario
        loop {
            println!("");
            printout_progress("Generating scenario...");

            // Generate new scenario
            let (game_state, scenario) = self.agent.generate_scenario().await?;

            // PRINT GENERATED BACKGROUND
            println!("::Generated scenario::\n{:?}", scenario);

            println!("");

            if get_yes_or_no("Play this scenario? (y/n) ") {
                return Ok((game_state, scenario));
            }
        }
    }

    async fn explore_location(
        &mut self,
        location: &str,
        scenario: &Scenario,
        game_state: &mut GameState,
    ) -> Result<(), Error> {
        let npcs = game_state.list_npcs_in_facility(location);
        println!("");
        println!("Select the action.");
        npcs.iter().enumerate().for_each(|(i, v)| {
            println!("{}. {}", i + 1, v);
        });

        println!("{}. go out", npcs.len() + 1);
        let Ok(selection) = get_user_response("Type the number : ").parse::<usize>() else {
            return Ok(());
        };

        if selection > 0 && selection <= npcs.len() {
            println!("");
            printout_progress("Generating dialogue...");
            let npc_name = npcs[selection - 1].as_str();

            let background_prompt = scenario.construct_background_message();
            let game_state_prompt =
                game_state.construct_game_state(&scenario, game_state.visited_count(), npc_name);
            game_state.visit(&npc_name);

            match self
                .agent
                .talk_to(background_prompt, game_state_prompt, &npc_name)
                .await
            {
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
            game_state.go_out();
            return Ok(());
        }
        Ok(())
    }

    fn choose_new_location(&mut self, game_state: &mut GameState) {
        println!("");
        println!("Where will you visit?");
        let selections = game_state.list_of_facilities();
        selections.iter().enumerate().for_each(|(i, v)| {
            println!("{}. {}", i + 1, v);
        });

        let Ok(selection) = get_user_response("Type the number : ").parse::<usize>() else {
            return;
        };
        if selection > 0 && selection <= selections.len() {
            game_state.move_to_facility(selections.get(selection - 1).unwrap().as_str());
        }
    }

    pub async fn play_game(&mut self) -> Result<(), Error> {
        let (mut game_state, scenario) = self.generate_scenario().await?;

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
        while !game_state.has_found_treasure() {
            if let Some(location) = game_state.current_location() {
                self.explore_location(location.as_str(), &scenario, &mut game_state)
                    .await?;
            } else {
                self.choose_new_location(&mut game_state);
            }
        }
        printout_progress(&format!("After a long journey filled with challenges and revelations, you finally uncover the hidden treasure beneath the {}.", treasure_location));

        Ok(())
    }
}

pub fn main() {
    dotenv::dotenv().ok();

    let api_key = std::env::var("OPEN_API_KEY").expect("Failed to find OPEN_API_KEY");
    let mut simulator = Simulator::new(&api_key);

    loop {
        println!("");
        if get_yes_or_no("Start new simulation of 'Find Treasure'? (y/n) : ") {
            let _ = ai_agents::sync::block_on(simulator.play_game());
        } else {
            break;
        }
    }
}
