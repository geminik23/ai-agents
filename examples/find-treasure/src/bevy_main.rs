use ai_agents::{sync::RwLock, Backend, Model};
use bevy::{
    prelude::*,
    tasks::{block_on, futures_lite::future, AsyncComputeTaskPool, Task},
};
use bevy_egui::{
    egui::{self, Color32, Margin, RichText},
    EguiContexts, EguiPlugin,
};
use find_treasure::{FindTreasureAgent, FindTreasureParam, GameState, Scenario};

use std::sync::Arc;

use bevy::ecs::system::Resource;

#[derive(Resource)]
pub struct GameCore {
    pub model: Arc<RwLock<FindTreasureAgent>>,
    pub scenario: Option<Scenario>,
}

impl GameCore {
    pub fn new(backend: Backend) -> Self {
        let model = Model::new(backend).unwrap();

        Self {
            model: Arc::new(RwLock::new(FindTreasureAgent::new(
                model,
                FindTreasureParam::default(),
            ))),
            scenario: None,
        }
    }
}

// AsyncComputeTaskPool
//
// Where to store the model?
//
#[derive(Component)]
struct AsyncGenDialogue(Task<(String, Entity)>);
#[derive(Component)]
struct AsyncGenScenario(Task<(Scenario, GameState, Entity)>);

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, Hash, States)]
enum AppState {
    #[default]
    SelectBackend,
    ScenarioGeneration,
    InGame,
    OnNPC,
    GameEnd,
}

#[derive(Resource)]
struct BackendSelection {
    current_backend: String,
    api_key: String,
}

impl Default for BackendSelection {
    fn default() -> Self {
        BackendSelection {
            current_backend: "ChatGPT".to_string(), // Default to ChatGPT
            api_key: String::new(),                 // Empty API key by default
        }
    }
}

#[derive(Resource, Default)]
struct ScenarioParam {
    text: String,
    num_npc: u32,
    num_facilities: u32,
    num_clues: u32,
    generating: bool,
}

pub fn main() {
    dotenv::dotenv().ok();
    App::new()
        .init_state::<AppState>()
        .init_resource::<BackendSelection>()
        .init_resource::<ScenarioParam>()
        .add_plugins(DefaultPlugins)
        .add_plugins(EguiPlugin)
        .add_systems(Startup, (configure_visuals_system, setup))
        .add_systems(
            Update,
            backend_selection_ui.run_if(in_state(AppState::SelectBackend)),
        )
        .add_systems(
            Update,
            (scenario_generation_ui, handle_scenario_gen_task)
                .run_if(in_state(AppState::ScenarioGeneration)),
        )
        .add_systems(
            Update,
            (play_game, handle_dailogue_gen_task).run_if(in_state(AppState::InGame)),
        )
        .add_systems(Update, dialogue_ui.run_if(in_state(AppState::OnNPC)))
        .add_systems(Update, game_end_ui.run_if(in_state(AppState::GameEnd)))
        .run();
}

fn setup(mut _commands: Commands) {}

fn configure_visuals_system(mut contexts: EguiContexts) {
    contexts.ctx_mut().set_visuals(egui::Visuals {
        window_rounding: 0.0.into(),
        ..Default::default()
    });
}

fn backend_selection_ui(
    mut commands: Commands,
    mut next_state: ResMut<NextState<AppState>>,
    mut backend: ResMut<BackendSelection>,
    mut egui_context: EguiContexts,
) {
    egui::Window::new("Find Treasure")
        .collapsible(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .frame(egui::Frame::default().inner_margin(Margin::same(20.0)))
        .show(egui_context.ctx_mut(), |ui| {
            ui.heading("Backend Selection");
            ui.spacing();

            // ComboBox for backend selection
            egui::ComboBox::from_label("Select Backend")
                .selected_text(backend.current_backend.clone())
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut backend.current_backend,
                        "ChatGPT".to_string(),
                        "ChatGPT",
                    );
                    ui.selectable_value(
                        &mut backend.current_backend,
                        "llama2".to_string(),
                        "llama2",
                    );
                });

            // show OpenAI API Key input
            if backend.current_backend == "ChatGPT" {
                ui.label("OpenAI API KEY: ");
                ui.text_edit_singleline(&mut backend.api_key);
            }

            if ui.button("Next").clicked() {
                // Model
                let backend = Backend::ChatGPT {
                    api_key: backend.api_key.clone(),
                    model: "gpt-3.5-turbo".into(),
                };

                commands.insert_resource(GameCore::new(backend));
                next_state.set(AppState::ScenarioGeneration);
            }
        });
}

fn scenario_generation_ui(
    mut commands: Commands,
    mut next_state: ResMut<NextState<AppState>>,
    mut egui_context: EguiContexts,
    mut scenario: ResMut<ScenarioParam>,
    core: Res<GameCore>,
) {
    //
    egui::Window::new("Find Treasure")
        .collapsible(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]) // Center the window
        .scroll2([false, true])
        .frame(egui::Frame::default().inner_margin(Margin::same(20.0)))
        .show(egui_context.ctx_mut(), |ui| {
            ui.heading("Scenario Generation");

            if scenario.generating {
                ui.label(RichText::new("Generating scenario....").color(Color32::RED));
            } else {
                if !scenario.text.is_empty() {
                    ui.label(RichText::new(&scenario.text).color(Color32::BLUE));
                }
            }

            // Disabled button logic
            let start_game_button = egui::Button::new("Start Game");

            // .enabled(!scenario.text.is_empty());
            if ui
                .add_enabled(
                    !scenario.generating && !scenario.text.is_empty(),
                    start_game_button,
                )
                .clicked()
            {
                next_state.set(AppState::InGame);
            }

            ui.separator(); // Visual separation

            // Input fields for parameters
            ui.horizontal(|ui| {
                ui.label("The Number of NPCs:");
                ui.add_enabled(
                    !scenario.generating,
                    egui::DragValue::new(&mut scenario.num_npc),
                );
            });
            ui.horizontal(|ui| {
                ui.label("The Number of Facilities:");
                ui.add_enabled(
                    !scenario.generating,
                    egui::DragValue::new(&mut scenario.num_facilities),
                );
            });
            ui.horizontal(|ui| {
                ui.label("The Number of Clues:");
                ui.add_enabled(
                    !scenario.generating,
                    egui::DragValue::new(&mut scenario.num_clues),
                );
            });

            // Generate scenario button
            let generate_button = egui::Button::new("Generate Scenario");
            if ui
                .add_enabled(!scenario.generating, generate_button)
                .clicked()
            {
                let param = FindTreasureParam {
                    num_npcs: scenario.num_npc,
                    num_facilities: scenario.num_facilities,
                    num_clues: scenario.num_clues,
                };

                // log::info!("{:?}", param);

                let thread_pool = AsyncComputeTaskPool::get();
                let entity = commands.spawn_empty().id();

                let model = core.model.clone();
                let e = entity.clone();
                let task = thread_pool.spawn(async move {
                    let mut model = model.write().await;
                    model.update_param(param);
                    let (game_state, scenario) = model.generate_scenario().await.unwrap();
                    (scenario, game_state, e)
                });
                commands.entity(entity).insert(AsyncGenScenario(task));

                scenario.generating = true;
            }
        });
}

fn handle_scenario_gen_task(
    mut commands: Commands,
    mut scenario_param: ResMut<ScenarioParam>,
    mut async_tasks: Query<&mut AsyncGenScenario>,
) {
    for mut task in &mut async_tasks {
        if let Some((scenario, mut game_state, entity)) = block_on(future::poll_once(&mut task.0)) {
            commands.entity(entity).despawn();
            let scenario_text = format!("In the quaint town of {}, legends speak of hidden treasure buried beneath its grounds. {} This place holds secrets waiting to be unearthed by a daring treasure hunter. Your quest begins amidst its lively streets and shadowy alleys, where every encounter could lead you closer to fortune—or into the depths of mystery.\n\nWelcome to the mystical town of {}. Rumors of hidden treasure have drawn many to its streets.", scenario.town.name, scenario.town.description, scenario.town.name);
            game_state.set_game_introduction(scenario_text.clone());
            scenario_param.generating = false;
            scenario_param.text = scenario.to_string();

            // insert resource
            commands.insert_resource(game_state);
            commands.insert_resource(scenario);
        }
    }
}

fn handle_dailogue_gen_task(
    mut commands: Commands,
    mut next_state: ResMut<NextState<AppState>>,
    mut async_tasks: Query<&mut AsyncGenDialogue>,
    mut game_state: ResMut<GameState>,
) {
    for mut task in &mut async_tasks {
        if let Some((dialogue, entity)) = block_on(future::poll_once(&mut task.0)) {
            commands.entity(entity).despawn();
            game_state.set_dialogue(dialogue);
            next_state.set(AppState::OnNPC);
        }
    }
}

fn dialogue_ui(
    mut egui_context: EguiContexts,
    mut next_state: ResMut<NextState<AppState>>,
    game_state: Res<GameState>,
) {
    let Some(npc) = game_state.current_npc() else {
        next_state.set(AppState::InGame);
        return;
    };
    egui::Window::new("NPC")
        .collapsible(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]) // Center the window
        .frame(egui::Frame::default().inner_margin(Margin::same(20.0))) // Add margins
        .show(egui_context.ctx_mut(), |ui| {
            ui.separator();
            ui.label(&format!("{}: {}", npc, game_state.dialogue()));

            if ui
                .add_sized(
                    [ui.available_width(), 0.0],
                    egui::Button::new(&format!("Back")).wrap(false),
                )
                .clicked()
            {
                next_state.set(AppState::InGame);
            }
        });
}

fn play_game(
    mut commands: Commands,
    mut egui_context: EguiContexts,
    mut game_state: ResMut<GameState>,
    scenario: Res<Scenario>,
    game_core: Res<GameCore>,
    mut next_state: ResMut<NextState<AppState>>,
) {
    egui::Window::new("Gameplay")
        .collapsible(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]) // Center the window
        .frame(egui::Frame::default().inner_margin(Margin::same(20.0))) // Add margins
        .show(egui_context.ctx_mut(), |ui| {
            let cur_location = game_state.current_location();

            let location = match &cur_location {
                Some(location) => location,
                None => "Entrance",
            };

            ui.heading(format!("Location: {}", location));
            ui.separator();

            if cur_location.is_some() {
                ui.label("Select the action");
                ui.separator();

                //
                //
                let npcs = game_state.list_npcs_in_facility(location);
                npcs.iter().enumerate().for_each(|(i, cur_npc)| {
                    if ui
                        .add_sized(
                            [ui.available_width(), 0.0],
                            egui::Button::new(&format!("{}. {}", i + 1, cur_npc)).wrap(false),
                        )
                        .clicked()
                    {
                        // generate the dialogue.
                        //
                        let thread_pool = AsyncComputeTaskPool::get();
                        let entity = commands.spawn_empty().id();

                        // let game_state_prompt = game_state.construct_game_state(scene_response, completed_order_count, cur_npc)
                        // {
                        //
                        //
                        // }

                        let background_prompt = scenario.construct_background_message();
                        let game_state_prompt = game_state.construct_game_state(
                            &scenario,
                            game_state.visited_count(),
                            cur_npc,
                        );
                        game_state.visit(&cur_npc);

                        let model = game_core.model.clone();
                        let e = entity.clone();
                        let cur_npc = cur_npc.clone();
                        let task = thread_pool.spawn(async move {
                            let mut model = model.write().await;

                            let result = model
                                .talk_to(background_prompt, game_state_prompt, &cur_npc)
                                .await;
                            let result = match result {
                                Ok(result) => result,
                                Err(err) => {
                                    log::error!(
                                        "Failed to generate the dialogue with {} - {:?}",
                                        cur_npc,
                                        err
                                    );
                                    "".into()
                                }
                            };
                            (result, e)
                        });
                        commands.entity(entity).insert(AsyncGenDialogue(task));
                    }
                });

                if ui
                    .add_sized(
                        [ui.available_width(), 0.0],
                        egui::Button::new(format!("{}. Go out", npcs.len() + 1)).wrap(false),
                    )
                    .clicked()
                {
                    game_state.go_out();
                }
            } else {
                ui.label(game_state.game_introduction());
                ui.separator();
                // choose new location
                ui.label("Where will you visit?");
                ui.separator();

                let selections = game_state.list_of_facilities();
                selections.iter().enumerate().for_each(|(i, v)| {
                    if ui
                        .add_sized(
                            [ui.available_width(), 0.0],
                            egui::Button::new(&format!("{}. {}", i + 1, v)).wrap(false),
                        )
                        .clicked()
                    {
                        game_state.move_to_facility(v);

                        if game_state.has_found_treasure() {
                            next_state.set(AppState::GameEnd);
                        }
                    }
                });
            }
        });
}

fn game_end_ui(
    mut egui_context: EguiContexts,
    game_state: Res<GameState>,
    mut next_state: ResMut<NextState<AppState>>,
) {
    egui::Window::new("GameEnd")
        .collapsible(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]) // Center the window
        .frame(egui::Frame::default().inner_margin(Margin::same(20.0))) // Add margins
        .show(egui_context.ctx_mut(), |ui| {
            ui.label(&format!("After a long journey filled with challenges and revelations, you finally uncover the hidden treasure beneath the {}.", game_state.treasure_place()));

            if ui
                .add_sized(
                    [ui.available_width(), 0.0],
                    egui::Button::new(&format!("To main")).wrap(false),
                )
                .clicked()
            {
                next_state.set(AppState::SelectBackend);
            }
        });
}
