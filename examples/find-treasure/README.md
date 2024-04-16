#  Find Treasure - Dynamic Content Generation in Gaming

"Find Treasure" is a simple one episode RPG game where scenarios and NPC dialogues are dynamically generated using OpenAI's GPT-3.5 model. The game tasks players with finding hidden treasure in a town by interacting with NPCs and exploring facilities. This project utilizes the `ai-agents` Rust library and a text-based interface built with Bevy and EGUI.


## Play Methods

The game can be experienced through two different interfaces:

1. **Command Line**: Utilizing `crossterm` for traditional terminal-based interaction.
2. **GUI (Text based)**: Built with `bevy` and `bevy_egui`, providing a graphical user interface.


## Running the Game

To play "Find Treasure," use one of the following commands depending on your preferred interface:

1. For the command line version:
   ```
   cargo run --features commandline
   ```
2. For the GUI version in Bevy:
   ```
   cargo run --features inbevy
   ```


## Preview

![play-gif]()


## Game Features

- **Dynamic Scenario Generation**: Each playthrough generates unique towns, facilities, and NPC arrangements.
- **Generated NPC Dialogue**: Dialogue with NPCs are not scripted but are generated on-the-fly using a large language model, providing a different experience.


## Challenges & Solutions

### **Consistency in Scenario Generation**

- **Problem**: Occasionally failed to adhere to specified numbers of NPCs and facilities.
- **Solution**: Adopted a cascading generation approach, where scenarios are not generated all at once but sequentially. This method allows for better adherence to the game specifications by building each component (town, facilities, NPCs) step-by-step.

### **Name Diversity**

- **Problem**: The AI model frequently reused names (NPCs, Town, Facilities).
- **Solution**: Implemented a pre-generation strategy to create a diverse pool of names, ensuring unique and varied NPC identities through selective AI assignment.


## Future Work

- **Advanced AI Integration and Dialogue System**: Enhance the complexity of NPC interactions by developing a dynamic dialogue system where NPCs can offer multiple responses and engage in more realistic back-and-forth conversations based on varied player reactions.
- **Voice Interaction**: Expand the interaction modes by integrating text-to-speech (TTS) and speech-to-text (STT), allowing players to converse with NPCs using voice.


This is one of the examples featured in the `ai-agents` library, illustrating practical applications of AI in game development.
