pub mod agents;
pub mod models;
pub mod modules;
pub mod sync;

pub trait ToKeywordString {
    fn to_keyword_string() -> String;
}

pub mod prelude {
    pub use super::models::{AgentModuleTrait, AgentTrait};
    pub use super::sllm::message::MessageBuilder;
    pub use super::ToKeywordString;
    pub use ai_agent_macro::*;
}
pub use sllm;

#[cfg(test)]
mod tests {
    use sllm::Model;

    pub fn get_model() -> Model {
        dotenv::dotenv().ok();
        sllm::Model::new(sllm::Backend::ChatGPT {
            api_key: std::env::var("OPEN_API_KEY").unwrap(),
            model: "gpt-3.5-turbo".into(),
        })
        .unwrap()
    }

    use super::ToKeywordString;
    use ai_agent_macro::KeywordString;

    #[allow(dead_code)]
    #[derive(KeywordString)]
    struct SubStruct {
        prop1: i32,
        prop2: f32,
        prop3: String,
    }

    #[allow(dead_code)]
    #[derive(KeywordString)]
    struct TestStruct {
        sub: SubStruct,
        prop: Vec<SubStruct>,
    }

    #[ignore]
    #[test]
    fn test_print_keyword() {
        assert_eq!(
            TestStruct::to_keyword_string(),
            "{sub{prop1, prop2, prop3}, prop[{prop1, prop2, prop3}]}"
        );
    }
}
