use std::{collections::HashMap, fmt, sync::Arc};

use tera::{Context, Tera};

pub use crate::traits::MessageBuilder;

#[derive(Clone)]
pub enum PromptMessageGroup {
    KeyValue {
        title: String,
        messages: Vec<(String, Arc<dyn Fn() -> String + Send + Sync>)>,
    },
    Templated {
        template: String,
        context: HashMap<String, String>,
    },
    Simple(String),
}

impl From<String> for PromptMessageGroup {
    fn from(value: String) -> Self {
        PromptMessageGroup::Simple(value)
    }
}

impl From<&str> for PromptMessageGroup {
    fn from(value: &str) -> Self {
        PromptMessageGroup::Simple(value.to_string())
    }
}
// #[derive(Clone)]
// pub struct PromptMessageGroup {
//     title: String,
//     messages: Vec<(String, Arc<dyn Fn() -> String + Send + Sync>)>,
// }

impl std::fmt::Debug for PromptMessageGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PromptMessageGroup::KeyValue { title, messages } => {
                f.debug_struct("KeyValue")
                    .field("title", &title)
                    .field(
                        "messages",
                        &messages
                            .iter()
                            .map(|(key, _)| key.clone())
                            .collect::<Vec<String>>(),
                    ) // can't display closures, so only the keys.
                    .finish()
            }
            PromptMessageGroup::Templated { template, context } => f
                .debug_struct("Templated")
                .field("template", &template)
                .field("context", &context)
                .finish(),
            PromptMessageGroup::Simple(message) => f.debug_tuple("Simple").field(message).finish(),
        }
    }
}

impl PromptMessageGroup {
    pub fn new_key_value(title: &str) -> Self {
        PromptMessageGroup::KeyValue {
            title: title.into(),
            messages: Vec::new(),
        }
    }

    pub fn new_templated(template: &str, context: HashMap<String, String>) -> Self {
        PromptMessageGroup::Templated {
            template: template.into(),
            context,
        }
    }

    pub fn new_simple(message: String) -> Self {
        PromptMessageGroup::Simple(message)
    }

    // A method to add a static message
    pub fn add_message(&mut self, key: &str, value: &str) {
        match self {
            PromptMessageGroup::KeyValue { messages, .. } => {
                let v = value.to_string();
                let value_arc = Arc::new(move || v.clone());
                messages.push((key.into(), value_arc));
            }
            _ => panic!("add_message is only valid for KeyValue variants"),
        }
    }

    // A method to add a dynamic message
    pub fn add_message_dyn<F>(&mut self, key: &str, value: F)
    where
        F: Fn() -> String + 'static + Send + Sync,
    {
        match self {
            PromptMessageGroup::KeyValue { messages, .. } => {
                messages.push((key.into(), Arc::new(value)));
            }
            _ => panic!("add_message_dyn is only valid for KeyValue variants"),
        }
    }
}

impl MessageBuilder for PromptMessageGroup {
    fn build(&mut self) -> String {
        match self {
            PromptMessageGroup::KeyValue { title, messages } => {
                let rendered_messages = messages
                    .iter()
                    .map(|(key, value_fn)| format!("{}: {}", key, value_fn()))
                    .collect::<Vec<String>>()
                    .join("\n");

                if title.is_empty() {
                    rendered_messages
                } else {
                    format!("[{}]\n{}", title, rendered_messages)
                }
            }
            PromptMessageGroup::Templated { template, context } => {
                let mut tera = Tera::default();
                tera.add_raw_template("template", template).unwrap();
                let mut tera_context = Context::new();
                for (key, value) in context {
                    tera_context.insert(key, value);
                }
                tera.render("template", &tera_context).unwrap()
            }
            PromptMessageGroup::Simple(message) => message.clone(),
        }
    }
}

// one time
pub struct PromptMessageBuilder<T>
where
    T: IntoIterator,
    T::Item: MessageBuilder,
{
    groups: Option<T>,
}

impl<T> PromptMessageBuilder<T>
where
    T: IntoIterator,
    T::Item: MessageBuilder,
{
    // Constructor for a new PromptMessageBuilder with an iterable of items that implement MessageBuilder.
    pub fn new(groups: T) -> Self {
        PromptMessageBuilder {
            groups: Some(groups),
        }
    }
}

impl<T> MessageBuilder for PromptMessageBuilder<T>
where
    T: IntoIterator,
    T::Item: MessageBuilder,
{
    fn build(&mut self) -> String {
        let groups = self
            .groups
            .take()
            .expect("Groups should not be taken more than once");

        groups
            .into_iter()
            .map(|mut group| group.build())
            .collect::<Vec<String>>()
            .join("\n\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_value_insert_and_build() {
        let mut group = PromptMessageGroup::new_key_value("Test Group");
        group.add_message_dyn("Key1", || "Value1".to_string());
        group.add_message("Key2", "Value2");

        let output = group.build();
        let expected_output = "[Test Group]\nKey1: Value1\nKey2: Value2";
        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_templated_build() {
        let mut context = HashMap::new();
        context.insert("name".to_string(), "World".to_string());
        let mut group = PromptMessageGroup::new_templated("Hello, {{ name }}!", context);
        let output = group.build();
        let expected_output = "Hello, World!";
        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_simple_build() {
        let mut group = PromptMessageGroup::new_simple("Just a simple message.".to_string());
        let output = group.build();
        let expected_output = "Just a simple message.";
        assert_eq!(output, expected_output);
    }
}
