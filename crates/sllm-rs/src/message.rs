use std::{fmt, sync::Arc};

use serde::Serialize;
use tera::{Context, Tera};

pub use crate::traits::MessageBuilder;

#[derive(Clone)]
pub struct TemplatedMessage {
    template: String,
    context: Context,
}

impl TemplatedMessage {
    pub fn new(template: &str) -> Self {
        Self {
            template: template.to_string(),
            context: Context::new(),
        }
    }

    pub fn insert<T: Serialize + ?Sized, S: Into<String>>(&mut self, key: S, val: &T) {
        self.context.insert(key, val);
    }

    pub fn remove(&mut self, index: &str) -> bool {
        self.context.remove(index).is_some()
    }

    // TODO get
}

#[derive(Clone)]
pub enum PromptMessage {
    KeyValue {
        title: String,
        messages: Vec<(String, Arc<dyn Fn() -> String + Send + Sync>)>,
    },
    Templated(TemplatedMessage),
    Simple(String),
}

impl From<TemplatedMessage> for PromptMessage {
    fn from(value: TemplatedMessage) -> Self {
        PromptMessage::Templated(value)
    }
}

impl From<String> for PromptMessage {
    fn from(value: String) -> Self {
        PromptMessage::Simple(value)
    }
}

impl From<&str> for PromptMessage {
    fn from(value: &str) -> Self {
        PromptMessage::Simple(value.to_string())
    }
}
// #[derive(Clone)]
// pub struct PromptMessageGroup {
//     title: String,
//     messages: Vec<(String, Arc<dyn Fn() -> String + Send + Sync>)>,
// }

impl std::fmt::Debug for PromptMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PromptMessage::KeyValue { title, messages } => {
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
            PromptMessage::Templated(templated_msg) => f
                .debug_struct("Templated")
                .field("template", &templated_msg.template)
                .field("context", &templated_msg.context)
                .finish(),
            PromptMessage::Simple(message) => f.debug_tuple("Simple").field(message).finish(),
        }
    }
}

impl PromptMessage {
    pub fn new_key_value(title: &str) -> Self {
        PromptMessage::KeyValue {
            title: title.into(),
            messages: Vec::new(),
        }
    }

    pub fn new_templated(templated_msg: TemplatedMessage) -> Self {
        PromptMessage::Templated(templated_msg)
    }

    pub fn new_simple(message: String) -> Self {
        PromptMessage::Simple(message)
    }

    // A method to add a static message
    pub fn add_message(&mut self, key: &str, value: &str) {
        match self {
            PromptMessage::KeyValue { messages, .. } => {
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
            PromptMessage::KeyValue { messages, .. } => {
                messages.push((key.into(), Arc::new(value)));
            }
            _ => panic!("add_message_dyn is only valid for KeyValue variants"),
        }
    }
}

impl MessageBuilder for PromptMessage {
    fn build(&mut self) -> String {
        match self {
            PromptMessage::KeyValue { title, messages } => {
                let rendered_messages = messages
                    .iter()
                    .map(|(key, value_fn)| {
                        if key.is_empty() {
                            format!("{}", value_fn())
                        } else {
                            format!("{}: {}", key, value_fn())
                        }
                    })
                    .collect::<Vec<String>>()
                    .join("\n");

                if title.is_empty() {
                    rendered_messages
                } else {
                    format!("[{}]\n{}", title, rendered_messages)
                }
            }
            PromptMessage::Templated(templated_msg) => {
                let mut tera = Tera::default();
                tera.add_raw_template("template", &templated_msg.template)
                    .unwrap();
                tera.render("template", &templated_msg.context).unwrap()
            }
            PromptMessage::Simple(message) => message.clone(),
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
        let mut group = PromptMessage::new_key_value("Test Group");
        group.add_message_dyn("Key1", || "Value1".to_string());
        group.add_message("Key2", "Value2");

        let output = group.build();
        let expected_output = "[Test Group]\nKey1: Value1\nKey2: Value2";
        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_templated_build() {
        let mut msg = TemplatedMessage::new("Hello, {{ name }}!");
        msg.insert("name", "World");
        let mut group = PromptMessage::new_templated(msg);
        let output = group.build();
        let expected_output = "Hello, World!";
        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_simple_build() {
        let mut group = PromptMessage::new_simple("Just a simple message.".to_string());
        let output = group.build();
        let expected_output = "Just a simple message.";
        assert_eq!(output, expected_output);
    }
}
