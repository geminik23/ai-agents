use std::sync::Arc;

pub use crate::traits::MessageBuilder;

#[derive(Clone)]
pub struct PromptMessageGroup {
    title: String,
    messages: Vec<(String, Arc<dyn Fn() -> String + Send + Sync>)>,
}

impl std::fmt::Debug for PromptMessageGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PromptMessageGroup")
            .field("title", &self.title)
            // Optionally, you can show the number of messages or their titles
            // but not the closures themselves as they do not implement Debug
            .field(
                "messages",
                &self
                    .messages
                    .iter()
                    .map(|(title, _)| title)
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl PromptMessageGroup {
    // Constructor for a new PromptMessageGroup with a title.
    pub fn new(title: &str) -> Self {
        PromptMessageGroup {
            title: title.to_string(),
            messages: Vec::new(),
        }
    }

    // Inserts a static message into the group.
    pub fn insert(&mut self, key: &str, value: &str) {
        let v = value.to_string();
        self.messages
            .push((key.to_string(), Arc::new(move || v.clone())));
    }

    // Inserts a dynamic message into the group.
    pub fn insert_dyn<F>(&mut self, key: &str, value: F)
    where
        F: Fn() -> String + 'static + Send + Sync,
    {
        self.messages.push((key.to_string(), Arc::new(value)));
    }

    // Removes a message by key.
    pub fn remove_key(&mut self, key: &str) {
        self.messages.retain(|(k, _)| k != key);
    }
}

impl MessageBuilder for PromptMessageGroup {
    fn build(&mut self) -> String {
        let messages = self
            .messages
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

        if self.title.is_empty() {
            messages
        } else {
            format!("[{}]\n{}", self.title, messages)
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

// pub struct PromptMessageBuilder {
//     groups: Vec<PromptMessageGroup>,
// }
//
// impl PromptMessageBuilder {
//     // Constructor for a new PromptMessageBuilder with groups.
//     pub fn new(groups: Vec<PromptMessageGroup>) -> Self {
//         PromptMessageBuilder { groups }
//     }
// }
//
// impl MessageBuilder for PromptMessageBuilder {
//     fn build(&self) -> String {
//         self.groups
//             .iter()
//             .map(|group| group.build())
//             .collect::<Vec<String>>()
//             .join("\n\n")
//     }
// }

// pub struct PromptMessageBuilder<'a> {
//     groups: Vec<&'a PromptMessageGroup>,
// }
//
// impl<'a> PromptMessageBuilder<'a> {
//     // Constructor for a new PromptMessageBuilder with groups.
//     pub fn new(groups: Vec<&'a PromptMessageGroup>) -> Self {
//         PromptMessageBuilder { groups }
//     }
// }
//
// impl<'a> MessageBuilder for PromptMessageBuilder<'a> {
//     fn build(&self) -> String {
//         self.groups
//             .iter()
//             .map(|group| group.build())
//             .collect::<Vec<String>>()
//             .join("\n\n")
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_static_message() {
        let mut group = PromptMessageGroup::new("Test Group");
        group.insert("Key1", "Value1");
        group.insert("Key2", "Value2");

        // Assuming a method to retrieve a message for testing purposes
        // This part is purely illustrative; actual implementation may vary
        assert_eq!(group.messages[0].0, "Key1");
        assert_eq!((group.messages[0].1)(), "Value1");
        assert_eq!(group.messages[1].0, "Key2");
        assert_eq!((group.messages[1].1)(), "Value2");
    }

    #[test]
    fn test_insert_dynamic_message() {
        let dynamic_value = || "Dynamic Value".to_string();
        let mut group = PromptMessageGroup::new("Group");
        group.insert_dyn("Key", dynamic_value);

        assert_eq!(group.messages[0].0, "Key");
        assert_eq!((group.messages[0].1)(), "Dynamic Value");
    }

    #[test]
    fn test_remove_key() {
        let mut group = PromptMessageGroup::new("Group");
        group.insert("Key", "Value");
        assert_eq!(group.messages.len(), 1);
        group.remove_key("Key");
        assert!(group.messages.is_empty());
    }

    #[test]
    fn test_build_output() {
        let mut group1 = PromptMessageGroup::new("Group1");
        group1.insert("Key1", "Value1");
        let mut group2 = PromptMessageGroup::new("Group2");
        group2.insert("Key2", "Value2");

        let output = PromptMessageBuilder::new(vec![group1, group2]).build();

        let expected_output = "[Group1]\nKey1: Value1\n\n[Group2]\nKey2: Value2";
        assert_eq!(output, expected_output);
    }
}
