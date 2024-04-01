use std::{collections::HashMap, sync::Arc};

use crate::{
    error::Error,
    sync::RwLock,
    traits::{Adapter, UnitProcess},
    ModuleParam,
};

struct Edge {
    to: String,
    adapter: Option<Arc<dyn Adapter>>,
}

pub struct PipelineNet {
    nodes: HashMap<String, Arc<RwLock<dyn UnitProcess + Send + Sync>>>,
    edges: HashMap<String, Vec<Edge>>,
    groups: HashMap<String, String>, // Maps group names to input node names for each group.
}

impl PipelineNet {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            groups: HashMap::new(),
        }
    }

    // Add a node that implements `UnitProcess`
    pub fn add_node(&mut self, name: &str, node: Arc<RwLock<dyn UnitProcess + Send + Sync>>) {
        self.nodes.insert(name.into(), node);
    }
    // Add an edge between nodes
    pub fn add_edge(&mut self, from: &str, to: &str) {
        let edge = Edge {
            to: to.to_string(),
            adapter: None,
        };
        self.edges.entry(from.to_string()).or_default().push(edge);
    }

    // Add an edge between nodes with an adapter
    pub fn add_edge_with_adapter<A: Adapter + 'static>(
        &mut self,
        from: &str,
        to: &str,
        adapter: A,
    ) {
        let edge = Edge {
            to: to.to_string(),
            adapter: Some(Arc::new(adapter)),
        };
        self.edges.entry(from.to_string()).or_default().push(edge);
    }

    // Set group with input node.
    pub fn set_group_input(&mut self, group_name: &str, input_node_name: &str) {
        self.groups
            .insert(group_name.into(), input_node_name.into());
    }

    // Process a group starting from the group's input node, collecting the each results.
    pub async fn process_group(
        &self,
        group_name: &str,
        initial_input: ModuleParam,
    ) -> Result<HashMap<String, ModuleParam>, Error> {
        let input_node_name = self
            .groups
            .get(group_name)
            .ok_or_else(|| Error::NotFound(group_name.to_string()))?;

        let mut results = HashMap::new();
        let mut stack = vec![(input_node_name.as_str(), initial_input)];

        // bfs
        while let Some((current_node_name, input)) = stack.pop() {
            if results.contains_key(current_node_name) {
                continue; // Skip if visited
            }

            let node = self
                .nodes
                .get(current_node_name)
                .ok_or_else(|| Error::NotFound(current_node_name.to_string()))?;

            let processed_input = node.read().await.process(input).await?;

            // let processed_input = node.process(input).await?;

            results.insert(current_node_name.to_string(), processed_input.clone());

            if let Some(edges) = self.edges.get(current_node_name) {
                for edge in edges {
                    let adapted_input = edge
                        .adapter
                        .as_ref()
                        .map(|adapter| adapter.adapt(processed_input.clone()))
                        .unwrap_or_else(|| processed_input.clone());
                    stack.push((&edge.to, adapted_input));
                }
            }
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    //
    //
    use super::*;
    use crate::sync::block_on;
    use async_trait::async_trait;
    use std::sync::Arc;

    // Mock implementation
    #[derive(Default)]
    struct MockUnitProcess;

    #[async_trait]
    impl UnitProcess for MockUnitProcess {
        fn get_name(&self) -> &str {
            "MockUnit"
        }
        async fn process(&self, input: ModuleParam) -> Result<ModuleParam, Error> {
            Ok(input)
        }
    }

    struct MockAdapter;

    impl Adapter for MockAdapter {
        fn adapt(&self, input: ModuleParam) -> ModuleParam {
            input
        }
    }

    #[test]
    fn test_pipeline_net() {
        let mut pipeline = PipelineNet::new();

        // Mock input for processing
        let mock_input: &str = "TestInput";
        let initial_input = ModuleParam::Str(mock_input.into());

        // Add nodes
        let node1 = Arc::new(RwLock::new(MockUnitProcess::default()));
        let node2 = Arc::new(RwLock::new(MockUnitProcess::default()));

        pipeline.add_node("node1", node1);
        pipeline.add_node("node2", node2);

        pipeline.add_edge_with_adapter("node1", "node2", |v: ModuleParam| {
            if let ModuleParam::Str(param) = v.clone() {
                assert_eq!(param, "TestInput");
            }
            v
        });

        // Set group input
        pipeline.set_group_input("group1", "node1");

        block_on(async move {
            let results = pipeline
                .process_group("group1", initial_input)
                .await
                .expect("Failed to process group");

            assert!(results.contains_key("node1"));
            assert!(results.contains_key("node2"));
            assert_eq!(
                results.get("node1").unwrap().as_string().unwrap(),
                mock_input
            );
            assert_eq!(
                results.get("node2").unwrap().as_string().unwrap(),
                mock_input
            );
        });
    }
}
