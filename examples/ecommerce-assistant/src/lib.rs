use ai_agents::units::ModelUnit;
use ai_agents::{prelude::*, sync::RwLock, Model};
use ai_agents::{units::DialogueUnit, PromptManager};
use ai_agents::{Error, ModuleParam, PipelineNet};
use serde::Serialize;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct CustomerInfo {
    name: String,
    order_id: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct OrderInfo {
    customer_name: String,
    order_id: String,
    order_status: String,
}

impl OrderInfo {
    pub fn new(customer_name: String, order_id: String, order_status: String) -> Self {
        Self {
            customer_name,
            order_id,
            order_status,
        }
    }
}

const COMMAND_DESCRIPTION: &'static str = r#"Respond in the following format for specific queries only. CMD["command:data", ...]

When a customer mentions content that matches a keyword type, generate your answer in the format described above. Here are the "keyword - command" mappings:

Keywords:
Customer Name - CNAME
Order Number - ORDER_ID

Example Answers:
CMD["CNAME:John Lee"]
CMD["ORDER_ID:#253523"]
CMD["CNAME:John Lee", "ORDER_ID:#253523"]"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PromptType {
    WithCommand,
    WithOrderInfo,
}

pub struct EcommerceChatAssistant {
    pipeline_net: PipelineNet,

    dialogue: Arc<RwLock<DialogueUnit>>,
    prompt: PromptManager<PromptType>,

    order_info: Option<OrderInfo>,
    received_customer_info: Option<CustomerInfo>,
}

impl EcommerceChatAssistant {
    pub fn new(model: Model, company: &str) -> Self {
        let unit = Arc::new(RwLock::new(DialogueUnit::new("dialogue")));
        let model_unit = Arc::new(RwLock::new(ModelUnit::new("chatgpt", model)));

        // Construct pipeline network.
        let mut net = PipelineNet::new();
        net.add_node("in", unit.clone());
        net.add_node("out", model_unit);
        net.add_edge("in", "out");
        net.set_group_input("dialogue", "in");

        // Prompt setting
        let mut prompt = PromptManager::new();

        let mut ctx_background = PromptMessageGroup::new("Background");
        ctx_background.insert("", &format!("You are an online assistant for the e-commerce company, {}. Your role is to provide support in a friendly and natural manner.", company));
        ctx_background.insert("", " To accurately determine the status of an order, it is essential to obtain the customer's name and order number.");

        let mut ctx_command = PromptMessageGroup::new("Command");
        ctx_command.insert("", COMMAND_DESCRIPTION);

        let mut ctx_rule = PromptMessageGroup::new("");
        ctx_rule.insert("", "If there is order information available that corresponds to the provided customer's name and order ID, the assistant must ignore the 'Command' and give answer based on the order status.");

        prompt.insert_prompt("b", ctx_background);
        prompt.insert_prompt("c", ctx_command);
        prompt.insert_prompt("r", ctx_rule);

        prompt.register_pattern(PromptType::WithCommand, "b c r");
        prompt.register_pattern(PromptType::WithOrderInfo, "b r");

        Self {
            dialogue: unit,
            order_info: None,
            prompt,
            pipeline_net: net,
            received_customer_info: None,
        }
    }

    // conditionally

    pub async fn reset(&mut self) {
        self.dialogue.write().await.clear_dialogue();
        self.order_info = None;
        self.received_customer_info = None;
    }

    pub fn update_order_info(&mut self, order_info: OrderInfo) {
        self.order_info = Some(order_info);
    }

    fn get_background(&mut self) -> Vec<PromptMessageGroup> {
        // set background
        let mut prompt = self.prompt.get(if self.received_customer_info.is_some() {
            PromptType::WithOrderInfo
        } else {
            PromptType::WithCommand
        });

        self.order_info.as_ref().map(|v| {
            let mut group = PromptMessageGroup::new("Order List");
            group.insert("", &{
                let this = &v;
                serde_json::to_string(this).unwrap()
            });
            prompt.push(group);
        });
        prompt
    }

    fn parse_command(&mut self, res: &str) -> Option<CustomerInfo> {
        // try to parsing into Vec<String>
        let cmd: Vec<String> = serde_json::from_str(&res[3..]).unwrap_or(Vec::new());

        let mut cinfo = CustomerInfo {
            name: "".into(),
            order_id: "".into(),
        };

        for mut cmd in cmd
            .into_iter()
            .map(|s| s.split(":").map(|v| v.to_string()).collect::<Vec<_>>())
            .filter(|v| v.len() == 2)
        {
            match cmd[0].as_str() {
                "CNAME" => {
                    cinfo.name = cmd.pop().unwrap();
                }
                "ORDER_ID" => {
                    cinfo.order_id = cmd.pop().unwrap();
                }
                _ => {}
            }
        }
        Some(cinfo)
    }

    pub async fn process_message(
        &mut self,
        message: Option<String>,
    ) -> Result<Option<String>, Error> {
        match message {
            Some(msg) => {
                // add customer's message
                {
                    let mut unit = self.dialogue.write().await;
                    unit.add_dialogue("Customer", msg.as_str());
                    unit.set_responder_name("Assistant");
                }

                let initial_input = ModuleParam::MessageBuilders(self.get_background());
                let results = self
                    .pipeline_net
                    .process_group("dialogue", initial_input)
                    .await?;

                let mut response = results.get("out").unwrap().as_string().cloned();
                // receive the response.
                // let mut response = self.agent.get_result().as_string().clone();

                // Hijack the response of agent
                match &response {
                    Some(res) => {
                        // check if response is command.
                        if res.starts_with("CMD[") {
                            self.received_customer_info = self.parse_command(res);
                            response = None;
                        } else {
                            // Update the last response.
                            self.dialogue.write().await.add_dialogue("Assistant", res)
                        }
                    }
                    None => {
                        // ERROR?
                    }
                }

                Ok(response)
            }
            None => {
                let initial_input = ModuleParam::MessageBuilders(self.get_background());
                {
                    let mut unit = self.dialogue.write().await;
                    unit.set_responder_name("Assistant");
                }

                let results = self
                    .pipeline_net
                    .process_group("dialogue", initial_input)
                    .await?;

                let response = results.get("out").unwrap().as_string().cloned();
                // let response = self.agent.get_result().as_string().cloned();
                if let Some(res) = &response {
                    self.dialogue.write().await.add_dialogue("Assistant", res);
                }
                Ok(response)
            }
        }
    }
}
