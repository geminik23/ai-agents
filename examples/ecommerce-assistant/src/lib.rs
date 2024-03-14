
use ai_agents::{models::Error, sllm::Model};

#[derive(Debug, Clone)]
pub struct OrderInfo {
    customer_name: String,
    order_id: String,
    order_status: String,
}

impl OrderInfo {
    pub fn new(customer_name: String, order_id: String, order_status: String) -> Self { Self { customer_name, order_id, order_status } }
}


pub struct EcommerceChatAssistant{
    model:Model,
}

impl EcommerceChatAssistant{
    pub fn new(model:Model)->Self { 
        Self { 
            model
        } 
    }

    pub async fn process_message(&mut self, message:Option<String>) -> Result<Option<String>, Error> {
        Ok(None)
    }
    

    pub fn update_order_info(&mut self, order_info: OrderInfo){
        //
    }

}


