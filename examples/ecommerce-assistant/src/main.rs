mod utils;
use ai_agents::{Backend, Error, Model};
use ecommerce_assistant::{EcommerceChatAssistant, OrderInfo};

fn get_order_info() -> OrderInfo {
    // input the order information
    loop {
        println!("");
        let res = utils::get_user_response("Input the order information(name/order_id/status) to simulate (eg. John Lee/340124/Shipped) : ");
        let info = res
            .split("/")
            .into_iter()
            .map(|v| v.trim())
            .filter(|v| !v.is_empty())
            .collect::<Vec<_>>();
        if info.len() != 3 {
            eprintln!("Wrong format");
            // invalid format
            continue;
        }

        return OrderInfo::new(info[0].into(), info[1].into(), info[2].into());
    }
}

async fn run() -> Result<(), Error> {
    let api_key = std::env::var("OPEN_API_KEY").expect("Failed to find OPEN_API_KEY");
    let config = Backend::ChatGPT {
        api_key,
        model: "gpt-3.5-turbo".to_string(),
    };

    let model = Model::new(config).unwrap();
    let mut agent = EcommerceChatAssistant::new(model, "ECommer");

    println!("");
    println!("Welcome to the E-Commerce Chat Assistant Simulation!");
    println!("If you want to quit, type 'q'.");
    println!(
        "-------------------------------------------------------------------------------------\n"
    );

    let order_info = get_order_info();

    let mut message = None;
    //
    loop {
        utils::printout_progress("Requesting...");
        match agent.process_message(message).await? {
            Some(response) => {
                utils::printout_assistant(&response);

                let customer_msg = utils::get_user_response("Customer: ");

                if customer_msg == "q" {
                    break;
                }

                message = Some(customer_msg);
            }
            None => {
                // update the order information
                agent.update_order_info(order_info.clone());
                message = None;
            }
        }
    }
    Ok(())
}

fn main() -> Result<(), Error> {
    dotenv::dotenv().ok();
    env_logger::init();

    ai_agents::sync::block_on(run())?;
    Ok(())
}
