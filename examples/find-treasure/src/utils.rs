use crossterm::{style, ExecutableCommand};

pub fn printout_text(text: &str, color: style::Color) {
    let mut stdout: std::io::Stdout = std::io::stdout();
    stdout.execute(style::SetForegroundColor(color)).unwrap();
    println!("{}", text);
    stdout.execute(style::ResetColor).unwrap();
}

pub fn printout_progress(text: &str) {
    printout_text(text, style::Color::DarkRed)
}

pub fn get_yes_or_no(prompt: &str) -> bool {
    match get_user_response(prompt).to_lowercase().as_str() {
        "y" => true,
        _ => false,
    }
}

pub fn get_user_response(prompt: &str) -> String {
    let mut stdout: std::io::Stdout = std::io::stdout();

    stdout
        .execute(style::SetForegroundColor(style::Color::DarkBlue))
        .unwrap();
    print!("{}", prompt);

    stdout.execute(style::ResetColor).unwrap();

    let mut response: String = String::new();
    std::io::stdin()
        .read_line(&mut response)
        .expect("Failed to read response");

    response.trim().to_string()
}
