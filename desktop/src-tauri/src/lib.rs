use std::io::Write;
use std::net::TcpListener;
use std::time::Duration;

use tauri::Manager;
use tauri_plugin_shell::process::CommandEvent;
use tauri_plugin_shell::ShellExt;
use tokio::net::TcpStream;
use tokio::time::sleep;

const STREAMLIT_HOST: &str = "127.0.0.1";

fn log_streamlit(message: impl AsRef<str>) {
    let log_path = std::env::temp_dir().join("otitenet-tauri.log");
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
    {
        let _ = writeln!(file, "{}", message.as_ref());
    }
    eprintln!("{}", message.as_ref());
}

fn choose_streamlit_port() -> std::io::Result<u16> {
    let listener = TcpListener::bind((STREAMLIT_HOST, 0))?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(port)
}

async fn wait_for_streamlit(port: u16) -> bool {
    let address = format!("{STREAMLIT_HOST}:{port}");
    log_streamlit(format!("waiting for Streamlit at http://{address}"));

    for attempt in 1..=90 {
        if TcpStream::connect(&address).await.is_ok() {
            log_streamlit(format!(
                "Streamlit is accepting connections at http://{address} after attempt {attempt}"
            ));
            return true;
        }
        sleep(Duration::from_millis(500)).await;
    }

    log_streamlit(format!(
        "timed out waiting for Streamlit at http://{address}"
    ));
    false
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let main_window = app
                .get_webview_window("main")
                .expect("main webview window is missing");

            let streamlit_port = choose_streamlit_port()?;

            let sidecar = app
                .shell()
                .sidecar("otitenet-streamlit")?
                .env("OTITENET_STREAMLIT_PORT", streamlit_port.to_string())
                .env("OTITENET_STREAMLIT_APP", "app_offline.py");

            tauri::async_runtime::spawn(async move {
                log_streamlit(format!(
                    "starting Streamlit sidecar on http://{STREAMLIT_HOST}:{streamlit_port}"
                ));

                let (mut rx, child) = sidecar.spawn().expect("failed to start Streamlit sidecar");
                log_streamlit(format!("Streamlit sidecar pid={}", child.pid()));

                tauri::async_runtime::spawn(async move {
                    while let Some(event) = rx.recv().await {
                        match event {
                            CommandEvent::Stdout(line) => log_streamlit(format!(
                                "streamlit stdout: {}",
                                String::from_utf8_lossy(&line).trim_end()
                            )),
                            CommandEvent::Stderr(line) => log_streamlit(format!(
                                "streamlit stderr: {}",
                                String::from_utf8_lossy(&line).trim_end()
                            )),
                            CommandEvent::Error(error) => {
                                log_streamlit(format!("streamlit event error: {error}"))
                            }
                            CommandEvent::Terminated(payload) => log_streamlit(format!(
                                "streamlit terminated: code={:?}, signal={:?}",
                                payload.code, payload.signal
                            )),
                            _ => log_streamlit("streamlit emitted an unknown event"),
                        }
                    }
                });

                if !wait_for_streamlit(streamlit_port).await {
                    return;
                }

                main_window
                    .eval(&format!(
                        "window.location.replace('http://{STREAMLIT_HOST}:{streamlit_port}')"
                    ))
                    .expect("failed to navigate to Streamlit");

                let _child = child;
                std::future::pending::<()>().await;
            });

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running Tauri application");
}
