//! A small HTTP/1.1 server.
//!
//! Only what an inference endpoint needs: a request line, headers, a `Content-Length`
//! body, JSON responses and server-sent events. No async runtime — generation is
//! serialised through one engine thread anyway, so a thread per connection costs
//! nothing and keeps the streaming path a plain blocking write.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;

pub struct Request {
    pub method: String,
    pub path: String,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
}

impl Request {
    pub fn read(stream: &mut BufReader<TcpStream>) -> Result<Option<Self>, String> {
        let mut line = String::new();
        // a clean EOF here is a client that closed the connection, not an error
        if stream.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
            return Ok(None);
        }
        let mut parts = line.split_whitespace();
        let method = parts.next().unwrap_or_default().to_string();
        let path = parts.next().unwrap_or_default().to_string();
        if method.is_empty() || path.is_empty() {
            return Err("malformed request line".into());
        }

        let mut headers = HashMap::new();
        loop {
            let mut h = String::new();
            if stream.read_line(&mut h).map_err(|e| e.to_string())? == 0 {
                break;
            }
            let h = h.trim_end();
            if h.is_empty() {
                break;
            }
            if let Some((k, v)) = h.split_once(':') {
                headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
            }
        }

        let len: usize = headers
            .get("content-length")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let mut body = vec![0u8; len];
        if len > 0 {
            stream.read_exact(&mut body).map_err(|e| e.to_string())?;
        }
        Ok(Some(Self { method, path, headers, body }))
    }

    /// Path without its query string.
    pub fn route(&self) -> &str {
        self.path.split('?').next().unwrap_or(&self.path)
    }
}

pub fn write_json(w: &mut TcpStream, status: u16, body: &str) -> std::io::Result<()> {
    write!(
        w,
        "HTTP/1.1 {status} {}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Connection: close\r\n\r\n",
        reason(status),
        body.len()
    )?;
    w.write_all(body.as_bytes())?;
    w.flush()
}

pub fn write_error(w: &mut TcpStream, status: u16, message: &str) -> std::io::Result<()> {
    let body = serde_json::json!({
        "error": { "message": message, "type": "invalid_request_error", "code": status }
    });
    write_json(w, status, &body.to_string())
}

/// Begin a server-sent event stream.
pub fn begin_sse(w: &mut TcpStream) -> std::io::Result<()> {
    write!(
        w,
        "HTTP/1.1 200 OK\r\n\
         Content-Type: text/event-stream\r\n\
         Cache-Control: no-cache\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Connection: close\r\n\r\n"
    )?;
    w.flush()
}

pub fn sse_data(w: &mut TcpStream, payload: &str) -> std::io::Result<()> {
    write!(w, "data: {payload}\n\n")?;
    w.flush()
}

pub fn sse_done(w: &mut TcpStream) -> std::io::Result<()> {
    write!(w, "data: [DONE]\n\n")?;
    w.flush()
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_drops_the_query_string() {
        let r = Request {
            method: "GET".into(),
            path: "/v1/models?limit=1".into(),
            headers: HashMap::new(),
            body: vec![],
        };
        assert_eq!(r.route(), "/v1/models");
    }
}
