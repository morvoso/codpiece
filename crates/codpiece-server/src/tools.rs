//! Qwen3.8 tool calls: parsing the model's emitted format back into the
//! OpenAI `tool_calls` shape.
//!
//! The chat template instructs the model to answer with
//!
//! ```text
//! <tool_call>
//! <function=get_weather>
//! <parameter=city>
//! Berlin
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! — an XML-ish framing, not the JSON blob earlier Qwen generations used.
//! The template renders string arguments raw and every other type through
//! `tojson`, so parsing inverts that: a parameter declared `string` in the
//! tool schema stays text, anything else is parsed as JSON. Without a schema
//! the value is parsed as JSON when it can be and kept as text otherwise.

use serde_json::{json, Map, Value};

pub const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";

#[derive(Debug, PartialEq)]
pub struct ToolCall {
    pub name: String,
    /// always a JSON object
    pub arguments: Value,
}

/// Splits generated text into the content that precedes any tool call and the
/// calls themselves. Text after a call is dropped: the template forbids it
/// ("You may provide optional reasoning ... BEFORE the function call, but NOT
/// after"), and keeping it would put prose in a message whose content field
/// clients ignore once `tool_calls` is set.
pub fn parse(text: &str, tools: Option<&Value>) -> (String, Vec<ToolCall>) {
    let Some(first) = text.find(OPEN) else {
        return (text.to_string(), Vec::new());
    };
    let content = text[..first].trim_end().to_string();
    let mut calls = Vec::new();
    let mut rest = &text[first..];
    while let Some(open) = rest.find(OPEN) {
        let body_start = open + OPEN.len();
        // an unterminated block means generation stopped mid-call; drop it
        let Some(close) = rest[body_start..].find(CLOSE) else { break };
        if let Some(call) = parse_one(&rest[body_start..body_start + close], tools) {
            calls.push(call);
        }
        rest = &rest[body_start + close + CLOSE.len()..];
    }
    (content, calls)
}

fn parse_one(block: &str, tools: Option<&Value>) -> Option<ToolCall> {
    const FN_OPEN: &str = "<function=";
    let at = block.find(FN_OPEN)? + FN_OPEN.len();
    let name_end = block[at..].find('>')? + at;
    let name = block[at..name_end].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let body = match block[name_end..].find("</function>") {
        Some(e) => &block[name_end + 1..name_end + e],
        None => &block[name_end + 1..],
    };

    const P_OPEN: &str = "<parameter=";
    const P_CLOSE: &str = "</parameter>";
    let mut args = Map::new();
    let mut rest = body;
    while let Some(p) = rest.find(P_OPEN) {
        let key_start = p + P_OPEN.len();
        let Some(key_end) = rest[key_start..].find('>').map(|i| i + key_start) else { break };
        let key = rest[key_start..key_end].trim().to_string();
        let val_start = key_end + 1;
        let (raw, next) = match rest[val_start..].find(P_CLOSE) {
            Some(e) => (&rest[val_start..val_start + e], val_start + e + P_CLOSE.len()),
            // a truncated final parameter still carries a usable value
            None => (&rest[val_start..], rest.len()),
        };
        // the template writes "\n" after the tag and before the closer; the
        // value itself may legitimately span lines or hold blank lines
        let raw = raw.strip_prefix('\n').unwrap_or(raw);
        let raw = raw.strip_suffix('\n').unwrap_or(raw);
        if !key.is_empty() {
            args.insert(key.clone(), coerce(raw, param_type(tools, &name, &key)));
        }
        rest = &rest[next..];
    }
    Some(ToolCall { name, arguments: Value::Object(args) })
}

/// Invert the template's rendering rule for one value.
fn coerce(raw: &str, declared: Option<&str>) -> Value {
    if declared == Some("string") {
        return Value::String(raw.to_string());
    }
    match serde_json::from_str::<Value>(raw) {
        // a bare word parses as nothing, but a quoted string parses as a
        // string the template would have written without quotes — so a
        // JSON string here means the model quoted it deliberately
        Ok(v) => v,
        Err(_) => Value::String(raw.to_string()),
    }
}

/// The declared JSON type of one parameter, if the request carried a schema.
fn param_type<'a>(tools: Option<&'a Value>, func: &str, key: &str) -> Option<&'a str> {
    let arr = tools?.as_array()?;
    let f = arr.iter().find_map(|t| {
        let f = t.get("function").unwrap_or(t);
        (f.get("name").and_then(Value::as_str) == Some(func)).then_some(f)
    })?;
    let ty = f.get("parameters")?.get("properties")?.get(key)?.get("type")?;
    match ty {
        Value::String(s) => Some(s.as_str()),
        // nullable parameters are declared as ["string", "null"]
        Value::Array(a) => a.iter().find_map(|v| v.as_str()).filter(|s| *s != "null"),
        _ => None,
    }
}

/// OpenAI wire shape. Arguments travel as a JSON *string*, per the spec.
pub fn to_openai(calls: &[ToolCall], id: &str) -> Value {
    shape(calls, id, false)
}

/// Same, for a streaming delta, where each call additionally carries the
/// `index` clients use to assemble calls across chunks.
pub fn to_openai_delta(calls: &[ToolCall], id: &str) -> Value {
    shape(calls, id, true)
}

fn shape(calls: &[ToolCall], id: &str, indexed: bool) -> Value {
    Value::Array(
        calls
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let mut v = json!({
                    "id": format!("call_{id}_{i}"),
                    "type": "function",
                    "function": {
                        "name": c.name,
                        "arguments": c.arguments.to_string(),
                    }
                });
                if indexed {
                    v["index"] = json!(i);
                }
                v
            })
            .collect(),
    )
}

/// Streaming counterpart of [`parse`]: emits content until a tool call opens,
/// then swallows the rest for parsing at the end. The marker can straddle
/// chunk boundaries, so a possible prefix of it is held back.
pub struct ToolStream {
    in_call: bool,
    held: String,
    call_text: String,
}

impl ToolStream {
    pub fn new() -> Self {
        Self { in_call: false, held: String::new(), call_text: String::new() }
    }

    /// Content safe to emit as a delta right now.
    pub fn push(&mut self, chunk: &str) -> String {
        if self.in_call {
            self.call_text.push_str(chunk);
            return String::new();
        }
        self.held.push_str(chunk);
        if let Some(at) = self.held.find(OPEN) {
            let ready = self.held[..at].to_string();
            self.call_text = self.held[at..].to_string();
            self.held.clear();
            self.in_call = true;
            return ready;
        }
        let keep = (1..OPEN.len())
            .rev()
            .find(|k| self.held.ends_with(&OPEN[..*k]))
            .unwrap_or(0);
        let cut = self.held.len() - keep;
        let ready = self.held[..cut].to_string();
        self.held = self.held[cut..].to_string();
        ready
    }

    /// (trailing content, calls) once the generation ends.
    pub fn flush(&mut self, tools: Option<&Value>) -> (String, Vec<ToolCall>) {
        if !self.in_call {
            return (std::mem::take(&mut self.held), Vec::new());
        }
        let (_, calls) = parse(&std::mem::take(&mut self.call_text), tools);
        (String::new(), calls)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOOLS: &str = r#"[{"type":"function","function":{"name":"get_weather",
        "parameters":{"properties":{"city":{"type":"string"},"days":{"type":"integer"},
        "metric":{"type":"boolean"}}}}}]"#;

    fn tools() -> Value {
        serde_json::from_str(TOOLS).unwrap()
    }

    #[test]
    fn parses_the_templates_own_format() {
        let text = "Let me check.\n\n<tool_call>\n<function=get_weather>\n\
                    <parameter=city>\nBerlin\n</parameter>\n\
                    <parameter=days>\n3\n</parameter>\n\
                    <parameter=metric>\ntrue\n</parameter>\n\
                    </function>\n</tool_call>";
        let (content, calls) = parse(text, Some(&tools()));
        assert_eq!(content, "Let me check.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments["city"], json!("Berlin"));
        assert_eq!(calls[0].arguments["days"], json!(3));
        assert_eq!(calls[0].arguments["metric"], json!(true));
    }

    #[test]
    fn a_declared_string_that_looks_numeric_stays_a_string() {
        let text = "<tool_call>\n<function=get_weather>\n\
                    <parameter=city>\n12345\n</parameter>\n</function>\n</tool_call>";
        let (_, calls) = parse(text, Some(&tools()));
        assert_eq!(calls[0].arguments["city"], json!("12345"));
        // without a schema the same text parses as a number
        let (_, calls) = parse(text, None);
        assert_eq!(calls[0].arguments["city"], json!(12345));
    }

    #[test]
    fn values_may_span_lines_and_hold_blank_lines() {
        let text = "<tool_call>\n<function=write>\n<parameter=body>\nline one\n\nline two\n\
                    </parameter>\n</function>\n</tool_call>";
        let (_, calls) = parse(text, None);
        assert_eq!(calls[0].arguments["body"], json!("line one\n\nline two"));
    }

    #[test]
    fn parses_several_calls_in_one_turn() {
        let text = "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n\
                    <tool_call>\n<function=b>\n<parameter=y>\n2\n</parameter>\n</function>\n</tool_call>";
        let (content, calls) = parse(text, None);
        assert!(content.is_empty());
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
    }

    #[test]
    fn an_unterminated_call_is_dropped_not_half_parsed() {
        let text = "thinking\n<tool_call>\n<function=a>\n<parameter=x>\n1\n";
        let (content, calls) = parse(text, None);
        assert_eq!(content, "thinking");
        assert!(calls.is_empty());
    }

    #[test]
    fn plain_prose_is_left_alone() {
        let (content, calls) = parse("no calls here", Some(&tools()));
        assert_eq!(content, "no calls here");
        assert!(calls.is_empty());
    }

    #[test]
    fn streaming_holds_back_a_split_marker() {
        let mut s = ToolStream::new();
        assert_eq!(s.push("Sure. "), "Sure. ");
        // marker arrives in pieces; nothing after it may be emitted as content
        assert_eq!(s.push("<tool"), "");
        assert_eq!(s.push("_call>\n<function=a>\n<parameter=x>\n1\n"), "");
        assert_eq!(s.push("</parameter>\n</function>\n</tool_call>"), "");
        let (tail, calls) = s.flush(None);
        assert!(tail.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["x"], json!(1));
    }

    #[test]
    fn streaming_releases_a_false_alarm_prefix() {
        let mut s = ToolStream::new();
        assert_eq!(s.push("a <too"), "a ");
        assert_eq!(s.push("l box"), "<tool box");
        let (tail, calls) = s.flush(None);
        assert_eq!(tail, "");
        assert!(calls.is_empty());
    }

    #[test]
    fn openai_shape_carries_arguments_as_a_json_string() {
        let calls = vec![ToolCall { name: "a".into(), arguments: json!({"x": 1}) }];
        let v = to_openai(&calls, "abc");
        assert_eq!(v[0]["type"], json!("function"));
        assert_eq!(v[0]["function"]["name"], json!("a"));
        assert_eq!(v[0]["function"]["arguments"], json!("{\"x\":1}"));
    }
}
