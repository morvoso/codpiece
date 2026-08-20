//! Applying the model's own chat template.
//!
//! The template ships inside the GGUF and is authoritative: it decides the exact byte
//! sequence the model sees, so guessing at it — even a format as familiar-looking as
//! `<|im_start|>role\n...<|im_end|>` — risks changing the model's behaviour silently.
//! Qwen3.5's is ~10 KB of Jinja using macros, `namespace()` counters, `is iterable` /
//! `is mapping` tests and whitespace control, which is why this renders it rather than
//! reimplementing it.

use minijinja::{context, Environment, Error, ErrorKind, Value};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    /// Either a plain string or the OpenAI content-parts array; the template handles
    /// both, so it is passed through as-is.
    #[serde(default)]
    pub content: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>,
    /// The `<think>` block of a prior assistant turn, kept separate the way the Qwen
    /// and DeepSeek APIs do. The template re-renders it verbatim inside the think tags
    /// (`preserve_thinking` defaults on), which is what lets a re-rendered conversation
    /// reproduce the exact token stream the session's recurrent state holds — without
    /// it every multi-turn chat with thinking re-prefills from scratch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

pub struct ChatTemplate {
    env: Environment<'static>,
}

impl ChatTemplate {
    pub fn new(source: &str) -> Result<Self, String> {
        let mut env = Environment::new();
        // Hugging Face chat templates are written against Jinja2-on-Python and freely
        // call Python string methods (`startswith`, `split`, `strip`), which are not
        // Jinja constructs and which minijinja does not implement. This is the shim for
        // exactly that; hand-rolling Python string semantics here would risk changing
        // the rendered prompt in ways nothing would catch.
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        // The template calls raise_exception() for inputs it refuses (images in a
        // system message, say). Surfacing it as a template error turns that into a
        // 400 rather than a panic or a silently malformed prompt.
        env.add_function("raise_exception", |msg: String| -> Result<Value, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, msg))
        });
        env.set_keep_trailing_newline(true);
        env.add_template_owned("chat".to_string(), source.to_string())
            .map_err(|e| format!("chat template failed to parse: {e}"))?;
        Ok(Self { env })
    }

    /// Render a conversation into the exact prompt string the model expects.
    pub fn render(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        tools: Option<&serde_json::Value>,
        enable_thinking: bool,
        template_kwargs: Option<&serde_json::Value>,
    ) -> Result<String, String> {
        let tmpl = self
            .env
            .get_template("chat")
            .map_err(|e| format!("chat template missing: {e}"))?;
        // The client's chat_template_kwargs (reasoning_effort, preserve_thinking, ...)
        // are template variables, not request fields — this template reads
        // reasoning_effort|default('xhigh'), so without forwarding them every request
        // renders at xhigh and the model over-thinks. Spread them into the context
        // alongside the ones codpiece controls; enable_thinking here wins if both set it.
        let mut ctx = std::collections::BTreeMap::<String, Value>::new();
        if let Some(serde_json::Value::Object(m)) = template_kwargs {
            for (k, v) in m {
                ctx.insert(k.clone(), Value::from_serialize(v));
            }
        }
        ctx.insert("messages".into(), Value::from_serialize(messages));
        ctx.insert("add_generation_prompt".into(), Value::from(add_generation_prompt));
        ctx.insert("tools".into(), tools.map(Value::from_serialize).unwrap_or(Value::from(())));
        ctx.insert("enable_thinking".into(), Value::from(enable_thinking));
        ctx.insert("add_vision_id".into(), Value::from(false));
        tmpl.render(ctx)
        .map_err(|e| {
            // minijinja chains causes; the root is what says which line refused
            let mut s = format!("chat template failed to render: {e}");
            let mut src = std::error::Error::source(&e);
            while let Some(e) = src {
                s.push_str(&format!(": {e}"));
                src = std::error::Error::source(e);
            }
            s
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn template() -> ChatTemplate {
        // the real template, extracted from the production GGUF
        let src = include_str!("../tests/data/qwen35-chat-template.jinja");
        ChatTemplate::new(src).expect("the production template must parse")
    }

    fn msg(role: &str, text: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: serde_json::Value::String(text.to_string()),
            name: None,
            tool_calls: None,
            reasoning_content: None,
        }
    }

    #[test]
    fn renders_a_plain_exchange() {
        let out = template()
            .render(&[msg("user", "Hi there")], true, None, true, None)
            .expect("render");
        assert!(out.contains("<|im_start|>user"), "{out}");
        assert!(out.contains("Hi there"), "{out}");
        // add_generation_prompt must leave the assistant turn open to continue, so the
        // last turn opened in the prompt is the assistant's
        let last = out.rfind("<|im_start|>").expect("a turn");
        assert!(out[last..].starts_with("<|im_start|>assistant"), "tail: {:?}", &out[last..]);
    }

    /// With thinking enabled this model's template opens a `<think>` block as part of
    /// the generation prompt, so generation continues *inside* it. Worth pinning: it
    /// changes where the model starts writing, and it is the kind of detail a
    /// hand-written prompt format would miss.
    #[test]
    fn thinking_mode_changes_the_generation_prompt() {
        let on = template()
            .render(&[msg("user", "Hi")], true, None, true, None)
            .expect("render");
        let off = template()
            .render(&[msg("user", "Hi")], true, None, false, None)
            .expect("render");
        assert!(on.trim_end().ends_with("<think>"), "thinking on, tail: {:?}", tail(&on));
        assert_ne!(on, off, "enable_thinking must change the prompt");
        assert!(!off.trim_end().ends_with("<think>"), "thinking off, tail: {:?}", tail(&off));
    }

    fn tail(s: &str) -> &str {
        &s[s.len().saturating_sub(48)..]
    }

    #[test]
    fn a_system_message_is_kept() {
        let out = template()
            .render(
                &[msg("system", "You are terse."), msg("user", "Hi")],
                true,
                None,
                true,
                None,
            )
            .expect("render");
        assert!(out.contains("You are terse."), "{out}");
        assert!(out.contains("<|im_start|>system"), "{out}");
    }

    #[test]
    fn multi_turn_preserves_order() {
        let out = template()
            .render(
                &[
                    msg("user", "first question"),
                    msg("assistant", "first answer"),
                    msg("user", "second question"),
                ],
                true,
                None,
                true,
                None,
            )
            .expect("render");
        let a = out.find("first question").expect("q1");
        let b = out.find("first answer").expect("a1");
        let c = out.find("second question").expect("q2");
        assert!(a < b && b < c, "turns out of order in:\n{out}");
    }

    #[test]
    fn without_generation_prompt_the_turn_is_closed() {
        let out = template()
            .render(&[msg("user", "Hi")], false, None, true, None)
            .expect("render");
        assert!(!out.trim_end().ends_with("<|im_start|>assistant"), "{out}");
    }

    #[test]
    fn content_parts_are_accepted() {
        // the OpenAI array form, which the template handles through render_content
        let m = ChatMessage {
            role: "user".into(),
            content: serde_json::json!([{"type": "text", "text": "part one"}]),
            name: None,
            tool_calls: None,
            reasoning_content: None,
        };
        let out = template().render(&[m], true, None, true, None).expect("render");
        assert!(out.contains("part one"), "{out}");
    }

    /// Does the second turn's rendering extend the first turn's byte-for-byte?
    ///
    /// This decides whether cross-turn prefix reuse works for /v1/chat with thinking
    /// on. Some chat templates strip `<think>` blocks from prior assistant turns on
    /// re-render; if this one does, the re-rendered conversation diverges from the
    /// session's history at the first assistant turn and every turn re-prefills.
    #[test]
    fn rerendering_extends_the_previous_render() {
        let t = template();
        let turn1 = t
            .render(&[msg("user", "What is 2+2?")], true, None, true, None)
            .expect("turn 1");
        // what the model would have produced, thinking included
        let reply = "<think>
Two plus two.
</think>

Four.";
        let turn2 = t
            .render(
                &[
                    msg("user", "What is 2+2?"),
                    msg("assistant", reply),
                    msg("user", "And doubled?"),
                ],
                true,
                None,
                true,
                None,
            )
            .expect("turn 2");
        let keeps_think = turn2.contains("Two plus two.");
        let extends = turn2.starts_with(&turn1);
        eprintln!("prior <think> kept on re-render: {keeps_think}");
        eprintln!("turn 2 extends turn 1 byte-for-byte: {extends}");
        eprintln!("--- turn1 tail: {:?}", &turn1[turn1.len().saturating_sub(60)..]);
        let cut = turn1.len().min(turn2.len());
        let div = turn1
            .bytes()
            .zip(turn2.bytes())
            .position(|(a, b)| a != b)
            .unwrap_or(cut);
        eprintln!("--- divergence at byte {div} of {}", turn1.len());
        eprintln!("--- turn2 around it: {:?}", &turn2[div.saturating_sub(40)..(div + 60).min(turn2.len())]);
        // the assertion records whichever behaviour is real; failure = it changed
        assert!(
            keeps_think || !extends,
            "template behaviour changed: think stripped yet still extends?"
        );
    }

    /// The round trip that makes multi-turn chat warm with thinking on.
    ///
    /// Turn 1 renders and the model generates from inside the opened `<think>` block.
    /// If the client sends the assistant turn back with `reasoning_content` split out
    /// (as the API now returns it), the re-render must equal turn 1's render plus the
    /// generated bytes — byte for byte — or the session's prefix match misses and the
    /// whole conversation re-prefills.
    #[test]
    fn reasoning_round_trip_is_byte_exact() {
        let t = template();
        let turn1 = t
            .render(&[msg("user", "What is 2+2?")], true, None, true, None)
            .expect("turn 1");
        assert!(turn1.ends_with("<think>\n"), "tail: {:?}", &turn1[turn1.len() - 30..]);

        // what a well-formed generation looks like, as raw text from inside the block
        let generated = "The user asks 2+2.\n</think>\n\nFour.";
        let (reasoning, content) = {
            let (a, b) = generated.split_once("</think>").unwrap();
            (a.trim().to_string(), b.trim().to_string())
        };

        let mut asst = msg("assistant", &content);
        asst.reasoning_content = Some(reasoning);
        let turn2 = t
            .render(
                &[msg("user", "What is 2+2?"), asst, msg("user", "And doubled?")],
                true,
                None,
                true,
                None,
            )
            .expect("turn 2");

        let expected_prefix = format!("{turn1}{generated}");
        assert!(
            turn2.starts_with(&expected_prefix),
            "re-render diverges from the generated stream:\n exp ..{:?}\n got ..{:?}",
            &expected_prefix[expected_prefix.len().saturating_sub(70)..],
            &turn2[..expected_prefix.len().min(turn2.len())]
                [expected_prefix.len().saturating_sub(70).min(turn2.len())..]
        );
    }

    #[test]
    fn raise_exception_becomes_an_error_not_a_panic() {
        // the template refuses images inside a system message
        let m = ChatMessage {
            role: "system".into(),
            content: serde_json::json!([{"type": "image", "image": "x"}]),
            name: None,
            tool_calls: None,
            reasoning_content: None,
        };
        let err = template().render(&[m], true, None, true, None).unwrap_err();
        assert!(err.to_lowercase().contains("system message"), "{err}");
    }
}
