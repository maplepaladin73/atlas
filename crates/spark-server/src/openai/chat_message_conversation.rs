// SPDX-License-Identifier: AGPL-3.0-only

use super::IncomingMessage;
use serde_json::{Value, json};

impl IncomingMessage {
    pub fn to_conversation_items(&self) -> Vec<Value> {
        if self.role == "tool" {
            return vec![json!({
                "type": "function_call_output", "call_id": self.tool_call_id,
                "name": self.name, "output": self.content.text,
            })];
        }
        let mut items = Vec::new();
        if self.tool_calls.is_none()
            || !self.content.text.is_empty()
            || self.reasoning_content.is_some()
        {
            let mut item = json!({
                "type": "message", "role": self.role,
                "content": [{"type": "input_text", "text": self.content.text}],
            });
            if let Some(reasoning) = &self.reasoning_content {
                item["reasoning_content"] = json!(reasoning);
            }
            items.push(item);
        }
        for call in self.tool_calls.iter().flatten() {
            items.push(json!({
                "type": "function_call", "call_id": call.id,
                "name": call.function.name, "arguments": call.function.arguments,
            }));
        }
        items
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversation_tool_call_replay() {
        for item in [
            json!({"type":"function_call", "call_id":"call_1", "name":"lookup", "arguments":"{}"}),
            json!({"type":"function_call_output", "call_id":"call_1", "output":"found", "name":"lookup"}),
            json!({"type":"function_call_output", "call_id":"call_1", "output":""}),
        ] {
            let expected = IncomingMessage::from_responses_input_item(&item)
                .unwrap()
                .unwrap();
            let store = crate::conversation_store::ConversationStore::with_config(
                10,
                std::time::Duration::from_secs(60),
            );
            let id = store.create(expected.to_conversation_items(), Default::default());
            let stored = store.get(&id).unwrap();
            let actual = IncomingMessage::from_conversation_item(&stored.items[0])
                .expect("tool item must replay");
            assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
        }
    }

    #[test]
    fn conversation_tool_call_replay_invalid_items() {
        for item in [
            json!({"type":"function_call"}),
            json!({"type":"unknown"}),
            json!(null),
        ] {
            assert!(IncomingMessage::from_conversation_item(&item).is_none());
        }
    }
}
