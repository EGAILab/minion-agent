use std::{
    collections::{HashMap, HashSet},
    time::{SystemTime, UNIX_EPOCH},
};

use super::{
    AssistantContentBlock, AssistantMessage, Message, ModelIdentity, StopReason, TextBlock,
    ThinkingBlock, ToolCall, ToolResultContentBlock, ToolResultMessage, UserContent,
    UserContentBlock,
};

const USER_IMAGE_PLACEHOLDER: &str = "(image omitted: model does not support images)";
const TOOL_IMAGE_PLACEHOLDER: &str = "(tool image omitted: model does not support images)";

/// Target identity and capability used by provider-neutral message transformation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransformTarget {
    identity: ModelIdentity,
    supports_images: bool,
}

impl TransformTarget {
    pub fn new(identity: ModelIdentity, supports_images: bool) -> Self {
        Self {
            identity,
            supports_images,
        }
    }

    pub fn identity(&self) -> &ModelIdentity {
        &self.identity
    }

    pub fn supports_images(&self) -> bool {
        self.supports_images
    }
}

/// Target-API policy injected into generic transformation.
pub trait ToolCallIdNormalizer {
    fn normalize(
        &mut self,
        id: &str,
        target: &TransformTarget,
        source: &AssistantMessage,
    ) -> String;
}

impl<F> ToolCallIdNormalizer for F
where
    F: FnMut(&str, &TransformTarget, &AssistantMessage) -> String,
{
    fn normalize(
        &mut self,
        id: &str,
        target: &TransformTarget,
        source: &AssistantMessage,
    ) -> String {
        self(id, target, source)
    }
}

/// Transform provider-neutral history for one target model.
pub fn transform_messages(
    messages: &[Message],
    target: &TransformTarget,
    normalizer: Option<&mut dyn ToolCallIdNormalizer>,
) -> Vec<Message> {
    let image_aware = if target.supports_images() {
        messages.to_vec()
    } else {
        messages.iter().cloned().map(downgrade_images).collect()
    };
    let content_transformed = transform_content(messages, image_aware, target, normalizer);
    synthesize_orphans(content_transformed)
}

fn transform_content(
    source: &[Message],
    messages: Vec<Message>,
    target: &TransformTarget,
    mut normalizer: Option<&mut dyn ToolCallIdNormalizer>,
) -> Vec<Message> {
    let mut id_map = HashMap::new();
    messages
        .into_iter()
        .enumerate()
        .map(|(index, message)| match message {
            Message::Assistant(mut assistant) => {
                let source_assistant = match &source[index] {
                    Message::Assistant(source) => source.as_ref(),
                    _ => unreachable!("message role is unchanged before content transformation"),
                };
                let same_model = is_same_model(source_assistant, target);
                assistant.content = assistant
                    .content
                    .into_iter()
                    .flat_map(|block| {
                        transform_assistant_block(
                            block,
                            same_model,
                            target,
                            source_assistant,
                            &mut normalizer,
                            &mut id_map,
                        )
                    })
                    .collect();
                Message::Assistant(assistant)
            }
            Message::ToolResult(mut result) => {
                if let Some(normalized) = id_map.get(&result.tool_call_id)
                    && !normalized.is_empty()
                    && normalized != &result.tool_call_id
                {
                    result.tool_call_id.clone_from(normalized);
                }
                Message::ToolResult(result)
            }
            user => user,
        })
        .collect()
}

fn is_same_model(source: &AssistantMessage, target: &TransformTarget) -> bool {
    source.provider == target.identity().provider()
        && source.api == target.identity().api()
        && source.model == target.identity().model_id()
}

fn transform_assistant_block(
    block: AssistantContentBlock,
    same_model: bool,
    target: &TransformTarget,
    source: &AssistantMessage,
    normalizer: &mut Option<&mut dyn ToolCallIdNormalizer>,
    id_map: &mut HashMap<String, String>,
) -> Vec<AssistantContentBlock> {
    match block {
        AssistantContentBlock::Thinking(thinking) => transform_thinking(thinking, same_model),
        AssistantContentBlock::Text(text) if !same_model => {
            vec![AssistantContentBlock::Text(TextBlock::new(text.text))]
        }
        AssistantContentBlock::ToolCall(mut call) => {
            let original_id = call.id.clone();
            if !same_model
                && call
                    .thought_signature
                    .as_deref()
                    .is_some_and(|s| !s.is_empty())
            {
                call.thought_signature = None;
            }
            if !same_model && let Some(policy) = normalizer.as_deref_mut() {
                let normalized = policy.normalize(&original_id, target, source);
                if normalized != original_id {
                    id_map.insert(original_id, normalized.clone());
                    call.id = normalized;
                }
            }
            vec![AssistantContentBlock::ToolCall(call)]
        }
        unchanged => vec![unchanged],
    }
}

fn transform_thinking(block: ThinkingBlock, same_model: bool) -> Vec<AssistantContentBlock> {
    if block.redacted {
        return if same_model {
            vec![AssistantContentBlock::Thinking(block)]
        } else {
            Vec::new()
        };
    }
    if same_model
        && block
            .thinking_signature
            .as_deref()
            .is_some_and(|s| !s.is_empty())
    {
        return vec![AssistantContentBlock::Thinking(block)];
    }
    if block.thinking.trim().is_empty() {
        return Vec::new();
    }
    if same_model {
        vec![AssistantContentBlock::Thinking(block)]
    } else {
        vec![AssistantContentBlock::Text(TextBlock::new(block.thinking))]
    }
}

fn synthesize_orphans(messages: Vec<Message>) -> Vec<Message> {
    let mut output = Vec::new();
    let mut pending = Vec::new();
    let mut resolved = HashSet::new();

    for message in messages {
        match message {
            Message::Assistant(assistant) => {
                flush_orphans(&mut output, &mut pending, &mut resolved);
                if matches!(
                    assistant.stop_reason,
                    StopReason::Error | StopReason::Aborted
                ) {
                    continue;
                }
                pending.extend(assistant.content.iter().filter_map(|block| match block {
                    AssistantContentBlock::ToolCall(call) => Some(call.clone()),
                    _ => None,
                }));
                output.push(Message::Assistant(assistant));
            }
            Message::ToolResult(result) => {
                resolved.insert(result.tool_call_id.clone());
                output.push(Message::ToolResult(result));
            }
            user => {
                flush_orphans(&mut output, &mut pending, &mut resolved);
                output.push(user);
            }
        }
    }
    flush_orphans(&mut output, &mut pending, &mut resolved);
    output
}

fn flush_orphans(
    output: &mut Vec<Message>,
    pending: &mut Vec<ToolCall>,
    resolved: &mut HashSet<String>,
) {
    for call in pending.drain(..) {
        if !resolved.contains(&call.id) {
            output.push(Message::ToolResult(Box::new(ToolResultMessage::new(
                call.id,
                call.name,
                vec![ToolResultContentBlock::Text(TextBlock::new(
                    "No result provided",
                ))],
                true,
                now_millis(),
            ))));
        }
    }
    resolved.clear();
}

fn now_millis() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock must not precede the Unix epoch")
        .as_millis() as f64
}

fn downgrade_images(message: Message) -> Message {
    match message {
        Message::User(mut message) => {
            if let UserContent::Blocks(blocks) = message.content {
                message.content = UserContent::Blocks(downgrade_user_blocks(blocks));
            }
            Message::User(message)
        }
        Message::ToolResult(mut message) => {
            message.content = downgrade_tool_blocks(message.content);
            Message::ToolResult(message)
        }
        assistant => assistant,
    }
}

fn downgrade_user_blocks(blocks: Vec<UserContentBlock>) -> Vec<UserContentBlock> {
    let mut output = Vec::new();
    let mut previous_was_placeholder = false;
    for block in blocks {
        match block {
            UserContentBlock::Image(_) => {
                if !previous_was_placeholder {
                    output.push(UserContentBlock::Text(TextBlock::new(
                        USER_IMAGE_PLACEHOLDER,
                    )));
                }
                previous_was_placeholder = true;
            }
            UserContentBlock::Text(text) => {
                previous_was_placeholder = text.text == USER_IMAGE_PLACEHOLDER;
                output.push(UserContentBlock::Text(text));
            }
        }
    }
    output
}

fn downgrade_tool_blocks(blocks: Vec<ToolResultContentBlock>) -> Vec<ToolResultContentBlock> {
    let mut output = Vec::new();
    let mut previous_was_placeholder = false;
    for block in blocks {
        match block {
            ToolResultContentBlock::Image(_) => {
                if !previous_was_placeholder {
                    output.push(ToolResultContentBlock::Text(TextBlock::new(
                        TOOL_IMAGE_PLACEHOLDER,
                    )));
                }
                previous_was_placeholder = true;
            }
            ToolResultContentBlock::Text(text) => {
                previous_was_placeholder = text.text == TOOL_IMAGE_PLACEHOLDER;
                output.push(ToolResultContentBlock::Text(text));
            }
        }
    }
    output
}
