#![allow(unused_imports)]

use flutter_rust_bridge::frb;
use std::fmt::Display;
use tokenizers::{Result, Tokenizer};
// use toktkn::{BPETokenizer, Tokenizer, TokenizerConfig};

type Token = i64;

pub struct OnnxInputWrapper {
    pub input_ids: Vec<Token>,
    pub attention_mask: Vec<Token>,
    pub token_type_ids: Vec<Token>,
}

pub struct TokenizerWrapper(Tokenizer);

impl TokenizerWrapper {
    #[frb(sync)]
    pub fn from_pretrained(model_id: String) -> anyhow::Result<Self> {
        let tokenizer = Tokenizer::from_pretrained(model_id, None)
            .map_err(|_| anyhow::anyhow!("failed to load tokenizer"))?;
        Ok(Self(tokenizer))
    }

    #[frb(sync)]
    pub fn tokenize(&self, prompt: String) -> anyhow::Result<OnnxInputWrapper> {
        let encoding = self
            .0
            .encode(prompt, false)
            .map_err(|_| anyhow::anyhow!("couldn't encode"))?;

        let encoded = OnnxInputWrapper {
            input_ids: encoding.get_ids().iter().map(|&i| i as Token).collect(),
            attention_mask: encoding
                .get_attention_mask()
                .iter()
                .map(|&i| i as Token)
                .collect(),
            token_type_ids: encoding
                .get_type_ids()
                .iter()
                .map(|&i| i as Token)
                .collect(),
        };
        Ok(encoded)
    }
}

#[frb(sync)]
pub fn tokenize(prompt: String) -> OnnxInputWrapper {
    _tokenize_internal_impl(prompt).unwrap()
}

#[frb(ignore)]
pub fn _tokenize_internal_impl(prompt: String) -> Result<OnnxInputWrapper> {
    let tokenizer = Tokenizer::from_pretrained("intfloat/e5-small-v2", None)?;

    let encoding = tokenizer.encode(prompt, false)?;
    Ok(OnnxInputWrapper {
        input_ids: encoding.get_ids().iter().map(|&i| i as Token).collect(),
        attention_mask: encoding
            .get_attention_mask()
            .iter()
            .map(|&i| i as Token)
            .collect(),
        token_type_ids: encoding
            .get_type_ids()
            .iter()
            .map(|&i| i as Token)
            .collect(),
    })
}
