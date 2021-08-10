#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Encoder model wrappers based on HuggingFace code
"""

import logging
from typing import Tuple
<<<<<<< HEAD
import nltk
=======
>>>>>>> 892e63999c2f7b4b9f710ef70d4e95c8d306956e

import torch
from torch import Tensor as T
from torch import nn
from transformers.modeling_bert import BertConfig, BertModel
from transformers.optimization import AdamW
from transformers.tokenization_bert import BertTokenizer
from transformers.tokenization_roberta import RobertaTokenizer

from dpr.utils.data_utils import Tensorizer
from .biencoder import BiEncoder
from .reader import Reader

logger = logging.getLogger(__name__)


def get_bert_biencoder_components(args, inference_only: bool = False, **kwargs):
    dropout = args.dropout if hasattr(args, "dropout") else 0.0
    question_encoder = HFBertEncoder.init_encoder(
        args.pretrained_model_cfg,
        projection_dim=args.projection_dim,
        dropout=dropout,
        **kwargs
    )
    ctx_encoder = HFBertEncoder.init_encoder(
        args.pretrained_model_cfg,
        projection_dim=args.projection_dim,
        dropout=dropout,
        **kwargs
    )

    fix_ctx_encoder = (
        args.fix_ctx_encoder if hasattr(args, "fix_ctx_encoder") else False
    )
    biencoder = BiEncoder(
        question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder
    )

    optimizer = (
        get_optimizer(
            biencoder,
            learning_rate=args.learning_rate,
            adam_eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_bert_tensorizer(args)

<<<<<<< HEAD
    augument_tensorizer = get_bert_augument_tensorizer(args)


    return tensorizer, biencoder, optimizer , augument_tensorizer
=======
    return tensorizer, biencoder, optimizer
>>>>>>> 892e63999c2f7b4b9f710ef70d4e95c8d306956e


def get_bert_reader_components(args, inference_only: bool = False, **kwargs):
    dropout = args.dropout if hasattr(args, "dropout") else 0.0
    encoder = HFBertEncoder.init_encoder(
        args.pretrained_model_cfg, projection_dim=args.projection_dim, dropout=dropout
    )

    hidden_size = encoder.config.hidden_size
    reader = Reader(encoder, hidden_size)

    optimizer = (
        get_optimizer(
            reader,
            learning_rate=args.learning_rate,
            adam_eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_bert_tensorizer(args)
    return tensorizer, reader, optimizer


def get_bert_tensorizer(args, tokenizer=None):
    if not tokenizer:
        tokenizer = get_bert_tokenizer(
            args.pretrained_model_cfg, do_lower_case=args.do_lower_case
        )
    return BertTensorizer(tokenizer, args.sequence_length)

<<<<<<< HEAD
def get_bert_augument_tensorizer(args, tokenizer=None):
    if not tokenizer:
        tokenizer = get_bert_tokenizer(
            args.pretrained_model_cfg, do_lower_case=args.do_lower_case
        )
    return BertAugumentTensorizer(tokenizer, args.sequence_length , args.max_mask_length)


=======
>>>>>>> 892e63999c2f7b4b9f710ef70d4e95c8d306956e

def get_roberta_tensorizer(args, tokenizer=None):
    if not tokenizer:
        tokenizer = get_roberta_tokenizer(
            args.pretrained_model_cfg, do_lower_case=args.do_lower_case
        )
    return RobertaTensorizer(tokenizer, args.sequence_length)


def get_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-5,
    adam_eps: float = 1e-8,
    weight_decay: float = 0.0,
<<<<<<< HEAD
    ) -> torch.optim.Optimizer:
=======
) -> torch.optim.Optimizer:
>>>>>>> 892e63999c2f7b4b9f710ef70d4e95c8d306956e
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)
    return optimizer


def get_bert_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    return BertTokenizer.from_pretrained(
        pretrained_cfg_name, do_lower_case=do_lower_case
    )


<<<<<<< HEAD


=======
>>>>>>> 892e63999c2f7b4b9f710ef70d4e95c8d306956e
def get_roberta_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    # still uses HF code for tokenizer since they are the same
    return RobertaTokenizer.from_pretrained(
        pretrained_cfg_name, do_lower_case=do_lower_case
    )


class HFBertEncoder(BertModel):
    def __init__(self, config, project_dim: int = 0):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.encode_proj = (
            nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        )
        self.init_weights()

    @classmethod
    def init_encoder(
        cls, cfg_name: str, projection_dim: int = 0, dropout: float = 0.1, **kwargs
    ) -> BertModel:
        cfg = BertConfig.from_pretrained(cfg_name if cfg_name else "bert-base-uncased")
<<<<<<< HEAD
=======
        print(f'init using {cfg_name}')
>>>>>>> 892e63999c2f7b4b9f710ef70d4e95c8d306956e
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        return cls.from_pretrained(
            cfg_name, config=cfg, project_dim=projection_dim, **kwargs
        )

    def forward(
        self, input_ids: T, token_type_ids: T, attention_mask: T
    ) -> Tuple[T, ...]:
        if self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
        else:
            hidden_states = None
            sequence_output, pooled_output = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )

        pooled_output = sequence_output[:, 0, :]
        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size

<<<<<<< HEAD
    
# class BertAugumentTensorizer(Tensorizer):
#     def __init__(
#         self, tokenizer: BertTokenizer, max_length: int, pad_to_max: bool = True
#     ):
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.pad_to_max = pad_to_max

#     def text_to_tensor(
#         self, text: str, title: str = None, add_special_tokens: bool = True
#     ):
#         text = text.strip()

#         # tokenizer automatic padding is explicitly disabled since its inconsistent behavior
#         if title:
#             token_ids = self.tokenizer.encode(
#                 title,
#                 text_pair=text,
#                 add_special_tokens=add_special_tokens,
#                 max_length=self.max_length,
#                 pad_to_max_length=False,
#                 truncation=True,
#             )
#         else:
#             token_ids = self.tokenizer.encode(
#                 text,
#                 add_special_tokens=add_special_tokens,
#                 max_length=self.max_length,
#                 pad_to_max_length=False,
#                 truncation=True,
#             )

#         seq_len = self.max_length
#         if self.pad_to_max and len(token_ids) < seq_len:
#             token_ids = token_ids + [self.tokenizer.pad_token_id] * (
#                 seq_len - len(token_ids)
#             )
#         if len(token_ids) > seq_len:
#             token_ids = token_ids[0:seq_len]
#             token_ids[-1] = self.tokenizer.sep_token_id

#         return torch.tensor(token_ids)

#     def get_pair_separator_ids(self) -> T:
#         return torch.tensor([self.tokenizer.sep_token_id])

#     def get_pad_id(self) -> int:
#         return self.tokenizer.pad_token_id

#     def get_attn_mask(self, tokens_tensor: T) -> T:
#         return tokens_tensor != self.get_pad_id()

#     def is_sub_word_id(self, token_id: int):
#         token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
#         return token.startswith("##") or token.startswith(" ##")

#     def to_string(self, token_ids, skip_special_tokens=True):
#         return self.tokenizer.decode(token_ids, skip_special_tokens=True)

#     def set_pad_to_max(self, do_pad: bool):
#         self.pad_to_max = do_pad


class BertAugumentTensorizer(Tensorizer):
    def __init__(
        self, 
        tokenizer: BertTokenizer, 
        max_length: int, 
        max_mask_length: int,
        pad_to_max: bool = True, 
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max = pad_to_max
        self.max_mask_length = max_mask_length
        self.nltk_stopwords = nltk.corpus.stopwords.words("english")


    def text_to_tensor(
        self, text: str, title: str = None, add_special_tokens: bool = True
    ):
        text = text.strip()
        # tokenizer automatic padding is explicitly disabled since its inconsistent behavior
        if title:
            token_ids = self.tokenizer.encode(
                title,
                text_pair=text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length,
                pad_to_max_length=False,
                truncation=True,
            )
        else:
            # original colbert augument
            # token_ids = self.tokenizer.encode(
            #     text,
            #     add_special_tokens=add_special_tokens,
            #     max_length=self.max_length,
            #     pad_to_max_length=False,
            #     truncation=True,
            # )


            # mask_v2
            # tokens = self.tokenizer.tokenize(text)
            # pos = nltk.pos_tag(tokens)
            # new_tokens = []
            # for i in range(len(tokens)):
            #     p = pos[i]
            #     token = tokens[i]
            #     # if token is Noun or adj and is not stop words and not subword token
            #     if ('NN' in p[1] or 'JJ' in p[1]) and (token not in self.nltk_stopwords) and '#' not in token: 
            #         new_tokens.append(token)
            #         new_tokens.append('[MASK]')
            #     else:
            #         new_tokens.append(token)

            # # print(len(tokens))
            # tokens = ['[CLS]'] + new_tokens + ['[SEP]']
            # if len(tokens) >= self.max_mask_length:
            #     tokens = tokens[:self.max_mask_length]
            #     tokens[-1] = '[SEP]'
            # else:
            #     tokens = tokens + (['[MASK]'] * (self.max_mask_length - len(tokens)))
            # token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # mask_v1
            text_pieces = text.split()
            pos = nltk.pos_tag(text_pieces)
            new_text_pieces = []
            for i in range(len(text_pieces)):
                word = text_pieces[i]
                p = pos[i]
                if ('NN' in p[1] or 'JJ' in p[1]) and (word not in self.nltk_stopwords): 
                    new_text_pieces.append(word)
                    new_text_pieces.append('[MASK]')
                else:
                    new_text_pieces.append(word)
            new_text_pieces = ['[CLS]'] + new_text_pieces + ['[SEP]']
            text = ' '.join(new_text_pieces)
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) >= self.max_mask_length:
                tokens = tokens[:self.max_mask_length]
                tokens[-1] = '[SEP]'
            else:
                tokens = tokens + (['[MASK]'] * (self.max_mask_length - len(tokens)))
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            # print(new_text_pieces)
            # print(text)
            # print(tokens)
            # print(token_ids)
            # input()

        seq_len = self.max_length
        # colbert add mask token
        if len(token_ids) < seq_len:
            token_ids = token_ids + [self.tokenizer.mask_token_id] * (
                self.max_mask_length
            )

        if self.pad_to_max and len(token_ids) < seq_len:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (
                seq_len - len(token_ids)
            )
        
        if len(token_ids) > seq_len:
            token_ids = token_ids[0:seq_len]
            token_ids[-1] = self.tokenizer.sep_token_id


        # seq_len = self.max_length
        # if self.pad_to_max and len(token_ids) < seq_len:
        #     token_ids = token_ids + [self.tokenizer.pad_token_id] * (
        #         seq_len - len(token_ids)
        #     )
        # if len(token_ids) > seq_len:
        #     token_ids = token_ids[0:seq_len]
        #     token_ids[-1] = self.tokenizer.sep_token_id

        # print(text)
        # print(tokens)
        # print(token_ids)
        # input()

        return torch.tensor(token_ids)

    def get_pair_separator_ids(self) -> T:
        return torch.tensor([self.tokenizer.sep_token_id])

    def get_pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    def get_mask_id(self) ->int:
        return self.tokenizer.mask_token_id

    def get_attn_mask(self, tokens_tensor: T) -> T:
        return tokens_tensor != self.get_pad_id()
        # mask = tokens_tensor != self.get_mask_id()

        # batch_mask = []
        # for ids in tokens_tensor:
        #     flag = True
        #     mask =  []
        #     for i in range(len(ids)):
        #         if flag:
        #             mask.append(1)
        #         else:
        #             mask.append(0)
        #         if ids[i] == 102:
        #             flag = False
        #     batch_mask.append(mask)

        # batch_mask = torch.tensor(batch_mask)
        # return batch_mask.cuda()

    def is_sub_word_id(self, token_id: int):
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    def to_string(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def set_pad_to_max(self, do_pad: bool):
        self.pad_to_max = do_pad


=======
>>>>>>> 892e63999c2f7b4b9f710ef70d4e95c8d306956e

class BertTensorizer(Tensorizer):
    def __init__(
        self, tokenizer: BertTokenizer, max_length: int, pad_to_max: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max = pad_to_max

    def text_to_tensor(
        self, text: str, title: str = None, add_special_tokens: bool = True
    ):
        text = text.strip()

        # tokenizer automatic padding is explicitly disabled since its inconsistent behavior
        if title:
            token_ids = self.tokenizer.encode(
                title,
                text_pair=text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length,
                pad_to_max_length=False,
                truncation=True,
            )
        else:
            token_ids = self.tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length,
                pad_to_max_length=False,
                truncation=True,
            )

        seq_len = self.max_length
        if self.pad_to_max and len(token_ids) < seq_len:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (
                seq_len - len(token_ids)
            )
        if len(token_ids) > seq_len:
            token_ids = token_ids[0:seq_len]
            token_ids[-1] = self.tokenizer.sep_token_id

        return torch.tensor(token_ids)

    def get_pair_separator_ids(self) -> T:
        return torch.tensor([self.tokenizer.sep_token_id])

    def get_pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    def get_attn_mask(self, tokens_tensor: T) -> T:
        return tokens_tensor != self.get_pad_id()

    def is_sub_word_id(self, token_id: int):
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    def to_string(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def set_pad_to_max(self, do_pad: bool):
        self.pad_to_max = do_pad


class RobertaTensorizer(BertTensorizer):
    def __init__(self, tokenizer, max_length: int, pad_to_max: bool = True):
        super(RobertaTensorizer, self).__init__(
            tokenizer, max_length, pad_to_max=pad_to_max
        )
