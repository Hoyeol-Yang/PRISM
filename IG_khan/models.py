import math
import os
import numpy as np
import torch, sys
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class KHANModel(nn.Module):

    def __init__(self, vocab_size: int, embed_size: int, nhead: int, d_hid: int, nlayers: int, dropout: float, num_class: int, knowledge_indices, alpha, beta):
        super(KHANModel, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(embed_size)

        self.embeddings = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.embed_size = embed_size

        self.pos_encoder = PositionalEncoding(embed_size, dropout, 2400)
        self.title_pos_encoder = PositionalEncoding(embed_size, dropout, 100)
        self.knowledge_encoder = KnowledgeEncoding(vocab_size, embed_size, knowledge_indices, alpha, beta, dropout)

        title_encoder_layers = TransformerEncoderLayer(embed_size, nhead, d_hid, dropout, batch_first=True)
        self.title_transformer = TransformerEncoder(title_encoder_layers, nlayers)

        word_encoder_layers = TransformerEncoderLayer(embed_size, nhead, d_hid, dropout, batch_first=True)
        self.word_transformer = TransformerEncoder(word_encoder_layers, nlayers)

        sentence_encoder_layers = TransformerEncoderLayer(embed_size, nhead, d_hid, dropout, batch_first=True)
        self.sentence_transformer = TransformerEncoder(sentence_encoder_layers, nlayers)

        self.title_multihead_attention = nn.MultiheadAttention(embed_size, nhead, dropout, batch_first=True)

        self.classifier = nn.Linear(embed_size, num_class)
        self.init_weights()


    def init_weights(self) -> None:
        initrange = 0.5
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        self.classifier.weight.data.uniform_(-initrange, initrange)
        self.classifier.bias.data.zero_()

    def forward(self, sentences: Tensor, titles: Tensor, return_attention: bool = False) -> Tensor:
        """
        Args:
            sentences: Tensor, shape [batch_size, sentence_len, word_len]
            return_attention: bool, if True, return attention weights along with output
        Returns:
            output: Tensor, shape[batch_size, num_class]
            (optional) attention_dict: dict with 'word_attention' and 'title_doc_attention'
        """

        isHierarchy = True
        isTitle = True

        # Storage for attention weights when return_attention=True
        word_attentions = [] if return_attention else None
        word_raw_scores = [] if return_attention else None  # raw scores before softmax

        if isHierarchy == True:
            title_embeddings = self.embeddings(titles).to(torch.long) * math.sqrt(self.embed_size)
            title_embeddings = self.title_pos_encoder(title_embeddings)
            title_embeddings = self.title_transformer(title_embeddings)
            title_embeddings = title_embeddings.mean(dim=1).unsqueeze(1)

            sentence_embeddings = []
            for texts in sentences: # batch_size (# of articles in a batch)
                word_embeddings = self.embeddings(texts) * math.sqrt(self.embed_size)
                residual = word_embeddings
                word_embeddings = self.knowledge_encoder(word_embeddings, texts)
                word_embeddings = word_embeddings + residual

                word_embeddings = self.pos_encoder(word_embeddings)
                word_embeddings = self.word_transformer(word_embeddings)

                # Compute word-level attention scores AFTER transformer
                # This captures contextual importance of each word
                if return_attention:
                    # Self-attention on transformer output: Q*K^T / sqrt(d)
                    attn_scores = torch.matmul(word_embeddings, word_embeddings.transpose(-2, -1))
                    attn_scores = attn_scores / math.sqrt(self.embed_size)
                    attn_weights = F.softmax(attn_scores, dim=-1)  # [num_sentences, seq_len, seq_len]
                    word_attentions.append(attn_weights.detach().cpu())
                    word_raw_scores.append(attn_scores.detach().cpu())  # raw scores before softmax

                sentence_embedding = word_embeddings.mean(dim=1)
                sentence_embeddings.append(sentence_embedding)

            sentence_embeddings = torch.stack(sentence_embeddings)
            sentence_embeddings = self.pos_encoder(sentence_embeddings)
            sentence_embeddings = self.sentence_transformer(sentence_embeddings)

            # Compute sentence-level attention scores AFTER transformer
            sentence_attention = None
            sentence_raw_scores = None
            if return_attention:
                # Self-attention on sentence embeddings: Q*K^T / sqrt(d)
                sent_attn_scores = torch.matmul(sentence_embeddings, sentence_embeddings.transpose(-2, -1))
                sent_attn_scores = sent_attn_scores / math.sqrt(self.embed_size)
                sent_attn_weights = F.softmax(sent_attn_scores, dim=-1)  # [batch, num_sentences, num_sentences]
                sentence_attention = sent_attn_weights.detach().cpu()
                sentence_raw_scores = sent_attn_scores.detach().cpu()

            # title-attention
            doc_embeddings = sentence_embeddings.mean(dim=1)
            title_doc_attention = None
            if isTitle == True:
                title_attn_out, title_attn_weights = self.title_multihead_attention(
                    title_embeddings, sentence_embeddings, sentence_embeddings
                )
                if return_attention:
                    title_doc_attention = title_attn_weights.detach().cpu()
                doc_embeddings = title_attn_out.squeeze(1) + doc_embeddings

            output = self.classifier(doc_embeddings)

            if return_attention:
                return output, {
                    'word_attention': word_attentions,  # List of [num_sentences, seq_len, seq_len] - softmax applied
                    'word_raw_scores': word_raw_scores,  # List of [num_sentences, seq_len, seq_len] - raw scores
                    'sentence_attention': sentence_attention,  # [batch, num_sentences, num_sentences] - softmax applied
                    'sentence_raw_scores': sentence_raw_scores,  # [batch, num_sentences, num_sentences] - raw scores
                    'title_doc_attention': title_doc_attention  # [batch, num_heads, 1, num_sentences]
                }
            return output

        else:
            texts = torch.flatten(sentences, start_dim=1)
            word_embeddings = self.embeddings(texts) * math.sqrt(self.embed_size)
            emb_with_pos = self.pos_encoder(word_embeddings)
            word_embeddings = self.word_transformer(emb_with_pos)
            doc_embeddings = word_embeddings.mean(dim=1)

            output = self.classifier(doc_embeddings)
            return output



class KnowledgeEncoding(nn.Module):

    def __init__(self, vocab_size: int, embed_size: int, knowledge_indices, alpha: float, beta: float, dropout: float = 0.3):
        super().__init__()

        self.alpha = alpha
        self.beta = beta

        # Use absolute path relative to models.py location (independent of working directory)
        _MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(_MODEL_DIR, 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f'knowledge_cache_v{vocab_size}_e{embed_size}.npz')

        # Load from cache if available
        if os.path.exists(cache_file):
            print(f'  - Loading cached Knowledge Embeddings from {cache_file}...')
            cached = np.load(cache_file)
            common_knowledge = cached['common']
            demo_knowledge = cached['demo']
            rep_knowledge = cached['rep']
        else:
            # Compute and cache on first run
            # YAGO uses RotatE; Political KG uses ModE (original paper settings)
            _pretrained = os.path.join(_MODEL_DIR, 'pre-trained')
            common_knowledge_path = os.path.join(_pretrained, 'YAGO.RotatE.')
            demo_knowledge_path = os.path.join(_pretrained, 'liberal.ModE.')
            rep_knowledge_path = os.path.join(_pretrained, 'conservative.ModE.')

            if embed_size == 128:
                common_knowledge_path += '128/entity_embedding.npy'
                demo_knowledge_path += '128/entity_embedding.npy'
                rep_knowledge_path += '128/entity_embedding.npy'
            elif embed_size == 256:
                common_knowledge_path += '256/entity_embedding.npy'
                demo_knowledge_path += '256/entity_embedding.npy'
                rep_knowledge_path += '256/entity_embedding.npy'
            elif embed_size == 512:
                common_knowledge_path += '512/entity_embedding.npy'
                demo_knowledge_path += '512/entity_embedding.npy'
                rep_knowledge_path += '512/entity_embedding.npy'
            else:
                print ('Wrong embedding dimension! Dimension should be 128, 256, 512, or 1024')
                sys.exit(1)

            common_pre_trained = np.load(common_knowledge_path)
            demo_pre_trained = np.load(demo_knowledge_path)
            rep_pre_trained = np.load(rep_knowledge_path)

            # Vectorized mapping (much faster than a plain loop)
            print('  - Building Knowledge Embeddings (first time, will be cached)...')

            # Convert knowledge_indices to numpy arrays
            common_indices = np.array(knowledge_indices['common'])
            demo_indices = np.array(knowledge_indices['demo'])
            rep_indices = np.array(knowledge_indices['rep'])

            # Initialize embedding matrices
            common_knowledge = np.zeros((vocab_size, embed_size), dtype=np.float32)
            demo_knowledge = np.zeros((vocab_size, embed_size), dtype=np.float32)
            rep_knowledge = np.zeros((vocab_size, embed_size), dtype=np.float32)

            # Fill in embeddings
            for j, vocab_idx in enumerate(common_indices):
                if vocab_idx > 0 and vocab_idx < vocab_size:
                    common_knowledge[vocab_idx] = common_pre_trained[j]

            for j, vocab_idx in enumerate(rep_indices):
                if vocab_idx > 0 and vocab_idx < vocab_size:
                    rep_knowledge[vocab_idx] = rep_pre_trained[j]

            for j, vocab_idx in enumerate(demo_indices):
                if vocab_idx > 0 and vocab_idx < vocab_size:
                    demo_knowledge[vocab_idx] = demo_pre_trained[j]

            # Save to cache
            print(f'  - Saving cache to {cache_file}...')
            np.savez(cache_file, common=common_knowledge, demo=demo_knowledge, rep=rep_knowledge)

        self.common_knowledge = nn.Embedding.from_pretrained(torch.FloatTensor(common_knowledge))
        self.demo_knowledge = nn.Embedding.from_pretrained(torch.FloatTensor(demo_knowledge))
        self.rep_knowledge = nn.Embedding.from_pretrained(torch.FloatTensor(rep_knowledge))

        self.fuse_knowledge_fc = nn.Linear(embed_size*2, embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.5
        self.fuse_knowledge_fc.weight.data.uniform_(-initrange, initrange)
        self.fuse_knowledge_fc.bias.data.zero_()

    def forward(self, word_embeddings: Tensor, texts: Tensor) -> Tensor:

        emb_with_ckwldg = (word_embeddings * self.alpha) + (self.common_knowledge(texts) * (1-self.alpha))

        demo_knwldg = (emb_with_ckwldg * self.beta) + (self.demo_knowledge(texts) * (1-self.beta))
        rep_knwldg = (emb_with_ckwldg * self.beta) + (self.rep_knowledge(texts) * (1-self.beta))

        # Concatenate and pass through FC layer
        emb_with_knowledge = self.fuse_knowledge_fc(torch.cat((demo_knwldg, rep_knwldg), 2))
        return self.dropout(emb_with_knowledge)



class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.3, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)
