# crf_model.py
import torch
import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF


class BanglaBertCRF(nn.Module):
    """
    BERT + CRF model for Named Entity Recognition.
    
    Architecture:
        BERT → Dropout → Linear (emission scores) → CRF (Viterbi decode)
    
    The CRF layer adds a learnable transition matrix that enforces globally
    valid label sequences (e.g., I-DIS cannot follow O without B-DIS first),
    which dramatically improves boundary detection in NER tasks.
    """

    def __init__(self, model_name_or_path: str, num_labels: int):
        super(BanglaBertCRF, self).__init__()
        self.num_labels = num_labels

        # Load base transformer WITHOUT the classification head
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        self.dropout = nn.Dropout(0.1)

        # Linear layer maps hidden states to per-token emission scores
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

        # CRF layer: learns valid label transition probabilities
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """
        Forward pass compatible with HuggingFace Trainer.

        During training (labels is not None):
            Returns (loss, predictions) where loss is CRF NLL loss.
        
        During inference (labels is None):
            Returns (predictions,) — Viterbi decoded optimal paths.
        """
        # 1. Get contextual hidden states from BERT backbone
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(bert_output.last_hidden_state)

        # 2. Compute emission scores (logits before CRF)
        emissions = self.classifier(sequence_output)  # shape: [B, seq_len, num_labels]

        # 3. CRF mask — True for real tokens, False for padding
        mask = attention_mask.bool()

        # 4. Viterbi decode (always computed for both train & inference)
        decoded_paths = self.crf.decode(emissions, mask=mask)

        # 5. Pad decoded paths back to full seq_len (Trainer needs consistent shapes)
        seq_len = emissions.size(1)
        preds = torch.zeros(emissions.size(0), seq_len, dtype=torch.long, device=emissions.device)
        for i, path in enumerate(decoded_paths):
            preds[i, :len(path)] = torch.tensor(path, dtype=torch.long, device=emissions.device)

        if labels is not None:
            # CRITICAL: CRF crashes if label indices are out of range.
            # HuggingFace uses -100 for subword/padding tokens.
            # We replace -100 with 0 (the 'O' tag) only for the CRF loss pass.
            # compute_metrics will still correctly ignore them via the l != -100 check.
            safe_labels = labels.clone()
            safe_labels[safe_labels == -100] = 0

            # CRF computes exact sequence probability via dynamic programming
            # Returns NEGATIVE log-likelihood — we negate to get loss
            loss = -self.crf(emissions, safe_labels, mask=mask, reduction='mean')

            return (loss, preds)

        else:
            # Inference: just return Viterbi paths
            return (preds,)

    def get_emissions(self, input_ids, attention_mask=None):
        """
        Utility method to extract raw emission logits (pre-CRF) for use in
        Knowledge Distillation. The KD student uses these as soft targets.
        """
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(bert_output.last_hidden_state)
        return self.classifier(sequence_output)
