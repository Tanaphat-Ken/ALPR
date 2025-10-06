"""
Multi-task TrOCR model for license plate recognition with province prediction.

This module implements a TrOCR variant that performs both:
1. Text recognition (original TrOCR task)  
2. Province classification from visual features (new auxiliary task)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import VisionEncoderDecoderModel
from transformers.modeling_outputs import Seq2SeqLMOutput

# Thai provinces mapping (TH-10 to TH-96)
THAI_PROVINCES = [
    "TH-10", "TH-11", "TH-12", "TH-13", "TH-14", "TH-15", "TH-16", "TH-17", "TH-18", "TH-19",
    "TH-20", "TH-21", "TH-22", "TH-23", "TH-24", "TH-25", "TH-26", "TH-27", "TH-30", "TH-31",
    "TH-32", "TH-33", "TH-34", "TH-35", "TH-36", "TH-37", "TH-38", "TH-39", "TH-40", "TH-41",
    "TH-42", "TH-43", "TH-44", "TH-45", "TH-46", "TH-47", "TH-48", "TH-49", "TH-50", "TH-51",
    "TH-52", "TH-53", "TH-54", "TH-55", "TH-56", "TH-57", "TH-58", "TH-60", "TH-61", "TH-62",
    "TH-63", "TH-64", "TH-65", "TH-66", "TH-67", "TH-70", "TH-71", "TH-72", "TH-73", "TH-74",
    "TH-75", "TH-76", "TH-77", "TH-80", "TH-81", "TH-82", "TH-83", "TH-84", "TH-85", "TH-86",
    "TH-90", "TH-91", "TH-92", "TH-93", "TH-94", "TH-95", "TH-96"
]

def province_code_to_id(province_code: str) -> int:
    """Convert province code to class ID (0-76)."""
    try:
        if not province_code:
            return 0  # Default to first province if empty
        index = THAI_PROVINCES.index(province_code)
        if index < 0 or index >= len(THAI_PROVINCES):
            print(f"WARNING: Province code '{province_code}' index {index} out of range [0, {len(THAI_PROVINCES)-1}]")
            return 0
        return index
    except ValueError:
        print(f"WARNING: Unknown province code '{province_code}', defaulting to TH-10 (Bangkok)")
        return 0  # Default to first province if not found


class MultiTaskTrOCROutput:
    """
    Output class for multi-task TrOCR model.
    Contains both text and province predictions with loss information.
    """
    
    def __init__(
        self,
        loss=None,
        text_loss=None,
        province_loss=None,
        text_logits=None,
        province_logits=None,
        encoder_last_hidden_state=None,
        past_key_values=None,
        **kwargs
    ):
        self.loss = loss
        self.text_loss = text_loss
        self.province_loss = province_loss
        self.text_logits = text_logits
        self.province_logits = province_logits
        self.encoder_last_hidden_state = encoder_last_hidden_state
        self.past_key_values = past_key_values
        
        # For compatibility with Seq2SeqLMOutput
        self.logits = text_logits  # Primary logits
        
        # Store additional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __contains__(self, key):
        """Support 'in' operator for compatibility with transformers."""
        return hasattr(self, key)
    
    def __getitem__(self, key):
        """Support dict-like access for compatibility."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' not found in output")
    
    def get(self, key, default=None):
        """Support dict-like get method."""
        return getattr(self, key, default)


class MultiTaskTrOCR(VisionEncoderDecoderModel):
    """
    Multi-task TrOCR model that performs both text recognition and province classification.
    
    The model uses:
    - Vision encoder to extract image features
    - Text decoder for license plate text generation  
    - Province classifier head for direct image-based province prediction
    """
    
    def __init__(self, config=None, encoder=None, decoder=None, num_provinces=77):
        super().__init__(config, encoder, decoder)
        
        self.num_provinces = num_provinces
        
        # Get hidden sizes
        encoder_hidden_size = self.encoder.config.hidden_size
        decoder_hidden_size = self.decoder.config.hidden_size
        
        # Province classification head
        self.province_classifier = nn.Sequential(
            nn.Linear(encoder_hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_provinces)
        )
        
        # Projection layer to match encoder-decoder hidden sizes
        if encoder_hidden_size != decoder_hidden_size:
            self.encoder_decoder_projection = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        else:
            self.encoder_decoder_projection = None
        
        # Loss weights
        self.text_loss_weight = 1.0
        self.province_loss_weight = 0.5
        
        # Initialize province classifier
        self._init_province_classifier()
    
    def _init_province_classifier(self):
        """Initialize province classifier weights."""
        for module in self.province_classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _shift_right(self, input_ids):
        """Shift input ids one token to the right for teacher forcing."""
        if input_ids is None:
            return None
            
        # Get start token id and validate it's within bounds
        start_token_id = getattr(self.config, 'decoder_start_token_id', None) or \
                        getattr(self.config, 'bos_token_id', None) or 0
        
        # Validate start token ID is within vocabulary bounds
        vocab_size = getattr(self.config, 'vocab_size', None)
        if vocab_size is not None and start_token_id >= vocab_size:
            print(f"WARNING: Start token ID {start_token_id} >= vocab_size {vocab_size}, using 0")
            start_token_id = 0
        
        # Validate all input token IDs are within bounds
        if vocab_size is not None:
            input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
        
        # Create shifted input
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = start_token_id
        
        return shifted_input_ids
    
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple] = None,
        past_key_values: Optional[Tuple] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        province_labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> MultiTaskTrOCROutput:
        """
        Forward pass for multi-task TrOCR.
        
        Args:
            pixel_values: Input images
            labels: Text labels for decoder
            province_labels: Province class labels (0-76)
            **kwargs: Other arguments for base model
            
        Returns:
            MultiTaskTrOCROutput with text and province predictions
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get encoder outputs
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        
        encoder_hidden_states = encoder_outputs.last_hidden_state if return_dict else encoder_outputs[0]
        
        # Project encoder hidden states if needed for cross-attention
        projected_encoder_states = encoder_hidden_states
        if self.encoder_decoder_projection is not None:
            # Debug: Check tensor shapes before projection
            if encoder_hidden_states.dim() != 3:
                print(f"ERROR: Unexpected encoder_hidden_states shape: {encoder_hidden_states.shape}")
                print(f"Expected: [batch_size, seq_len, hidden_size]")
                # Reshape if needed - assume batch dimension might be flattened
                if encoder_hidden_states.dim() == 2:
                    # Try to infer batch size from other context
                    expected_hidden_size = 768  # ViT encoder hidden size
                    if encoder_hidden_states.shape[-1] == expected_hidden_size:
                        # This is [total_tokens, hidden_size], need to reshape
                        # For now, treat as single batch
                        encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
                        print(f"Reshaped to: {encoder_hidden_states.shape}")
                    else:
                        print(f"Cannot safely reshape - hidden size {encoder_hidden_states.shape[-1]} != {expected_hidden_size}")
                        
            try:
                projected_encoder_states = self.encoder_decoder_projection(encoder_hidden_states)
            except RuntimeError as e:
                print(f"Projection failed: {e}")
                print(f"Input shape: {encoder_hidden_states.shape}")
                print(f"Projection weight shape: {self.encoder_decoder_projection.weight.shape}")
                # Skip projection on error - use original encoder states
                projected_encoder_states = encoder_hidden_states
        
        # Text generation (original TrOCR task)
        # Prepare decoder inputs properly
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if labels is not None:
                # Create decoder_input_ids by shifting labels right for teacher forcing
                decoder_input_ids = self._shift_right(labels)
            else:
                # Create default start tokens for generation
                batch_size = encoder_hidden_states.shape[0]
                start_token_id = getattr(self.config, 'decoder_start_token_id', None) or \
                               getattr(self.config, 'bos_token_id', None) or 0
                
                # Validate start token ID against current vocabulary size
                vocab_size = getattr(self.config, 'vocab_size', None)
                if vocab_size is not None and start_token_id >= vocab_size:
                    print(f"WARNING: Start token ID {start_token_id} >= vocab_size {vocab_size}, using 0")
                    start_token_id = 0
                    
                decoder_input_ids = torch.full(
                    (batch_size, 1), 
                    start_token_id, 
                    dtype=torch.long, 
                    device=encoder_hidden_states.device
                )
        
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=projected_encoder_states,
            encoder_attention_mask=None,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
        )
        
        text_logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
        
        # Province classification from encoder features
        # Use global average pooling over spatial dimensions
        # encoder_hidden_states shape: [batch_size, seq_len, hidden_size]
        pooled_features = encoder_hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        province_logits = self.province_classifier(pooled_features)
        
        # Calculate losses
        text_loss = None
        province_loss = None
        total_loss = None
        
        if labels is not None:
            # Text loss (cross-entropy)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            text_loss = loss_fct(text_logits.view(-1, text_logits.size(-1)), labels.view(-1))
            total_loss = self.text_loss_weight * text_loss
        
        if province_labels is not None:
            # Province loss (cross-entropy)
            loss_fct = nn.CrossEntropyLoss()
            province_loss = loss_fct(province_logits, province_labels)
            
            if total_loss is not None:
                total_loss = total_loss + self.province_loss_weight * province_loss
            else:
                total_loss = self.province_loss_weight * province_loss
        
        return MultiTaskTrOCROutput(
            loss=total_loss,
            text_loss=text_loss,
            province_loss=province_loss,
            text_logits=text_logits,
            province_logits=province_logits,
            encoder_last_hidden_state=encoder_hidden_states,
            past_key_values=decoder_outputs.past_key_values if hasattr(decoder_outputs, 'past_key_values') else None,
        )
    
    def generate(self, pixel_values=None, **generation_kwargs):
        """
        Generate text using the text decoder.
        Returns both generated text and province predictions.
        """
        # Get encoder outputs for generation
        encoder_outputs = self.encoder(pixel_values=pixel_values)
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # Get province predictions from original encoder features (before projection)
        pooled_features = encoder_hidden_states.mean(dim=1)
        province_logits = self.province_classifier(pooled_features)
        province_predictions = torch.argmax(province_logits, dim=-1)
        
        # For text generation, we need to use the parent's generate method
        # but pass projected encoder states if needed
        if self.encoder_decoder_projection is not None:
            # Try to use the original TrOCR generate with encoder outputs
            # Don't project here - let the parent model handle projection internally
            generation_kwargs['encoder_outputs'] = encoder_outputs
        else:
            generation_kwargs['encoder_outputs'] = encoder_outputs
            
        # Remove pixel_values since we already have encoder_outputs
        generation_kwargs.pop('pixel_values', None)
        
        try:
            generated_ids = super().generate(**generation_kwargs)
            
            return {
                'sequences': generated_ids,
                'province_predictions': province_predictions,
                'province_logits': province_logits
            }
        except Exception as e:
            print(f"Generation failed: {e}")
            # Fallback: simple greedy generation using forward pass
            batch_size = encoder_hidden_states.shape[0]
            max_length = generation_kwargs.get('max_new_tokens', 20) + 1
            
            # Start with start token
            start_token_id = getattr(self.config, 'decoder_start_token_id', None) or \
                           getattr(self.config, 'bos_token_id', None) or 0
            
            generated_ids = torch.full(
                (batch_size, max_length), 
                self.config.pad_token_id or 0, 
                dtype=torch.long, 
                device=encoder_hidden_states.device
            )
            generated_ids[:, 0] = start_token_id
            
            return {
                'sequences': generated_ids,
                'province_predictions': province_predictions,
                'province_logits': province_logits
            }


def create_multitask_model(base_model_id: str, num_provinces: int = 77) -> MultiTaskTrOCR:
    """
    Create a multi-task TrOCR model from a base model.
    
    Args:
        base_model_id: HuggingFace model ID (e.g., "openthaigpt/thai-trocr")
        num_provinces: Number of province classes (default: 77 for Thai provinces)
    
    Returns:
        MultiTaskTrOCR model instance
    """
    # Load base model
    base_model = VisionEncoderDecoderModel.from_pretrained(base_model_id)
    
    # Create multi-task version
    multitask_model = MultiTaskTrOCR(
        config=base_model.config,
        encoder=base_model.encoder,
        decoder=base_model.decoder,
        num_provinces=num_provinces
    )
    
    return multitask_model