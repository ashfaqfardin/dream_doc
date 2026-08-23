import torch
import math
import torch.nn.functional as F

def apply_rotary_emb(x, freqs_cis, use_real=True, use_real_unbounded_dim=-1):
    if use_real:
        cos, sin = freqs_cis
        cos = cos[None, None].to(x.device)
        sin = sin[None, None].to(x.device)

        if use_real_unbounded_dim == -1:
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbounded_dim == -2:
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


class TransformerBlock(torch.nn.Module):
    def __init__(self, orig_module, module_name):
        super().__init__()

        # Registering Modules
        self.orig_module = orig_module

        ## Normalization modules - Latent
        self.norm1 = orig_module.norm1
        self.norm2 = orig_module.norm2

        ## Normalization modules - Context
        self.norm1_context = orig_module.norm1_context
        self.norm2_context = orig_module.norm2_context

        ## Attention module
        self.attn = orig_module.attn

        ## Feed-Forward
        self.ff = orig_module.ff
        self.ff_context = orig_module.ff_context

        ## Other hyperparams
        self._chunk_size = getattr(orig_module, '_chunk_size', None)
        self._chunk_dim = getattr(orig_module, '_chunk_dim', 0)

        ## Module name
        self.module_name = module_name

    def calculate_attention_image_text(self, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # Predictions from latent
        query = self.attn.to_q(hidden_states)
        key = self.attn.to_k(hidden_states)
        value = self.attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.attn.heads

        query = query.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)

        if self.attn.norm_q is not None:
            query = self.attn.norm_q(query)

        if self.attn.norm_k is not None:
            key = self.attn.norm_k(key)

        if encoder_hidden_states is not None:
            # Predictions from context
            context_query_proj = self.attn.add_q_proj(encoder_hidden_states)
            context_key_proj = self.attn.add_k_proj(encoder_hidden_states)
            context_value_proj = self.attn.add_v_proj(encoder_hidden_states)

            context_query_proj = context_query_proj.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)
            context_key_proj = context_key_proj.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)
            context_value_proj = context_value_proj.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)

            if self.attn.norm_added_q:
                context_query_proj = self.attn.norm_added_q(context_query_proj)
            if self.attn.norm_added_k:
                context_key_proj = self.attn.norm_added_k(context_key_proj)

            # Combine predictions
            query = torch.cat([context_query_proj, query], dim=2)
            key = torch.cat([context_key_proj, key], dim=2)
            value = torch.cat([context_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # Compute Attention
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, :encoder_hidden_states.shape[1]], # Context
                hidden_states[:, encoder_hidden_states.shape[1]:]  # Image
            )

            # Projection
            hidden_states = self.attn.to_out[0](hidden_states)
            # Dropout
            hidden_states = self.attn.to_out[1](hidden_states)
            encoder_hidden_states = self.attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states

        return hidden_states

    def forward_block(self,
                hidden_states, # Latent input
                encoder_hidden_states, # Context input
                temb, # Time embedding
                image_rotary_emb=None,
                joint_attention_kwargs=None):

        # Normalize Latents
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        # Normalize Context
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(encoder_hidden_states, emb=temb)

        # Attention
        joint_attention_kwargs = joint_attention_kwargs or {}

        attn_output, context_attn_output = self.calculate_attention_image_text(hidden_states=norm_hidden_states,
                                                                    encoder_hidden_states=norm_encoder_hidden_states,
                                                                    image_rotary_emb=image_rotary_emb)

        # Process latent output
        attn_output = gate_msa.unsqueeze(1) * attn_output

        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process context output
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        context_ff_output = c_gate_mlp.unsqueeze(1) * context_ff_output

        encoder_hidden_states = encoder_hidden_states + context_ff_output

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states

    def forward(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None):
        if (joint_attention_kwargs["current_timestep_idx"] <= joint_attention_kwargs["stop_timestep_idx"]) and (joint_attention_kwargs["current_timestep_idx"] >= joint_attention_kwargs["start_timestep_idx"]):
            encoder_hidden_states, hidden_states_out = self.forward_attention_combine(hidden_states, encoder_hidden_states, temb, image_rotary_emb=image_rotary_emb, joint_attention_kwargs=joint_attention_kwargs)
        else:
            encoder_hidden_states, hidden_states_out = self.forward_block(hidden_states, encoder_hidden_states, temb, image_rotary_emb=image_rotary_emb, joint_attention_kwargs=joint_attention_kwargs)

        return encoder_hidden_states, hidden_states_out

    def get_cross_attention_mask(self, image_features, text_features, joint_attention_kwargs):
        batch_size, _, _ = image_features.shape

        query = self.attn.to_q(image_features)

        inner_dim = query.shape[-1]
        head_dim = inner_dim // self.attn.heads

        query = query.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)

        if self.attn.norm_q is not None:
            query = self.attn.norm_q(query)

        key = self.attn.add_k_proj(text_features)
        key = key.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)
        if self.attn.norm_added_k:
            key = self.attn.norm_added_k(key)

        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        scale_factor = math.sqrt(query.size(-1))
        attention_scores = attention_scores / scale_factor

        attention_map = torch.softmax(attention_scores, dim=-1).mean(dim=1)[:, :, 0]

        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

        attention_map = F.sigmoid(10 * (attention_map - 0.5)).unsqueeze(-1)

        threshold = joint_attention_kwargs.get("attention_threshold", 0.5)
        attention_map = (attention_map >= threshold).to(attention_map.dtype)


        return attention_map


    def forward_attention_combine(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None):
        edit_encoder_hidden_states = joint_attention_kwargs["edit_prompt_embeds"]
        neg_encoder_hidden_states = joint_attention_kwargs["neg_prompt_embeds"]
        temb_edit = joint_attention_kwargs["temb_edit"]
        edit_content_scale = joint_attention_kwargs["edit_content_scale"]

        # Normalize Latents
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        # Normalize Context
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(encoder_hidden_states, emb=temb_edit)

        # Normalize Edit Context
        norm_edit_hidden_states, c_gate_msa_edit, c_shift_mlp_edit, c_scale_mlp_edit, c_gate_mlp_edit = self.norm1_context(edit_encoder_hidden_states, emb=temb_edit)

        # Normalize Neg Context
        norm_neg_hidden_states, c_gate_msa_neg, c_shift_mlp_neg, c_scale_mlp_neg, c_gate_mlp_neg = self.norm1_context(neg_encoder_hidden_states, emb=temb_edit)

        # Attention
        joint_attention_kwargs = joint_attention_kwargs or {}

        attn_output, context_attn_output = self.calculate_attention_image_text(hidden_states=norm_hidden_states,
                                                                    encoder_hidden_states=norm_encoder_hidden_states,
                                                                    image_rotary_emb=image_rotary_emb)

        # Attention with edit condition
        attn_output_edit, context_attn_output_edit = self.calculate_attention_image_text(hidden_states=norm_hidden_states, encoder_hidden_states=norm_edit_hidden_states, image_rotary_emb=image_rotary_emb)

        # Attention with neg condition
        attn_output_neg, context_attn_output_neg = self.calculate_attention_image_text(hidden_states=norm_hidden_states, encoder_hidden_states=norm_neg_hidden_states, image_rotary_emb=image_rotary_emb)

        # Projecting the edit condition
        attn_prod = torch.sum(attn_output_edit * attn_output_neg, dim=2, keepdim=True)
        attn_norm_squared = torch.sum(attn_output_neg * attn_output_neg, dim=2, keepdim=True)

        attention_mask = self.get_cross_attention_mask(norm_hidden_states, norm_edit_hidden_states, joint_attention_kwargs)

        epsilon = 1e-10

        projection = (attn_prod / (attn_norm_squared + epsilon)) * attn_output_neg

        orthogonal_dir = attn_output_edit - projection
        orthogonal_dir = orthogonal_dir * attention_mask

        # Interpolation here
        orig_attn_norm = torch.norm(attn_output, dim=-1, keepdim=True)
        attn_output = attn_output + edit_content_scale * (orthogonal_dir)

        attn_output = attn_output / torch.norm(attn_output, dim=-1, keepdim=True) * orig_attn_norm

        # Process latent output
        attn_output = gate_msa.unsqueeze(1) * attn_output

        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process context output
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        context_ff_output = c_gate_mlp.unsqueeze(1) * context_ff_output

        encoder_hidden_states = encoder_hidden_states + context_ff_output

        # Process edit context output
        context_attn_output_edit = c_gate_msa_edit.unsqueeze(1) * context_attn_output_edit
        edit_encoder_hidden_states = edit_encoder_hidden_states + context_attn_output_edit

        norm_edit_hidden_states = self.norm2_context(edit_encoder_hidden_states)
        norm_edit_hidden_states = norm_edit_hidden_states * (1 + c_scale_mlp_edit[:, None]) + c_shift_mlp_edit[:, None]

        context_edit_ff_output = self.ff_context(norm_edit_hidden_states)
        context_edit_ff_output = c_gate_mlp_edit.unsqueeze(1) * context_edit_ff_output

        edit_encoder_hidden_states = edit_encoder_hidden_states + context_edit_ff_output

        # Process edit context output
        context_attn_output_neg = c_gate_msa_neg.unsqueeze(1) * context_attn_output_neg
        neg_encoder_hidden_states = neg_encoder_hidden_states + context_attn_output_neg

        norm_neg_hidden_states = self.norm2_context(neg_encoder_hidden_states)
        norm_neg_hidden_states = norm_neg_hidden_states * (1 + c_scale_mlp_neg[:, None]) + c_shift_mlp_neg[:, None]

        context_neg_ff_output = self.ff_context(norm_neg_hidden_states)
        context_neg_ff_output = c_gate_mlp_neg.unsqueeze(1) * context_neg_ff_output

        neg_encoder_hidden_states = neg_encoder_hidden_states + context_neg_ff_output

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
            edit_encoder_hidden_states = edit_encoder_hidden_states.clip(-65504, 65504)
            neg_encoder_hidden_states = neg_encoder_hidden_states.clip(-65504, 65504)

        joint_attention_kwargs["edit_prompt_embeds"] = edit_encoder_hidden_states
        joint_attention_kwargs["neg_prompt_embeds"] = neg_encoder_hidden_states

        return encoder_hidden_states, hidden_states


class SingleTransformerBlock(torch.nn.Module):
    def __init__(self, orig_module, module_name):
        super().__init__()

        # Registering Modules
        self.orig_module = orig_module

        ## Normalization Module
        self.norm = orig_module.norm

        ## Projection Modules
        self.mlp_hidden_dim = orig_module.mlp_hidden_dim
        self.proj_mlp = orig_module.proj_mlp
        self.proj_out = orig_module.proj_out
        self.act_mlp = orig_module.act_mlp

        ## Attention Module
        self.attn = orig_module.attn


    def calculate_attention(self, hidden_states, attention_mask=None, image_rotary_emb=None):
        batch_size, _, _ = hidden_states.shape

        # Latent projections
        query = self.attn.to_q(hidden_states)
        key = self.attn.to_k(hidden_states)
        value = self.attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.attn.heads

        query = query.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)

        if self.attn.norm_q is not None:
            query = self.attn.norm_q(query)
        if self.attn.norm_k is not None:
            key = self.attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        return hidden_states


    def forward_block(self, hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None):
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        joint_attention_kwargs = joint_attention_kwargs or {}

        attn_output = self.calculate_attention(hidden_states=norm_hidden_states,
                                               image_rotary_emb=image_rotary_emb)

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states

    def forward(self, hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None,
                encoder_hidden_states=None):
        # Newer diffusers (>=0.32) passes text and image tokens separately; concat before processing.
        txt_seq_len = 0
        if encoder_hidden_states is not None:
            txt_seq_len = encoder_hidden_states.shape[1]
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        hidden_states_out = self.forward_block(hidden_states, temb, image_rotary_emb=image_rotary_emb,
                                               joint_attention_kwargs=joint_attention_kwargs)

        if txt_seq_len > 0:
            return hidden_states_out[:, :txt_seq_len], hidden_states_out[:, txt_seq_len:]
        return hidden_states_out
