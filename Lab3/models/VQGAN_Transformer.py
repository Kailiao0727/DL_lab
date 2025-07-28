import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        mapping, indices, _ = self.vqgan.encode(x)
        indices = indices.reshape(mapping.shape[0], -1)
        return indices
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        def linear(r): return 1 - r
        def cosine(r): return np.cos(r * np.pi / 2)
        def square(r): return 1 - r ** 2
        if mode == "linear":
            return linear
        elif mode == "cosine":
            return cosine
        elif mode == "square":
            return square
        else:
            raise NotImplementedError

##TODO2 step1-3:            
    def forward(self, x):
        
        z_indices= self.encode_to_z(x) #ground truth

        B, N = z_indices.shape
        device = z_indices.device

        # Random masking
        mask_ratio = np.random.uniform(0.3, 0.7)
        rand_mask = torch.rand(B, N, device=device) < mask_ratio
        masked_input = z_indices.clone()
        masked_input[rand_mask] = self.mask_token_id

        logits = self.transformer(masked_input)  #transformer predict the probability of tokens

        return logits, z_indices    
    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, z_indices, mask, num_mask, r = 1.0, mask_func="cosine"):
        B, N = z_indices.shape
        device = z_indices.device
        z_indices_with_mask = mask * self.mask_token_id + (~mask) * z_indices
        logits = self.transformer(z_indices_with_mask)
        #Apply softmax to convert logits into a probability distribution across the last dimension.
        
        logits[..., self.mask_token_id] = float('-inf')
        logits = torch.softmax(logits, dim=-1)

        #FIND MAX probability for each token value
        z_indices_predict_prob, z_indices_predict = logits.max(dim=-1)

        mask_ratio = self.gamma_func(mask_func)(r)

        #predicted probabilities add temperature annealing gumbel noise as confidence
        g = -torch.empty_like(z_indices_predict_prob).exponential_().log()  # gumbel noise
        temperature = self.choice_temperature * (1 - mask_ratio)
        confidence = z_indices_predict_prob + temperature * g
        
        #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        confidence[~mask] = float('inf')
        #sort the confidence for the rank 
        sorted_confidence, sorted_indices = torch.sort(confidence, dim=-1)
        #define how much the iteration remain predicted tokens by mask scheduling
        next_mask = mask.clone()
        current_masked = mask.sum(dim=1)
        num_unmask = (mask.shape[1] - num_mask*mask_ratio).long()
        # print(f"Mask ratio: {mask_ratio}, Current masked tokens: {current_masked}, num_unmask: {num_unmask}")

        
        if num_unmask[0] > 0:
            # keep only the top num_mask tokens masked
            unmask_idx = sorted_indices[0, -num_unmask[0]:]
            next_mask[0, unmask_idx] = False
        # print(f"Next mask shape: {next_mask.shape}, Next mask sum: {next_mask.sum(dim=1)}")
        

        ##At the end of the decoding process, add back the original(non-masked) token values
        z_indices_predict = mask * z_indices_predict + (~mask) * z_indices
        
        return z_indices_predict, next_mask
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
