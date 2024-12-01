# expert-token-resonance-pytorch
Unofficial implementation of Expert Token Resonance in PyTorch (and later Triton)

Based on the the excellent paper:  
### Expert-Token Resonance: Redefining MoE Routing through Affinity-Driven Active Selection  
Jing Li, Zhijie Sun, Dachao Lin, Xuan He, Yi Lin, Binfan Zheng, Li Zeng, Rongqian Zhao, Xin Chen

https://arxiv.org/abs/2406.00023  
(backup link:  https://doi.org/10.48550/arXiv.2406.00023  )


Status:  
1 - An initial working version for PyTorch eager.  I have finally been able to implement equation 4 properly,   
    and verified it results in cosine similarity of [-1,1]. (matmuls' didn't want to match earlier).  
  
2 - Next: Planning to use it in llm-base test bed to make sure it trains as expected. 




