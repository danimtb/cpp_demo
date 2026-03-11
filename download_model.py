import torch

# MiDas_small, 256
# MiDas, 384
# DPT_Hybrid, 384
# DPT_Large, 384
# DPT_BEiT_L_512, 512

model_type = "DPT_Large"
model = torch.hub.load("intel-isl/MiDaS", model_type)
model.eval()

example_input = torch.rand(1, 3, 384, 384)
    
with torch.no_grad():
    # check_trace=False ignores the 'len()' warnings in the ViT backbone
    traced_model = torch.jit.trace(model, example_input, check_trace=False)

output_path = f"models/{model_type}.pt"
traced_model.save(output_path)
