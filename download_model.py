import torch

model_name = "DPT_Hybrid"

model = torch.hub.load("intel-isl/MiDaS", model_name)

model_type = "DPT_Hybrid"

device = torch.device("cpu")
model.to(device)
model.eval()

torch.save(model.state_dict(), f"models/{model_name}.pt")
