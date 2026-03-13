import torch

model_name = "DPT_Hybrid"

model = torch.hub.load("intel-isl/MiDaS", model_name)


device = torch.device("cpu")
model.to(device)
model.eval()

torch.save(model.state_dict(), f"{model_name}.pt")
