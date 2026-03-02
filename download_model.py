import torch

model = torch.hub.load("depth-anything/Depth-Anything-V2", "depth_anything_v2_vits", pretrained=True)
model.eval()

example_input = torch.rand(1, 3, 518, 518)

with torch.no_grad():
    # check_trace=False evita el error de "Constant nodes" que tuviste
    traced_model = torch.jit.trace(model, example_input, check_trace=False)

traced_model.save("depth_anything_v2.pt")
print("Modelo exportado con éxito para LibTorch 2.9.1")
