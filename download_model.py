import torch

# Cargamos el modelo
model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
model.eval()

# Creamos la entrada de ejemplo
example_input = torch.rand(1, 3, 384, 384)

# Forzamos el trazado ignorando los chequeos de consistencia
with torch.no_grad():
    traced_script_module = torch.jit.trace(model, example_input, check_trace=False)

traced_script_module.save("MiDaS_small.pt")
print("¡Solucionado! 'MiDaS_small.pt' generado correctamente.")