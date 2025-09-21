## Entorno y prueba rápida (Windows PowerShell)

1) Crear y activar entorno virtual e instalar dependencias

```powershell
./setup_env.ps1 -EnvName venv
# Para CUDA (ajustar a tu versión):
# ./setup_env.ps1 -EnvName venv -CUDA
```

2) Probar el modelo híbrido (smoke test)

```powershell
./venv/Scripts/Activate.ps1
python smoke_test.py
```

Deberías ver un "Smoke test OK" si todo está bien. El test crea una configuración mínima, construye `ECGHybridVariableBeforeBiTrans` y valida la forma de salida.

### Notas
- Para GPUs NVIDIA, instala la build de PyTorch con CUDA según `https://pytorch.org/get-started/locally/`.
- Ajusta `num_leads` y `trans_dim` en el `smoke_test.py` para que coincidan con tus datos reales.
- Si usas datasets y entrenador del directorio `ECGTransForm/`, integra el nuevo modelo importando `from ModeloNuevo import ECGHybridVariableBeforeBiTrans` y reemplazando la instancia del modelo.


