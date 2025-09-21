param(
    [string]$EnvName = "venv",
    [string]$Python = "python", 
    [switch]$CUDA
)
${ErrorActionPreference} = "Stop"

Write-Host "Creando entorno virtual: $EnvName" -ForegroundColor Cyan
& $Python -m venv $EnvName

Write-Host "Activando entorno virtual" -ForegroundColor Cyan
. "$EnvName\Scripts\Activate.ps1"

Write-Host "Actualizando pip" -ForegroundColor Cyan
pip install --upgrade pip

if ($CUDA) {
    Write-Host "Instalando PyTorch con CUDA (ajusta versión/URL si es necesario)" -ForegroundColor Yellow
    # Ejemplo: CUDA 12.1 (ajustar según GPU/driver):
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install torchvision --index-url https://download.pytorch.org/whl/cu121
}

Write-Host "Instalando dependencias de requirements.txt" -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "Listo. Para activar el entorno en futuras sesiones:" -ForegroundColor Green
Write-Host ".\\$EnvName\\Scripts\\Activate.ps1" -ForegroundColor Green


