Write-Host "🚀 Multimodal AI Production Setup (Windows PowerShell)"
Write-Host "========================================================="

if (-not (Get-Command "python" -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Python is not found in PATH." -ForegroundColor Red
    exit 1
}

Write-Host "📦 Creating virtual environment..." -ForegroundColor Cyan
python -m venv .venv
. .venv\Scripts\Activate.ps1

Write-Host "📦 Installing dependencies..." -ForegroundColor Cyan
python -m pip install --upgrade pip
pip install -r requirements.txt

if (-not (Test-Path ".env")) {
    Copy-Item "config\config.example.yaml" -Destination ".env" -ErrorAction SilentlyContinue
    Copy-Item ".env.example" -Destination ".env" -ErrorAction SilentlyContinue
    Write-Host "📝 Created .env file (please review and edit)" -ForegroundColor Yellow
}

Write-Host "📁 Creating required directories..." -ForegroundColor Cyan
$dirs = @(
    "models/base", "models/adapters", "models/fine-tuned", "models/merged",
    "data/raw", "data/processed", "data/eval", "data/memory", "data/embeddings/chromadb", "data/generated",
    "logs"
)

foreach ($d in $dirs) {
    if (-not (Test-Path $d)) {
        New-Item -Path $d -ItemType Directory -Force | Out-Null
    }
}

Write-Host "✅ Setup Complete!" -ForegroundColor Green
Write-Host "Next steps:"
Write-Host "  1. Edit .env and config\config.yaml"
Write-Host "  2. Download model: python scripts\download_model.py --all"
Write-Host "  3. Start training: python -m src.training.train --config config\config.yaml"
Write-Host "  4. Start server: python -m src.api.server"
