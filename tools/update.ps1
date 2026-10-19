# EmoStudio Blur — обновление до последней версии с GitHub.
# Запускается из update.bat или кнопкой «Обновить» в программе.
param(
    [switch]$Relaunch,       # после обновления снова открыть программу
    [int]$WaitPid = 0,       # дождаться, пока закроется окно программы
    [switch]$SkipShortcuts   # не трогать ярлыки (для проверки обновления на копии)
)
$ErrorActionPreference = "Continue"
$Repo = "Hesoyam-228/emostudio-blur"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
try { $Host.UI.RawUI.WindowTitle = "EmoStudio Blur — обновление" } catch {}

function Say([string]$text, [string]$color = "Gray") { Write-Host $text -ForegroundColor $color }
function Step([int]$n, [string]$text) { Write-Host ""; Write-Host "[$n/3] $text" -ForegroundColor Magenta }
function Done([int]$code) {
    if ($Relaunch) { Write-Host ""; Read-Host "Нажми Enter, чтобы закрыть" | Out-Null }
    exit $code
}
function Fail([string]$text, [bool]$untouched = $true) {
    Write-Host ""
    Say "ОШИБКА: $text" "Red"
    if ($untouched) { Say "Программа не изменена, можно пользоваться старой версией." "DarkGray" }
    else { Say "Файлы уже обновлены. Запусти install.bat, он доустановит остальное." "DarkGray" }
    Done 1
}
function Read-Version { $v = Get-Content (Join-Path $Root "version.txt") -ErrorAction SilentlyContinue | Select-Object -First 1; if ($v) { $v.Trim() } else { "?" } }

Say ""
Say "   ███  EmoStudio Blur — обновление" "Magenta"
Say "   ────────────────────────────────────" "DarkGray"

if ($WaitPid) {
    Say "Жду, пока закроется программа..." "DarkGray"
    try { Wait-Process -Id $WaitPid -Timeout 30 -ErrorAction Stop } catch {}
}
$old = Read-Version

# 1. Новые файлы
if (Test-Path (Join-Path $Root ".git")) {
    Step 1 "Забираю новую версию через git"
    git -C $Root pull --ff-only
    if ($LASTEXITCODE -ne 0) { Fail "git pull не прошёл (возможно, в папке есть свои изменения)" }
} else {
    Step 1 "Скачиваю новую версию с GitHub"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $tmp = Join-Path $env:TEMP "EmoStudioBlur-update"
    Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory $tmp | Out-Null
    $zip = Join-Path $tmp "main.zip"
    try {
        $ProgressPreference = "SilentlyContinue"
        Invoke-WebRequest "https://github.com/$Repo/archive/refs/heads/main.zip" -OutFile $zip -UseBasicParsing
    } catch { Fail "не удалось скачать: $($_.Exception.Message). Проверь интернет." }
    try { Expand-Archive $zip -DestinationPath $tmp -Force } catch { Fail "скачанный архив повреждён, попробуй ещё раз" }
    $src = Get-ChildItem $tmp -Directory | Select-Object -First 1
    if (-not $src -or -not (Test-Path (Join-Path $src.FullName "src\app.py"))) { Fail "в архиве не EmoStudio Blur" }
    # копируем поверх: venv, настройки и арты пользователя не трогаем, лишнего не удаляем
    robocopy $src.FullName $Root /E /XD venv .git __pycache__ /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { Fail "не удалось скопировать новые файлы (код robocopy $LASTEXITCODE)" $false }
    Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
}
$new = Read-Version
Say "Версия: $old -> $new" "Green"

# 2. Библиотеки и модели
$venvPy = Join-Path $Root "venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Say "Окружения venv нет — запускаю полную установку." "Yellow"
    & (Join-Path $PSScriptRoot "setup.ps1")
    exit $LASTEXITCODE
}
Step 2 "Обновляю библиотеки и модели"
& $venvPy -m pip install -r (Join-Path $Root "requirements.txt") --disable-pip-version-check -q
if ($LASTEXITCODE -ne 0) { Fail "не удалось обновить библиотеки" $false }
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$env:HF_HUB_VERBOSITY = "error"
$env:PYTHONWARNINGS = "ignore"
& $venvPy (Join-Path $Root "src\censor_core.py") --download
if ($LASTEXITCODE -ne 0) { Fail "не удалось скачать модели (проверь интернет)" $false }

# 3. Ярлыки (иконка могла поменяться)
Step 3 "Обновляю ярлыки"
if ($SkipShortcuts) { Say "пропущено" "DarkGray" }
else { & $venvPy (Join-Path $Root "src\shortcuts.py") | Out-Null; Say "Готово" "Green" }

Write-Host ""
Say "   EmoStudio Blur обновлён до версии $new" "Green"
if ($Relaunch) {
    Start-Process -FilePath (Join-Path $Root "venv\Scripts\pythonw.exe") `
        -ArgumentList "`"$(Join-Path $Root 'src\app.py')`"" -WorkingDirectory $Root
    Start-Sleep -Seconds 2
}
exit 0
