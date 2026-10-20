# EmoStudio Blur — установка. Запускается из install.bat.
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
try { $Host.UI.RawUI.WindowTitle = "EmoStudio Blur — установка" } catch {}

function Say([string]$text, [string]$color = "Gray") { Write-Host $text -ForegroundColor $color }
function Step([int]$n, [string]$text) { Write-Host ""; Write-Host "[$n/5] $text" -ForegroundColor Magenta }
function Ask([string]$question) {
    # «н» — это Y в русской раскладке
    $answer = Read-Host "$question [Y/N]"
    if ($null -eq $answer) { return $false }
    return ($answer.Trim().ToLower() -in @("y", "yes", "д", "да", "н"))
}
function Fail([string]$text) {
    Write-Host ""
    Say "ОШИБКА: $text" "Red"
    Say "Подробности смотри выше. Если не получается — открой README.md, раздел «Если что-то не работает»." "DarkGray"
    exit 1
}

function Get-PyVersion([string]$exe, [string[]]$pyArgs) {
    try {
        $v = & $exe @pyArgs -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $v) { return ([string]$v).Trim() }
    } catch {}
    return $null
}

function Find-Python {
    $candidates = @()
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($v in "3.12", "3.11", "3.10", "3.13") { $candidates += , @("py", @("-$v")) }
    }
    foreach ($name in "python", "python3") {
        if (Get-Command $name -ErrorAction SilentlyContinue) { $candidates += , @($name, @()) }
    }
    foreach ($p in Get-ChildItem "$env:LOCALAPPDATA\Programs\Python\Python3*\python.exe" -ErrorAction SilentlyContinue) {
        $candidates += , @($p.FullName, @())
    }
    foreach ($c in $candidates) {
        $ver = Get-PyVersion $c[0] $c[1]
        if ($ver) {
            $mm = [version]$ver
            if ($mm -ge [version]"3.10" -and $mm -lt [version]"3.14") {
                return @{ Exe = $c[0]; Args = $c[1]; Ver = $ver }
            }
        }
    }
    return $null
}

Say ""
Say "   ███  EmoStudio Blur" "Magenta"
Say "        a u t o   b l u r" "DarkMagenta"
Say "   ────────────────────────────────────" "DarkGray"
Say "   Установка займёт несколько минут. Нужен интернет." "Gray"

# 1. Python
Step 1 "Ищу Python 3.10–3.13"
$py = Find-Python
if (-not $py) {
    Say "Подходящий Python не найден." "Yellow"
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        if (Ask "Установить Python 3.12 автоматически?") {
            winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
            $py = Find-Python
        }
    }
    if (-not $py) {
        Say "Поставь Python вручную с python.org (при установке отметь «Add python.exe to PATH»)," "Yellow"
        Say "а потом снова запусти install.bat." "Yellow"
        Start-Process "https://www.python.org/downloads/"
        exit 1
    }
}
Say "Найден Python $($py.Ver)" "Green"

# 2. Окружение
Step 2 "Готовлю отдельное окружение (папка venv)"
$venvPy = Join-Path $Root "venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    $pyArgs = @($py.Args) + @("-m", "venv", "venv")
    & $py.Exe @pyArgs
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPy)) { Fail "не удалось создать окружение venv" }
    Say "Окружение создано" "Green"
} else {
    Say "Окружение уже есть" "Green"
}

# 3. Библиотеки
Step 3 "Ставлю библиотеки (это самый долгий шаг)"
& $venvPy -m pip install --upgrade pip --disable-pip-version-check -q
& $venvPy -m pip install -r requirements.txt --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { Fail "не удалось установить библиотеки" }
Say "Библиотеки готовы" "Green"

# 4. Ярлыки — раньше моделей, чтобы ярлык был, даже если окно закроют во время долгого скачивания
Step 4 "Создаю ярлыки на рабочем столе и в меню «Пуск»"
& $venvPy (Join-Path $Root "src\shortcuts.py")
if ($LASTEXITCODE -ne 0) {
    Say "Ярлыки создать не получилось — запускай через «EmoStudio Blur.bat»." "Yellow"
    Say "Подробности: %APPDATA%\EmoStudio Blur\shortcuts.log" "DarkGray"
}

# 5. Модели
Step 5 "Скачиваю модели распознавания (около 100 МБ)"
Say "Это может занять несколько минут. Не закрывай окно, даже если кажется, что ничего не происходит." "Yellow"
$env:HF_HUB_VERBOSITY = "error"
$env:PYTHONWARNINGS = "ignore"
& $venvPy (Join-Path $Root "src\censor_core.py") --download
if ($LASTEXITCODE -ne 0) {
    Say "Модели пока не скачались (нет интернета или huggingface.co недоступен)." "Yellow"
    Say "Ничего страшного: программа докачает их сама при первом запуске." "Yellow"
}

Write-Host ""
Say "   Готово! EmoStudio Blur установлен." "Green"
Say "   Запуск — ярлык «EmoStudio Blur» на рабочем столе." "Gray"
Write-Host ""
if (Ask "Запустить сейчас?") {
    Start-Process -FilePath (Join-Path $Root "venv\Scripts\pythonw.exe") `
        -ArgumentList "`"$(Join-Path $Root 'src\app.py')`"" -WorkingDirectory $Root
}
exit 0
