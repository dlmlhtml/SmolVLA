# Run in Administrator PowerShell. This installs WSL/Ubuntu, not Python on Windows.
$ErrorActionPreference = "Stop"
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Open PowerShell as Administrator and run this script again."
}
if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw "WSL command is unavailable. Follow https://learn.microsoft.com/windows/wsl/install"
}
& wsl.exe --install -d Ubuntu-24.04 --no-launch
if ($LASTEXITCODE -notin @(0, 3010)) {
    throw "WSL installation failed (exit $LASTEXITCODE). See the output above."
}
Write-Host "Restart Windows if requested. Then run: wsl -d Ubuntu-24.04"
Write-Host "Create your Ubuntu username/password on first launch."
Write-Host "Continue with WINDOWS_4060.md, section 2, inside Ubuntu."
