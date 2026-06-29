param(
    [string]$Destination = "external/ascend-devkit",
    [string]$Branch = "master"
)

$ErrorActionPreference = "Stop"

if (Test-Path -LiteralPath $Destination) {
    Write-Host "Destination already exists: $Destination"
    Write-Host "Run 'git -C $Destination pull' if you need to update it."
    exit 0
}

git clone --depth 1 --filter=blob:none --sparse --branch $Branch `
    https://gitee.com/HUAWEI-ASCEND/ascend-devkit.git $Destination

Push-Location $Destination
git sparse-checkout set src/E2E-Sample/Car
Pop-Location

Write-Host "Fetched car source to $Destination/src/E2E-Sample/Car"

