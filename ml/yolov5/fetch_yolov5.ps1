param(
    [string]$Destination = "external/yolov5",
    [string]$Branch = "v7.0"
)

$ErrorActionPreference = "Stop"

if (Test-Path -LiteralPath $Destination) {
    Write-Host "Destination already exists: $Destination"
    exit 0
}

git clone --depth 1 --branch $Branch https://github.com/ultralytics/yolov5.git $Destination

