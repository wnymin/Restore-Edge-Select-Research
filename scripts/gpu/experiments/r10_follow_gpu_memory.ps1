param(
    [Parameter(Mandatory = $true)][int]$WorkerId
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$counters = '\GPU Process Memory(*)\Dedicated Usage', '\GPU Process Memory(*)\Shared Usage', '\GPU Process Memory(*)\Total Committed',
    '\Memory\Available Bytes', '\Memory\Committed Bytes', '\Memory\Cache Bytes'
$prefix = 'pid_' + $WorkerId + '_'
[Console]::WriteLine('host_utc,pid,gpu_instance,counter,bytes')
while (Get-Process -Id $WorkerId -ErrorAction SilentlyContinue) {
    try {
        $capture = Get-Counter -Counter $counters -ErrorAction Stop
    } catch {
        # A process-specific wildcard can disappear during the collection.
        if (-not (Get-Process -Id $WorkerId -ErrorAction SilentlyContinue)) { break }
        [Console]::Error.WriteLine(('{0},counter sample missed,{1}' -f [DateTime]::UtcNow.ToString('o'), $_.FullyQualifiedErrorId))
        Start-Sleep -Milliseconds 400
        continue
    }
    $stamp = [DateTime]::UtcNow.ToString('o')
    foreach ($row in $capture.CounterSamples) {
        if ($row.Status -ne 0) { continue }
        $name = $row.Path.Substring($row.Path.LastIndexOf('\') + 1).ToLowerInvariant().Replace(' ', '_')
        if ($row.Path -match '\\memory\\') {
            [Console]::WriteLine(('{0},0,system,{1},{2}' -f $stamp, $name, [int64]$row.CookedValue))
        } elseif ($row.InstanceName -and $row.InstanceName.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
            [Console]::WriteLine(('{0},{1},{2},{3},{4}' -f $stamp, $WorkerId, $row.InstanceName, $name, [int64]$row.CookedValue))
        }
    }
    Start-Sleep -Milliseconds 800
}
