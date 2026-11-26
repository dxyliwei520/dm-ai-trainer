# GPU / RAM desktop monitor (minimal, read-only)
# Double-click the desktop shortcut to start; close the window to exit.
# Reads NVML (nvml.dll) + GlobalMemoryStatusEx in-process every 1.5 s.
# No subprocess, no GPU work, does not touch any training/eval process.

param([switch]$Snapshot)   # -Snapshot = render once to %TEMP%\gpu-monitor.png and exit (debug only)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName PresentationFramework, PresentationCore, WindowsBase

if (-not ('GpuMon.Nv' -as [type])) {
    Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

namespace GpuMon {
    public static class Nv {
        [StructLayout(LayoutKind.Sequential)] public struct Mem { public ulong total; public ulong free; public ulong used; }
        [StructLayout(LayoutKind.Sequential)] public struct Util { public uint gpu; public uint mem; }

        [DllImport("nvml.dll", EntryPoint = "nvmlInit_v2")] public static extern int Init();
        [DllImport("nvml.dll", EntryPoint = "nvmlDeviceGetHandleByIndex_v2")] public static extern int Handle(uint index, out IntPtr dev);
        [DllImport("nvml.dll", EntryPoint = "nvmlDeviceGetMemoryInfo")] public static extern int MemInfo(IntPtr dev, out Mem m);
        [DllImport("nvml.dll", EntryPoint = "nvmlDeviceGetUtilizationRates")] public static extern int UtilRates(IntPtr dev, out Util u);
        [DllImport("nvml.dll", EntryPoint = "nvmlDeviceGetTemperature")] public static extern int Temp(IntPtr dev, uint sensor, out uint t);
    }

    public static class Sys {
        [StructLayout(LayoutKind.Sequential)] public struct MSX {
            public uint dwLength; public uint dwMemoryLoad;
            public ulong ullTotalPhys; public ulong ullAvailPhys;
            public ulong ullTotalPageFile; public ulong ullAvailPageFile;
            public ulong ullTotalVirtual; public ulong ullAvailVirtual; public ulong ullAvailExtendedVirtual;
        }
        [DllImport("kernel32.dll", SetLastError = true)] public static extern bool GlobalMemoryStatusEx(ref MSX m);

        public static double[] Ram() {
            MSX m = new MSX();
            m.dwLength = (uint)Marshal.SizeOf(typeof(MSX));
            if (!GlobalMemoryStatusEx(ref m)) return new double[] { 0, 0, 0 };
            return new double[] { m.ullTotalPhys, m.ullTotalPhys - m.ullAvailPhys, m.dwMemoryLoad };
        }
    }
}
"@
}

$xaml = @'
<Window xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        Width="330" SizeToContent="Height"
        WindowStyle="None" AllowsTransparency="True" Background="Transparent"
        ResizeMode="NoResize" Topmost="True" ShowInTaskbar="True"
        Title="GPU Monitor" FontFamily="Segoe UI">
  <Border Background="#E8121218" CornerRadius="10" BorderBrush="#38FFFFFF" BorderThickness="1" Padding="12,8,12,10">
    <StackPanel>
      <DockPanel Margin="0,0,0,6">
        <TextBlock x:Name="BtnClose" DockPanel.Dock="Right" Text="&#x2715;" FontSize="11"
                   Foreground="#70FFFFFF" Cursor="Hand" VerticalAlignment="Top" Margin="4,0,0,0"/>
        <TextBlock x:Name="TxtTitle" Text="GPU Monitor" FontSize="11" Foreground="#C8FFFFFF"
                   VerticalAlignment="Center" Margin="0,0,0,0"/>
      </DockPanel>

      <Grid Margin="0,1">
        <Grid.ColumnDefinitions>
          <ColumnDefinition Width="42"/><ColumnDefinition Width="166"/><ColumnDefinition Width="*"/>
        </Grid.ColumnDefinitions>
        <TextBlock Text="VRAM" FontSize="11" Foreground="#A0FFFFFF" VerticalAlignment="Center"/>
        <Border Grid.Column="1" Height="8" Background="#24FFFFFF" CornerRadius="4" VerticalAlignment="Center">
          <Border x:Name="FillVram" Width="0" HorizontalAlignment="Left" Background="#3B82F6" CornerRadius="4"/>
        </Border>
        <TextBlock x:Name="TxtVram" Grid.Column="2" Text="--" FontSize="11" Foreground="#FFFFFFFF"
                   Margin="8,0,0,0" VerticalAlignment="Center"/>
      </Grid>

      <Grid Margin="0,4">
        <Grid.ColumnDefinitions>
          <ColumnDefinition Width="42"/><ColumnDefinition Width="166"/><ColumnDefinition Width="*"/>
        </Grid.ColumnDefinitions>
        <TextBlock Text="RAM" FontSize="11" Foreground="#A0FFFFFF" VerticalAlignment="Center"/>
        <Border Grid.Column="1" Height="8" Background="#24FFFFFF" CornerRadius="4" VerticalAlignment="Center">
          <Border x:Name="FillRam" Width="0" HorizontalAlignment="Left" Background="#22C55E" CornerRadius="4"/>
        </Border>
        <TextBlock x:Name="TxtRam" Grid.Column="2" Text="--" FontSize="11" Foreground="#FFFFFFFF"
                   Margin="8,0,0,0" VerticalAlignment="Center"/>
      </Grid>

      <TextBlock x:Name="TxtFooter" Text="..." FontSize="10" Foreground="#8AFFFFFF" Margin="0,6,0,0"/>
    </StackPanel>
  </Border>
</Window>
'@

$win = [Windows.Markup.XamlReader]::Load((New-Object System.Xml.XmlNodeReader ([xml]$xaml)))

$fillVram = $win.FindName('FillVram')
$fillRam  = $win.FindName('FillRam')
$txtVram  = $win.FindName('TxtVram')
$txtRam   = $win.FindName('TxtRam')
$txtFoot  = $win.FindName('TxtFooter')
$btnClose = $win.FindName('BtnClose')

# --- GPU via NVML (init once) ---
$script:dev  = [IntPtr]::Zero
$script:nvOk = $false
try {
    if ([GpuMon.Nv]::Init() -eq 0) { $script:nvOk = ([GpuMon.Nv]::Handle(0, [ref]$script:dev) -eq 0) }
} catch { $script:nvOk = $false }

$BAR = 166.0
$GB  = 1GB

$update = {
    try {
        $gpuTxt = 'n/a'
        $fillVram.Background = [Windows.Media.BrushConverter]::new().ConvertFromString('#3B82F6')
        if ($script:nvOk) {
            $m = New-Object 'GpuMon.Nv+Mem'
            if ([GpuMon.Nv]::MemInfo($script:dev, [ref]$m) -eq 0 -and $m.total -gt 0) {
                $pct = [double]$m.used / [double]$m.total * 100
                $fillVram.Width = [math]::Round([math]::Min($BAR, $BAR * $pct / 100), 1)
                $txtVram.Text = '{0:N1}/{1:N1} GB  {2:N0}%' -f ($m.used / $GB), ($m.total / $GB), $pct
                if ($pct -ge 90) { $fillVram.Background = [Windows.Media.BrushConverter]::new().ConvertFromString('#EF4444') }
            }
            $u = New-Object 'GpuMon.Nv+Util'
            $util = '?'
            if ([GpuMon.Nv]::UtilRates($script:dev, [ref]$u) -eq 0) { $util = $u.gpu }
            $t = [uint32]0
            $temp = '?'
            if ([GpuMon.Nv]::Temp($script:dev, 0, [ref]$t) -eq 0) { $temp = $t }
            $txtFoot.Text = 'GPU {0}%  {1} C   poll 1.5s' -f $util, $temp
        } else {
            $txtVram.Text = 'NVML unavailable'
            $txtFoot.Text = 'GPU n/a'
        }

        $r = [GpuMon.Sys]::Ram()
        if ($r[0] -gt 0) {
            $pct = [double]$r[1] / $r[0] * 100
            $fillRam.Width = [math]::Round([math]::Min($BAR, $BAR * $pct / 100), 1)
            $txtRam.Text = '{0:N1}/{1:N1} GB  {2:N0}%' -f ($r[1] / $GB), ($r[0] / $GB), $pct
            if ($pct -ge 90) { $fillRam.Background = [Windows.Media.BrushConverter]::new().ConvertFromString('#EF4444') }
        }
    } catch { }
}

# --- window behaviour ---
$win.Add_Loaded({
    $wa = [System.Windows.SystemParameters]::WorkArea
    $win.Left = $wa.Right - $win.ActualWidth - 22
    $win.Top  = $wa.Bottom - $win.ActualHeight - 22
})
$win.Add_MouseLeftButtonDown({ if ($_.ButtonState -eq 'Pressed') { $win.DragMove() } })
$win.Add_PreviewKeyDown({ if ($_.Key -eq 'Escape') { $win.Close() } })
$btnClose.Add_MouseLeftButtonDown({ $_.Handled = $true; $win.Close() })

& $update

if ($Snapshot) {
    $win.Show()
    $win.Dispatcher.Invoke([Windows.Threading.DispatcherPriority]::ApplicationIdle, [Action]{}) | Out-Null
    & $update
    $win.UpdateLayout()
    $bmp = New-Object Windows.Media.Imaging.RenderTargetBitmap ([int]$win.ActualWidth), ([int]$win.ActualHeight), 96, 96, ([Windows.Media.PixelFormats]::Pbgra32)
    $bmp.Render($win)
    $enc = New-Object Windows.Media.Imaging.PngBitmapEncoder
    $enc.Frames.Add([Windows.Media.Imaging.BitmapFrame]::Create($bmp))
    $path = Join-Path $env:TEMP 'gpu-monitor.png'
    $fs = [System.IO.File]::Create($path); $enc.Save($fs); $fs.Close()
    $win.Close()
    "snapshot: $path"
    return
}

$timer = New-Object System.Windows.Threading.DispatcherTimer
$timer.Interval = [TimeSpan]::FromMilliseconds(1500)
$timer.Add_Tick({ & $update })
$timer.Start()

[void]$win.ShowDialog()
