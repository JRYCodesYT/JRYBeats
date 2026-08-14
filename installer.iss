#ifndef MyAppVersion
  #define MyAppVersion "1.0.0"
#endif

[Setup]
AppName=JRYBeats
AppVersion={#MyAppVersion}
DefaultDirName={autopf}\JRYBeats
DefaultGroupName=JRYBeats

OutputDir=installer-output
OutputBaseFilename=JRYBeats_{#MyAppVersion}_Windows_x64-setup

Compression=lzma
SolidCompression=yes

ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

UninstallDisplayName=JRYBeats
WizardStyle=modern

[Files]
Source: "dist\JRYBeats.exe"; DestDir: "{app}"; Flags: ignoreversion

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Icons]
Name: "{autoprograms}\JRYBeats"; Filename: "{app}\JRYBeats.exe"
Name: "{autodesktop}\JRYBeats"; Filename: "{app}\JRYBeats.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\JRYBeats.exe"; Description: "Launch JRYBeats"; Flags: nowait postinstall skipifsilent
