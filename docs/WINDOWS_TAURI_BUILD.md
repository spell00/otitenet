# Windows Tauri Build

Otitenet uses Tauri v2 plus a Python/Streamlit sidecar. The production Windows
installer should be built on real Windows. In this repo, the recommended path is
the manual GitHub Actions workflow `Build Windows Desktop`.

## Recommended: Windows CI

From GitHub Actions, run:

```text
Build Windows Desktop
```

Inputs:

```text
variant=compact
```

Use `variant=exact` only when you need the full PyTorch runtime.

The workflow runs on `windows-latest`, builds the PyInstaller sidecar, copies it
to Tauri's expected Windows sidecar path, builds the NSIS installer, and uploads:

```text
otitenet-windows-sidecar-compact
otitenet-windows-installer-compact
```

## Native Windows Commands

On a Windows machine from the project root:

```powershell
$env:OTITENET_DESKTOP_VARIANT = "compact"
$env:OTITENET_PYINSTALLER_ONEFILE = "1"

python -m pip install --upgrade pip
python -m pip install -r requirements-desktop.txt
python -m pip install -r requirements-export.txt
python -m PyInstaller packaging/pyinstaller/otitenet_streamlit.spec --clean -y

New-Item -ItemType Directory -Force -Path "desktop/src-tauri/binaries" | Out-Null
Copy-Item "dist/otitenet-streamlit.exe" "desktop/src-tauri/binaries/otitenet-streamlit-x86_64-pc-windows-msvc.exe" -Force

npm ci
Push-Location desktop
npm ci
npm run tauri:build -- --bundles nsis
Pop-Location
```

The Windows NSIS installer is written under:

```text
desktop/src-tauri/target/release/bundle/nsis/
```

## Ubuntu Server Notes

The Ubuntu server can build the Ubuntu `.deb` installer natively. It should not
be the production Windows build path.

Wine can be useful for quick debugging, but on headless servers it can hang while
starting Wine services or fail before `python.exe --version`. If that happens,
stop using Wine and run the Windows CI workflow instead.

If you still need to debug Wine locally, install a current WineHQ build first.
Ubuntu's default Wine 6.x is too old for Python 3.11 Windows wheels and can fail
with missing UCRT functions such as:

```text
api-ms-win-crt-runtime-l1-1-0.dll.fetestexcept
```

After upgrading Wine, recreate the prefix instead of trying to repair an old
one. Old prefixes can fail to load core DLLs such as `win32u.dll`, `user32.dll`,
or `shell32.dll`.
