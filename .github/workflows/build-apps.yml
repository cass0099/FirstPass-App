name: Build Apps

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build-windows:
    runs-on: windows-latest

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install "PySide6[webengine]"
        pip install pyinstaller

    - name: List Files
      run: |
        echo "Current directory:"
        dir
        echo "Main.py location:"
        dir main.py

    - name: Build with PyInstaller
      run: |
        pyinstaller --onefile --name FirstPass main.py

    - name: Upload Windows artifact
      uses: actions/upload-artifact@v4
      with:
        name: FirstPass-Windows
        path: dist/FirstPass.exe

  build-macos:
    runs-on: macos-latest

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install "PySide6[webengine]"
        pip install pyinstaller

    - name: List Files
      run: |
        echo "Current directory:"
        ls -la
        echo "Main.py location:"
        ls -la main.py

    - name: Build with PyInstaller
      run: |
        pyinstaller --onefile --name FirstPass main.py

    - name: Create DMG
      run: |
        cd dist
        hdiutil create -volname "FirstPass" -srcfolder FirstPass -ov -format UDZO FirstPass.dmg

    - name: Upload Mac artifacts
      uses: actions/upload-artifact@v4
      with:
        name: FirstPass-macOS
        path: |
          dist/FirstPass
          dist/FirstPass.dmg