# TODO (manual steps)

These steps are optional but recommended for a smooth VS Code experience.

1) VS Code: install extensions

- "Python" (ms-python.python)
- "Ruff" (charliermarsh.ruff)
- "CMake Tools" (ms-vscode.cmake-tools)

2) VS Code: open the workspace at the repo root

- File -> Open Folder... -> select this repo

3) VS Code: select the repo venv interpreter

- Command Palette -> "Python: Select Interpreter"
- Pick: `.venv/bin/python`

4) (Optional) Use CMake preset

- Command Palette -> "CMake: Select Configure Preset"
- Select: `dev`
- Then: "CMake: Build"
