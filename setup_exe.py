import PyInstaller.__main__
import os
import shutil

# Clean up previous builds
if os.path.exists('build'):
    shutil.rmtree('build')
if os.path.exists('dist'):
    shutil.rmtree('dist')

# Define the command for PyInstaller
# --onefile: Create a single executable file
# --clean: Clean PyInstaller cache
# --name: Name of the executable
# --add-data: Include data files (format: source;destination for Windows, source:destination for Mac/Linux)
# Note: Since this script might run on Mac or Windows, we need to handle the separator.

separator = ';' if os.name == 'nt' else ':'

datas = [
    (f'app.py{separator}.'),
    (f'utils.py{separator}.'),
    (f'model_loader.py{separator}.'),
    (f'best_stgnn_hybrid.pth{separator}.'),
    (f'gb_model.pkl{separator}.'),
    (f'scaler.pkl{separator}.'),
    (f'sample_data.csv{separator}.'),
    (f'.streamlit{separator}.streamlit'), # Include config if exists
]

# Hidden imports that PyInstaller might miss
hidden_imports = [
    'streamlit',
    'pandas',
    'numpy',
    'scikit-learn',
    'torch',
    'openai',
    'reportlab',
    'altair',
    'pyarrow',
    'watchdog',
    'smmap',
    'git',
    'blinker',
    'cachetools',
    'click',
    'jinja2',
    'markdown',
    'markupsafe',
    'packaging',
    'pillow',
    'protobuf',
    'pydeck',
    'pympler',
    'requests',
    'rich',
    'tenacity',
    'toml',
    'tornado',
    'typing_extensions',
    'tzlocal',
    'validators',
]

args = [
    'run_app.py', # The entry point script
    '--name=CognitiveAlertSystem',
    '--onefile',
    '--clean',
    '--windowed', # Hide console window (remove if you want to see errors)
]

# Add data files
for source, dest in datas:
    if os.path.exists(source.split(separator)[0]):
        args.append(f'--add-data={source}')

# Add hidden imports
for module in hidden_imports:
    args.append(f'--hidden-import={module}')

# Additional hooks path if needed (standard streamlit hook usually suffices)
# args.append('--additional-hooks-dir=hooks')

print("Starting build process...")
PyInstaller.__main__.run(args)
print("Build complete. Executable is in the 'dist' folder.")
