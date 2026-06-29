"""Minimal test of _cutlpk import."""
import os
import sys

# Must add DLL directories BEFORE any import that triggers DLL loading
for dll_dir in [
    r"C:\gurobi1203\win64\bin",
    r"C:\Users\WYKge\Desktop\clustering_paper_2026\experiment\cuPDLPx\build-msvc",
    r"C:\Users\WYKge\Desktop\clustering_paper_2026\experiment\cuPDLPx\build-msvc\Release",
    r"C:\Users\WYKge\Desktop\clustering_paper_2026\experiment\cuPDLPx\build-msvc\_deps\pslp-build\Release",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
]:
    if os.path.isdir(dll_dir):
        os.add_dll_directory(dll_dir)
        print(f"Added DLL dir: {dll_dir}")

sys.path.insert(0, r"c:\Users\WYKge\Desktop\clustering_paper_2026\experiment\cutLPK\python_api")

try:
    from cutlpk import OrdinaryKMeans
    print("import OK!")
except Exception as e:
    print(f"FAIL: {e}")
