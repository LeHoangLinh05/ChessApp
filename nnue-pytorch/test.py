# test_dll_load.py (đặt trong D:\AI\nnue-pytorch\)
import ctypes
import os

dll_path_absolute = r"D:\AI\nnue-pytorch\training_data_loader.dll"

print(f"Attempting to load: {dll_path_absolute}")
if not os.path.exists(dll_path_absolute):
    print(f"ERROR: DLL file does NOT exist at the specified path!")
else:
    print(f"SUCCESS: DLL file found at the specified path.")
    try:
        my_lib = ctypes.CDLL(dll_path_absolute)
        print(f"SUCCESS: Successfully loaded '{os.path.basename(dll_path_absolute)}'.")
    except OSError as e:
        print(f"ERROR: Failed to load '{os.path.basename(dll_path_absolute)}'.")
        print(f"OSError details: {e}")
        print("This usually means a dependent DLL is missing or there's an architecture mismatch.")
        print("Please check with a tool like 'Dependencies' (https://github.com/lucasg/Dependencies)")
        print("and ensure Microsoft Visual C++ Redistributable for Visual Studio 2015-2022 is installed.")