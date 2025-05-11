from ctypes import *

# Load NNUE probe and init weights
nnue_instance = None
try:
    nnue_instance = cdll.LoadLibrary('D:/AI/Chess/ChessApp/Chess/libnnueprobe.dll')
    nnue_instance.nnue_init(b'D:/AI/Chess/ChessApp/Chess/net_epoch3.nnue')
    print("NNUE loaded and initialized from nnue_interface.py")
except Exception as e:
    print(f"Error initializing NNUE in nnue_interface.py: {e}")
    nnue_instance = None

# Wrapper function để gọi evaluate từ instance đã load
def nnue_evaluate_fen_bytes(fen_bytes):
    if nnue_instance:
        return nnue_instance.nnue_evaluate_fen(fen_bytes)
    return 0 # Hoặc raise lỗi nếu NNUE không được load