import os
import subprocess


def install_dependencies():
    print("[Prepare] Bat dau cai dat thu vien cho chay local...")
    packages = ["sentence-transformers", "faiss-cpu", "datasets"]
    cmd = ["python", "-m", "pip", "install", *packages]
    subprocess.check_call(cmd)


def ensure_save_dir(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    print(f"[Prepare] Thu muc data san sang: {save_dir}")


if __name__ == "__main__":
    save_dir = "./data"
    install_dependencies()
    ensure_save_dir(save_dir)
