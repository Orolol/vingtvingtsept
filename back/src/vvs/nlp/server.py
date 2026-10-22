"""Lancement de llama-server (llama.cpp) et garde-fous GPU.

Le GPU est partagé (~/AGENTS.md) et le bureau tourne sur la même carte : on ne lance le
serveur que si `MIN_FREE_MIB` sont libres, on vérifie après chargement que la VRAM totale
utilisée reste sous `MAX_USED_MIB`, et on n'arrête que le processus qu'on a lancé. vLLM est
exclu sur ce chantier (gel de la machine pendant la capture des graphes CUDA, 2026-10-22).

Relance manuelle (depuis n'importe où) :

    bash -c '"$(cd back && uv run python -c "from vvs.nlp.server import shell_command; \
print(shell_command())")"'

ou directement la commande renvoyée par `shell_command()`.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from vvs import paths

LLAMA_SERVER = Path.home() / "workspace/llama.cpp/build/bin/llama-server"
GGUF = (Path.home() / ".cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots"
        / "f1bfb127c64f7072bdd2cad55f258b9c8b2910fe/Qwen3.8-27B-UD-Q5_K_XL.gguf")
PORT = 8765
MIN_FREE_MIB = 26 * 1024
MAX_USED_MIB = 27 * 1024
LOG = paths.RUNS / "logs" / "llama-server-press-nlp.log"


@dataclass(frozen=True)
class ServerConfig:
    model: Path = GGUF
    ctx: int = 8192
    parallel: int = 8
    batch: int = 512
    ubatch: int = 512
    port: int = PORT

    def argv(self) -> list[str]:
        return [str(LLAMA_SERVER), "-m", str(self.model), "-ngl", "999", "-c", str(self.ctx),
                "--parallel", str(self.parallel), "-b", str(self.batch), "-ub", str(self.ubatch),
                "--flash-attn", "on", "--port", str(self.port), "--host", "127.0.0.1"]


def shell_command(cfg: ServerConfig | None = None) -> str:
    cfg = cfg or ServerConfig()
    return f"nohup {shlex.join(cfg.argv())} > {shlex.quote(str(LOG))} 2>&1 &"


def gpu_memory_mib() -> tuple[int, int]:
    """(utilisée, libre) en Mio, d'après nvidia-smi."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
    used, free = (int(x) for x in out.split(","))
    return used, free


def check_before_launch() -> tuple[int, int]:
    used, free = gpu_memory_mib()
    if free < MIN_FREE_MIB:
        raise RuntimeError(f"VRAM libre {free} Mio < {MIN_FREE_MIB} : ne pas lancer, attendre")
    return used, free


def check_after_load() -> int:
    used, _ = gpu_memory_mib()
    if used > MAX_USED_MIB:
        raise RuntimeError(f"VRAM utilisée {used} Mio > {MAX_USED_MIB} : arrêter le serveur et "
                           "réduire -c ou --parallel")
    return used
