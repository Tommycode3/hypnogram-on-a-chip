"""Epoch every downloaded night into data/cache/*.npz (Fpz-Cz, 30 s, uV)."""
import numpy as np

from tinysleep.data import ALL, STAGES, build_cache, load_nights

if __name__ == "__main__":
    build_cache(ALL)
    nights = load_nights(ALL)
    y = np.concatenate([n.y for n in nights])
    print(f"{len(nights)} nights, {len(y)} epochs")
    for i, s in enumerate(STAGES):
        print(f"  {s:4s} {np.sum(y == i):6d} ({np.mean(y == i):.1%})")
