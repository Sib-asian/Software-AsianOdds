"""
Utility script per calcolare e visualizzare una mappa FTLE del flusso Double Gyre
utilizzando la libreria `numbacs`.

Esecuzione:
    python3 analysis/numbacs_double_gyre.py

Prerequisiti:
    pip install numbacs matplotlib numpy

Il codice è idempotente: produce un'immagine PNG nella cartella `analysis/output`.
"""

from __future__ import annotations

import pathlib
from math import copysign

import matplotlib.pyplot as plt
import numpy as np

try:
    from numbacs.diagnostics import ftle_grid_2D
    from numbacs.flows import get_predefined_flow
    from numbacs.integration import flowmap_grid_2D
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "numbacs non risulta installato. Installa prima il pacchetto:\n"
        "    pip install numbacs"
    ) from exc


def compute_double_gyre_ftle(
    t0: float = 0.0,
    T: float = -10.0,
    nx: int = 401,
    ny: int = 201,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcola la mappa FTLE per il flusso Double Gyre.

    Args:
        t0: tempo di partenza delle particelle.
        T: intervallo di integrazione; negativo = integrazione all'indietro.
        nx: numero di punti lungo l'asse x.
        ny: numero di punti lungo l'asse y.

    Returns:
        tuple (x, y, ftle) dove:
            x: coordinate x della griglia.
            y: coordinate y della griglia.
            ftle: matrice 2D con i valori FTLE.
    """
    int_direction = copysign(1.0, T)
    funcptr, params, domain = get_predefined_flow(
        "double_gyre", int_direction=int_direction
    )

    x = np.linspace(domain[0][0], domain[0][1], nx)
    y = np.linspace(domain[1][0], domain[1][1], ny)

    flowmap = flowmap_grid_2D(funcptr, t0, T, x, y, params)

    dx = abs(x[1] - x[0])
    dy = abs(y[1] - y[0])
    ftle = ftle_grid_2D(flowmap, T, dx, dy)

    return x, y, ftle


def plot_ftle(x: np.ndarray, y: np.ndarray, ftle: np.ndarray) -> pathlib.Path:
    """
    Crea un grafico contour della mappa FTLE e salva il risultato in PNG.
    """
    output_dir = pathlib.Path(__file__).with_suffix("").parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "double_gyre_ftle.png"

    fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
    contour = ax.contourf(x, y, ftle.T, levels=80, cmap="viridis")
    fig.colorbar(contour, ax=ax, label="FTLE")

    ax.set_title("FTLE – Double Gyre (T = -10)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return output_path


def main() -> None:
    x, y, ftle = compute_double_gyre_ftle()
    output_path = plot_ftle(x, y, ftle)
    print(f"FTLE Double Gyre salvata in: {output_path}")


if __name__ == "__main__":
    main()
