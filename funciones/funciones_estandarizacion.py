import pandas as pd
import re
from IPython.display import display, HTML

from .funciones import *

def limpiar_id(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)
    s = s.replace(" ", "")
    return s

def reporte_duplicados(df, id_col, nombre):
    dup = df[df[id_col].duplicated(keep=False)].sort_values(id_col)
    print(f"{nombre}: {dup[id_col].nunique()} IDs duplicados; {len(dup)} filas implicadas.")
    return dup


def resumen_missing(df, nombre, altura=450, espacio=40):
    out = (pd.DataFrame({
        "Variable": df.columns,
        "Faltantes": df.isna().sum().values,
        "% faltantes": df.isna().mean().values * 100,
        "Disponibles": df.notna().sum().values,
    })
        .sort_values("% faltantes", ascending=False)
        .reset_index(drop=True))

    subtitulo(nombre)
    mostrar_tabla(
        out,
        alto=altura,
        espacio=espacio,
        formato={"% faltantes": "{:.2f}%", "Faltantes": "{:.0f}", "Disponibles": "{:.0f}"},
    )
    return out

