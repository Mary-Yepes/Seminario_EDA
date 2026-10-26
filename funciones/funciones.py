#%pip install pandas numpy matplotlib scipy statsmodels scikit-learn openpyxl
from .funciones_normalizacion import normalizar_texto
from IPython.display import display, Markdown

from pathlib import Path
import re
import unicodedata
import warnings
from IPython.display import display, HTML

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import openpyxl

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 300)
pd.set_option("display.max_colwidth", 1000)
pd.set_option("display.width", 4000)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

#Para los gráficos
plt.rcParams.update({
    "figure.figsize": (7, 4.5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})


def mostrar_columnas(df, altura=400):
    tabla = pd.DataFrame({
        "N.º": range(1, len(df.columns) + 1),
        "Nombre de la columna": df.columns
    })
    display(HTML(
        f'<div style="max-height: {altura}px; overflow: auto; margin: 0.4rem 0 1.5rem 0;">'
        f'{tabla.to_html(index=False, border=0)}</div>'
    ))



def mostrar_df(df, anchas=None):
    """anchas: {columna: px} para columnas de texto largo."""
    estilos = [
        {"selector": "",
         "props": [("table-layout", "fixed"),
                   ("width", "max-content"),
                   ("border-collapse", "collapse")]},
        {"selector": "th",
         "props": [("white-space", "normal"),
                   ("text-align", "center !important"),
                   ("min-width", "300px"),
                   ("max-width", "300px"),
                   ("overflow-wrap", "break-word"),
                   ("line-height", "1.2"),
                   ("padding", "5px 8px"),
                   ("vertical-align", "middle")]},
        {"selector": "td",
         "props": [("white-space", "nowrap"),
                   ("overflow", "hidden"),
                   ("text-overflow", "ellipsis"),
                   ("min-width", "150px"),
                   ("max-width", "300px"),
                   ("vertical-align", "top"),
                   ("padding", "4px 8px")]},
    ]

    estilo = (df.style
                .hide(axis="index")
                .set_table_attributes('class="dataframe"')
                .set_table_styles(estilos))

    for col, px in (anchas or {}).items():
        if col in df.columns:
            estilo = estilo.set_properties(subset=[col], **{
                "min-width": f"{px}px", "max-width": f"{px}px",
                "white-space": "normal", "vertical-align": "top"})

    display(HTML(f'<div style="overflow-x: auto; margin: 0.4rem 0 1.5rem 0;">{estilo.to_html()}</div>'))


def mostrar_valores_unicos(df, altura=400):
    tabla = df.nunique(dropna=False).rename("n_unicos").reset_index()
    tabla.columns = ["Nombre de la columna", "Valores únicos"]
    display(HTML(
        f'<div style="max-height: {altura}px; overflow: auto; margin: 0.4rem 0 1.5rem 0;">'
        f'{tabla.to_html(index=False, border=0)}</div>'
    ))



def subtitulo(texto, espacio_antes=False):
    arriba = "&nbsp;\n\n" if espacio_antes else ""
    display(Markdown(f"{arriba}**{texto}**"))


def inspeccion_inicial(df, nombre):
    display(Markdown(f"#### {nombre}\n\n{df.shape[0]} filas × {df.shape[1]} columnas"))

    subtitulo("Primeras filas")
    mostrar_df(df.head())

    subtitulo("Últimas filas")
    mostrar_df(df.tail())

    subtitulo("Columnas")
    mostrar_columnas(df)

    subtitulo("Tipos de datos y valores no nulos")
    df.info()

    subtitulo("Valores únicos por columna", espacio_antes=True)
    mostrar_valores_unicos(df)


def extraer_numero(x):
    if pd.isna(x):
        return np.nan

    s = normalizar_texto(x)

    if s in {"", "-", "0", "0.0", "na", "nan", "n/a", "no aplica"}:
        return np.nan

    nums = re.findall(r"\d+(?:[\.,]\d+)?", s)

    if not nums:
        return np.nan

    return float(nums[0].replace(",", "."))


#clasificamos la situación de cada paciente
def clasificar_consistencia_aoo(fila):

    #Si no hay AOO, no se puede evaluar
    if pd.isna(fila["aoo"]):
        return "AOO no disponible"

    tiene_neuro = pd.notna(fila["edad_eval_neuro"])
    tiene_hx = pd.notna(fila["edad_eval_hx"])

    #Si no tenemos ninguna edad de referencia
    if not tiene_neuro and not tiene_hx:
        return "Sin edad de evaluación"

    inc_neuro = fila["inconsistente_con_neuro"]
    inc_hx = fila["inconsistente_con_hx"]

    if inc_neuro and inc_hx:
        return "Inconsistente en ambas fuentes"

    if inc_neuro and not inc_hx:
        return "Inconsistente solo en neuro"

    if inc_hx and not inc_neuro:
        return "Inconsistente solo en hxgen"

    return "Sin inconsistencia"


def tabla_extremos(df, titulo, color):
    """Tabla compacta con título y encabezado de color, compatible con modo claro y oscuro."""
    return (df.style
            .hide(axis="index")
            .set_table_attributes('class="dataframe"')
            .format({"AOO (años)": "{:.0f}"})
            .set_caption(titulo)
            .set_table_styles([
                {"selector": "caption",
                 "props": "caption-side: top; font-weight: 700; white-space: nowrap; "
                          "text-align: center; padding-bottom: 6px;"},
                {"selector": "th",
                 "props": f"background-color: {color} !important; color: white !important; "
                          "text-align: center !important;"},
                {"selector": "td",
                 "props": "text-align: center; padding: 4px 18px;"},
            ])
            .to_html())



def mostrar_tabla(df, alto=400, renombrar=None, formato=None, alinear="left",
                  decimales=2, ajustar_texto=False, una_linea=None,
                  centrar=False, espacio=0, multilinea=None, anchos=None):
    """Muestra un DataFrame sin índice, con scroll y encabezado fijo."""
    if renombrar:
        df = df.rename(columns=renombrar)
    salto = "normal" if ajustar_texto else "nowrap"
    margen = "margin: 0 auto;" if centrar else "margin-left: 0;"
    estilo = (
        df.style
        .hide(axis="index")
        .set_table_attributes('class="dataframe"')
        .format(formato or {}, precision=decimales, na_rep="—")
        .set_table_styles([
            {"selector": "",
             "props": f"width: auto; {margen}"},
            {"selector": "th",
             "props": "position: sticky; top: 0; z-index: 1; "
                      "background-color: var(--vscode-editor-background, var(--jp-layout-color0, var(--pst-color-background, white))); "
                      "color: var(--vscode-editor-foreground, var(--jp-ui-font-color0, var(--pst-color-text-base, black))); "
                      "border-bottom: 1px solid var(--vscode-editorWidget-border, rgba(128,128,128,.4));"},
            {"selector": "td, th",
             "props": f"text-align: {alinear}; padding: 4px 14px; white-space: {salto}; "
                      "vertical-align: top;"},
            {"selector": "tbody tr:nth-child(even)",
             "props": "background-color: rgba(128,128,128,.12);"},
            {"selector": "th",                                  # al final para que gane
             "props": "text-align: center !important;"},
        ])
    )
    if una_linea:
        estilo = estilo.set_properties(subset=una_linea, **{"white-space": "nowrap !important"})
    if multilinea:
        idx = [df.columns.get_loc(c) + 1 for c in multilinea if c in df.columns]
        extra = [{"selector": f"td.col{i-1}",
                  "props": "white-space: pre-line !important; min-width: 180px;"}
                 for i in idx]
        estilo = estilo.set_table_styles(extra, overwrite=False)
    for col, px in (anchos or {}).items():
        if col in df.columns:
            estilo = estilo.set_properties(subset=[col], **{
                "min-width": f"{px}px", "white-space": "normal !important"})
    display(HTML(
        f'<div style="max-height: {alto}px; overflow: auto; margin: 0.6rem 0 {espacio}px 0;">'
        f'{estilo.to_html()}</div>'
    ))


nombres_faltantes = {
    "variable": "Variable",
    "n_faltantes": "N.º faltantes",
    "pct_faltantes": "% faltantes",
    "decision_preliminar": "Decisión preliminar"
}
