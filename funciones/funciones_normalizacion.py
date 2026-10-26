import re
import unicodedata
import pandas as pd


def normalizar_texto(s):
    if pd.isna(s):
        return s

    s = str(s).replace("\xa0", " ").strip()
    s = s.replace("ñ", "ni").replace("Ñ", "Ni")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(
        ch for ch in s
        if not unicodedata.combining(ch)
    )
    s = s.lower()
    s = re.sub(r"\s+", " ", s)

    return s


def normalizar_columna(s):
    s = normalizar_texto(s)
    s = s.replace("<br>", " ")
    s = re.sub(r"[¿?()/,:;.%+\\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_ ]", "", s)
    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s).strip("_")

    return s


def nombres_unicos(cols):
    vistos = {}
    salida = []

    for c in cols:
        base = normalizar_columna(c)

        if base not in vistos:
            vistos[base] = 1
            salida.append(base)
        else:
            vistos[base] += 1
            salida.append(f"{base}_{vistos[base]}")

    return salida


def mapa_columnas(df):
    nuevas = nombres_unicos(df.columns)

    mapa = pd.DataFrame({
        "Original": list(df.columns),
        "Normalizada": nuevas
    })

    return mapa, nuevas

def normalizar_para_match(s):
    s = str(s).replace("\xa0", " ").replace("<br>", " ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(
        ch for ch in s
        if not unicodedata.combining(ch)
    )
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s