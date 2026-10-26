import pandas as pd
import re
import openpyxl
import numpy as np

from .funciones_normalizacion import *


def _clave_codigo(v):
    if pd.isna(v): return None
    if isinstance(v,(int,np.integer)): return str(int(v))
    if isinstance(v,(float,np.floating)) and float(v).is_integer(): return str(int(v))
    s=str(v).strip()
    if re.fullmatch(r"-?\d+\.0",s): return str(int(float(s)))
    return s

def _normalizar_etiqueta_documentada(texto):
    s=re.sub(r"\s+"," ",str(texto)).strip(" .;:-")
    base=normalizar_para_match(s)
    if base in {"si","sí"}: return "Sí"
    if base=="no": return "No"
    if "desconocid" in base: return "Desconocido"
    return s[:1].upper()+s[1:] if s else s

def _lineas_utiles_comentario(texto):
    lineas=[]
    for linea in str(texto or "").replace("\r","\n").split("\n"):
        linea=re.sub(r"\s+"," ",linea).strip()
        if not linea or linea=="======" or linea.startswith("ID#"): continue
        if re.search(r"\(20\d{2}-\d{2}-\d{2}",linea): continue
        if re.fullmatch(r"[A-Za-zÁÉÍÓÚáéíóúÑñ ]+:",linea): continue
        lineas.append(linea)
    return lineas


def _extraer_codificacion(texto,encabezado=""):
    lineas=_lineas_utiles_comentario(texto); pares=[]; ultimo=None
    for linea in lineas:
        inicios=list(re.finditer(r"(?<![\w/.-])(GDS\s*)?(-?\d+(?:\s*-\s*\d+)?)\s*(?:[:.=]\s*|\s+)",linea,re.I))
        if inicios:
            for j,m in enumerate(inicios):
                codigo=re.sub(r"\s+","",m.group(2))
                fin=inicios[j+1].start() if j+1<len(inicios) else len(linea)
                etiqueta=linea[m.end():fin].strip(" .;:-")
                if etiqueta:
                    pares.append([codigo,_normalizar_etiqueta_documentada(etiqueta)])
                    ultimo=len(pares)-1
        elif ultimo is not None and not re.match(r"^[A-ZÁÉÍÓÚ ]{4,}$",linea):
            pares[ultimo][1]=re.sub(r"\s+"," ",pares[ultimo][1]+" "+linea).strip()
    for m in re.finditer(r"(?<!\d)(\d+)\s*=\s*([^):,;]+)",str(encabezado),re.I):
        pares.append([m.group(1),_normalizar_etiqueta_documentada(m.group(2))])
    salida={}
    for codigo,etiqueta in pares:
        if codigo not in salida: salida[codigo]=etiqueta
    return salida

def extraer_metadatos_excel(PATH_NEURO, PATH_MOCA, PATH_HXGEN):
    fuentes=[("neuropsicologia",PATH_NEURO),("moca",PATH_MOCA),("hxgen",PATH_HXGEN)]
    registros=[]
    for fuente,ruta in fuentes:
        wb=openpyxl.load_workbook(ruta,data_only=False,read_only=False)
        for ws in wb.worksheets:
            for celda in ws[1]:
                if celda.value is None: continue
                original=str(celda.value)
                comentario=celda.comment.text.strip() if celda.comment else ""
                registros.append({
                    "fuente":fuente,"hoja":ws.title,"celda_encabezado":celda.coordinate,
                    "original_excel":original,"normalizada":normalizar_columna(original),
                    "comentario_excel":comentario,
                    "etiquetas_codigos_dict":_extraer_codificacion(comentario,original)
                })
    return pd.DataFrame(registros)


def _mapa_codigos_columna(col, inventario):
    if col not in inventario["columna"].values: return {}
    d=inventario.set_index("columna").at[col,"etiquetas_codigos_dict"]
    return d if isinstance(d,dict) else {}


def _etiquetar_valor(v,mapa, CODIGO_NO_REGISTRADO, ETIQUETA_NO_REGISTRADO):
    if pd.isna(v): return np.nan
    clave=_clave_codigo(v)
    if clave==CODIGO_NO_REGISTRADO: return ETIQUETA_NO_REGISTRADO
    if clave in mapa: return mapa[clave]
    try:
        num=float(clave)
        for rango,etiqueta in mapa.items():
            if re.fullmatch(r"-?\d+-\d+",rango):
                a,b=map(float,rango.split("-"))
                if a<=num<=b: return etiqueta
    except (ValueError,TypeError): pass
    canon=normalizar_para_match(clave)
    if canon=="si": return "Sí"
    if canon=="no": return "No"
    return clave

def serie_etiquetada(col, base_full, CODIGO_NO_REGISTRADO, ETIQUETA_NO_REGISTRADO,
                     inventario, incluir_especiales=True):
    s = base_full[col].copy()
    tipo = inventario.set_index("columna").at[col, "tipo_analitico"]
    mapa = _mapa_codigos_columna(col, inventario)
    if tipo.startswith("cuantitativa"):
        especiales = {k for k, v in mapa.items() if v in {"Desconocido", "No registrado"}}
        return s.mask(s.map(_clave_codigo).isin(especiales | {CODIGO_NO_REGISTRADO}))
    out = s.map(lambda v: _etiquetar_valor(v, mapa, CODIGO_NO_REGISTRADO, ETIQUETA_NO_REGISTRADO))
    out.name = col
    if not incluir_especiales:
        out = out.mask(out.isin(["Desconocido", "No registrado"]))
    return out


def serie_analitica(col, inventario, CODIGO_NO_REGISTRADO, ETIQUETA_NO_REGISTRADO, base_full):
    tipo = inventario.set_index("columna").at[col, "tipo_analitico"]
    if tipo.startswith("cuantitativa") or tipo == "ordinal":
        s = base_full[col].copy()
        mapa = _mapa_codigos_columna(col, inventario)
        especiales = {k for k, v in mapa.items() if v in {"Desconocido", "No registrado"}}
        return s.mask(s.map(_clave_codigo).isin(especiales | {CODIGO_NO_REGISTRADO}))
    return serie_etiquetada(col, base_full, CODIGO_NO_REGISTRADO, ETIQUETA_NO_REGISTRADO,
                            inventario, incluir_especiales=False)

def codigos_observados(col, base_full):
    return sorted({_clave_codigo(v) for v in base_full[col].dropna()}, key=str)

def documentado(v, doc):
    if v in doc: return True
    try:
        num=float(v)
        return any(float(a)<=num<=float(b) for a,b in (k.split("-") for k in doc if re.fullmatch(r"-?\d+-\d+",k)))
    except (ValueError,TypeError): return False




