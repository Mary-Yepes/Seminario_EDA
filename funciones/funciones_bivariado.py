import math,textwrap,re
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
from IPython.display import display,Markdown

from .funciones_codificaciones import *
from .funciones_univariado import *
from .funciones_codificaciones import *
from .funciones_codificaciones import _mapa_codigos_columna
from .funciones_codificaciones import _mapa_codigos_columna, _clave_codigo

BC="#315e73"; BC2="#789cab"; BPANEL="#eeefed"; BFONDO="#f6f5f2"
BES = "#d8cfad"   # beige: riesgo Potencial

def titulo_biv(x,w=42):
    return "\n".join(textwrap.wrap(str(x),width=w,break_long_words=False))

def clave_biv(v):
    if pd.isna(v): return None
    if isinstance(v,(int,np.integer)): return str(int(v))
    if isinstance(v,(float,np.floating)) and float(v).is_integer(): return str(int(v))
    s=str(v).strip()
    try:
        f=float(s)
        return str(int(f)) if f.is_integer() else s
    except: return s

def mapa_bivariado(col, inventario, base_full):
    """Mapa documentado + convención binaria confirmada 0=No, 1=Sí, 9=Desconocido."""
    fila = inventario.loc[inventario["columna"].eq(col)]
    if fila.empty: return {}

    mapa = _mapa_codigos_columna(col, inventario)
    if mapa: return {clave_biv(k): v for k, v in mapa.items()}

    tipo = fila["tipo_analitico"].iloc[0]
    vals = {clave_biv(v) for v in base_full[col].dropna()}
    vals.discard("-")

    if tipo == "categórica binaria" and vals.issubset({"0", "1", "9"}):
        return {"0": "No", "1": "Sí", "9": "Desconocido"}

    return {}


def serie_plot_bivariada(col, inventario, base_full):
    """Serie etiquetada para tablas/gráficos; no modifica base_full."""
    mapa=mapa_bivariado(col, inventario, base_full)

    def trad(v):
        if pd.isna(v):return np.nan
        k=clave_biv(v)
        if k=="-":return np.nan

        if k in mapa:
            e=mapa[k]
            return np.nan if e in {"Desconocido","No registrado"} else e

        canon=normalizar_para_match(k)
        if canon=="si":return "Sí"
        if canon=="no":return "No"

        return f"Código {k} (sin etiqueta documentada)"

    out=base_full[col].map(trad)
    out.name=col
    return out

def evaluar_codificacion(col, tipo, inventario, base_full):
    if tipo not in ("categórica nominal", "categórica binaria", "ordinal"): return True, ""
    if col == "neuro__tmta": return False, "TMTA requiere validar su codificación"

    mapa = mapa_bivariado(col, inventario, base_full)

    if mapa: return True, ""
    return False, "Codificación pendiente de validación"

def direccion_resultado(prueba,e):
    if pd.isna(e):return ""
    if prueba=="Spearman":
        if e>0:return "Valores mayores asociados con AOO mayor"
        if e<0:return "Valores mayores asociados con AOO menor"
        return "Sin dirección monotónica"
    return "Comparar medianas de AOO entre grupos"

def prueba_bivariada_aoo(col, inventario, base_full, CODIGO_NO_REGISTRADO, ETIQUETA_NO_REGISTRADO,
                         y_col="aoo", n_min=10, n_min_grupo=5):
    meta=inventario.set_index("columna")

    if col not in meta.index:
        return {"variable":col,"columna":col,"evaluable":False,"motivo":"Fuera del inventario"}

    tipo=meta.at[col,"tipo_analitico"]
    familia=meta.at[col,"familia"]
    leakage=meta.at[col,"riesgo_leakage_aoo"]
    nombre = etiqueta_bonita(col, inventario)
    y=pd.to_numeric(base_full[y_col],errors="coerce")
    cod_ok, alerta = evaluar_codificacion(col, tipo, inventario, base_full)

    base={
        "variable":nombre,"columna":col,"familia":familia,
        "tipo_analitico":tipo,"riesgo_leakage_aoo":leakage,
        "codificacion_validada":cod_ok,"alerta_codificacion":alerta
    }

    if col=="neuro__tmta":
        return {**base,"evaluable":False,"motivo":alerta}

    if tipo in ("texto libre / alta cardinalidad","identificador","variable auxiliar del EDA"):
        return {**base,"evaluable":False,"motivo":"Tipo no apto para prueba formal"}

    # CUANTITATIVAS
    if tipo.startswith("cuantitativa"):
        x = pd.to_numeric(serie_analitica(col, inventario, CODIGO_NO_REGISTRADO,
                                  ETIQUETA_NO_REGISTRADO, base_full), errors="coerce")
        d=pd.DataFrame({"x":x,"y":y}).dropna(); n=len(d)

        if d.x.nunique()<=1:
            return {**base,"evaluable":False,"motivo":"Sin variación observable","n":n}

        if n<n_min:
            return {**base,"evaluable":False,"motivo":f"n insuficiente ({n}<{n_min})","n":n}

        rho,p=stats.spearmanr(d.x,d.y)

        return {
            **base,"evaluable":True,"prueba":"Spearman","n":n,
            "estadistico":rho,"p_valor":p,
            "medida_efecto":rho,"medida_efecto_nombre":"rho de Spearman",
            "direccion":direccion_resultado("Spearman",rho),
            "medianas_por_grupo":np.nan,"nota":alerta
        }

    # ORDINALES: Spearman solo si orden/codificación están validados
    if tipo=="ordinal":
        raw = serie_analitica(col, inventario, CODIGO_NO_REGISTRADO,
                      ETIQUETA_NO_REGISTRADO, base_full)
        xn=pd.to_numeric(raw,errors="coerce")
        nobs=raw.notna().sum()

        if cod_ok and nobs and xn.notna().sum()/nobs>=.8:
            d=pd.DataFrame({"x":xn,"y":y}).dropna(); n=len(d)

            if d.x.nunique()<=1:
                return {**base,"evaluable":False,"motivo":"Sin variación observable","n":n}

            if n<n_min:
                return {**base,"evaluable":False,"motivo":f"n insuficiente ({n}<{n_min})","n":n}

            rho,p=stats.spearmanr(d.x,d.y)

            return {
                **base,"evaluable":True,"prueba":"Spearman","n":n,
                "estadistico":rho,"p_valor":p,
                "medida_efecto":rho,"medida_efecto_nombre":"rho de Spearman",
                "direccion":direccion_resultado("Spearman",rho),
                "medianas_por_grupo":np.nan,"nota":alerta
            }

    # CATEGÓRICAS
    x=serie_plot_bivariada(col, inventario, base_full)
    d=pd.DataFrame({"grupo":x,"y":y}).dropna()
    d["grupo"]=d["grupo"].astype(str)

    conteo=d.grupo.value_counts()
    validos=conteo[conteo>=n_min_grupo].index
    d=d[d.grupo.isin(validos)].copy()

    k=d.grupo.nunique(); n=len(d)

    if k<2:
        return {
            **base,"evaluable":False,
            "motivo":f"Menos de 2 categorías con n≥{n_min_grupo}","n":n
        }

    muestras=[g.y.to_numpy() for _,g in d.groupby("grupo")]
    medianas=d.groupby("grupo").y.median().round(1).to_dict()

    if k==2:
        a,b=muestras
        u,p=stats.mannwhitneyu(a,b,alternative="two-sided")
        rb=1-(2*u)/(len(a)*len(b))

        return {
            **base,"evaluable":True,"prueba":"U de Mann-Whitney","n":n,
            "estadistico":u,"p_valor":p,
            "medida_efecto":rb,"medida_efecto_nombre":"r biserial por rangos",
            "direccion":"Comparar medianas de AOO entre grupos",
            "medianas_por_grupo":medianas,"nota":alerta
        }

    h,p=stats.kruskal(*muestras)
    eps2=(h-k+1)/(n-k) if n>k else np.nan

    return {
        **base,"evaluable":True,"prueba":f"Kruskal-Wallis (k={k})","n":n,
        "estadistico":h,"p_valor":p,
        "medida_efecto":eps2,"medida_efecto_nombre":"épsilon cuadrado",
        "direccion":"Comparar medianas de AOO entre grupos",
        "medianas_por_grupo":medianas,"nota":alerta
    }
    
    
def panel_bivariado_numerico(ax,x,y,titulo):
    d=pd.DataFrame({
        "x":pd.to_numeric(x,errors="coerce"),
        "y":pd.to_numeric(y,errors="coerce")
    }).dropna()

    ax.set_facecolor(BPANEL)
    ax.set_title(titulo_biv(titulo),loc="left",fontsize=10,fontweight="bold",pad=10)

    if len(d)<3:
        ax.text(.5,.5,"Datos insuficientes",ha="center",va="center",transform=ax.transAxes)
        return

    ax.scatter(d.x,d.y,s=22,color=BC,alpha=.55,edgecolor="white",linewidth=.4)

    try:
        m,b=np.polyfit(d.x,d.y,1)
        xs=np.linspace(d.x.min(),d.x.max(),50)
        ax.plot(xs,m*xs+b,color="#8d806c",lw=2,ls="--")
    except: pass

    rho,p=stats.spearmanr(d.x,d.y)

    ax.text(
        .02,.96,f"ρ={rho:.2f}   p={p:.3g}   n={len(d)}",
        transform=ax.transAxes,va="top",fontsize=8,
        bbox=dict(boxstyle="round",fc="white",ec="none",alpha=.8)
    )

    ax.set_xlabel(titulo_biv(titulo,35),fontsize=8)
    ax.set_ylabel("AOO (años)",fontsize=8)
    ax.grid(alpha=.22)
    ax.spines[["top","right"]].set_visible(False)
    
    
    
def panel_bivariado_categorico(ax,cat,y,titulo,max_categorias=8,n_min_grupo=5):
    d=pd.DataFrame({
        "grupo":cat,
        "y":pd.to_numeric(y,errors="coerce")
    }).dropna()

    d["grupo"]=d.grupo.astype(str)
    counts=d.grupo.value_counts()
    d=d[d.grupo.isin(counts[counts>=n_min_grupo].index)]

    ax.set_facecolor(BPANEL)
    ax.set_title(titulo_biv(titulo),loc="left",fontsize=10,fontweight="bold",pad=10)

    if d.empty or d.grupo.nunique()<2:
        ax.text(.5,.5,"Datos/grupos insuficientes",ha="center",va="center",transform=ax.transAxes)
        return

    orden=d.groupby("grupo").y.median().sort_values(ascending=False).index
    cats=list(orden)[:max_categorias]
    datos=[d.loc[d.grupo.eq(c),"y"].to_numpy() for c in cats]
    etiquetas=[f"{titulo_biv(c,24)} (n={len(z)})" for c,z in zip(cats,datos)]

    bp=ax.boxplot(datos,vert=False,patch_artist=True,widths=.55)

    for j,b in enumerate(bp["boxes"]):
        b.set_facecolor(BC if j%2==0 else BC2)
        b.set_edgecolor(BC)

    for e in ("whiskers","caps"):
        for l in bp[e]:l.set_color(BC)

    for m in bp["medians"]:
        m.set_color("#8d806c")
        m.set_linewidth(2)

    ax.set_yticks(range(1,len(cats)+1),etiquetas,fontsize=8)
    ax.set_xlabel("AOO (años)",fontsize=8)
    ax.grid(axis="x",alpha=.22)
    ax.grid(axis="y",visible=False)
    ax.spines[["top","right","left"]].set_visible(False)
    
    
    
def analisis_bivariado_familia(columnas, inventario, base_full, titulo,
                               CODIGO_NO_REGISTRADO, ETIQUETA_NO_REGISTRADO, acumulador=None):
    if not columnas:
        display(Markdown("Sin variables integradas."))
        return pd.DataFrame()

    meta = inventario.set_index("columna")
    resultados = [prueba_bivariada_aoo(c, inventario, base_full,
                                       CODIGO_NO_REGISTRADO, ETIQUETA_NO_REGISTRADO)
                  for c in columnas]
    tabla = pd.DataFrame(resultados)

    mostrar = [
        "variable", "tipo_analitico", "riesgo_leakage_aoo", "evaluable",
        "prueba", "n", "estadistico", "medida_efecto", "medida_efecto_nombre",
        "direccion", "medianas_por_grupo", "codificacion_validada",
        "alerta_codificacion", "p_valor", "motivo", "nota",
    ]
    for c in mostrar:
        if c not in tabla:
            tabla[c] = np.nan

    # ── Tabla de pruebas (solo vista; `tabla` y el acumulador no se modifican) ──
    subtitulo(f"Pruebas de asociación con AOO — {titulo}")

    tabla_vista = tabla[mostrar].copy()
    tabla_vista["medianas_por_grupo"] = tabla_vista["medianas_por_grupo"].apply(
        lambda d: "\n".join(f"{k}: {v}" for k, v in d.items()) if isinstance(d, dict) else ""
    )
    for c in ["evaluable", "codificacion_validada", "alerta_codificacion"]:
        es_bool = tabla_vista[c].map(type).eq(bool)
        tabla_vista[c] = tabla_vista[c].where(~es_bool,
                                              tabla_vista[c].map({True: "Sí", False: "No"}))
    tabla_vista["n"] = pd.to_numeric(tabla_vista["n"], errors="coerce").astype("Int64")
    tabla_vista = tabla_vista.replace({"nan": np.nan, "None": np.nan, "": np.nan})

    mostrar_tabla(
        tabla_vista,
        alto=400,
        renombrar={
            "variable": "Variable",
            "tipo_analitico": "Tipo analítico",
            "riesgo_leakage_aoo": "Riesgo leakage AOO",
            "evaluable": "Evaluable",
            "prueba": "Prueba",
            "n": "n",
            "estadistico": "Estadístico",
            "medida_efecto": "Medida de efecto",
            "medida_efecto_nombre": "Tipo de efecto",
            "direccion": "Dirección",
            "medianas_por_grupo": "Medianas por grupo",
            "codificacion_validada": "Codificación validada",
            "alerta_codificacion": "Alerta codificación",
            "p_valor": "p valor",
            "motivo": "Motivo",
            "nota": "Nota",
        },
        formato={"Estadístico": "{:.2f}", "Medida de efecto": "{:.3f}", "p valor": "{:.4f}"},
        ajustar_texto=True,
        una_linea=["Tipo analítico", "Prueba", "n", "Evaluable",
                   "Estadístico", "Medida de efecto", "p valor"],
        anchos={"Variable": 340, "Riesgo leakage AOO": 170,
                "Medianas por grupo": 450, "Dirección": 220,
                "Motivo": 260, "Nota": 260},
        espacio=24,
    )

    if acumulador is not None:
        acumulador.extend(resultados)

    # ── Gráficos ──
    graficables = tabla.loc[tabla.evaluable.fillna(False), "columna"].tolist()

    for ini in range(0, len(graficables), 4):
        lote = graficables[ini:ini + 4]
        nr = math.ceil(len(lote) / 2)
        alto = 4.4 * nr + 1.0   # 1 pulgada extra reservada para el título general

        fig, axes = plt.subplots(nr, 2, figsize=(14, alto), squeeze=False)
        fig.patch.set_facecolor(BFONDO)
        fig.suptitle(f"{titulo} vs. AOO", fontsize=14, fontweight="bold",
                     color="#263b43", y=1 - 0.25 / alto)

        for ax, col in zip(axes.flat, lote):
            tipo = meta.at[col, "tipo_analitico"]
            nombre = etiqueta_bonita(col, inventario)
            y = base_full["aoo"]

            if tipo.startswith("cuantitativa") or tipo == "ordinal":
                resultado = tabla.loc[tabla.columna.eq(col)].iloc[0]
                if resultado["prueba"] == "Spearman":
                    x = serie_analitica(col, inventario, CODIGO_NO_REGISTRADO,
                                        ETIQUETA_NO_REGISTRADO, base_full)
                    # Pocos valores distintos (niveles, escalas): boxplot por nivel
                    if pd.to_numeric(x, errors="coerce").nunique() <= 10:
                        panel_bivariado_ordinal(
                            ax, x, y, nombre,
                            etiquetas=mapa_bivariado(col, inventario, base_full)
                        )
                    else:
                        panel_bivariado_numerico(ax, x, y, nombre)
                else:
                    panel_bivariado_categorico(
                        ax, serie_plot_bivariada(col, inventario, base_full), y, nombre)
            else:
                panel_bivariado_categorico(
                    ax, serie_plot_bivariada(col, inventario, base_full), y, nombre)

        for ax in axes.flat[len(lote):]:
            ax.axis("off")

        fig.subplots_adjust(left=.15, right=.95, top=1 - 1.25 / alto, bottom=.9 / alto,
                            hspace=.60, wspace=.48)
        plt.show()

    return tabla

           
           
           
def bootstrap_ic_efecto(col, prueba, inventario, base_full, CODIGO_NO_REGISTRADO,
                        ETIQUETA_NO_REGISTRADO, n_boot=1000, seed=42, n_min_grupo=5):
    rng=np.random.default_rng(seed)
    y=base_full["aoo"]; vals=[]

    if prueba=="Spearman":
        x = pd.to_numeric(serie_analitica(col, inventario, CODIGO_NO_REGISTRADO,
                                      ETIQUETA_NO_REGISTRADO, base_full), errors="coerce")
        d=pd.DataFrame({"x":x,"y":y}).dropna()

        if len(d)<10:return np.nan,np.nan

        for _ in range(n_boot):
            b=d.iloc[rng.integers(0,len(d),len(d))]
            r=stats.spearmanr(b.x,b.y).statistic
            if np.isfinite(r):vals.append(r)

    else:
        x=serie_plot_bivariada(col, inventario, base_full)
        d=pd.DataFrame({"g":x,"y":y}).dropna()
        d["g"]=d.g.astype(str)

        counts=d.g.value_counts()
        d=d[d.g.isin(counts[counts>=n_min_grupo].index)]
        grupos=list(d.g.unique())

        if len(grupos)<2:return np.nan,np.nan

        for _ in range(n_boot):
            m=[
                rng.choice(d.loc[d.g.eq(g),"y"],size=(d.g==g).sum(),replace=True)
                for g in grupos
            ]

            if len(grupos)==2:
                u=stats.mannwhitneyu(m[0],m[1],alternative="two-sided").statistic
                vals.append(1-2*u/(len(m[0])*len(m[1])))
            else:
                h=stats.kruskal(*m).statistic
                n=sum(map(len,m)); k=len(m)
                vals.append((h-k+1)/(n-k) if n>k else np.nan)

    vals=np.asarray(vals,float)
    vals=vals[np.isfinite(vals)]

    return tuple(np.quantile(vals,[.025,.975])) if len(vals)>=100 else (np.nan,np.nan)


def posthoc_dunn(col, inventario, base_full, ajuste="holm", n_min_grupo=5):
    x=serie_plot_bivariada(col, inventario, base_full); y=base_full["aoo"]
    d=pd.DataFrame({"grupo":x,"y":y}).dropna()
    d["grupo"]=d.grupo.astype(str)

    counts=d.grupo.value_counts()
    d=d[d.grupo.isin(counts[counts>=n_min_grupo].index)].copy()

    if d.grupo.nunique()<3:return pd.DataFrame()

    d["rango"]=stats.rankdata(d.y)
    N=len(d); grupos=sorted(d.grupo.unique())
    emp=d.y.value_counts().to_numpy()
    tie=1-(np.sum(emp**3-emp)/(N**3-N)) if N>1 else np.nan
    var=(N*(N+1)/12)*tie
    filas=[]

    for i,g1 in enumerate(grupos):
        for g2 in grupos[i+1:]:
            a=d[d.grupo.eq(g1)]
            b=d[d.grupo.eq(g2)]

            z=(a.rango.mean()-b.rango.mean())/np.sqrt(
                var*(1/len(a)+1/len(b))
            )
            p=2*stats.norm.sf(abs(z))

            filas.append({
                "grupo_1":g1,"grupo_2":g2,
                "n_1":len(a),"n_2":len(b),
                "z":z,"p_sin_ajuste":p
            })

    out=pd.DataFrame(filas)

    if len(out):
        out["p_ajustado_holm"]=multipletests(
            out["p_sin_ajuste"],method=ajuste
        )[1]
        out["significativo_005"]=out["p_ajustado_holm"]<.05

    return out



def redactar_hallazgos_bivariados(tabla, nombre, max_destacados=4, max_grupos=5):
    def cuantas(k, singular, plural):
        return f"{k} {singular if k == 1 else plural}"

    grupo = nombre.lower().removeprefix("variables ").strip()

    if tabla is None or tabla.empty:
        return f"En el grupo «{grupo}» no hubo variables disponibles."

    ev = tabla.loc[tabla.evaluable.fillna(False).astype(bool)].copy()
    noev = tabla.loc[~tabla.evaluable.fillna(False).astype(bool)]

    if ev.empty:
        return f"En el grupo «{grupo}» ninguna variable reunió condiciones mínimas para comparación formal con AOO."

    sen = ev.loc[ev.p_valor < .05].sort_values("p_valor")
    txt = f"Se analizaron {cuantas(len(ev), 'variable', 'variables')} del grupo «{grupo}». "

    if sen.empty:
        txt += "Ninguna mostró señal con p<0.05 antes de corregir por comparaciones múltiples."
    else:
        txt += (f"{cuantas(len(sen), 'variable mostró', 'variables mostraron')} una señal estadística "
                "antes de corregir por las múltiples comparaciones. ")
        for _, r in sen.head(max_destacados).iterrows():
            if r.prueba == "Spearman":
                fuerza = ("marcada" if abs(r.medida_efecto) >= .5 else
                          "moderada" if abs(r.medida_efecto) >= .3 else "débil")
                sentido = "mayor" if r.medida_efecto > 0 else "menor"
                frase = (f"{r.variable} presentó una tendencia {fuerza}: sus valores más altos "
                         f"coincidieron con una AOO {sentido}")
            else:
                med = r.medianas_por_grupo if isinstance(r.medianas_por_grupo, dict) else {}
                items = [f"{str(k).rstrip('.')}: {v} años" for k, v in list(med.items())[:max_grupos]]
                extra = len(med) - max_grupos
                if extra > 0:
                    items.append(f"y {cuantas(extra, 'grupo adicional', 'grupos adicionales')}")
                frase = f"{r.variable} mostró diferencias de AOO entre sus categorías"
                if items:
                    frase += " (" + "; ".join(items) + ")"
            if str(r.riesgo_leakage_aoo).lower() not in ("", "nan", "none", "bajo", "ninguno"):
                frase += "; su utilidad como covariable exige confirmar la temporalidad"
            if r.codificacion_validada is False:
                frase += "; la interpretación de las categorías queda pendiente de validar su codificación"
            txt += frase + ". "

        resto = len(sen) - max_destacados
        if resto > 0:
            txt += f"{cuantas(resto, 'señal preliminar adicional se detalla', 'señales preliminares adicionales se detallan')} en la tabla. "
        txt += "Estas señales deben contrastarse con el ajuste FDR del resumen consolidado. "

    if len(noev):
        nombres = ", ".join(noev.variable.astype(str).head(3))
        if len(noev) == 1:
            txt += f"No pudo evaluarse formalmente 1 variable: {nombres}."
        else:
            txt += f"No pudieron evaluarse formalmente {len(noev)} variables, entre ellas {nombres}."
    return txt

def mostrar_hallazgos_bivariados(tabla,nombre):
    display(Markdown(redactar_hallazgos_bivariados(tabla,nombre)))



def diagnostico_conversion_numerica(col, inventario, base_full, CODIGO_NO_REGISTRADO,
                                    ETIQUETA_NO_REGISTRADO, max_ejemplos=15):
    valores = base_full[col].dropna().map(_clave_codigo)
    n = len(valores)

    directo = pd.to_numeric(valores, errors="coerce")
    limpio = pd.to_numeric(
        valores.str.replace(r'["\'+\s]', "", regex=True)
               .str.replace(",", ".", regex=False)
               .replace({CODIGO_NO_REGISTRADO: np.nan, "": np.nan}),
        errors="coerce",
    )

    no_convertibles = (valores[directo.isna()]
                       .replace({CODIGO_NO_REGISTRADO: ETIQUETA_NO_REGISTRADO})
                       .value_counts().head(max_ejemplos).to_dict())

    return pd.DataFrame([{
        "variable": etiqueta_bonita(col, inventario),
        "n_no_faltante": n,
        "convertible_directo_pct": round(100 * directo.notna().sum() / n, 1) if n else np.nan,
        "convertible_limpio_pct": round(100 * limpio.notna().sum() / n, 1) if n else np.nan,
        "n_ceros": int((limpio == 0).sum()),
        "ejemplos_no_convertibles": no_convertibles,
    }])


def panel_bivariado_ordinal(ax, x, y, titulo, etiquetas=None, n_min_grupo=5, seed=0):
    """Boxplot de AOO por nivel, en el orden natural de la variable.

    Para variables numéricas con pocos valores (escalas, niveles, códigos ordenados).
    Los niveles con menos de `n_min_grupo` pacientes se muestran como puntos.
    """
    d = pd.DataFrame({
        "x": pd.to_numeric(x, errors="coerce"),
        "y": pd.to_numeric(y, errors="coerce"),
    }).dropna()

    ax.set_facecolor(BPANEL)
    ax.set_title(titulo_biv(titulo), loc="left", fontsize=10, fontweight="bold", pad=10)

    if len(d) < 3 or d.x.nunique() < 2:
        ax.text(.5, .5, "Datos insuficientes", ha="center", va="center", transform=ax.transAxes)
        return

    niveles = sorted(d.x.unique())
    pos = np.arange(1, len(niveles) + 1)
    rng = np.random.default_rng(seed)
    grandes = []

    for p, nivel in zip(pos, niveles):
        z = d.loc[d.x.eq(nivel), "y"].to_numpy()
        if len(z) >= n_min_grupo:
            grandes.append((p, z))
        else:
            ax.scatter(p + rng.uniform(-.08, .08, len(z)), z, s=26, color=BC2,
                       edgecolor=BC, linewidth=.6, zorder=3)

    if grandes:
        bp = ax.boxplot([z for _, z in grandes], positions=[p for p, _ in grandes],
                        widths=.55, patch_artist=True, manage_ticks=False)
        for j, b in enumerate(bp["boxes"]):
            b.set_facecolor(BC if j % 2 == 0 else BC2)
            b.set_edgecolor(BC)
        for e in ("whiskers", "caps"):
            for l in bp[e]:
                l.set_color(BC)
        for m in bp["medians"]:
            m.set_color("#8d806c"); m.set_linewidth(2)
        for f in bp["fliers"]:
            f.set_markeredgecolor(BC)

    # Tendencia de las medianas (solo niveles con n >= n_min_grupo)
    if len(grandes) >= 2:
        ax.plot([p for p, _ in grandes], [np.median(z) for _, z in grandes],
                color="#8d806c", lw=1.6, ls="--", marker="o", ms=4, zorder=4)

    def nombre_nivel(v):
        k = str(int(v)) if float(v).is_integer() else str(v)
        return titulo_biv(etiquetas.get(k, k), 11) if etiquetas else k

    conteos = d.x.value_counts()
    ax.set_xticks(pos, [f"{nombre_nivel(v)}\n(n={conteos[v]})" for v in niveles], fontsize=7)

    rho, p = stats.spearmanr(d.x, d.y)
    ax.text(.02, .96, f"ρ={rho:.2f}   p={p:.3g}   n={len(d)}", transform=ax.transAxes,
            va="top", fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="none", alpha=.8))

    ax.set_xlabel(titulo_biv(titulo, 35), fontsize=8)
    ax.set_ylabel("AOO (años)", fontsize=8)
    ax.grid(axis="y", alpha=.22); ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)



def grupo_interpretacion_fdr(riesgo):
    if riesgo == 'Bajo': return '1. Significativas tras FDR — riesgo Bajo'
    if riesgo in ('Revisar temporalidad','Bajo / revisar temporalidad','Revisar'):
        return '2. Significativas tras FDR — requiere revisar temporalidad'
    return '3. Significativas tras FDR — riesgo Potencial/Alto'


def mostrar_resultados_biv(tabla, renombrar=None, alto=420):
    """Vista legible de la tabla de pruebas (no modifica los resultados)."""
    vista = tabla.copy()
    for c in vista.columns:
        no_nulos = vista[c].dropna()
        if len(no_nulos) and no_nulos.map(type).eq(bool).all():      # True/False -> Sí/No
            vista[c] = vista[c].map({True: "Sí", False: "No"})
    if "n" in vista.columns:                                         # 334.000000 -> 334
        vista["n"] = pd.to_numeric(vista["n"], errors="coerce").astype("Int64")
    vista = vista.replace({"nan": np.nan, "None": np.nan})           # "nan" de texto -> —
    mostrar_tabla(vista, alto=alto, renombrar=renombrar, decimales=3,
                  ajustar_texto=True, espacio=24)