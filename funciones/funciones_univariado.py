from .funciones_normalizacion import *
from .funciones_codificaciones import *
from .funciones import *
from .funciones import mostrar_tabla
import matplotlib.pyplot as plt
from functools import partial
from IPython.display import display, Markdown
import textwrap

import math

C1="#315e73"; C2="#789cab"; C3="#b8cbd1"; CES="#d8cfad"
FONDO="#f6f5f2"; PANEL="#eeefed"; PALETA=[C1,C2,C3,"#52798a","#9eb8c2"]



def etiqueta_bonita(col, inventario):
    especiales = {'aoo':'Edad de inicio de Alzheimer (AOO)', 'neuro__mmse':'MMSE',
                  'moca__moca':'MoCA', 'neuro__fast':'FAST',
                  'anios_fumando_limpio':'Años de tabaquismo (99 = desconocido)'}
    if col in especiales: return especiales[col]
    if 'columna' in inventario.columns and col in set(inventario['columna']):
        original = inventario.loc[inventario['columna'].eq(col), 'original'].iloc[0]
        return re.sub(r'\s+', ' ', str(original)).strip()
    s = col.removeprefix('neuro__').removeprefix('hx__').removeprefix('moca__')
    return s.replace('_', ' ').strip().capitalize()


def columna_integrada(fila):
    prefijo = {'neuropsicologia':'neuro__', 'moca':'moca__', 'hxgen':'hx__'}.get(fila['fuente'], '')
    return prefijo + fila['normalizada']


# Se reusan los subgrupos existentes; se rescatan discrepancias de espacios
# usando el normalizador ya definido y se dejan visibles los casos inciertos.
def subgrupo_revisado(GRUPOS_NEURO, GRUPOS_DEMO, neuro_norm, fila):
    if fila['subgrupo']: return fila['subgrupo']
    clave = normalizar_para_match(fila['original'])
    # El encabezado real y GRUPOS_DEMO difieren en un espacio antes de
    # '(comparten)'. La pregunta corresponde inequívocamente a historia familiar.
    if fila['fuente'] == 'neuropsicologia' and 'miembros afectados de la familia' in clave and 'padres biologicos' in clave:
        return 'historia_familiar'
    grupos = GRUPOS_NEURO if fila['columna'] in neuro_norm else GRUPOS_DEMO
    for nombre, variables in grupos.items():
        if clave in {normalizar_para_match(v) for v in variables}: return nombre
    return 'sin_asignacion_validada'


def tipo_analitico(fila, base_full, binarias, ordinales):
    c = fila['columna']; nombre = fila['normalizada']; s = base_full[c]
    if nombre in ('codigo','id') or c == 'id': return 'identificador'
    if nombre in ('edad_de_inicio','edad_de_inicio_actualizada'): return 'variable auxiliar del EDA'
    if any(x in nombre for x in ('medicamento_del_sujeto','otro_idoma','mes_y_ano_de_nacimiento','mes_y_anio_de_nacimiento')): return 'texto libre / alta cardinalidad'
    if nombre in ('edad','peso','talla','peso_en_kilogramo','altura_en_centimetro', 'tiempo_anos_de_conocerce','tiempo_anios_de_conocerce') \
        or nombre.startswith('numero_total_de_anios_que_ha_fumado'): return 'cuantitativa continua'
    if any(x in nombre for x in ordinales): return 'ordinal'
    if any(x in nombre for x in binarias) and s.nunique(dropna=True) <= 3: return 'categórica binaria'
    if nombre in ('edad','peso','talla','peso_en_kilogramo','altura_en_centimetro','tiempo_anos_de_conocerce') or nombre.startswith('numero_total_de_anios_que_ha_fumado'): return 'cuantitativa continua'
    if fila['fuente'] == 'neuropsicologia' and fila['grupo_principal'] == 'neuropsicologica':
        if any(x in nombre for x in ('fecha','quejas_subjetivas')): return 'texto libre / alta cardinalidad'
        return 'cuantitativa discreta' if pd.to_numeric(s, errors='coerce').notna().sum() >= .8*s.notna().sum() else 'categórica nominal'
    if c == 'moca__moca': return 'cuantitativa discreta'
    if nombre in ('enfermedad_actual','que_sintomas_presenta','antecedentes'): return 'texto libre / alta cardinalidad'
    if s.nunique(dropna=True) <= 12: return 'categórica nominal'
    return 'texto libre / alta cardinalidad'


def cortar(txt,w=27,n=2):
    p=textwrap.wrap(str(txt),width=w,break_long_words=False)
    if len(p)>n: p=p[:n]; p[-1]+="…"
    return "\n".join(p)


def tabla_cat(s):
    s=s.copy(); n=len(s)
    esp=s.astype(str).isin(["Desconocido","No registrado"])
    nv=(s.notna()&~esp).sum(); filas=[]
    for cat,cant in s.dropna().astype(str).value_counts().items():
        especial=cat in ["Desconocido","No registrado"]
        pct=100*cant/(n if especial else max(nv,1))
        filas.append([cat,int(cant),pct,"total" if especial else "válido",especial])
    t=pd.DataFrame(filas,columns=["categoria","n","pct","base","especial"])
    if t.empty:return t
    return pd.concat([t[~t.especial].sort_values("n",ascending=False),t[t.especial]],ignore_index=True)


def colores_cat(t):
    out=[]; i=0
    for esp in t.especial:
        if esp: out.append(CES)
        else: out.append(PALETA[i%len(PALETA)]); i+=1
    return out


def modo_cat(t):
    if t.empty:return "vacio"
    if len(t)<=3 and t.categoria.astype(str).str.len().max()<=22:return "donut"
    return "bar"


def panel_donut(ax,titulo,t):
    ax.set_facecolor(PANEL)
    wedges,_,aut=ax.pie(
        t.n,colors=colores_cat(t),startangle=90,counterclock=False,radius=.82,
        wedgeprops=dict(width=.34,edgecolor=FONDO,linewidth=1.4),
        autopct=lambda p:f"{p:.1f}%" if p>=4 else "",pctdistance=.79
    )
    for x in aut:x.set_fontsize(8)
    if len(t)==1:
        r=t.iloc[0]
        ax.text(0,.06,cortar(r.categoria,20,2),ha="center",fontsize=13,fontweight="bold",color=C1)
        ax.text(0,-.11,f"{r.n} pacientes",ha="center",fontsize=9,color="#536168")
    else:
        leg=[f"{cortar(r.categoria,20,2)} — {r.n} ({r.pct:.1f}% {r.base})" for _,r in t.iterrows()]
        ax.legend(wedges,leg,loc="center left",bbox_to_anchor=(.92,.5),frameon=False,fontsize=8)
    ax.set_title(
    "\n".join(textwrap.wrap(titulo, width=42)),
    loc="left",
    fontsize=10,
    fontweight="bold",
    pad=10
)
    
    
    








def panel_bar(ax,titulo,t):
    ax.set_facecolor(PANEL); y=np.arange(len(t))
    ax.barh(y,t.pct,color=colores_cat(t),height=.66,edgecolor="white")
    ax.set_yticks(y,[cortar(x,27,2) for x in t.categoria]); ax.invert_yaxis()
    ax.set_xlim(0,max(105,t.pct.max()+15)); ax.set_xlabel("%",fontsize=8)
    for yy,(_,r) in zip(y,t.iterrows()):
        ax.text(r.pct+1,yy,f"{r.n} ({r.pct:.1f}% {r.base})",va="center",fontsize=8)
    ax.set_title(
    "\n".join(textwrap.wrap(titulo, width=42)),
    loc="left",
    fontsize=10,
    fontweight="bold",
    pad=10
)
    ax.grid(axis="x",alpha=.18); ax.grid(axis="y",visible=False)
    ax.spines[["top","right","left"]].set_visible(False); ax.tick_params(axis="y",length=0)

def panel_num(ax,titulo,s):
    x=pd.to_numeric(s,errors="coerce").dropna(); ax.set_facecolor(PANEL)
    if x.empty: ax.text(.5,.5,"Sin datos válidos",ha="center"); return
    bins=max(5,min(15,int(np.ceil(1+3.322*np.log10(len(x))))))
    ax.hist(x,bins=bins,color=C1,edgecolor="white")
    med=x.median(); ax.axvline(med,color="#8d806c",ls="--",lw=1.8,label=f"Mediana {med:.1f}")
    ax.legend(frameon=False,fontsize=8)
    ax.set_title(
    "\n".join(textwrap.wrap(titulo, width=42)),
    loc="left",
    fontsize=10,
    fontweight="bold",
    pad=10
)
    ax.set_xlabel("Valor observado",fontsize=8); ax.set_ylabel("Pacientes",fontsize=8)
    ax.grid(axis="y",alpha=.18); ax.spines[["top","right"]].set_visible(False)

def dibujar(ax,col, inventario, serie_univ):
    fila=inventario.loc[inventario.columna.eq(col)].iloc[0]
    tipo=fila.tipo_analitico; titulo=fila.etiqueta; s=serie_univ(col)
    if tipo.startswith("cuantitativa"): panel_num(ax,titulo,s); return
    t=tabla_cat(s)
    if modo_cat(t)=="donut": panel_donut(ax,titulo,t)
    else: panel_bar(ax,titulo,t)


def analisis_familia(columnas, inventario, base_full, titulo,
                     CODIGO_NO_REGISTRADO, ETIQUETA_NO_REGISTRADO,
                     ncols=2, por_pagina=4):
    serie_univ = partial(serie_univariada, inventario=inventario, base_full=base_full,
                         CODIGO_NO_REGISTRADO=CODIGO_NO_REGISTRADO,
                         ETIQUETA_NO_REGISTRADO=ETIQUETA_NO_REGISTRADO)
    etiqueta = partial(etiqueta_bonita, inventario=inventario)

    meta = inventario.set_index("columna")
    graficables, no_graf = [], []
    for c in columnas:
        if c not in base_full.columns:
            continue
        tipo = meta.loc[c, "tipo_analitico"]
        if tipo in ["texto libre / alta cardinalidad", "identificador", "variable auxiliar del EDA"]:
            no_graf.append(c)
        else:
            graficables.append(c)

    display(Markdown(
        f"Variables integradas en la familia: **{len(columnas)}** · "
        f"graficadas: **{len(graficables)}** · resumidas sin gráfico: **{len(no_graf)}**"
    ))

    for ini in range(0, len(graficables), por_pagina):
        lote = graficables[ini:ini + por_pagina]
        nr = math.ceil(len(lote) / ncols)
        fig, axes = plt.subplots(nr, ncols, figsize=(14, 4.1 * nr), squeeze=False)
        fig.patch.set_facecolor(FONDO)
        fig.suptitle(titulo, fontsize=15, fontweight="bold", color="#2c4855", y=.98)

        for ax, c in zip(axes.flat, lote):
            dibujar(ax, c, inventario, serie_univ)
        for ax in axes.flat[len(lote):]:
            ax.axis("off")

        fig.subplots_adjust(left=.13, right=.93, top=.90, bottom=.08, hspace=.65, wspace=.50)
        plt.show()

    if no_graf:
        subtitulo("Variables no graficadas por alta cardinalidad o tipo de dato", espacio_antes=True)
        series = {c: serie_univ(c) for c in no_graf}
        mostrar_tabla(
            pd.DataFrame([{
                "variable": etiqueta(c),
                "tipo": meta.loc[c, "tipo_analitico"],
                "n válido": series[c].notna().sum(),
                "faltantes": series[c].isna().sum(),
                "n únicos": series[c].nunique(dropna=True),
            } for c in no_graf]),
            alto=300,
            renombrar={
                "variable": "Variable",
                "tipo": "Tipo",
                "n válido": "n válido",
                "faltantes": "Faltantes",
                "n únicos": "n únicos",
            },
            ajustar_texto=True,
            una_linea=["Variable", "Tipo"],
            espacio=24,
            centrar=True,
        )
        
        
        
def ejecutar_familia(familia, titulo, inventario, base_full,
                     CODIGO_NO_REGISTRADO, ETIQUETA_NO_REGISTRADO):
    serie_univ = partial(serie_univariada, inventario=inventario, base_full=base_full,
                         CODIGO_NO_REGISTRADO=CODIGO_NO_REGISTRADO,
                         ETIQUETA_NO_REGISTRADO=ETIQUETA_NO_REGISTRADO)

    columnas = inventario.loc[
        (inventario["familia"] == familia) &
        (~inventario["tipo_analitico"].isin(["identificador", "variable auxiliar del EDA"])),
        "columna"
    ].tolist()

    resumen_cat = []
    for col in columnas:
        fila = inventario.loc[inventario["columna"].eq(col)]
        if fila.empty:
            continue
        tipo = fila["tipo_analitico"].iloc[0]
        if tipo not in ["categórica nominal", "categórica binaria", "ordinal"]:
            continue

        s = serie_univ(col)
        t = tabla_cat(s)
        if t.empty:
            continue

        validas = t.loc[~t["especial"]]
        principal = validas.iloc[0] if len(validas) else t.iloc[0]

        resumen_cat.append({
            "variable": etiqueta_bonita(col, inventario),
            "categoría predominante": principal["categoria"],
            "n": int(principal["n"]),
            "% válido": round(principal["pct"], 1),
            "faltantes": int(s.isna().sum()),
        })

    if resumen_cat:
        subtitulo("Resumen de variables categóricas")
        mostrar_tabla(
            pd.DataFrame(resumen_cat),
            alto=350,
            renombrar={
                "variable": "Variable",
                "categoría predominante": "Categoría predominante",
                "faltantes": "Faltantes",
            },
            formato={"% válido": "{:.1f}"},
            ajustar_texto=True,
            una_linea=["n", "% válido", "Faltantes"],
            anchos={"Variable": 320, "Categoría predominante": 260},
            espacio=24,
        )

    subtitulo("Distribuciones")
    analisis_familia(columnas, inventario, base_full, titulo,
                     CODIGO_NO_REGISTRADO, ETIQUETA_NO_REGISTRADO)
    
    
def serie_univariada(col, inventario, base_full, CODIGO_NO_REGISTRADO, ETIQUETA_NO_REGISTRADO):
    tipo = inventario.set_index("columna").at[col, "tipo_analitico"]
    if tipo.startswith("cuantitativa"):
        return serie_analitica(col, inventario, CODIGO_NO_REGISTRADO,
                               ETIQUETA_NO_REGISTRADO, base_full)
    return serie_etiquetada(col, base_full, CODIGO_NO_REGISTRADO,
                            ETIQUETA_NO_REGISTRADO, inventario,
                            incluir_especiales=True)


def resumen_numerico(s):
    x = pd.to_numeric(s, errors="coerce")
    v = x.dropna()
    n_total = len(x)
    media, de = v.mean(), v.std()
    q1, q3 = v.quantile(.25), v.quantile(.75)
    return pd.DataFrame([{
        "n_valido": int(v.size),
        "n_faltante": int(n_total - v.size),
        "porcentaje_faltante": round(100 * (n_total - v.size) / n_total, 1),
        "media": round(media, 2),
        "mediana": round(v.median(), 2),
        "DE": round(de, 2),
        "minimo": v.min(),
        "Q1": q1,
        "Q3": q3,
        "maximo": v.max(),
        "RIQ": q3 - q1,
        "CV": round(de / media, 2) if media else np.nan,
    }])


def tabla_frecuencias_numerica(s, bins=10):
    # Intervalos de amplitud uniforme (estadísticos, no puntos de corte clínicos)
    v = pd.to_numeric(s, errors="coerce").dropna()
    fi = pd.cut(v, bins=bins).value_counts(sort=False)
    n = fi.sum()
    t = pd.DataFrame({
        "intervalo": fi.index.astype(str),
        "fi": fi.values,
        "hi": (fi.values / n).round(3),
    })
    t["Fi"] = t["fi"].cumsum()
    t["porcentaje_acumulado"] = (100 * t["Fi"] / n).round(1)
    return t


def grafico_numerico(s, titulo):
    v = pd.to_numeric(s, errors="coerce").dropna()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2, 1]})

    ax1.hist(v, bins=tabla_frecuencias_numerica(v).shape[0], color="#315e73", edgecolor="white")
    ax1.axvline(v.median(), color="#8d806c", ls="--", lw=1.8, label=f"Mediana {v.median():.1f}")
    ax1.axvline(v.mean(), color="black", ls=":", lw=1.4, label=f"Media {v.mean():.1f}")
    ax1.set_title(titulo, loc="left", fontweight="bold")
    ax1.set_xlabel("Valor observado")
    ax1.set_ylabel("Pacientes")
    ax1.legend(frameon=False, fontsize=8)

    ax2.boxplot(v, vert=False, widths=.5, patch_artist=True,
                boxprops=dict(facecolor="#b8cbd1", color="#315e73"),
                medianprops=dict(color="#315e73", lw=2))
    ax2.set_yticks([])
    ax2.set_title("Diagrama de caja", loc="left", fontweight="bold")
    ax2.set_xlabel("Valor observado")

    plt.tight_layout()
    plt.show()            
    
def grafico_numerico_2(s, titulo, xlabel="Valor observado"):
    v = pd.to_numeric(s, errors="coerce").dropna()
    fig, ax = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}
    )

    # Histograma (arriba)
    ax[0].hist(v, bins=tabla_frecuencias_numerica(v).shape[0],
               color="#315e73", edgecolor="white")
    ax[0].axvline(v.median(), color="#8d806c", ls="--", lw=1.8,
                  label=f"Mediana {v.median():.1f}")
    ax[0].axvline(v.mean(), color="black", ls=":", lw=1.4,
                  label=f"Media {v.mean():.1f}")
    ax[0].set_title(titulo, loc="left", fontweight="bold")
    ax[0].set_ylabel("Pacientes")
    ax[0].legend(frameon=False, fontsize=8)

    # Boxplot horizontal (abajo)
    ax[1].boxplot(v, vert=False, widths=.5, patch_artist=True,
                  boxprops=dict(facecolor="#b8cbd1", color="#315e73"),
                  medianprops=dict(color="#315e73", lw=2))
    ax[1].set_yticks([])
    ax[1].set_xlabel(xlabel)

    plt.tight_layout()
    plt.show()    