from .funciones_normalizacion import normalizar_para_match, normalizar_columna
import pandas as pd
import re
import openpyxl


def dict_fuente(df, mapa, fuente):
    dic = mapa.copy()
    dic["fuente"] = fuente
    dic["dtype"] = [str(df[c].dtype) for c in df.columns]
    dic["n_unicos"] = [df[c].nunique(dropna=True) for c in df.columns]
    dic["n_faltantes"] = [df[c].isna().sum() for c in df.columns]
    dic["pct_faltantes"] = [df[c].isna().mean() * 100 for c in df.columns]
    return dic


def detectar_grupo(original, fuente, demo_norm_match, neuro_norm_match):
    clave = normalizar_para_match(original)
    if fuente == "neuropsicologia":
        if clave in demo_norm_match:
            return "demografica_social"
        if clave in neuro_norm_match:
            return "neuropsicologica"
    if fuente == "moca" and clave == "moca":
        return "neuropsicologica"
    if fuente == "hxgen":
        return "clinica_hxgen"
    return "otra"

def detectar_subgrupo(original, fuente, grupos_demo_norm, grupos_neuro_norm):
    clave = normalizar_para_match(original)
    if fuente == "neuropsicologia":
        for grupo, variables in grupos_demo_norm.items():
            if clave in variables:
                return grupo
        for grupo, variables in grupos_neuro_norm.items():
            if clave in variables:
                return grupo
    if fuente == "moca" and clave == "moca":
        return "cognicion_global"
    if fuente == "hxgen":
        return "historia_clinica_genetica"
    return ""

def asignar_uso(fila):
    grupo = fila["grupo_principal"]
    subgrupo = fila["subgrupo"]
    variable = normalizar_para_match(fila["Original"])
    if "edad de inicio" == variable:
        return "Outcome principal"
    if "edad de inicio actualizada" == variable:
        return "Complemento para completar AOO faltante"
    if grupo == "demografica_social":
        return "Caracterización / covariable candidata"
    if grupo == "neuropsicologica":
        return "Caracterización clínica / reducción / análisis secundario"
    if grupo == "clinica_hxgen":
        return "Caracterización clínica / covariable candidata"
    return "Revisar"



def asignar_leakage(fila):

    fuente = fila["fuente"]
    grupo = fila["grupo_principal"]
    subgrupo = fila["subgrupo"]
    variable = normalizar_para_match(fila["Original"])

    # 1. NO APLICA
    if variable in {
        "codigo",
        "edad de inicio",
        "edad de inicio actualizada"
    }:
        return "No aplica"

    # 2. BAJO RIESGO: Variables razonablemente previas al inicio
    bajo_riesgo = {
        "genero",
        "mes y anio de nacimiento",
        "idioma",
        "otro idoma",
        "nivel educativo",
        "lateralidad",
        "hay miembros afectados de la familia padres biologicos hermanos consanguineos completos comparten ambos padres o hijos biologicos",
        "familiar afectado"
    }
    if variable in bajo_riesgo:
        return "Bajo"

    # 3. ALTO RIESGO: Medidas claramente posteriores o contemporáneas al estado de enfermedad
    alto_riesgo = {
        "mmse",
        "moca",
        "fast",
        "esc deterioroglobal",
        "quejas subjetivas",
        "barthel",
        "lawton y brody",
        "ikatz",
        "yesavage",
        "enfermedad actual",
        "que sintomas presenta",
        "el paciente necesita apoyo para realizar las siguientes tareas",
        "actualmente se encuentra en algun tratamiento farmacologico",
        "si la respuesta anterior fue si que medicamentos esta consumiendo"
    }
    if variable in alto_riesgo:
        return "Alto"

    # 4. POTENCIAL: Batería neuropsicológica completa
    if grupo == "neuropsicologica":
        return "Potencial"

    # 5. REVISAR TEMPORALIDAD: Comorbilidades, hábitos, cirugía, medicación, residencia, antropometría, etc.
    if fuente == "hxgen":
        return "Revisar temporalidad"

    if subgrupo in {
        "cardiovasculares_cerebrovasculares",
        "neurologicos",
        "metabolicos_otros",
        "psiquiatricos",
        "tabaquismo",
        "medicamentos",
        "antropometria",
        "informante_red_apoyo"
    }:
        return "Revisar temporalidad"
    if grupo == "demografica_social":
        return "Bajo / revisar temporalidad"
    return "Revisar"



def asignar_descripcion(fila): #Asigna una descripción preliminar a cada variable.

    variable = normalizar_para_match(fila["Original"])
    fuente = fila["fuente"]

    # 1. IDENTIFICACIÓN Y VARIABLES SOCIODEMOGRÁFICAS
    descripciones = {
        "codigo":
            "Identificador asignado al participante.",
        "genero":
            "Género o sexo registrado del participante.",
        "dx":
            "Diagnóstico registrado para el participante.",
        "mes y ano de nacimiento":
            "Mes y año de nacimiento del participante.",
        "edad":
            "Edad registrada del participante.",
        "idioma":
            "Idioma principal registrado para el participante.",
        "otro idoma":
            "Registro de otro idioma reportado por el participante.",
        "nivel educativo":
            "Nivel educativo alcanzado por el participante.",
        "estado civil":
            "Estado civil registrado del participante.",
        "estado habitacional":
            "Situación habitacional registrada del participante.",
        "nivel de independencia":
            "Nivel de independencia registrado para realizar actividades cotidianas.",
        "tipo de residencia":
            "Tipo de residencia en la que vive el participante.",
        "lateralidad":
            "Lateralidad predominante registrada para el participante.",

        # 2. INFORMANTE Y RED DE APOYO
        "relacion del informante":
            "Relación o parentesco del informante con el participante.",
        "tiempo anos de conocerce":
            "Tiempo durante el cual el informante ha conocido al participante.",
        "vive el informante con el sujeto":
            "Indica si el informante convive con el participante.",
        "si el informante no vive con el sujeto con que frecuencia visita personalmente al sujeto":
            "Frecuencia con la que el informante visita personalmente al participante.",
        "si el informante no vive con el sujeto con que frecuencia tiene contacto telefonico con el":
            "Frecuencia de contacto telefónico entre el informante y el participante.",

        # 3. HISTORIA FAMILIAR Y EDAD DE INICIO
        "familiar afectado":
            "Registro del familiar reportado como afectado por la enfermedad.",
        "edad de inicio":
            "Edad registrada como inicio de la enfermedad.",
        "edad de inicio actualizada":
            "Valor actualizado o complementario de la edad de inicio.",

        # 4. ANTROPOMETRÍA
        "peso":
            "Peso registrado del participante.",
        "talla":
            "Talla o estatura registrada del participante.",

        # 5. MEDICAMENTOS
        "medicamento del sujeto 1":
            "Primer medicamento registrado para el participante.",
        "medicamento del sujeto 2":
            "Segundo medicamento registrado para el participante.",
        "medicamento del sujeto 3":
            "Tercer medicamento registrado para el participante.",
        "medicamento del sujeto 4":
            "Cuarto medicamento registrado para el participante.",
        "medicamento del sujeto 5":
            "Quinto medicamento registrado para el participante.",
        "medicamento del sujeto 6":
            "Sexto medicamento registrado para el participante.",
        "medicamento del sujeto 7":
            "Séptimo medicamento registrado para el participante.",
        "medicamento del sujeto 8":
            "Octavo medicamento registrado para el participante.",
        "medicamento del sujeto 9":
            "Noveno medicamento registrado para el participante.",
            
        # 6. TABAQUISMO
        "el sujeto ha fumado en los pasados 30 dias":
            "Indica si el participante reportó consumo reciente de cigarrillos.",
        "el sujeto ha fumado mas de 100 cigarros / cigarrillos en su vida":
            "Indica si el participante reportó haber fumado más de 100 cigarrillos durante su vida.",
        "numero total de anos que ha fumado el sujeto 99 = desconocido:":
            "Número total de años de consumo de cigarrillos registrado para el participante.",

        # 7. CARDIOVASCULARES Y CEREBROVASCULARES
        "ataque cardiaco o paro cardiaco":
            "Registro de antecedente de ataque cardíaco o paro cardíaco.",
        "fibrilacion auricular":
            "Registro de antecedente o presencia de fibrilación auricular.",
        "angioplastia/ endarterectomia/ stent":
            "Registro de antecedentes de procedimientos vasculares.",
        "bypass cardiaco/coronario":
            "Registro de antecedente de bypass cardíaco o coronario.",
        "marcapasos y/o desfibrilador":
            "Registro del uso o antecedente de marcapasos o desfibrilador.",
        "insuficiencia cardiaca congestiva":
            "Registro de antecedente o presencia de insuficiencia cardíaca congestiva.",
        "enfermedad cerebro-vascular ecv":
            "Registro de antecedente o presencia de enfermedad cerebrovascular.",


        # 8. NEUROLÓGICOS
        "parkinson":
            "Registro de antecedente o presencia de enfermedad de Parkinson.",
        "convulsiones":
            "Registro de antecedente o presencia de convulsiones.",
        "traumatismo de craneo-encefalico tce":
            "Registro de antecedente de traumatismo craneoencefálico.",

        # 9. OTRAS CONDICIONES MÉDICAS
        "diabetes":
            "Registro de antecedente o presencia de diabetes.",
        "cancer":
            "Registro de antecedente o presencia de cáncer.",
        "hipercholesterolemia":
            "Registro de antecedente o presencia de hipercolesterolemia.",
        "hipertension arterial":
            "Registro de antecedente o presencia de hipertensión arterial.",
        "niveles de vitamina d":
            "Información registrada relacionada con los niveles de vitamina D.",
        "deficiencia de b12":
            "Registro de antecedente o presencia de deficiencia de vitamina B12.",
        "enfermedad tiroidea":
            "Registro de antecedente o presencia de enfermedad tiroidea.",
        "artritis":
            "Registro de antecedente o presencia de artritis.",
        "artrosis":
            "Registro de antecedente o presencia de artrosis.",
        "osteoporosis":
            "Registro de antecedente o presencia de osteoporosis.",
        "incontinencia — urinaria":
            "Registro de presencia de incontinencia urinaria.",
        "incontinencia — fecal":
            "Registro de presencia de incontinencia fecal.",
        "apnea del sueno u otro":
            "Registro de apnea del sueño u otra condición relacionada.",

        # 10. PSIQUIÁTRICOS
        "trastorno de estres postraumatico tept":
            "Registro de antecedente o presencia de trastorno de estrés postraumático.",
        "trastorno bipolar":
            "Registro de antecedente o presencia de trastorno bipolar.",
        "esquizofrenia":
            "Registro de antecedente o presencia de esquizofrenia.",
        "depresion":
            "Registro de antecedente o presencia de depresión.",
        "ansiedad":
            "Registro de antecedente o presencia de ansiedad.",
        "trastorno obsesivo-compulsivo toc":
            "Registro de antecedente o presencia de trastorno obsesivo-compulsivo.",
        "eeg":
            "Información registrada relacionada con electroencefalograma.",


        # 11. EVALUACIÓN NEUROPSICOLÓGICA
        "hay valoracion nps anterior":
            "Indica si existe una valoración neuropsicológica previa.",
        "si tiene anteriormente valoracion nps en que fecha se aplico":
            "Fecha registrada de una valoración neuropsicológica previa.",
        "mmse":
            "Puntaje obtenido en una prueba de evaluación cognitiva global.",
        "cmweschler":
            "Resultado registrado de una prueba relacionada con memoria.",


        # 12. FLUIDEZ VERBAL
        "fvsa":
            "Resultado de una tarea de fluidez verbal.",
        "perdida categoria fvsa":
            "Registro de pérdida de categoría durante la tarea FVSa.",
        "perseveraciones fvsa":
            "Número o registro de perseveraciones observadas en FVSa.",
        "fvsc":
            "Resultado de una tarea de fluidez verbal.",
        "perdida categoria fvsc":
            "Registro de pérdida de categoría durante la tarea FVSc.",
        "perseveraciones fvsc":
            "Número o registro de perseveraciones observadas en FVSc.",
        "fvff":
            "Resultado de una tarea de fluidez verbal.",
        "perdida categoria fvff":
            "Registro de pérdida de categoría durante la tarea FVFf.",
        "perseveraciones fvff":
            "Número o registro de perseveraciones observadas en FVFf.",
        "fvfa":
            "Resultado de una tarea de fluidez verbal.",
        "perdida categoria fvfa":
            "Registro de pérdida de categoría durante la tarea FVFa.",
        "perseveraciones fvfa":
            "Número o registro de perseveraciones observadas en FVFa.",
        "fvfs":
            "Resultado de una tarea de fluidez verbal.",
        "perdida categoria fvfs":
            "Registro de pérdida de categoría durante la tarea FVFs.",
        "perseveraciones fvfs":
            "Número o registro de perseveraciones observadas en FVFs.",

        # 13. FIGURA COMPLEJA DE REY Y LENGUAJE
        "fcrocop":
            "Resultado obtenido en la tarea de copia de una figura compleja.",
        "fcroevo":
            "Resultado obtenido en la tarea de evocación de una figura compleja.",
        "token":
            "Resultado obtenido en una prueba relacionada con comprensión del lenguaje.",
            
        # 14. APRENDIZAJE Y MEMORIA - REY
        "taavrey1":
            "Resultado del primer ensayo de una prueba de aprendizaje y memoria verbal.",
        "taavrey2":
            "Resultado del segundo ensayo de una prueba de aprendizaje y memoria verbal.",
        "taavrey3":
            "Resultado del tercer ensayo de una prueba de aprendizaje y memoria verbal.",
        "taavrey4":
            "Resultado del cuarto ensayo de una prueba de aprendizaje y memoria verbal.",
        "taavrey5":
            "Resultado del quinto ensayo de una prueba de aprendizaje y memoria verbal.",
        "taavreyb1":
            "Resultado registrado en la lista o ensayo de interferencia de la prueba.",
        "taavrey6":
            "Resultado de un ensayo posterior de recuerdo en la prueba de memoria verbal.",
        "taavrey7":
            "Resultado de un ensayo posterior de recuerdo en la prueba de memoria verbal.",
        "taavrey rec si":
            "Número de respuestas afirmativas registradas en la fase de reconocimiento.",
        "taavrey rec no":
            "Número de respuestas negativas registradas en la fase de reconocimiento.",


        # 15. TEST DE BOSTON
        "tbn espontanea":
            "Resultado de denominación espontánea registrado en el Test de Boston.",
        "tbn semantica":
            "Resultado registrado con ayuda o clave semántica en el Test de Boston.",
        "tbn total":
            "Puntuación total registrada en el Test de Boston.",

        # 16. FUNCIÓN EJECUTIVA Y ATENCIÓN
        "tspal":
            "Resultado registrado en la condición de palabras de una prueba de atención y control ejecutivo.",
        "tscol":
            "Resultado registrado en la condición de colores de una prueba de atención y control ejecutivo.",
        "tspalcol":
            "Resultado registrado en una condición combinada de palabras y colores.",
        "tmta":
            "Resultado de la parte A de una prueba de seguimiento y atención.",
        "tmtb":
            "Resultado de la parte B de una prueba de seguimiento y función ejecutiva.",
        "teac":
            "Resultado registrado en una prueba relacionada con atención y funcionamiento ejecutivo.",

        # 17. MEMORIA SEMÁNTICA Y CONOCIMIENTO
        "tsmydoral":
            "Resultado de una evaluación oral relacionada con memoria semántica o conocimiento.",
        "tsmyd escr":
            "Resultado de una evaluación escrita relacionada con memoria semántica o conocimiento.",
        "benton":
            "Resultado registrado en la prueba de Benton.",
        "reloj":
            "Resultado obtenido en la prueba del dibujo del reloj.",

        # 18. WISCONSIN
        "twiscat":
            "Número o resultado de categorías alcanzadas en la prueba Wisconsin.",
        "twisenp":
            "Registro de errores no perseverativos en la prueba Wisconsin.",
        "twisep":
            "Registro de errores perseverativos en la prueba Wisconsin.",
        "twiscorresctas":
            "Número de respuestas correctas registrado en la prueba Wisconsin.",

        # 19. FUNCIONAL, AFECTIVO Y QUEJAS
        "quejas subjetivas":
            "Registro de quejas cognitivas subjetivas reportadas.",
        "yesavage":
            "Resultado obtenido en una escala de evaluación de síntomas depresivos.",
        "barthel":
            "Resultado de una escala de independencia en actividades de la vida diaria.",
        "lawton y brody":
            "Resultado de una escala de actividades instrumentales de la vida diaria.",
        "ikatz":
            "Resultado registrado en una escala de funcionamiento en actividades cotidianas.",
        "fast":
            "Resultado de una escala utilizada para valorar el estado funcional asociado al deterioro.",
        "esc deterioroglobal":
            "Resultado de una escala de valoración global del deterioro.",

        # 20. MOCA
        "moca":
            "Puntaje obtenido en una prueba de evaluación cognitiva global.",

        # 21. HXGEN
        "ciudad o municipio de residencia":
            "Ciudad o municipio de residencia registrado para el participante.",
        "enfermedad actual":
            "Enfermedad o condición actual registrada para el participante.",
        "que sintomas presenta":
            "Síntomas registrados para el participante.",
        "el paciente necesita apoyo para realizar las siguientes tareas":
            "Actividades para las que se reporta que el participante requiere apoyo.",
        "indique que estudios previos le han realizado.":
            "Estudios o evaluaciones previas realizadas al participante.",
        "si le han realizado estudios previos encontraron hallazgos cuales":
            "Hallazgos reportados en estudios previos.",
        "antecedentes":
            "Antecedentes clínicos registrados para el participante.",
        "actualmente se encuentra en algun tratamiento farmacologico":
            "Indica si el participante se encuentra actualmente en tratamiento farmacológico.",
        "si la respuesta anterior fue si que medicamentos esta consumiendo":
            "Medicamentos que el participante reporta consumir actualmente.",
        "usted fuma":
            "Indica si el participante reporta consumo actual de cigarrillos.",
        "si respuesta anterior fue afirmativa hace cuantos anos fuma":
            "Tiempo de consumo de cigarrillos reportado por el participante.",
        "con que frecuencia consume alcohol":
            "Frecuencia de consumo de alcohol reportada por el participante.",
        "le han realizado algun procedimiento quirurgico":
            "Indica si el participante reporta antecedentes de procedimientos quirúrgicos.",
        "si la respuesta anterior fue si. que cirugia le han realizado":
            "Procedimiento quirúrgico reportado por el participante.",
        "en que fecha le realizaron la cirugia":
            "Fecha registrada del procedimiento quirúrgico.",
        "sufre de algun tipo de alergia":
            "Indica si el participante reporta algún tipo de alergia.",
        "si la respuesta anterior es si. a que es alergico":
            "Sustancia o elemento al que se reporta alergia.",
        "peso en kilogramo":
            "Peso del participante registrado en kilogramos.",
        "altura en centimetro":
            "Altura del participante registrada en centímetros.",
        "alguien mas en su familia tiene o tuvo alzheimer":
            "Indica si se reportan familiares con antecedente de enfermedad de Alzheimer.",
        "si la respuesta anterior es afirmativa. cuantas personas han tenido o tienen alzheimer en su familia ?":
            "Número de familiares reportados con antecedente de enfermedad de Alzheimer.",
        "que parentesco tienen con usted":
            "Parentesco de los familiares reportados con enfermedad de Alzheimer."
    }

    # Las variables con encabezados muy largos se manejan por búsqueda de palabras
    if (
        "miembros afectados de la familia" in variable
        and "padres biologicos" in variable
    ):
        return "Indica si existen familiares biológicos reportados como afectados por la enfermedad."
    if "trastornos neuropsiquiatricos del desarrollo" in variable:
        return "Registro de antecedentes o presencia de trastornos neuropsiquiátricos del desarrollo."
  
    if variable in descripciones:
        return descripciones[variable]



# Incorporar al diccionario las codificaciones documentadas en los comentarios de Excel.
def _normalizar_etiqueta_diccionario(texto):
    s = re.sub(r"\s+", " ", str(texto)).strip(" .;:-")
    base = normalizar_para_match(s)
    if base == "si": return "Sí"
    if base == "no": return "No"
    if "desconocid" in base: return "Desconocido"
    return s[:1].upper() + s[1:] if s else s

def _extraer_codificacion_diccionario(texto, encabezado=""):
    lineas=[]
    for linea in str(texto or "").replace("\r", "\n").split("\n"):
        linea=re.sub(r"\s+", " ", linea).strip()
        if not linea or linea == "======" or linea.startswith("ID#"): continue
        if re.search(r"\(20\d{2}-\d{2}-\d{2}", linea): continue
        if re.fullmatch(r"[A-Za-zÁÉÍÓÚáéíóúÑñ ]+:", linea): continue
        lineas.append(linea)
    pares=[]; ultimo=None
    for linea in lineas:
        inicios=list(re.finditer(r"(?<![\w/.-])(GDS\s*)?(-?\d+(?:\s*-\s*\d+)?)\s*(?:[:.=]\s*|\s+)", linea, re.I))
        if inicios:
            for j,m in enumerate(inicios):
                codigo=re.sub(r"\s+", "", m.group(2))
                fin=inicios[j+1].start() if j+1 < len(inicios) else len(linea)
                etiqueta=linea[m.end():fin].strip(" .;:-")
                if etiqueta:
                    pares.append([codigo, _normalizar_etiqueta_diccionario(etiqueta)])
                    ultimo=len(pares)-1
        elif ultimo is not None and not re.match(r"^[A-ZÁÉÍÓÚ ]{4,}$", linea):
            pares[ultimo][1]=re.sub(r"\s+", " ", pares[ultimo][1]+" "+linea).strip()
    for m in re.finditer(r"(?<!\d)(\d+)\s*=\s*([^):,;]+)", str(encabezado), re.I):
        pares.append([m.group(1), _normalizar_etiqueta_diccionario(m.group(2))])
    salida={}
    for codigo, etiqueta in pares:
        if codigo not in salida: salida[codigo]=etiqueta
    return salida

def extraer_metadatos_excel_diccionario(PATH_NEURO, PATH_MOCA, PATH_HXGEN):
    fuentes=[("neuropsicologia", PATH_NEURO), ("moca", PATH_MOCA), ("hxgen", PATH_HXGEN)]
    registros=[]
    for fuente, ruta in fuentes:
        wb=openpyxl.load_workbook(ruta, data_only=False, read_only=False)
        for ws in wb.worksheets:
            for celda in ws[1]:
                if celda.value is None: continue
                original=str(celda.value)
                comentario=celda.comment.text.strip() if celda.comment else ""
                registros.append({
                    "fuente":fuente, "normalizada":normalizar_columna(original),
                    "comentario_excel":comentario,
                    "etiquetas_codigos_dict":_extraer_codificacion_diccionario(comentario, original)
                })
    return pd.DataFrame(registros)    


from IPython.display import display, HTML

def mostrar_diccionario(df, altura=500, anchas=("Descripción", "Codificación")):
    estilo = (
        df.style
        .hide(axis="index")
        .set_table_attributes('class="dataframe"')          # estilo del tema
        .format(precision=2)
        .set_table_styles([
            {"selector": "",
             "props": "width: max-content; table-layout: auto;"},
            {"selector": "th",
             "props": "position: sticky; top: 0; z-index: 1; "
                      "text-align: center !important; white-space: normal; "
                      "min-width: 110px; max-width: 220px; "
                      "background-color: var(--pst-color-background, var(--jp-layout-color0, white));"},
            {"selector": "td",
             "props": "text-align: left; vertical-align: top; white-space: normal; "
                      "overflow-wrap: break-word; min-width: 110px; max-width: 260px; padding: 6px 10px;"},
        ])
    )
    columnas_anchas = [c for c in anchas if c in df.columns]
    if columnas_anchas:
        estilo = estilo.set_properties(subset=columnas_anchas,
                                       **{"min-width": "320px", "max-width": "480px"})

    display(HTML(
        f'<div style="max-height: {altura}px; overflow: auto; margin: 0.4rem 0 1.5rem 0;">'
        f'{estilo.to_html()}</div>'
    ))