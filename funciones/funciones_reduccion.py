import pandas as pd

def convertir_numericas_probables(df, cols, umbral=.70):
    convertidas = []
    for c in cols:
        if c not in df.columns:
            continue

        s = df[c]
        n = s.notna().sum()

        if n == 0:
            continue

        sn = pd.to_numeric(
            s.astype(str)
             .str.replace(",", ".", regex=False)
             .str.replace(r"^\s*-\s*$", "", regex=True),
            errors="coerce"
        )

        if sn.notna().sum() / n >= umbral:
            df[c] = sn
            convertidas.append(c)

    return convertidas


