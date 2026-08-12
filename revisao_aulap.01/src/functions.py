def limpeza(df):
    #Remove linhas duplicadas
    df = df.drop_duplicates()

    # Mantém as linhas onde a quantidade de nulos <= metade do total de colunas
    df = df[df.isna().sum(axis=1) <= (len(df.columns) / 2)]
    return df


def traduzir(df):
    df.columns = ['espécie','ilha','comprimento_do_bico_mm',
                  'altura_do_bico_mm','comprimento_da_nadadeira_mm',
                  'massa_corporal_g','sexo']
    
    df = df.replace({'Male': 'Macho', 'Female': 'Fêmea'})
    return df

