import matplotlib.pyplot as plt
import seaborn as sns

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

def especie_ilha(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="ilha", hue="espécie", palette="viridis")

    plt.title("Distribuição das Espécies por Ilha", fontsize=14, fontweight="bold")
    plt.xlabel("Ilha", fontsize=12)
    plt.ylabel("Quantidade de Pinguins", fontsize=12)
    plt.legend(title="Espécie")
    plt.show()

def bico_comp_alt(df):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df,
    x="comprimento_do_bico_mm",
    y="altura_do_bico_mm",
    hue="espécie",
    palette="Set1",
    s=70,  # Tamanho dos pontos
    )

    plt.title(
    "Comprimento vs. Profundidade do Bico",
    fontsize=14,
    fontweight="bold",
    )
    plt.xlabel("Comprimento do Bico (mm)")
    plt.ylabel("Profundidade do Bico (mm)")
    plt.show()

def sexo_peso(df):
    plt.figure(figsize=(8, 6))
    sns.violinplot(
    data=df,
    x="espécie",
    y="massa_corporal_g",
    hue="sexo",
    split=True,  # Une fêmea e macho no mesmo "violão"
    palette="Set2",
    inner="quart",  # Mostra as linhas de quartis
    )

    plt.title(
    "Distribuição da Massa Corporal por Espécie e Sexo",
    fontsize=14,
    fontweight="bold",
    )
    plt.xlabel("Espécie")
    plt.ylabel("Massa Corporal (g)")
    plt.show()
