import dataset as d
import src.functions as f

'''EDA'''
print("Visão geral das 5 primeiras linhas:")
print(d.df.head())
print()
print("Dimensão do df:", d.df.shape)
print()
print("Informações gerais:")
print(d.df.info())
print()

'''LIMPEZA'''

f.limpeza(d.df)
print(f.traduzir(d.df))
f.especie_ilha(d.df)
f.bico_comp_alt(d.df)
f.sexo_peso(d.df)
f.matriz(d.df)
