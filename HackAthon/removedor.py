#remover 

with open("Wow... Loved this place. Crust is not good.txt", "r") as arq:
	with open("texto2.txt", 'w+') as arq2:
		for linha in arq:
			arq2.write(linha.replace("\n","").replace("\t"," "))
		arq2.seek(0)
		print(arq2.read())

