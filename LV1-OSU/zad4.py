import string

fhand = open('song.txt')
sadrzaj = fhand.read()

sadrzajBezZnakova = sadrzaj.translate(str.maketrans('','',string.punctuation)).lower()
rijeci = sadrzajBezZnakova.split()

rjecnik = {}

for r in rijeci:
    rjecnik[r] = rjecnik.get(r,0) + 1

    jedinstveneRijeci = []

    for kljuc in rjecnik:
        if rjecnik[kljuc] ==1:
            jedinstveneRijeci.append(kljuc)

brojJedinstvenihRijeci = len(jedinstveneRijeci)
print(f"Broj jedinstvenih rijeci: {brojJedinstvenihRijeci}")
print(f"{jedinstveneRijeci}")

fhand.close()