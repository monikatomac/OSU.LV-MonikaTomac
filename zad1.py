sati = int(input("Radni sati: "))
cijena = float(input("eura/h: "))

#ukupno = sati * cijena

def total_euro():
    return sati*cijena

ukupno = total_euro()
print("Ukupno:", ukupno, "eura")