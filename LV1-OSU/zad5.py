ham_rijeci = 0
ham_broj = 0
spam_rijeci = 0
spam_broj = 0
spam_usklicnici = 0

fhand = open("SMSSpamCollection.txt")

for line in fhand:
    line = line.strip()
    if line.startswith("ham"):
        ham_broj +=1
        ham_rijeci += len(line.split())-1
    elif line.startswith("spam"):
        spam_broj += 1
        spam_rijeci += len(line.split()) - 1
        if line.endswith("!"):
            spam_usklicnici += 1

fhand.close()

print("prosjek rijeci ham u porukama:", ham_rijeci/ham_broj)
print("prosjek rijeci spam u porukama:", spam_rijeci/spam_broj)
print("broj spam poruka sa usklicnikom:", spam_usklicnici)

