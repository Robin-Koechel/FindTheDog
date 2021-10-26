class Pet:
    from PIL import Image
    animal = ""
    name = ""
    place = ""
    dateOfLoss = ""
    findeFixPlakatNr = ""
    imagePet = ""
    def __init__(self,animal,name,place,dateOfLoss,findeFixPlakatNr,pathImage):
        self.animal = animal
        self.name = name
        self.place = place
        self.dateOfLoss = dateOfLoss
        self.findeFixPlakatNr = findeFixPlakatNr
        self.imagePet = self.Image.open(pathImage)

    def outAll(self):
        print(self.animal,self.name,self.place,self.dateOfLoss,self.findeFixPlakatNr)
