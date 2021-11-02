class Pet:
    animal = ""
    name = ""
    place = ""
    dateOfLoss = ""
    findeFixPlakatNr = ""
    imagePetPath = "F:\PYTHON\Projekte\FindTheDog\Scraping\\"
    def __init__(self,animal,name,place,dateOfLoss,findeFixPlakatNr,pathImage):
        self.animal = animal
        self.name = name
        self.place = place
        self.dateOfLoss = dateOfLoss
        self.findeFixPlakatNr = findeFixPlakatNr
        self.imagePetPath += pathImage

    def outAll(self):
        print(self.animal,self.name,self.place,self.dateOfLoss,self.findeFixPlakatNr, self.imagePetPath)
    def getImagePath(self):
        return self.imagePetPath
