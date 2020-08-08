import wikipedia
import os
import json

class WikipediaSave:

    def __init__(self):
        pass

    def saveData(self, entity):
        summary = wikipedia.summary(entity)
        filename = entity.replace(" ", "_") + ".txt"

        f = open(os.path.join(os.getcwd(), "Wiki_data/{}".format(filename)), "w+")
        f.write(summary)
        f.close()

        return json.dumps({
            "entity" : entity,
            "message" : "Success in file writing",
            "status" : 200
        })
