import json

# TODO: Validation of JSON schema may require installation of an additional library
with open('Config.json', 'r') as jsonFile:
    configs = json.load(jsonFile)
